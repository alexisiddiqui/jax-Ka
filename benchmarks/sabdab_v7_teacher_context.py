"""Qualify the sole paired cross-source native match at exported teacher scope."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

import numpy as np
import pyarrow.parquet as pq

from pkabench.runtime import require_compute,atomic_json,digest,config_hash
from pkabench.sabdab_labels import export_states,qualify_labels
from pkatrain.augmentation_store import AugmentationStore


def read(path):return json.loads(Path(path).read_text())


def normalise_pdb(text,mapping,key_map):
    forward={(c,int(n)):(oc,int(on),oi) for c,n,oc,on,oi in mapping['residues']}
    assigned={(c,int(n)):(kind,oc,int(on)) for c,n,kind,oc,on in mapping['termini']}
    atoms={}
    for line in text.splitlines():
        if line[:6].strip() not in ['ATOM','HETATM']:continue
        key=(line[21],int(line[22:26]));kind='residue'
        if key in assigned:
            kind,c,n=assigned[key];source=forward[c,n]
        else:source=forward[key]
        chain,number,insertion=key_map[source]
        atom=(chain,number,insertion,kind,line[17:20].strip(),line[12:16].strip(),line[76:78].strip())
        if atom in atoms:raise ValueError('Duplicate teacher atom key')
        atoms[atom]=tuple(float(line[s:s+8]) for s in [30,38,46])
    return atoms


def run(args):
    require_compute(threads=2,allow_comp1400=True)
    from biotite.structure.io import pdbx
    import biotite.structure as struc
    from jaxpropka.topology import load_topology
    comparisons=pq.read_table(args.root/'reuse/context-comparisons.parquet').to_pylist()
    paired=[r for r in comparisons if r['candidate_parent'].startswith('pinder:') and r['status']=='rigid_heavy_atom_identity']
    results=[]
    with AugmentationStore(args.root/'sabdab/prepared') as store:
        for pair in paired:
            uid=pair['unit_id'];cid=pair['candidate_parent'].split(':',1)[1]
            frame=store.raw(uid);meta=frame['metadata']['structure'];result=dict(unit_id=uid,candidate_parent=pair['candidate_parent'],teacher_reuse_approved=False)
            try:
                raw=frame['arrays']['canonical_cif'].tobytes().decode()
                atoms=pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(raw)),model=1,altloc='occupancy',include_bonds=True)
                topology=load_topology(atoms,gap_policy='cap',freeze_disulfides=True)
                real={tuple((x['chain'],x['number'],x['insertion'],x['group'])) for x in frame['metadata']['result']['true_termini']}
                original=args.runtime/'pretraining/pinder-pkai-fv-v1/labels-v3/entries'/cid
                curated=args.baseline/'fv-build/pretraining/pinder-pkai-fv-v1/labels-v3/entries'/cid
                native_meta=read(curated/'meta.json');original_meta=read(original/'meta.json')
                for name,expected in native_meta['outputs'].items():
                    if digest(curated/name)!=expected:raise ValueError('Curated teacher output pin changed')
                if digest(original/'teacher-inputs.tar.gz')!=original_meta['outputs']['teacher-inputs.tar.gz']:
                    raise ValueError('Original teacher export archive pin changed')
                teacher_manifest=read(args.runtime/'pretraining/pinder-pkai-fv-v1/labels-v3/manifest.json')
                if digest(args.runtime/'pretraining/pinder-pkai-fv-v1/labels-v3/manifest.json')!=native_meta['label_manifest_sha256']:
                    raise ValueError('Teacher protocol manifest pin changed')
                pilot_request=read(args.root.parent/'sabdab-v7-antibody-scope-fix-v1/labels-v1/entries/pdb_00001tjh_H_L/request.json')
                for path,expected in pilot_request['teacher_input_pins'].items():
                    if teacher_manifest['input_pins'].get(path)!=expected or digest(path)!=expected:
                        raise ValueError('Teacher package/model contexts differ')
                with __import__('gzip').open(curated/'AB.cif.gz','rt') as f:native_cif=pdbx.CIFFile.read(f)
                native=pdbx.get_structure(native_cif,model=1,altloc='occupancy',use_author_fields=True,include_bonds=True)
                native=load_topology(native,gap_policy='cap',freeze_disulfides=True).atoms
                chain_map={x['right']:x['left'] for x in pair['chain_map']};native_map={}
                for chain in dict.fromkeys(map(str,native.chain_id)):
                    sub=native[native.chain_id==chain]
                    for ordinal,i in enumerate(struc.get_residue_starts(sub),1):
                        native_map[chain,int(sub.res_id[i]),str(sub.ins_code[i]).strip()]=(chain_map[chain],ordinal,'')
                if {(native_map[tuple(k[:3])]+(k[3],)) for k in native_meta['real_termini']}!=real:
                    raise ValueError('True-terminal teacher context differs despite identical heavy coordinates')
                own_map={(k.chain,k.number,k.insertion):(k.chain,k.number,k.insertion) for k in topology.keys}
                state_signatures={}
                with tarfile.open(original/'teacher-inputs.tar.gz','r:gz') as archive, tempfile.TemporaryDirectory() as temporary:
                    old_mapping=json.loads(archive.extractfile('mapping.json').read())
                    mappings=export_states(topology,meta,real,Path(temporary))
                    for state in ['AB','A','B']:
                        old=archive.extractfile(state+'.pdb').read()
                        if hashlib.sha256(old).hexdigest()!=native_meta['teacher_input_sha256'][state]:raise ValueError('Pinned teacher state changed')
                        a=normalise_pdb(old.decode(),old_mapping[state],native_map)
                        b=normalise_pdb((Path(temporary)/(state+'.pdb')).read_text(),mappings[state],own_map)
                        if a!=b:raise ValueError('Exported teacher chemical atoms/termini/coordinates differ: '+state)
                        state_signatures[state]=config_hash([[list(k),list(v)] for k,v in sorted(a.items())])
                labels=copy.deepcopy(read(curated/'labels.json'))
                for model in ['pkai','pkai_plus']:
                    for state,values in labels[model].items():
                        labels[model][state]=[[*native_map[tuple(row[:3])],*row[3:]] for row in values]
                sites=copy.deepcopy(frame['metadata']['sites']);counts=qualify_labels(sites,labels)
                result.update(status='same_exported_teacher_context_verified',teacher_reuse_approved=True,
                    native_teacher_entry=str(curated),counts=counts,teacher_state_signatures=state_signatures,
                    residue_key_map=[dict(native=list(k),sabdab=list(v)) for k,v in sorted(native_map.items())],
                    labels_sha256=digest(curated/'labels.json'),model_pins=pilot_request['teacher_input_pins'],
                    protocol='fixed-coordinate CPU pKAI/pKAI+ AB/A/B; exact normalised teacher atom/terminal/coordinate equality',
                    teacher_context_id=config_hash(dict(states=state_signatures,models=pilot_request['teacher_input_pins'])))
            except Exception as exc:result.update(status='teacher_context_reuse_withheld',error=repr(exc))
            results.append(result)
    report=dict(status='paired_cross_source_teacher_context_checks_complete',records=results,
        approved=sum(r['teacher_reuse_approved'] for r in results),new_teacher_predictions=False,
        geometry_matches_do_not_authorise_other_reuses=True,
        input_pins={str(args.root/'reuse/context-comparisons.parquet'):digest(args.root/'reuse/context-comparisons.parquet')})
    atomic_json(args.root/'reuse/teacher-context.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','runtime','baseline']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
