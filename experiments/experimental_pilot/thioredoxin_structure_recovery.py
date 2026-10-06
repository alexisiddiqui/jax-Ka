"""Prepare state/construct-correct human-thioredoxin recovery structures."""
import json
import os
import urllib.request
from pathlib import Path

from pkabench.runtime import require_compute, atomic_json, digest

require_compute()
import numpy as np
from biotite.structure.io import pdbx
from pkabench.conformers import resolve
from pkabench.prep import CANONICAL, complete, topology, write_cif
from pkabench.supervision import inventory

runtime = Path(os.environ['PKABENCH_RUNTIME'])
out = runtime/'experimental/pkadr-human-thioredoxin-recovery-v4'
source = runtime/'experimental/pkadr-full-v1/sources/1ERT.cif'
targets = {
    '1TRW': dict(records=[916], site=('ASP',26), redox='reduced', construct='C62A/C69A/C73A/M74T'),
    '1TRS': dict(records=[917], site=('ASP',26), redox='oxidized', construct='C62A/C69A/C73A/M74T'),
    '4TRX': dict(records=[918], site=('CYS',32), redox='reduced', construct='M74T'),
}

out.mkdir(parents=True, exist_ok=False)

def sequence(cif, chain):
    asym=cif.block['struct_asym']; poly=cif.block['entity_poly']
    entity=dict(zip(asym['id'].as_array(str),asym['entity_id'].as_array(str)))[chain]
    return ''.join(dict(zip(poly['entity_id'].as_array(str),poly['pdbx_seq_one_letter_code_can'].as_array(str)))[entity].split())

reference_cif=pdbx.CIFFile.read(source)
reference_sequence=sequence(reference_cif,'A')
manifests=[]
for pdb, spec in targets.items():
    work=out/pdb; work.mkdir()
    raw=work/f'{pdb}.cif'
    url=f'https://files.rcsb.org/download/{pdb}.cif'
    raw.write_bytes(urllib.request.urlopen(url,timeout=60).read())
    cif=pdbx.CIFFile.read(raw); cat=cif.block['atom_site']
    polymer=(cat['label_seq_id'].as_array(str)!='.')&(cat['label_seq_id'].as_array(str)!='?')
    selected=sorted(set(cat['label_asym_id'].as_array(str)[polymer&(cat['auth_asym_id'].as_array(str)=='A')]))
    if len(selected)!=1: raise ValueError(f'{pdb}: author chain A maps to {selected}')
    chain=selected[0]
    models=cat['pdbx_PDB_model_num'].as_array(str); first=models[0]
    new=pdbx.CIFCategory()
    for name in cat: new[name]=pdbx.CIFColumn(cat[name].as_array(str)[models==first])
    cif.block['atom_site']=new
    cif,conformers=resolve(cif,[chain]); atomic_json(work/'conformers.json',conformers)
    evidence=inventory(cif,{'A':[chain],'B':[]}); atomic_json(work/'input_atom_mask.json',evidence)
    atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
    author=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=True)
    seq=sequence(cif,chain)
    atoms.res_id=author.res_id.copy(); atoms.ins_code=author.ins_code.copy()
    atoms=atoms[(atoms.chain_id==chain)&np.isin(atoms.res_name,list(CANONICAL))&~np.isin(np.char.upper(atoms.element),['H','D'])]
    write_cif(work/'observed.cif',atoms)
    fixed=complete(atoms,str(runtime/'envs/pypka/bin/pdb2pqr30'),work)
    top=topology(fixed); write_cif(work/'prepared.cif',fixed)
    group,number=spec['site']
    hits=[i for i,k in enumerate(top.keys) if k.chain==chain and k.number==number and k.insertion=='']
    if len(hits)!=1 or str(top.residue(hits[0]).res_name[0])!=group:
        raise ValueError(f'{pdb}: target mapping failed for {group}{number}')
    def residue_number(key):
        return int(''.join(c for c in key.split(':',1)[1] if c.isdigit()))
    active=any({residue_number(a),residue_number(b)}=={32,35} for a,b in top.metadata['disulfide_pairs'])
    if active != (spec['redox']=='oxidized'):
        raise ValueError(f'{pdb}: active-site disulfide={active}, expected {spec["redox"]}')
    if spec['construct']=='wild type' and seq!=reference_sequence:
        raise ValueError(f'{pdb}: proposed wild-type recovery sequence differs from 1ERT')
    if spec['construct']=='M74T':
        diffs=[(i+1,a,b) for i,(a,b) in enumerate(zip(reference_sequence,seq)) if a!=b]
        if len(seq)!=len(reference_sequence) or diffs!=[(74,'M','T')]:
            raise ValueError(f'{pdb}: unexpected construct differences {diffs}')
    elif spec['construct']!='wild type':
        diffs=[(i+1,a,b) for i,(a,b) in enumerate(zip(reference_sequence,seq)) if a!=b]
        if len(seq)!=len(reference_sequence) or diffs!=[(62,'C','A'),(69,'C','A'),(73,'C','A'),(74,'M','T')]:
            raise ValueError(f'{pdb}: unexpected construct differences {diffs}')
    manifests.append(dict(pdb=pdb,records=spec['records'],site=dict(group=group,resnum=number,chain=chain),
        redox_state=spec['redox'],construct=spec['construct'],sequence=seq,
        reference_1ERT_sequence=reference_sequence,source_url=url,source_sha256=digest(raw),
        prepared_sha256=digest(work/'prepared.cif'),disulfide_pairs=top.metadata['disulfide_pairs'],
        first_model_only=True,model_fit=False,job=os.environ['SLURM_JOB_ID']))

atomic_json(out/'manifest.json',dict(version='pkadr-human-thioredoxin-recovery-v4',structures=manifests,
    code_sha256=digest(Path(__file__)),model_fit=False,experimental_training=False,job=os.environ['SLURM_JOB_ID']))
print(json.dumps(json.loads((out/'manifest.json').read_text()),indent=2),flush=True)
