"""Conservative whole-pool reuse inventory at prepared native atom scope.

Sequence equality only prunes comparisons. Geometry matches require all heavy
atoms under one proper rigid transform and the same physical-chain partition.
No cross-source teacher reuse is inferred from geometry alone.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import io
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, config_hash, digest, require_compute
from pkatrain.augmentation_store import AugmentationStore
from sabdab_v7_native_scope import atom_signature, read_cif, source


def read(path):
    return json.loads(Path(path).read_text())


def write_rows(path, rows):
    columns=sorted({k for r in rows for k in r})
    pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in columns} for r in rows]),path,compression='zstd')


def context_key(row):
    return (row['n_atoms'],tuple(sorted(c['sequence'] for c in row['chains'])))


def chain_atoms(atoms):
    """Independent of source residue numbering; retain physical-chain scope."""
    import biotite.structure as struc
    from jaxpropka.parameters import THREE, ALPHABET
    aa=dict(zip(THREE,ALPHABET));out={}
    for cid in dict.fromkeys(map(str,atoms.chain_id)):
        sub=atoms[atoms.chain_id==cid];starts=struc.get_residue_starts(sub,add_exclusive_stop=True)
        signatures=[];coords=[];seq=''
        for ordinal,(s,e) in enumerate(zip(starts[:-1],starts[1:])):
            residue=sub[s:e];seq+=aa[str(residue.res_name[0])]
            order=sorted(range(len(residue)),key=lambda i:(str(residue.atom_name[i]),str(residue.element[i])))
            names=[(ordinal,str(residue.res_name[i]),str(residue.atom_name[i]),str(residue.element[i])) for i in order]
            if len(set(names))!=len(names):raise ValueError('Ambiguous repeated atom identity')
            signatures.extend(names);coords.extend(residue.coord[order])
        out[cid]=(seq,tuple(signatures),np.asarray(coords,dtype=float))
    return out


def compare_contexts(left,right,left_side,right_side,permutation_limit=720):
    """Prove one atom-preserving chain bijection and proper rigid transform."""
    a=chain_atoms(left);b=chain_atoms(right)
    if Counter((x[0],x[1]) for x in a.values())!=Counter((x[0],x[1]) for x in b.values()):
        return dict(status='different_heavy_atom_inventory')
    options=[[q for q in b if a[p][:2]==b[q][:2]] for p in a]
    count=int(np.prod([len(x) for x in options]))
    if count>permutation_limit:return dict(status='chain_symmetry_unassessed',candidate_bijections=count)
    best=None;partition_available=bool(left_side and right_side)
    for candidate in itertools.product(*options):
        if len(set(candidate))!=len(candidate):continue
        mapping=dict(zip(a,candidate));partition_match=None
        if partition_available:
            pairs={(left_side[p],right_side[q]) for p,q in mapping.items()}
            partition_match=(len(pairs)==2 and len({p for p,q in pairs})==2 and len({q for p,q in pairs})==2)
        x=np.concatenate([a[p][2] for p in a]);y=np.concatenate([b[mapping[p]][2] for p in a])
        xx=x-x.mean(0);yy=y-y.mean(0);u,_,vh=np.linalg.svd(xx.T@yy)
        correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vh)
        delta=xx@(u@correction@vh)-yy;dist=np.linalg.norm(delta,axis=1)
        result=dict(status='same_atoms_different_geometry',rmsd_A=float(np.sqrt(np.mean(dist**2))),
            max_atom_error_A=float(dist.max()),chain_map=[dict(left=p,right=q) for p,q in mapping.items()],
            biological_partition_match=partition_match)
        if best is None or result['max_atom_error_A']<best['max_atom_error_A']:best=result
        if result['max_atom_error_A']<=0.005 and result['rmsd_A']<=0.002:
            result['status']='rigid_heavy_atom_identity'
            if partition_match is False:result['status']='same_geometry_different_biological_partition'
            return result
    return best or dict(status='chain_bijection_unassessed')


STORE=None;NATIVE={};RUNTIME=None;BASELINE=None


def initialise(store,native,runtime,baseline):
    global STORE,NATIVE,RUNTIME,BASELINE
    STORE=AugmentationStore(store);NATIVE=native;RUNTIME=Path(runtime);BASELINE=Path(baseline)


@lru_cache(maxsize=12)
def load_sabdab(uid):
    from biotite.structure.io import pdbx
    frame=STORE.raw(uid);meta=frame['metadata']['structure']
    file=pdbx.CIFFile.read(io.StringIO(frame['arrays']['canonical_cif'].tobytes().decode()))
    atoms=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True,include_bonds=True)
    if atom_signature(atoms)!=meta['native_coordinate_sha256']:raise ValueError('Packed native atom identity changed')
    return atoms,meta


@lru_cache(maxsize=12)
def load_native(uid):
    from biotite.structure.io import pdbx
    from pkabench.conformers import resolve
    from jaxpropka.parameters import THREE
    from jaxpropka.topology import load_topology
    row=NATIVE[uid];cid=row['source_native_id']
    if row['source']=='pkpdb':
        entry=source(RUNTIME,'pretraining/pkpdb-full-v1/entries/'+cid)
        selected=[r['chain'] for r in read(entry/'defects.json')['sequences']]
        path=source(RUNTIME,'pretraining/pkpdb-v1/structures/'+cid[1:3]+'/'+cid+'.cif.gz')
        file,_=resolve(read_cif(path),selected)
        label=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=False)
        author=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True)
        atoms=label.copy();atoms.res_id=author.res_id.copy();atoms.ins_code=author.ins_code.copy()
        atoms=atoms[np.isin(atoms.res_name,list(THREE)) & np.isin(atoms.chain_id,selected)
            & ~np.isin(np.char.upper(atoms.element),['H','D'])]
    else:
        fv=row['input_scope_id']=='v6c-native-observed-Fv-coordinates'
        entry=(BASELINE/'fv-build/pretraining/pinder-pkai-fv-v1/labels-v3/entries'/cid
            if fv else source(RUNTIME,'pretraining/pinder-pkai-v1/entries/'+cid))
        file=read_cif(entry/'AB.cif.gz')
        raw=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True,include_bonds=True)
        atoms=load_topology(raw,gap_policy='cap',freeze_disulfides=True).atoms
    if atom_signature(atoms)!=row['prepared_student_context_sha256']:raise ValueError('Native context changed after full audit')
    for pin in row['input_pins']:
        if digest(pin['path'])!=pin['sha256']:raise ValueError('Native audit input changed: '+pin['path'])
    return atoms,{v['chain']:v['side'] for v in (row.get('side_by_chain') or [])}


def compare_one(task):
    left,right=task
    result=dict(unit_id=left,candidate_parent=right,teacher_reuse_approved=False)
    try:
        a,meta=load_sabdab(left)
        if right.startswith('sabdab:'):
            b,other=load_sabdab(right);side=other['side_by_chain']
        else:b,side=load_native(right)
        result.update(compare_contexts(a,b,meta['side_by_chain'],side))
        result['teacher_context_status']=('historical_site_only_not_paired_pkai' if right.startswith('pkpdb:')
            else 'requires_exact_teacher_input_terminus_and_protocol_proof')
    except Exception as exc:result.update(status='comparison_unassessed',error=repr(exc))
    return result


def qualify_pilot(row,store,root):
    """Same-unit byte/context reuse from the already curated native pilot."""
    uid=row['unit_id'];entry=root/'labels-v1/entries'/row['source_native_id']
    if not (entry/'receipt.json').exists():return None
    receipt=read(entry/'receipt.json');frame=store.raw(uid);meta=frame['metadata']['structure']
    if receipt['status']!='complete' or receipt['native_coordinate_sha256']!=meta['native_coordinate_sha256']:
        raise ValueError('Pilot native coordinate mismatch: '+uid)
    for name,expected in receipt['files_sha256'].items():
        if digest(entry/name)!=expected:raise ValueError('Pilot bytes changed: '+str(entry/name))
    evidence=read(entry/'source-evidence.json');terminal=read(entry/'terminal-evidence.json')
    # The same frozen source, biological-side map and crop boundary contract
    # are required. Labels were computed before the boundary mask correction;
    # correction changed no teacher atoms or natural termini.
    old_crop=read(entry/'crop-metadata.json');new_crop=meta['antibody_crop_metadata']
    for crop in [old_crop,new_crop]:
        if crop['sha256']!=config_hash({k:v for k,v in crop.items() if k!='sha256'}):raise ValueError('Crop contract hash changed')
        pin=crop.get('metadata_pin')
        if pin and digest(pin['path'])!=pin['sha256']:raise ValueError('Published metadata changed')
    # The pilot pins the original CSV path; the packed census pins its frozen
    # row through inputs.json. Both must carry the exact SAME source row and
    # retained-domain contract. Provenance location is not a scientific change.
    def crop_content(crop):return {k:v for k,v in crop.items() if k not in ['sha256','metadata_pin']}
    if crop_content(old_crop)!=crop_content(new_crop):raise ValueError('Pilot published crop contract mismatch')
    def scientific_evidence(value):
        value=json.loads(json.dumps(value))
        scope=value['antibody_scope'];scope.pop('record_sha256');scope.pop('source_metadata_pin')
        for omission in value['intentional_omissions']:omission.pop('scope_record_sha256')
        return value
    if scientific_evidence(evidence)!=scientific_evidence(frame['metadata']['evidence']):
        raise ValueError('Pilot source or masking evidence mismatch: '+uid)
    def terms(value):
        return sorted((v['chain'],v['number'],v['insertion'],v['group']) for v in value)
    if terms(row['true_termini'])!=sorted(map(tuple,terminal['real'])):raise ValueError('Pilot true termini differ')
    if terms(row['known_termini'])!=sorted(map(tuple,terminal['known'])):raise ValueError('Pilot known termini differ')
    request=read(entry/'request.json')
    for path,expected in request['teacher_input_pins'].items():
        if digest(path)!=expected:raise ValueError('Pilot teacher pin changed')
    return dict(unit_id=uid,teacher_status='verified_same_native_teacher_reuse',entry=str(entry),
        receipt_sha256=digest(entry/'receipt.json'),native_coordinate_sha256=meta['native_coordinate_sha256'],
        labels_sha256=digest(entry/'labels.json'),masks_recurated=True)


def run(args):
    require_compute(threads=args.workers+1,allow_comp1400=True)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);started=time.monotonic()
    native={r['unit_id']:r for r in pq.read_table(args.native/'units.parquet').to_pylist() if r['status']=='checked'}
    sabdab=[r for r in pq.read_table(args.sabdab/'units.parquet').to_pylist() if r['status']=='prepared']
    groups=defaultdict(list)
    for row in native.values():groups[context_key(row)].append(row['unit_id'])
    tasks=[]
    for row in sorted(sabdab,key=lambda r:r['unit_id']):
        key=context_key(row);tasks.extend((row['unit_id'],peer) for peer in groups[key]);groups[key].append(row['unit_id'])
    print(json.dumps(dict(candidate_pairs=len(tasks),by_source=dict(Counter(peer.split(':',1)[0] for uid,peer in tasks)))),flush=True)
    inventory=read(Path(read(args.audit/'inputs.json')['inventory'])/'inventory-manifest.json')
    with ProcessPoolExecutor(args.workers,initializer=initialise,
            initargs=(args.sabdab/'prepared',native,inventory['runtime'],inventory['baseline'])) as pool:
        results=[]
        for row in pool.map(compare_one,tasks,chunksize=8):
            results.append(row)
            if len(results)%100==0:print(json.dumps(dict(compared=len(results),total=len(tasks))),flush=True)
    write_rows(args.output/'context-comparisons.parquet',results)
    pilot=[];pilot_holds=[]
    with AugmentationStore(args.sabdab/'prepared') as store:
        for row in sabdab:
            try:
                proof=qualify_pilot(row,store,args.pilot)
                if proof:pilot.append(proof)
            except Exception as exc:pilot_holds.append(dict(unit_id=row['unit_id'],error=repr(exc)))
    atomic_json(args.output/'pilot-reuse.json',dict(verified=pilot,withheld=pilot_holds))
    summary=dict(status='conservative_prepared_context_reuse_screen_complete',
        prepared_sabdab=len(sabdab),native_units=len(native),candidate_pairs=len(tasks),
        counts=dict(Counter(r['status'] for r in results)),
        by_source={s:dict(Counter(r['status'] for r in results if r['candidate_parent'].startswith(s+':'))) for s in ['pkpdb','pinder','sabdab']},
        verified_same_native_pilot_teacher_reuses=len(pilot),pilot_reuse_withheld=pilot_holds,
        cross_source_teacher_reuse_approved=0,source_precedence=['pkpdb','sabdab','pinder'],
        precedence_gate='prepared context AND teacher context AND compatible operational role; never PDB ID alone',
        no_source_ownership_activated=True,no_new_teacher_or_surfaces=True,wall_s=time.monotonic()-started,
        comparisons_sha256=digest(args.output/'context-comparisons.parquet'),
        native_sha256=digest(args.native/'units.parquet'),sabdab_sha256=digest(args.sabdab/'units.parquet'))
    atomic_json(args.output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native','sabdab','audit','pilot','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);run(parser.parse_args())
