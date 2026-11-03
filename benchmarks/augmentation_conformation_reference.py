"""Compare saved ANM motions with independent experimental PINDER apo/holo pairs.

No training, minimisation, MSMS or APBS. All surfaces retain whole-chain
occlusion. Patch radii are Euclidean, not geodesic. Missing/mutated atoms are
excluded by a conservative neighbourhood mask, never by truncating a surface.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np
from scipy.spatial import cKDTree

from pkabench.runtime import atomic_json, digest, require_compute
from augmentation_surface_audit import (
    AMINO, CHEM, RADII, anchors, annotate, atom_exposure, checks, composition,
    crop, fit, frames, rms, shape, surface,
)

RADII_PATCH = (6., 9., 12.)
ONE = dict(zip(AMINO, 'ARNDCQEGHILKMFPSTWYV'))
SEED = 20261022


def correspondence(ref_names, moving_names, min_coverage=.9):
    """Global sequence alignment, exact identity on aligned residues, >=90% each."""
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner(mode='global', match_score=2., mismatch_score=-4.,
                             open_gap_score=-10., extend_gap_score=-.5)
    a = ''.join(ONE[n] for n in ref_names)
    b = ''.join(ONE[n] for n in moving_names)
    alignment = aligner.align(a, b)[0]
    indices = np.asarray(alignment.indices)
    mapped = indices[:, np.all(indices >= 0, axis=0)]
    if not mapped.shape[1]: raise ValueError('empty sequence correspondence')
    identity = float(np.mean([a[i] == b[j] for i, j in mapped.T]))
    coverage = (mapped.shape[1]/len(a), mapped.shape[1]/len(b))
    if identity != 1. or min(coverage) < min_coverage:
        raise ValueError(f'sequence identity/coverage: {identity}, {coverage}')
    return dict(zip(map(int, mapped[0]), map(int, mapped[1]))), dict(
        identity=identity, reference_coverage=coverage[0], moving_coverage=coverage[1])


def state(xyz, atom_names, owner, names, keys):
    xyz = np.asarray(xyz, float); owner = np.asarray(owner, int)
    if not np.isfinite(xyz).all(): raise ValueError('nonfinite coordinates')
    lookup = {(int(r), n): i for i, (r, n) in enumerate(zip(owner, atom_names))}
    if len(lookup) != len(xyz): raise ValueError('duplicate atom identities')
    bb = []
    for r in range(len(names)):
        if not all((r, n) in lookup for n in ('N', 'CA', 'C')):
            raise ValueError(f'incomplete backbone at {r}')
        bb.append([lookup[r, n] if (r, n) in lookup else lookup[r, 'C']
                   for n in ('N', 'CA', 'C', 'O')])
    bb = np.asarray(bb, int)
    if any(n[0] not in RADII for n in atom_names): raise ValueError('unsupported heavy element')
    if not np.isfinite(frames(xyz, bb)).all(): raise ValueError('degenerate residue frame')
    atomsets=[set() for _ in names]
    for r,n in zip(owner,atom_names):atomsets[int(r)].add(n)
    return dict(xyz=xyz, atom_names=list(atom_names), owner=owner, names=list(names),
                keys=keys, bb=bb, lookup=lookup,atomsets=atomsets)


def read_pdb(path):
    from biotite.structure.io.pdb import PDBFile
    import biotite.structure as st
    atoms = PDBFile.read(str(path)).get_structure(model=1, altloc='occupancy')
    atoms = atoms[np.isin(atoms.res_name, AMINO) & ~np.isin(atoms.element, ['H','D'])]
    if len(set(atoms.chain_id)) != 1: raise ValueError('PINDER file is not a single protein chain')
    starts = st.get_residue_starts(atoms, add_exclusive_stop=True)
    names, keys, owners = [], [], []
    for r, (s, e) in enumerate(zip(starts[:-1], starts[1:])):
        names.append(str(atoms.res_name[s]))
        keys.append([str(atoms.chain_id[s]), int(atoms.res_id[s]), str(atoms.ins_code[s])])
        owners.extend([r]*(e-s))
    return state(atoms.coord, list(map(str, atoms.atom_name)), owners, names, keys)


def load_unit(pilot, index):
    old = pilot/f'native-large/units/{index:03d}'
    info = json.loads((old/'preparation.json').read_text())
    report = json.loads((old/'report.json').read_text())
    heavy = np.flatnonzero(np.asarray(info['is_native']) & np.asarray(info['is_heavy']))
    native = np.load(old/'prepared.npz')['xyz'][heavy]
    ref = state(native, [info['atom_names'][i] for i in heavy],
                np.asarray(info['residue'])[heavy], info['names'], info['keys'])
    zero = dict(ref, xyz=np.load(old/'native/c00/min.npy')[heavy].astype(float))
    metrics = {m['candidate']:m for m in report['metrics']}
    if not metrics[0]['geometry_valid']: raise ValueError('zero-displacement repair failed geometry')
    return old, heavy, ref, zero, metrics, report


def cached_download(name, directory):
    if Path(name).name != name or not name.endswith('.pdb'):
        raise ValueError('unsafe PINDER filename')
    path = directory/name
    url = 'https://pinderdata.org/2024-02/pdbs/'+name
    if not path.exists():
        error = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(url, timeout=45) as response: data=response.read()
                if b'ATOM  ' not in data: raise ValueError('download is not a PDB file')
                path.write_bytes(data); error=None; break
            except Exception as exc: error=exc
        if error: raise error
    return path, dict(url=url, sha256=digest(path), bytes=path.stat().st_size)


def atom_mapping(a, b, residue_map):
    pairs=[]; bad_a=set(range(len(a['names'])))-set(residue_map)
    bad_b=set(range(len(b['names'])))-set(residue_map.values())
    for r,s in residue_map.items():
        an=a['atomsets'][r]
        bn=b['atomsets'][s]
        # A terminal OXT is also an inventory difference: mask it explicitly.
        if an != bn: bad_a.add(r); bad_b.add(s)
        pairs.extend((a['lookup'][r,n],b['lookup'][s,n]) for n in sorted(an & bn))
    return np.asarray(pairs,int),bad_a,bad_b


def make_surface(s, points):
    radius=np.asarray([RADII[n[0]]+1.4 for n in s['atom_names']])
    aa=np.asarray([AMINO.index(s['names'][r]) for r in s['owner']])
    chem=np.asarray([CHEM[s['names'][r]] for r in s['owner']])
    return annotate(surface(s['xyz'],s['bb'],s['owner'],radius,points),aa,chem)


def fixed_anchors(ref, surf, limit=32):
    # Choose the same anchor residues/centres for every radius and comparison.
    return anchors(ref['xyz'],ref['bb'],ref['owner'],surf,12.,limit)


def patch_definitions(a,b,resmap,anchors_fixed,sa,sb):
    """Mask any patch affected by unmatched residues/atoms or physical breaks."""
    _,bad_a,bad_b=atom_mapping(a,b,resmap)
    fa,fb=frames(a['xyz'],a['bb']),frames(b['xyz'],b['bb'])
    ca,cb=a['xyz'][a['bb'][:,1]],b['xyz'][b['bb'][:,1]]
    # Exclude boundary neighbourhoods around chain breaks as well as absent atoms.
    for st,bad in ((a,bad_a),(b,bad_b)):
        for i in range(len(st['names'])-1):
            if np.linalg.norm(st['xyz'][st['bb'][i,2]]-st['xyz'][st['bb'][i+1,0]])>2.:
                bad.update((i,i+1))
    ta=cKDTree(a['xyz'][np.isin(a['owner'],list(bad_a))]) if bad_a else None
    tb=cKDTree(b['xyz'][np.isin(b['owner'],list(bad_b))]) if bad_b else None
    definitions=[];excluded=[]
    for r,center,_ in anchors_fixed:
        if r not in resmap: continue
        s=resmap[r];mcenter=cb[s]+(center-ca[r])@fa[r].T@fb[s]
        for radius in RADII_PATCH:
            reason=None
            if ((ta is not None and ta.query(center)[0]<=radius+3.5)
                    or (tb is not None and tb.query(mcenter)[0]<=radius+3.5)):
                reason='unmatched atom/residue or chain-break neighbourhood'
            members=np.asarray([int(i) for i in np.flatnonzero(np.linalg.norm(ca-center,axis=1)<=radius)
                                if int(i) in resmap],int)
            if len(members)<4 or np.linalg.matrix_rank(ca[members]-ca[members].mean(0))<2:
                reason=reason or 'insufficient noncollinear corresponding backbone'
            pa,pb=crop(sa,center,radius),crop(sb,mcenter,radius)
            if min(len(pa['points']),len(pb['points']))<20:reason=reason or 'insufficient exposed surface samples'
            if reason:
                excluded.append(dict(anchor=r,radius_A=radius,reason=reason));continue
            definitions.append(dict(anchor=r,center=center,radius=radius,members=members))
    return definitions,excluded


def measure(a,b,resmap,sa,sb,definitions,parent,group,candidate,amplitude,npoints):
    ids=np.asarray(sorted(resmap),int);moving_ids=np.asarray([resmap[i] for i in ids],int)
    ca,cb=a['xyz'][a['bb'][:,1]],b['xyz'][b['bb'][:,1]]
    rotation,translation=fit(cb[moving_ids],ca[ids])
    pair_ids,_,_=atom_mapping(a,b,resmap)
    fa,fb=frames(a['xyz'],a['bb']),frames(b['xyz'],b['bb'])
    # Surface sample IDs refer to different inventories in experimental pairs.
    # Remap moving IDs to corresponding reference atoms; unmatched IDs stay disjoint.
    lookup={int(j):int(i) for i,j in pair_ids}
    sb=dict(sb)
    mapped=np.asarray([lookup.get(int(j),len(a['xyz'])+int(j)) for j in sb['atom']])
    sb['sample_id']=mapped*npoints+sb['sample_id']%npoints
    common=dict(parent=parent,group=group,candidate=candidate,amplitude_A=amplitude,
                global_ca_rmsd_A=rms(cb[moving_ids]@rotation+translation,ca[ids]))
    whole=dict(common,n_residues=len(a['names']),matched_residues=len(ids),
               global_surface_distance_A=shape(sa,sb,rotation,translation)['distance_A'],
               global_composition_TV=float(.5*np.abs(composition(sa)-composition(sb)).sum()))
    rows=[]
    for definition in definitions:
        r,center,radius,members=(definition[k] for k in ('anchor','center','radius','members'))
        s=resmap[r];mcenter=cb[s]+(center-ca[r])@fa[r].T@fb[s]
        pa,pb=crop(sa,center,radius),crop(sb,mcenter,radius)
        if min(len(pa['points']),len(pb['points']))<20:continue
        mi=np.asarray([resmap[int(i)] for i in members],int)
        lr,lt=fit(cb[mi],ca[members])
        # Compare exposed-area weights for the same atoms, independent of NN matching.
        pairs=pair_ids[np.isin(a['owner'][pair_ids[:,0]],members)]
        local=shape(pa,pb,lr,lt);glob=shape(pa,pb,rotation,translation)
        rows.append(dict(common,anchor=int(r),radius_A=radius,patch_residues=len(members),
            patch_ca_after_global_rmsd_A=rms(cb[mi]@rotation+translation,ca[members]),
            patch_ca_after_local_rmsd_A=rms(cb[mi]@lr+lt,ca[members]),
            local_surface_distance_A=local['distance_A'],
            global_surface_distance_A=glob['distance_A'],
            local_normal_angle_deg=local['normal_angle_deg'],
            local_chemistry_mismatch_fraction=local['chemistry_mismatch_fraction'],
            exposed_composition_TV=float(.5*np.abs(composition(pa)-composition(pb)).sum()),
            atom_exposure_change=1.-atom_exposure(sa['atom_area'][pairs[:,0]],sb['atom_area'][pairs[:,1]])))
    return whole,rows


def prepare(args):
    require_compute(threads=1)
    import pyarrow.dataset as ds
    sample=json.loads((args.pilot/'native-large/sample.json').read_text())
    units=[r for r in sample if r['dataset']=='pinder']
    index_path=args.pilot.parent/'pkabench/audits/pinder-index-v1/index.parquet'
    index=ds.dataset(str(index_path))
    cols=['id','cluster_id','uniprot_R','uniprot_L','holo_R_pdb','holo_L_pdb',
          'apo_R_pdb','apo_L_pdb','apo_R_quality','apo_L_quality','apo_R','apo_L']
    records=index.to_table(columns=cols,filter=ds.field('id').isin([u['id'] for u in units])).to_pylist()
    byid={r['id']:r for r in records}
    eligible=[]
    for unit in units:
        row=byid.get(unit['id']);side='R' if unit['state']=='A' else 'L'
        if row and row['apo_'+side+'_pdb']:
            eligible.append(dict(source=unit,index=unit['index'],side=side,index_record=row))
    # Interleave size bins, stable hash within each; no selection on motion/surface metrics.
    groups=[]
    for sizebin in range(5):
        group=[r for r in eligible if r['source']['bin']==sizebin]
        groups.append(sorted(group,key=lambda r:hashlib.sha256(f"{SEED}|{r['source']['id']}".encode()).hexdigest()))
    ranked=[g[i] for i in range(max(map(len,groups),default=0)) for g in groups if i<len(g)]
    args.output.mkdir(parents=True,exist_ok=True)
    downloads=args.output/'downloads';downloads.mkdir(exist_ok=True)
    accepted=[];excluded=[];seen=set()
    for unit in ranked:
        if len(accepted)>=args.pairs:break
        row=unit['index_record'];side=unit['side'];protein=row['uniprot_'+side]
        if not protein or protein=='UNDEFINED':protein=row['cluster_id']
        if protein in seen:continue
        try:
            old,heavy,ref,zero,metrics,report=load_unit(args.pilot,unit['index'])
            paths={};receipts={}
            for role in ('apo','holo'):
                paths[role],receipts[role]=cached_download(row[role+'_'+side+'_pdb'],downloads)
            apo,holo=read_pdb(paths['apo']),read_pdb(paths['holo'])
            mapping,alignment=correspondence(ref['names'],apo['names'])
            hmap,halign=correspondence(ref['names'],holo['names'])
            ai=np.asarray(sorted(hmap));bi=np.asarray([hmap[i] for i in ai])
            rotation,translation=fit(holo['xyz'][holo['bb'][bi,1]],ref['xyz'][ref['bb'][ai,1]])
            holo_rmsd=rms(holo['xyz'][holo['bb'][bi,1]]@rotation+translation,ref['xyz'][ref['bb'][ai,1]])
            if holo_rmsd>.05:raise ValueError(f'PINDER holo does not reproduce saved source: {holo_rmsd}')
            sa,sb=make_surface(ref,args.points),make_surface(apo,args.points)
            aa=fixed_anchors(ref,sa,args.anchors)
            definitions,missing=patch_definitions(ref,apo,mapping,aa,sa,sb)
            counts={str(r):sum(d['radius']==r for d in definitions) for r in RADII_PATCH}
            if min(counts.values())<4:raise ValueError(f'fewer than four comparable patches per radius: {counts}')
            parent='pinder/'+unit['source']['id']+'/'+unit['source']['state']
            unit.update(parent=parent,protein=protein,downloads=receipts,
                paths={k:str(v) for k,v in paths.items()},alignment=alignment,
                holo_source_ca_rmsd_A=holo_rmsd,comparable_patches=counts,
                masked_patches=missing,source_report_sha256=digest(old/'report.json'))
            accepted.append(unit);seen.add(protein)
            print(json.dumps(dict(accepted=len(accepted),parent=parent,patches=counts)),flush=True)
        except Exception as exc:
            excluded.append(dict(index=unit['index'],id=unit['source']['id'],error=type(exc).__name__+': '+str(exc)))
            print(json.dumps(excluded[-1]),flush=True)
    atomic_json(args.output/'sample.json',dict(parents=accepted,exclusions=excluded,
        available_links=len(eligible),requested_pairs=args.pairs,index_sha256=digest(index_path),
        config=dict(points=args.points,anchors=args.anchors,radii_A=RADII_PATCH,seed=SEED,
                    identity=1.,minimum_resolved_sequence_coverage=.9,minimum_patches_each_radius=4),
        selection='saved 500-parent pilot; PINDER links; interleaved length bins, hash order, unique UniProt; no motion-based selection'))
    print(json.dumps(dict(accepted=len(accepted),excluded=len(excluded),available_links=len(eligible))),flush=True)


def run_parent(args):
    require_compute(threads=1)
    started=time.monotonic()
    sample=json.loads((args.output/'sample.json').read_text());cfg=sample['config']
    unit=sample['parents'][args.index];parent=unit['parent']
    old,heavy,ref,zero,metrics,report=load_unit(args.pilot,unit['index'])
    apo=read_pdb(unit['paths']['apo']);mapping,alignment=correspondence(ref['names'],apo['names'])
    sa,sb=make_surface(ref,cfg['points']),make_surface(apo,cfg['points'])
    anchor=fixed_anchors(ref,sa,cfg['anchors'])
    definitions,missing=patch_definitions(ref,apo,mapping,anchor,sa,sb)
    def measured(a,b,rm,s1,s2,defs,group,c,amplitude):
        return measure(a,b,rm,s1,s2,defs,parent,group,c,amplitude,cfg['points'])
    wholes=[];patches=[];coverage=[]
    w,p=measured(ref,apo,mapping,sa,sb,definitions,'experimental',-1,0.)
    wholes.append(w);patches+=p
    identity={r:r for r in range(len(ref['names']))}
    sz=make_surface(zero,cfg['points'])
    # Reference anchors follow the backbone into the repaired zero state.
    fr,fz=frames(ref['xyz'],ref['bb']),frames(zero['xyz'],zero['bb'])
    cz=zero['xyz'][zero['bb'][:,1]];cr=ref['xyz'][ref['bb'][:,1]]
    zero_defs=[]
    for d in definitions:
        r=d['anchor'];d=dict(d)
        d['center']=cz[r]+(d['center']-cr[r])@fr[r].T@fz[r]
        zero_defs.append(d)
    w,p=measured(ref,zero,identity,sa,sz,definitions,'repair_only',0,0.)
    wholes.append(w);patches+=p
    for c in range(1,25):
        metric=metrics.get(c)
        if not metric or not metric['geometry_valid']:
            coverage.append(dict(candidate=c,geometry_valid=False));continue
        xyz=np.load(old/f'native/c{c:02d}/min.npy')[heavy].astype(float)
        moving=dict(ref,xyz=xyz);sc=make_surface(moving,cfg['points'])
        for group,reference,sref,defs in [('anm_vs_zero',zero,sz,zero_defs),('anm_vs_native',ref,sa,definitions)]:
            w,p=measured(reference,moving,identity,sref,sc,defs,group,c,metric['input_amplitude_A'])
            wholes.append(w);patches+=p
        coverage.append(dict(candidate=c,geometry_valid=True))
    # Same parent and patches at twice the sampling density, one fixed direction
    # per amplitude, establishes finite-surface-sampling sensitivity cheaply.
    sensitivity=[]
    if args.index<3:
        dense={}
        for label,s in [('native',ref),('zero',zero),('apo',apo)]:dense[label]=make_surface(s,2*cfg['points'])
        for label,a,b,rm,defs in [('experimental',ref,apo,mapping,definitions),('repair_only',ref,zero,identity,definitions)]:
            w,p=measure(a,b,rm,dense['native'],dense['apo' if label=='experimental' else 'zero'],defs,parent,label,-1 if label=='experimental' else 0,0.,2*cfg['points'])
            sensitivity+=p
        for c in (1,9,17):
            if not metrics.get(c,{}).get('geometry_valid'):continue
            moving=dict(ref,xyz=np.load(old/f'native/c{c:02d}/min.npy')[heavy].astype(float))
            w,p=measure(zero,moving,identity,dense['zero'],make_surface(moving,2*cfg['points']),zero_defs,parent,'anm_vs_zero',c,metrics[c]['input_amplitude_A'],2*cfg['points'])
            sensitivity+=p
    dest=args.output/'parents'/f"{args.index:02d}";dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/'structures.csv',wholes);write_csv(dest/'patches.csv',patches)
    write_csv(dest/'sensitivity.csv',sensitivity)
    atomic_json(dest/'result.json',dict(parent=parent,passed=True,alignment=alignment,
        wall_s=time.monotonic()-started,geometry=coverage,patches=len(patches),
        masked_patches=missing,script_sha256=digest(__file__),
        sample_sha256=digest(args.output/'sample.json'),input_report_sha256=digest(old/'report.json'),
        input_preparation_sha256=digest(old/'preparation.json'),input_prepared_sha256=digest(old/'prepared.npz'),
        downloads=unit['downloads'],candidate_coordinate_sha256={str(c):digest(old/f'native/c{c:02d}/min.npy') for c in range(25) if metrics.get(c,{}).get('geometry_valid')}))
    print(json.dumps(dict(parent=parent,patches=len(patches),wall_s=time.monotonic()-started)),flush=True)


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','parent'])
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=20);p.add_argument('--points',type=int,default=256)
    p.add_argument('--anchors',type=int,default=32);p.add_argument('--index',type=int,default=0)
    args=p.parse_args()
    (prepare if args.command=='prepare' else run_parent)(args)


if __name__=='__main__':main()
