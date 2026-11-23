"""Small, inactive SAbDab-native assembly and cache-reuse pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import io
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, config_hash, digest, require_compute
from pkabench.sabdab_units import prepare
from sabdab_v7_exposure import read
from sabdab_v7_roles import write_rows


def choose(official, assignments, units):
    eligible = {r['unit_id']: r for r in assignments if r['operational_role'] in {'train', 'validation'}}
    rows = [r for r in official if 'sabdab:' + r['INSTANCE'] in eligible]
    selected = {}; strata = defaultdict(list)
    def size(r):
        return sum(len(r['V'+s+'_numerable_seq']) for s in ['H','L']) + sum(len(s) for s,t in zip(r['agresolvedseqs'].split('/'),r['agtypes'].split('/')) if t in {'PROTEIN','PEPTIDE'})
    def add(r, reason):
        selected.setdefault(r['INSTANCE'], dict(row=r, reasons=[], estimated_residues=size(r)))['reasons'].append(reason)
    for r in rows:
        if r['type'] in {'FV','FAB','FAB+FC','SD-H'} and size(r) <= 1536:
            strata[(r['ab_ag_split'], 'SD-H' if r['type']=='SD-H' else 'paired',
                    'peptide' if 'PEPTIDE' in r['agtypes'].split('/') else 'protein')].append(r)
    for key, rr in sorted(strata.items()):
        rr.sort(key=lambda r:(size(r),r['INSTANCE']))
        for f in [.25,.75]:
            add(rr[int((len(rr)-1)*f)], 'size-stratum:' + repr(key))
    for split in ['train','test']:
        for typ in ['SD-L','VNAR']:
            rr = sorted((r for r in rows if r['ab_ag_split']==split and r['type']==typ and size(r)<=1536), key=lambda r:config_hash(r['INSTANCE']))
            if rr:add(rr[0], 'other-antibody-type:' + typ)
        for reason,predicate in [('multichain-antigen',lambda r:len(r['agchains'].split('/'))>1),
                                 ('scfv',lambda r:r.get('construct')=='SCFV')]:
            rr=sorted((r for r in rows if r['ab_ag_split']==split and size(r)<=1536 and predicate(r)), key=lambda r:config_hash(r['INSTANCE']))
            if rr:add(rr[0],reason)
    active_parent = defaultdict(list)
    for uid,a in eligible.items():
        if a['source'] != 'sabdab':active_parent[(a['source'],units[uid]['pdb_id'])].append(uid)
    for source in ['pkpdb','pinder']:
        rr=sorted((r for r in rows if (source,r['PDB_ID'][-4:].lower()) in active_parent and size(r)<=1536), key=lambda r:config_hash(r['INSTANCE']))
        for r in rr[:2]:add(r, 'existing-parent:' + source)
    return [selected[k] for k in sorted(selected)]


def run(args):
    require_compute(threads=args.workers,allow_comp1400=True)
    if args.output.exists():raise FileExistsError('Choose a new preparation pilot root')
    args.output.mkdir(parents=True); (args.output/'entries').mkdir()
    data=read(args.root/'inputs.json')
    assignments=pq.read_table(args.assignments).to_pylist()
    units={u['unit_id']:u for u in pq.read_table(args.root/'units.parquet').to_pylist()}
    selected=choose(data['official'],assignments,units)
    download=read(args.archive.parent/'download.json')
    if digest(args.archive)!=download['sha256']:raise ValueError('Pinned SAbDab archive changed')
    metadata={Path(v['row']['cif_file']).name:v for v in selected}
    if len(metadata)!=len(selected):raise ValueError('Ambiguous ML archive basenames')
    raw={}; started=time.monotonic()
    with tarfile.open(args.archive,'r|gz') as archive:
        for member in archive:
            name=Path(member.name).name
            if member.isfile() and name in metadata:
                if name in raw:raise ValueError('Duplicate archive member: '+name)
                raw[name]=archive.extractfile(member).read()
    if raw.keys()!=metadata.keys():raise ValueError('Selected ML coordinates missing from archive')
    archive_s=time.monotonic()-started
    from biotite.structure.io import pdbx
    results=[]; prepared=[]; parent_lookups=defaultdict(list)
    active={a['unit_id']:a for a in assignments if a['operational_role'] in {'train','validation'}}
    for uid,a in active.items():
        if a['source']!='sabdab':parent_lookups[units[uid]['pdb_id']].append(uid)
    for name,spec in sorted(metadata.items()):
        row=spec['row']; uid='sabdab:'+row['INSTANCE']; out=args.output/'entries'/row['INSTANCE'];out.mkdir()
        with gzip.open(out/'source.cif.gz','wb',compresslevel=1) as handle:handle.write(raw[name])
        started=time.monotonic()
        try:
            atoms,topology,side,meta=prepare(pdbx.CIFFile.read(io.StringIO(raw[name].decode())),row)
            meta.update(archive_member=name,source_cif_sha256=__import__('hashlib').sha256(raw[name]).hexdigest(),
                selection_reasons=spec['reasons'], candidate_assignment_sha256=digest(args.assignments))
            file=pdbx.CIFFile();pdbx.set_structure(file,atoms)
            with gzip.open(out/'AB.cif.gz','wt',compresslevel=1) as handle:file.write(handle)
            atomic_json(out/'structure.json',meta)
            np.savez_compressed(out/'side.npz',side=side)
            if topology.n_residues>1536:raise ValueError('Prepared input exceeds current production capacity')
            peers=sorted(parent_lookups[row['PDB_ID'][-4:].lower()])
            prepared.append(dict(unit_id=uid,entry=str(out),split=active[uid]['operational_role'],
                native_source_split=row['ab_ag_split'],component=row['ab_ag_cluster'],
                n=topology.n_residues,source='sabdab',data_kind='paired',
                cif_sha256=digest(out/'AB.cif.gz'),structure_sha256=digest(out/'structure.json')))
            results.append(dict(unit_id=uid,status='prepared',n_res=topology.n_residues,
                antibody_type=row['type'],construct=row.get('construct',''),
                partner_A_chains=sorted(k for k,v in meta['side_by_chain'].items() if v=='A'),
                partner_B_chains=sorted(k for k,v in meta['side_by_chain'].items() if v=='B'),
                existing_parent_candidates=peers,cache_reuse='not_established_by_parent_PDB',
                selection_reasons=spec['reasons'],prepare_s=time.monotonic()-started))
        except Exception as exc:
            results.append(dict(unit_id=uid,status='withheld_preparation',error=repr(exc),
                antibody_type=row['type'],construct=row.get('construct',''),selection_reasons=spec['reasons'],prepare_s=time.monotonic()-started))
    columns=sorted({k for r in results for k in r})
    write_rows(args.output/'preparation-results.parquet',[{k:r.get(k) for k in columns} for r in results])
    atomic_json(args.output/'prepared-units.json',dict(status='pilot_only_not_training_release',units=prepared,
        assignments_sha256=digest(args.assignments),archive_sha256=download['sha256']))
    summary=dict(status='native_preparation_pilot_complete',release_ready=False,selected=len(results),
        results=dict(Counter(r['status'] for r in results)), archive_scan_s=archive_s,
        selected_roles=dict(Counter(v['row']['ab_ag_split'] for v in selected)),
        failures=[r for r in results if r['status']!='prepared'],
        native_labels_or_features_built=False,active_release_changed=False,
        teacher_context_identity_established=False,code_sha256=digest(__file__))
    atomic_json(args.output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','assignments','archive','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
