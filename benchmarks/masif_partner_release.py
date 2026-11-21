"""Publish a compact, source-pinned PINDER Ab/Ag release from disjoint builds."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore,ReleaseWriter
from masif_partner_units import inventory_v5,UNIT_POLICY
from masif_surface_quality import SURFACE_POLICY
from masif_cache_audit import inspect_arrays,extent_metrics


def validate_native(runtime,unit,record,scratch):
    from masif_annotation_pilot import export_native
    path=Path(scratch)/'native.pdb'
    source=export_native(runtime,unit,path,surface_policy=SURFACE_POLICY)
    stored=record['metadata']['source']
    if source['pdb_sha256']!=stored['pdb_sha256'] or source['canonical_residue_ids']!=record['arrays']['canonical_residue_ids'].tolist():
        raise ValueError(unit['id']+': transferred mapping/native input differs on independent source export')
    atoms=np.asarray([[float(line[k:k+8]) for k in (30,38,46)] for line in path.read_text().splitlines() if line.startswith('ATOM')])
    extent=extent_metrics(record['arrays']['vertices'],atoms)
    if extent['atoms_outside_bbox_plus_2A'] or extent['atoms_outside_convex_hull_plus_2A']:
        raise ValueError(unit['id']+': native source extent lost by transferred mesh: '+str(extent))
    return extent


def validate(unit,record,profile):
    m=record['metadata'];a=record['arrays'];receipt=m['receipt'];source=m['source']
    for key in ['dataset','cid','state','role','unit_policy','source_files_sha256']:
        if m[key]!=unit[key]:raise ValueError('record source/partner differs: '+key)
    if m['unit_policy']!=UNIT_POLICY or receipt['surface_policy']!=SURFACE_POLICY or receipt['status']!='complete':
        raise ValueError('unvalidated partner/preparation policy')
    if any(receipt.get(k)!=v for k,v in profile.items()):raise ValueError('record model/runtime differs')
    if receipt['reference_patches_checked']!=3 or receipt['descriptor_max_reference_error']>1e-4:
        raise ValueError('descriptor reference validation failed')
    coverage=receipt['mesh_coverage']
    if coverage['source_atoms_outside_bbox_plus_2A'] or coverage['raw_surface_max_nearest_A']>3.:
        raise ValueError('surface coverage validation failed')
    canonical=a['canonical_residue_ids']
    if (canonical.tolist()!=source['canonical_residue_ids'] or len(canonical)!=unit['n']
            or source['actual_residues']!=unit['n'] or len(np.unique(canonical))!=len(canonical)):
        raise ValueError('canonical residue source mapping differs')
    hard,_,_=inspect_arrays(a)
    if hard:raise ValueError('invalid transferred surface: '+str(hard))


def publish(runtime,v5,builds,output,quarantine=None,recovery_builds=None):
    units,inventory=inventory_v5(v5,runtime);lookup={u['id']:u for u in units};records={};failures={};stores=[];profile=None;manifests={}
    try:
        recovered=[]
        for path,is_recovery in [(p,False) for p in builds]+[(p,True) for p in (recovery_builds or [])]:
            path=Path(path);report=json.loads((path/'report.json').read_text())
            current={k:report[k] for k in ['checkpoint_sha256','preprocessing_profile','paper_runtime_sha256']}
            if profile is not None and current!=profile:raise ValueError('mixed scientific model/runtime profiles')
            profile=current;store=AugmentationStore(path/'fingerprints');stores.append(store)
            manifests[str(path)]=digest(path/'fingerprints/manifest.json')
            complete={r['id'] for r in report['parents'] if r['status']=='complete'}
            if set(store.by_id)!=complete:raise ValueError('build payload/report inventory differs')
            for row in report['parents']:
                rid=row['id']
                if rid not in lookup:raise ValueError('unexpected source unit')
                if rid in records:
                    if not is_recovery or records[rid][1]['status']=='complete':
                        raise ValueError('duplicate successful source unit or unmarked recovery')
                    if row['status']=='complete':recovered.append(rid)
                elif is_recovery:raise ValueError('recovery unit was not an original failed request')
                records[rid]=(store,row)
                if row['status']!='complete':failures[rid]=row
                else:failures.pop(rid,None)
        if set(records)!=set(lookup):raise ValueError('missing requested native partner units')
        allowed={};plan_sha=None
        if quarantine:
            plan=json.loads(Path(quarantine).read_text());plan_sha=digest(quarantine)
            if plan['inventory_sha256']!=inventory['inventory_sha256']:raise ValueError('quarantine inventory differs')
            allowed={x['id']:x['reason'] for x in plan['failures']}
        if set(failures)!=set(allowed):
            raise ValueError('unreviewed native preparation failures: '+str(sorted(set(failures)^set(allowed))))
        held_pairs={rid.rsplit('/',1)[0] for rid in failures};selected=[u for u in units if u['id'].rsplit('/',1)[0] not in held_pairs]
        if not selected:raise ValueError('no complete partner pairs remain')
        pair_indices={};pairs={}
        # Validate every chosen frame and pairing before publishing the marker.
        with tempfile.TemporaryDirectory(prefix='masif-publish-native-',dir='/tmp') as scratch:
            for unit in selected:
                record=records[unit['id']][0].raw(unit['id']);validate(unit,record,profile)
                validate_native(runtime,unit,record,scratch)
                key=unit['id'].rsplit('/',1)[0];pairs.setdefault(key,{})[unit['role']]=unit
                pair_indices.setdefault(key,{})[unit['role']]=record['arrays']['canonical_residue_ids'].tolist()
        for key,pair in pair_indices.items():
            if set(pair)!={'Ab','Ag'}:raise ValueError('incomplete selected partner pair')
            a,b=map(set,(pair['Ab'],pair['Ag']))
            if a&b or a|b!=set(range(len(a)+len(b))):raise ValueError('Ab/Ag mapping does not partition the native training assembly')
        provenance=dict(kind='dense-native-masif-paper-v1',unit_policy=UNIT_POLICY,surface_policy=SURFACE_POLICY,
            source_inventory_sha256=inventory['inventory_sha256'],source_build_manifests=manifests,
            quarantine_plan_sha256=plan_sha,recovered_unit_ids=sorted(recovered),publisher_sha256=digest(__file__),**profile)
        with ReleaseWriter(output,provenance,zstd_level=1) as writer:
            for unit in sorted(selected,key=lambda x:x['id']):
                record=records[unit['id']][0].raw(unit['id'])
                writer.add(unit['id'],record['metadata'],record['arrays'],summary=unit)
            writer.finish(expected_ids=[u['id'] for u in selected])
        with AugmentationStore(output) as store:
            for unit in selected:validate(unit,store.raw(unit['id']),profile)
        pair_rows=[dict(id=key,cid=p['Ab']['cid'],split=p['Ab']['split'],group=p['Ab']['group'],
                       Ab=p['Ab']['id'],Ag=p['Ag']['id'],Ab_state=p['Ab']['state'],Ag_state=p['Ag']['state'],
                       Ab_chains=p['Ab']['source_chain_labels'],Ag_chains=p['Ag']['source_chain_labels']) for key,p in sorted(pairs.items())]
        held=[dict(id=u['id'],split=u['split'],cid=u['cid'],role=u['role'],
                   reason=allowed.get(u['id'],'paired partner quarantined'),
                   error=failures[u['id']].get('error') if u['id'] in failures else None)
              for u in units if u['id'].rsplit('/',1)[0] in held_pairs]
        atomic_json(Path(output)/'partner-pairs.json',dict(unit_policy=UNIT_POLICY,pairs=pair_rows))
        atomic_json(Path(output)/'quarantine.json',dict(inventory_sha256=inventory['inventory_sha256'],units=held))
        atomic_json(Path(output)/'quality.json',dict(status='complete_with_quarantine' if held else 'complete',
            requested_units=len(units),validated_units=len(selected),quarantined_units=len(held),pairs=len(pairs),
            recovered_units=len(recovered),
            unit_counts=dict(Counter(u['split']+'/'+u['role'] for u in selected)),
            source_inventory_sha256=inventory['inventory_sha256'],release_manifest_sha256=digest(Path(output)/'manifest.json'),
            partner_pairs_sha256=digest(Path(output)/'partner-pairs.json'),
            checks=['independent native export and convex-hull coverage','source maps and model/runtime','finite arrays and triangle geometry','raw MSMS coverage receipts',
                    'disjoint complete canonical partner partition','TF1/GPU reference','full transfer-release decode']))
    finally:
        for store in stores:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--v5-root',type=Path,required=True)
    p.add_argument('--build',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recovery',type=Path,nargs='+',help='explicit retries of originally failed requests; never replace successful units')
    p.add_argument('--quarantine',type=Path);a=p.parse_args();require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True)
    publish(a.runtime,a.v5_root,a.build,a.output,a.quarantine,a.recovery)
