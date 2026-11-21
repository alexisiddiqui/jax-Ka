"""Review preparation outliers and plan a complete-assembly MaSIF release."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore
from masif_pkpdb_recovery_release import validate
from masif_preparation_compare import frame,compare_arrays
from masif_pkpdb_abag import membership


def review(base,output):
    base=Path(base).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True);r=base/'masif-pkpdb-recovery-v1'
    panel=json.loads((r/'topology/replay-comparison.json').read_text());native=r/'topology/native-replay'
    units={u['id']:u for u in json.loads((r/'topology/native-inventory.json').read_text())['units']}
    old={u['id']:u for u in json.loads((base/'masif-pkpdb-cache-audit-v2/full/catalog.json').read_text())['entries']}
    report=json.loads((native/'report.json').read_text());profile={k:report[k] for k in ['checkpoint_sha256','paper_runtime_sha256','preprocessing_profile']}
    from pkatrain.production_sidechains import _pkpdb_atoms
    cases=[];overrides=[];hold=[]
    worst=sorted(panel['cases'],key=lambda c:c['old_to_current_native']['all']['median_cosine'])[:2]
    worst_ids={c['id'] for c in worst}
    with AugmentationStore(native/'fingerprints') as store:
        for case in panel['cases']:
            rid=case['id'];record=store.raw(rid);row=dict(case,worst_case=rid in worst_ids)
            try:
                row['independent_QA']=validate(units[rid],record,base.parent/'pkabench',profile)
                row['status']='validated'
            except Exception as exc:row.update(status='held',error=str(exc))
            bb,_,_=_pkpdb_atoms(base.parent/'pkabench',units[rid]['cid'])
            ids=record['metadata']['source']['added_terminal_oxygen_canonical_ids']
            centers=bb[ids,2] if ids else np.empty((0,3))
            row['terminal_local_comparison']=compare_arrays(frame(old[rid])['arrays'],record['arrays'],centers)
            row['residues']=len(record['arrays']['canonical_residue_ids'])
            # These were the two pre-identified topology/preparation extrema.
            # Upgrade only independently validated replays, without extrapolating
            # a numerical cosine threshold to all old records.
            if rid in worst_ids:
                if row['status']=='validated':overrides.append(rid)
                else:hold.append(rid)
            cases.append(row)
    atomic_json(output/'outlier-review.json',dict(status='complete',cases=cases,override_ids=sorted(overrides),hold_ids=hold,
        override_store=str(native/'fingerprints'),source_panel_sha256=digest(r/'topology/replay-comparison.json'),
        native_manifest_sha256=digest(native/'fingerprints/manifest.json'),
        decision='Reuse fully audited legacy frames with explicit original preparation tags; retain topology review flags. Replace the two pre-identified preparation extrema with independently validated current-profile outputs, or withhold them if validation fails.',
        limitations=['The diagnostic panel does not certify every nonmanifold surface',
            'Stable repeats cannot exclude a shared topology defect',
            'Terminal-local metrics here use actual added-OXT centers; the earlier topology-only comparison used an empty terminal-center list']))


def validation_chains(task):
    runtime,row=task
    if row.get('dataset')=='pinder':
        # Same reader arguments and load_topology call as production_inputs,
        # without importing the model-training module to obtain its CIF reader.
        import gzip
        from biotite.structure.io import pdbx
        from jaxpropka.topology import load_topology
        from pkatrain.production_graphs import source,PINDER
        path=source(Path(runtime),PINDER+'/entries/'+row['id'])/'AB.cif.gz'
        with gzip.open(path,'rt') as stream:file=pdbx.CIFFile.read(stream)
        atoms=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True,include_bonds=True)
        topology=load_topology(atoms,gap_policy='cap',freeze_disulfides=True);chain=np.asarray(topology.chain_index)
    else:
        from pkatrain.production_inputs import _residues
        _,chain,_=_residues('pkpdb',Path(runtime),row['id'])
    return row['id'],sorted(int(x) for x in np.unique(chain)),len(chain)


def route_parent(row,requirements,available,forced=None):
    missing=[u['id'] for u in requirements if u['id'] not in available]
    reasons=list(forced or [])
    if missing:reasons.append('missing_required_surfaces')
    if not requirements:reasons.append('no_canonical_protein_units')
    return dict(dataset=row['dataset'],cid=row['id'],split=row['split'],group=row['group'],
        status='held' if reasons else 'covered',required_units=requirements,missing_units=missing,reasons=reasons,
        canonical_residues=int(row['n_res']))


def plan(base,v5,output,workers=8):
    base=Path(base).resolve();v5=Path(v5);output=Path(output);runtime=base.parent/'pkabench';audit=base/'masif-pkpdb-cache-audit-v2'
    output.mkdir(parents=True,exist_ok=True);reviewed=json.loads((output/'outlier-review.json').read_text())
    members,hashes=membership(runtime);pkpdb={m['id']:dict(m,dataset='pkpdb') for m in members}
    pool=json.loads((v5/'pool/pool-v5/pool-v5.json').read_text());pinder=[];pins=dict(hashes)
    for split,name,key in [('train','pinder.tsv','pool'),('val','pinder-val.tsv','validation')]:
        path=v5/'pool/pool-v5'/name;sha=digest(path)
        if sha!=pool['datasets']['pinder'][key]['sha256']:raise ValueError('v5 PINDER membership changed')
        pins['v5/'+name]=sha
        pinder.extend(dict(m,dataset='pinder',split=split) for m in csv.DictReader(path.open(),delimiter='\t'))
    if not pool['base']['pkpdb'].startswith('unchanged:'):raise ValueError('v5 pKPDB is no longer inherited unchanged')
    cat=json.loads((audit/'full/catalog.json').read_text());summary=json.loads((audit/'full/summary.json').read_text())
    if digest(audit/'full/catalog.json')!=summary['catalog_sha256'] or digest(audit/'full/records.jsonl')!=summary['results_sha256']:
        raise ValueError('complete audit evidence changed')
    records={r['id']:r for r in (json.loads(line) for line in (audit/'full/records.jsonl').read_text().splitlines())}
    hard={rid for rid,r in records.items() if r.get('hard_rebuild_reasons') or r['status']!='audited'}|set(reviewed['hold_ids'])
    available={};sources={};common=None
    def add_store(path,allowed=None,label='current'):
        nonlocal common
        path=Path(path)
        with AugmentationStore(path) as store:
            p=store.manifest['provenance'];scientific={k:p[k] for k in ['checkpoint_sha256','paper_runtime_sha256']}
            scientific['preprocessing_profile']=p.get('preprocessing_profile','masif-paper-2019-v1')
            if common is not None and scientific!=common:raise ValueError('source model/runtime mismatch')
            common=scientific;sources[str(path)]=digest(path/'manifest.json')
            for row in store.rows:
                if allowed is not None and row['id'] not in allowed:continue
                s=dict(row['summary']);cid=s['cid'];dataset=s['dataset'];m=pkpdb[cid] if dataset=='pkpdb' else next_pinder[cid]
                if s['split']!=m['split']:raise ValueError('source surface split differs from pool')
                compact={k:s[k] for k in ['dataset','cid','split','state','role','chain_index','chain_indices','unit_policy','n'] if k in s}
                compact.update(group=m['group'],source_kind=label,surface_policy=p.get('surface_policy','original-v1'))
                if label=='legacy':
                    a=records[row['id']];compact.update(topology_review=a.get('review_reasons',[]),preparation_review=a.get('preparation_compatibility'))
                available[row['id']]=dict(row,path=str(path/row['shard']),source_root=str(path),summary=compact)
    next_pinder={m['id']:m for m in pinder}
    # Metadata readers check each original index/manifest; all selected frame
    # hashes are checked again while packing.
    allowed_legacy=set(records)-hard
    for root,sha in cat['source_manifests'].items():
        if digest(Path(root)/'manifest.json')!=sha:raise ValueError('legacy source manifest changed')
        add_store(root,allowed_legacy,'legacy')
    catalog={e['id']:e for e in cat['entries']}
    for rid,e in available.items():
        if any(e[k]!=catalog[rid][k] for k in ['sha256','bytes','raw_bytes','offset']):raise ValueError('legacy frame differs from audited catalog')
    sources_current=[base/'masif-pkpdb-abag-v1/release',base/'masif-pinder-abag-v1/release',
        base/'masif-pkpdb-recovery-v1/candidate-release',base/'masif-pkpdb-recovery-v1/resource-candidate-release']
    for path in sources_current:add_store(path)
    add_store(reviewed['override_store'],set(reviewed['override_ids']),'reviewed_preparation_override')
    # Protein chains omitted by the original source audit are not required.
    cohort=json.loads((base/'masif-pkpdb-train-v1/cohort.json').read_text());chain_lists=defaultdict(list)
    if cohort['training_tsv_sha256']!=hashes['pkpdb.tsv']:raise ValueError('original protein-chain cohort membership changed')
    for u in cohort['units']:chain_lists[u['cid']].append(u['chain_index'])
    val=[m for m in members if m['split']=='val'];validation_counts={}
    with ProcessPoolExecutor(workers) as executor:
        for cid,chains,n in executor.map(validation_chains,[(str(runtime),m) for m in val]):chain_lists[cid]=chains;validation_counts[cid]=n
    # Grouped inputs replace every corresponding per-chain feature, whether
    # the grouped output succeeded or remains explicitly quarantined.
    grouped_inventory=json.loads((base/'masif-pkpdb-abag-v1/inventory/inventory.json').read_text())
    group_specs=defaultdict(list)
    for u in grouped_inventory['units']:group_specs[u['cid']].append(dict(id=u['id'],chains=u['chain_indices'],role=u['role'],state=u['state']))
    for u in json.loads((base/'masif-pkpdb-recovery-v1/inventory/antibodies.json').read_text())['units']:
        group_specs[u['cid']].append(dict(id=u['id'],chains=u['chain_indices'],role='Ab',state='Ab'))
    parent_rows=[];selected=set();replacement_chains=set()
    for cid,m in sorted(pkpdb.items()):
        specs=group_specs[cid];used=[]
        for spec in specs:used.extend(spec['chains'])
        if len(used)!=len(set(used)):raise ValueError('overlapping grouped pKPDB chain selections')
        if not set(used)<=set(chain_lists[cid]):raise ValueError('grouped inventory contains absent canonical chain')
        replacement_chains.update('pkpdb/'+cid+'/chain-%04d'%i for i in used)
        req=list(specs)+[dict(id='pkpdb/'+cid+'/chain-%04d'%i,chains=[i],role='protein',state='chain-%04d'%i) for i in sorted(chain_lists[cid]) if i not in used]
        row=dict(m)
        if cid in validation_counts:row['n_res']=validation_counts[cid]
        item=route_parent(row,req,available)
        if item['status']=='covered':selected.update(u['id'] for u in req)
        parent_rows.append(item)
    # Expected Ab/Ag membership includes the quarantined pair too.
    from masif_partner_units import inventory_v5
    antibody_units,_=inventory_v5(v5,runtime);ab_specs=defaultdict(list)
    for u in antibody_units:ab_specs[u['cid']].append(dict(id=u['id'],state=u['state'],role=u['role']))
    # Pool n_res is source metadata and can include residues omitted by the
    # canonical protein topology. Match the actual production training order,
    # rather than withholding valid mapped partners over metadata counts.
    pinder_canonical={}
    with ProcessPoolExecutor(workers) as executor:
        tasks=[(str(runtime),m) for m in pinder if m['id'] in ab_specs]
        for cid,_,n in executor.map(validation_chains,tasks):pinder_canonical[cid]=n
    for m in pinder:
        req=ab_specs[m['id']] if m['stratum']=='Ab/Ag' else [dict(id='pinder/'+m['id']+'/'+state,state=state,role='protein') for state in ['A','B']]
        row=dict(m)
        if m['id'] in pinder_canonical:row['n_res']=pinder_canonical[m['id']]
        item=route_parent(row,req,available)
        item['membership_residue_count']=int(m['n_res'])
        if item['status']=='covered':selected.update(u['id'] for u in req)
        parent_rows.append(item)
    chosen=[available[rid] for rid in sorted(selected)];by_split={}
    for dataset in ['pkpdb','pinder']:
        for split in ['train','val']:
            rows=[p for p in parent_rows if p['dataset']==dataset and p['split']==split]
            by_split[dataset+'/'+split]=dict(parents=len(rows),covered=sum(r['status']=='covered' for r in rows),held=sum(r['status']!='covered' for r in rows),
                required_surfaces=sum(len(r['required_units']) for r in rows),available_surfaces=sum(u['id'] in available for r in rows for u in r['required_units']))
    atomic_json(output/'coverage.json',dict(format='masif-native-assembly-coverage-v1',membership_sha256=pins,parents=parent_rows))
    atomic_json(output/'selection.json',dict(format='masif-native-selected-frames-v1',entries=chosen,source_manifests=sources,
        scientific_profile=common,coverage_sha256=digest(output/'coverage.json'),audit_results_sha256=summary['results_sha256'],
        outlier_review_sha256=digest(output/'outlier-review.json')))
    atomic_json(output/'summary.json',dict(status='inventory_complete',coverage=by_split,selected_surfaces=len(chosen),
        selected_frame_bytes=sum(e['bytes'] for e in chosen),replaced_chain_ids=len(replacement_chains),
        profile_counts=dict(Counter(e['summary']['surface_policy'] for e in chosen)),
        retained_topology_review_records=sum(bool(e['summary'].get('topology_review')) for e in chosen),
        filtered_available_surfaces=len(set(available)-selected),
        policy='complete training assemblies only; required partner failure withholds the assembly; explicit mixed preparation profiles; no membership rewrite'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['review','plan']);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--v5',type=Path);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True)
    if a.command=='review':review(a.base,a.output)
    else:plan(a.base,a.v5,a.output,a.workers)
