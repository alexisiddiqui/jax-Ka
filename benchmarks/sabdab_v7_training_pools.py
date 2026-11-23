"""Inactive v7 source pools with operational groups and unchanged rank algorithm."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pkabench.runtime import atomic_json,config_hash,digest,require_compute
from pkabench.training_pools import min_fractions,SEED,FRACTIONS


def read(path):return json.loads(Path(path).read_text())


class Union:
    def __init__(self):self.parent={}
    def find(self,key):
        self.parent.setdefault(key,key)
        if self.parent[key]!=key:self.parent[key]=self.find(self.parent[key])
        return self.parent[key]
    def merge(self,a,b):
        x,y=self.find(a),self.find(b)
        if x!=y:self.parent[max(x,y)]=min(x,y)


def write_tsv(path,rows):
    columns=['id','source','operational_role','group','stratum','group_stratum','min_fraction','n_res','unit_id']
    with path.open('w') as f:
        w=csv.DictWriter(f,columns,delimiter='\t');w.writeheader();w.writerows([{k:r.get(k,'') for k in columns} for r in rows])


def run(args):
    require_compute(threads=2)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    plan={r['unit_id']:r for r in pq.read_table(args.audit/'build-manifest-v2/units.parquet').to_pylist()}
    units={r['unit_id']:r for r in pq.read_table(args.audit.parent/'sabdab-v7-roles-v2/audit/units.parquet').to_pylist()}
    assignments={r['unit_id']:r for r in pq.read_table(args.audit/'rescope/proposed-assignments.parquet').to_pylist()}
    original={r['unit_id']:r for r in pq.read_table(args.audit.parent/'sabdab-v7-exposure-v1/inventory-v3/units.parquet').to_pylist()}
    decisions={r['unit_id']:r for r in pq.read_table(args.ownership/'membership.parquet').to_pylist()}
    admission_holds=set(read(args.native_admission/'summary.json')['withheld_ids']) if args.native_admission else set()
    active=[];excluded=[];union=Union();component_roles={}
    for uid,r in sorted(plan.items()):
        include=r['eligible_for_build']
        include &= uid not in admission_holds
        if r['source']=='sabdab':include &= decisions.get(uid,{}).get('membership_action')=='include_native'
        if not include:
            excluded.append(dict(unit_id=uid,source=r['source'],operational_role=r['operational_role'],
                reason='no_qualified_interface_for_operational_role' if uid in admission_holds else decisions.get(uid,{}).get('membership_action',r['graph_action'])))
            continue
        components=assignments[uid]['matched_components'];role=r['operational_role']
        for component in components:
            if component in component_roles and component_roles[component]!=role:raise ValueError('Authoritative component spans operational roles')
            component_roles[component]=role;union.find(component)
        for c in components[1:]:union.merge(components[0],c)
        active.append((uid,r,components))
    rows=[]
    for uid,r,components in active:
        if components:
            group='published:'+union.find(components[0]);stratum='published_abag_component'
        else:
            group='native:'+r['source']+':'+config_hash(sorted(units[uid]['source_groups']))[:20]
            stratum=r['source']+':'+str(original[uid]['stratum'])
        rows.append(dict(unit_id=uid,id=r['source_native_id'],source=r['source'],operational_role=r['operational_role'],
            group=group,stratum=stratum,group_stratum=stratum,n_res=r['n_res'],
            matched_components=components,native_source_split=r['original_source_split'],
            native_component=units[uid]['native_component'],source_groups=units[uid]['source_groups'],release_active=False))
    train=[r for r in rows if r['operational_role']=='train'];val=[r for r in rows if r['operational_role']=='validation']
    if {r['group'] for r in train}&{r['group'] for r in val}:raise ValueError('Operational group spans training and validation')
    strata={r['group']:r['stratum'] for r in train}
    fractions=min_fractions(strata,seed=SEED,version='pool-v7-native-v1')
    for r in rows:r['min_fraction']=round(fractions[r['group']],9) if r['operational_role']=='train' else None
    for source in ['pkpdb','pinder','sabdab']:
        write_tsv(args.output/(source+'.tsv'),[r for r in rows if r['source']==source and r['operational_role']=='train'])
        write_tsv(args.output/(source+'-val.tsv'),[r for r in rows if r['source']==source and r['operational_role']=='validation'])
    pq.write_table(pa.Table.from_pylist(rows),args.output/'membership.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(excluded),args.output/'excluded.parquet',compression='zstd')
    report=dict(status='inactive_v7_operational_pools_complete',version='pool-v7-native-v1',seed=SEED,
        algorithm='pkabench.training_pools.min_fractions: seeded hash rank within stratum, (position+1)/group_count',
        group_policy='union of matched published components; otherwise original source-native assembly family group',
        stratum_policy='published_abag_component for mapped groups; source and existing stratum for unmapped general groups',
        counts={source:dict(Counter(r['operational_role'] for r in rows if r['source']==source)) for source in ['pkpdb','pinder','sabdab']},
        train_groups=len(strata),validation_groups=len({r['group'] for r in val}),
        fractions={str(f):dict(Counter(r['source'] for r in train if r['min_fraction']<=f+1e-12)) for f in FRACTIONS},
        no_new_inverse_group_weights=True,roles_unchanged=True,release_active=False,
        membership_sha256=digest(args.output/'membership.parquet'),
        input_pins={str(p):digest(p) for p in [args.audit/'build-manifest-v2/units.parquet',
            args.audit/'rescope/proposed-assignments.parquet',args.ownership/'membership.parquet']})
    atomic_json(args.output/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['audit','ownership','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--native-admission',type=Path);run(p.parse_args())
