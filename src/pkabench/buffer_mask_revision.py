"""Immutable buffer-only 15/20 A migration; preserves all structural preparations."""
import copy
import json
from pathlib import Path
from collections import Counter
from .runtime import require_compute, atomic_json, digest


def apply(source, out):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .glycan_buffer_policy import POLICY, BUFFERS, masks
    from .prep import read_cif
    from .schema import KEY, read_table, write_table
    source=Path(source).resolve(); out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    policy=copy.deepcopy(POLICY)
    policy.update(version='buffer-15-20-v3',buffer_train_A=15.,buffer_eval_A=20.)
    policy['buffer_rule']=policy['buffer_rule'].replace('20/25 A','15/20 A')
    policy['evidence']='User-approved 15/20 A operational buffer flags following buffer-propka-v2. Neutral alcohol/ether pilot only; other eligible buffer chemistry remains an extrapolation. Single-deletion outliers are not ruled out at 15 A.'
    manifest=json.loads((source/'manifest.json').read_text())
    atomic_json(out/'manifest.json',{'source_campaign':str(source),'source_manifest_sha256':digest(source/'manifest.json'),'source_masks_sha256':digest(source/'site_masks.parquet'),'candidates':manifest['candidates'],'policy':policy,'implementation_sha256':digest(Path(__file__)),'production_allowed':False})
    (out/'structures').mkdir(); (out/'rows').mkdir()
    for name in ('sites.parquet','natural-gap-tiers'):
        (out/name).symlink_to((source/name).resolve(),target_is_directory=(source/name).is_dir())
    identity=lambda r:tuple(r[k] for k in KEY)
    previous=pq.read_table(source/'site_masks.parquet').to_pylist(); bycomplex={}
    for r in previous: bycomplex.setdefault(r['complex_id'],[]).append(r)
    final=[]; structures=[]; changed_pairs=[]; gained=Counter(); accepted=0
    for candidate in manifest['candidates']:
        cid=candidate['complex_id']; oldrow=source/'rows'/f'{cid}.json'; row=json.loads(oldrow.read_text())
        if row['status']!='accepted':
            (out/'rows'/oldrow.name).symlink_to(oldrow.resolve()); continue
        accepted+=1; root=out/'structures'/cid; root.mkdir(); oldroot=source/'structures'/cid
        record=json.loads((oldroot/'removed_components.json').read_text()); components=record['components']
        has_buffer=any(c['name'] in BUFFERS for c in components)
        if has_buffer:
            for c in components:
                if c['name'] in BUFFERS:
                    assert c['policy_class']=='buffer'
                    c['train_radius_A']=15.; c['eval_radius_A']=20.
            sites=read_table(oldroot/'sites.parquet'); revised=masks(sites,read_cif(oldroot/'AB.cif'),components)
            before={identity(r):r for r in bycomplex[cid]}; current=[]
            for r in revised:
                old=before[identity(r)]; gap=old['natural_gap_tier'] in ('clean','uncertain')
                new=old|r|{'train_mask':r['component_train_mask'] and gap,'eval_mask':r['component_eval_mask'] and gap}
                for role in ('train','eval'):
                    assert not old[role+'_mask'] or new[role+'_mask']
                    gained[role]+=int(new[role+'_mask'] and not old[role+'_mask'])
                assert not new['eval_mask'] or new['train_mask']
                assert new['natural_gap_tier']==old['natural_gap_tier'] and new['interface']==old['interface']
                current.append(new)
            atomic_json(root/'component_masks.json',revised); changed_pairs.append(cid)
        else:
            current=bycomplex[cid]
            (root/'component_masks.json').symlink_to((oldroot/'component_masks.json').resolve())
        record['policy']=policy; atomic_json(root/'removed_components.json',record)
        provenance=json.loads((oldroot/'provenance.json').read_text())
        provenance.update(stripping_policy=policy,removed_components_sha256=digest(root/'removed_components.json'),source_campaign=str(source),reused_protein_preparation=True)
        atomic_json(root/'provenance.json',provenance)
        for path in oldroot.iterdir():
            if path.is_file() and path.name not in ('component_masks.json','removed_components.json','provenance.json'):
                (root/path.name).symlink_to(path.resolve())
        row['structure']['provenance']=json.dumps(provenance,sort_keys=True)
        atomic_json(out/'rows'/oldrow.name,row); structures.append(row['structure']); final.extend(current)
        assert (root/'AB.cif').resolve()==(oldroot/'AB.cif').resolve()
    assert len(final)==len(previous) and {identity(r) for r in final}=={identity(r) for r in previous}
    write_table(out/'structures.parquet','structures',structures)
    pq.write_table(pa.Table.from_pylist(final),out/'site_masks.parquet')
    assert digest(source/'site_masks.parquet')==json.loads((out/'manifest.json').read_text())['source_masks_sha256']
    report={'accepted':accepted,'candidates':len(manifest['candidates']),'buffer_pairs_recomputed':len(changed_pairs),'gained_sites':dict(gained),'policy':policy,'pipeline_errors':[], 'mask_sha256':digest(out/'site_masks.parquet'),'production_allowed':False,'checks_passed':True,'all_protein_geometry_reused':True,'no_sites_lost':True}
    atomic_json(out/'report.json',report); print(json.dumps(report),flush=True)


def recount(campaigns,out):
    require_compute()
    import os
    import pyarrow.parquet as pq
    from .glycan_buffer_policy import recount as original_recount
    original_recount(campaigns,out)
    out=Path(out); report=json.loads((out/'report.json').read_text())
    previous_path=Path(os.environ['PKABENCH_RUNTIME'])/'universe/glycan-buffer-v2-recount/eligibility-with-existing-assignments.parquet'
    before=pq.read_table(previous_path).to_pylist(); after=pq.read_table(out/'eligibility-with-existing-assignments.parquet').to_pylist()
    old={r['complex_id']:r for r in before}; summary={}
    for r in after:
        p=old[r['complex_id']]
        assert all(r[k]==p[k] for k in ('split','component_id','benchmark_eligible','preparation_status','preparation_code'))
        assert all(r[k]>=p[k] for k in ('train_sites','eval_sites','train_interface_sites','eval_interface_sites'))
    for split in ('train','val','test'):
        role='train' if split=='train' else 'eval'
        def stats(data):
            rr=[r for r in data if r['split']==split and r['benchmark_eligible']]
            return {'interface_pairs':sum(r[role+'_interface_sites']>0 for r in rr),'interface_sites':sum(r[role+'_interface_sites'] for r in rr),'sites':sum(r[role+'_sites'] for r in rr),'usable_groups':len({r['component_id'] for r in rr if r[role+'_interface_sites']>0})}
        summary[split]={'before':stats(before),'after':stats(after)}
    report['versus_buffer20_25']=summary
    report['verification']={'passed':True,'assignments_and_preparations_unchanged':True,'all_retention_counts_nondecreasing':True}
    atomic_json(out/'report.json',report); print(json.dumps({'versus_buffer20_25':summary,'verification':report['verification']},indent=2),flush=True)
