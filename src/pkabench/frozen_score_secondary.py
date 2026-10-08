"""Secondary pooled summaries and diagnostic figures; group macro stays primary."""
import json
from pathlib import Path
from collections import defaultdict
from .runtime import require_compute, atomic_json, digest
from .schema import key, read_table
from .frozen_score import pairs,measures,write_csv


def run(campaign):
    require_compute(); campaign=Path(campaign)
    manifest=json.loads((campaign/'manifest.json').read_text()); rows=read_table(campaign/'predictions.parquet'); sites=read_table(campaign/'sites.parquet')
    masks={key(r):r for r in read_table(campaign/'site_masks.parquet')}; annotations={key(r):r for r in sites}; assignments={r['complex_id']:r for r in read_table(campaign/'assignments.parquet')}
    ref=pairs(rows,sites,masks,'pypka'); models={m:pairs(rows,sites,masks,m) for m in manifest['methods'] if m!='pypka'}; common=set(ref)
    for m in models.values(): common &= set(m)
    pooled=[]; cancellation=[]; valid_teacher=set()
    for k in ref:
        if masks[k]['interface']: valid_teacher.add(k[0])
    for method,model in models.items():
        for split in ('train','val','test'):
            for scope,keys in (('pairwise',set(ref)&set(model)),('all_method_common',common)):
                for subset in ('interface','shell_0_20'):
                    kk=sorted(k for k in keys if assignments[k[0]]['split']==split and (masks[k]['interface'] if subset=='interface' else annotations[k]['min_partner_distance']<=20))
                    pooled.append({'method':method,'split':split,'scope':scope,'subset':subset,'n':len(kk),**measures([ref[k][0] for k in kk],[model[k][0] for k in kk],[model[k][1]-ref[k][1] for k in kk],[model[k][2]-ref[k][2] for k in kk])})
        for k in sorted(set(ref)&set(model)):
            if masks[k]['interface']:
                cancellation.append({'method':method,'complex_id':k[0],'split':assignments[k[0]]['split'],'ab_error':model[k][1]-ref[k][1],'free_error':model[k][2]-ref[k][2]})
    write_csv(campaign/'scores_pooled_secondary.csv',pooled); write_csv(campaign/'error_cancellation.csv',cancellation)
    atomic_json(campaign/'coverage_gate.json',{'teacher_valid_interface_pairs':len(valid_teacher),'selected_pairs':len(assignments),'teacher_interface_coverage_fraction':len(valid_teacher)/len(assignments),
        'teacher_80_percent_requirement_pass':len(valid_teacher)/len(assignments)>=.8,'definition':'At least one masked-eligible interface site with finite valid teacher AB/free midpoints. Full site coverage is reported separately; not an independent-accuracy gate.','implementation_sha256':digest(Path(__file__))})

    runtimes=[]
    sizes={r['complex_id']:r['n_residues'] for r in read_table(campaign/'structures.parquet')}
    for cid,n in sizes.items():
        for method in manifest['methods']:
            side=json.loads((campaign/'jobs'/method/f'{cid}.json').read_text())
            initial=campaign/'timeout-history'/cid/'initial.json'
            old=json.loads(initial.read_text()) if method=='jaxka' and initial.exists() else None
            runtimes.append({'complex_id':cid,'n_residues':n,'method':method,'status':side['status'],
                'final_attempt_seconds':side['wall_seconds'],'initial_timeout':old is not None,
                'total_observed_attempt_seconds':side['wall_seconds']+(old['wall_seconds'] if old else 0),
                'ab_seconds':side['state_seconds'].get('AB'),'a_seconds':side['state_seconds'].get('A'),'b_seconds':side['state_seconds'].get('B'),
                'node':side['node'],'job':side['job']})
    write_csv(campaign/'runtime_by_residues.csv',runtimes)
