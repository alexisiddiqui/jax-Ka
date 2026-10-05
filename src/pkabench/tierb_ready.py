"""Join export and native-order replay gates without changing prior results."""
import json,os,sys
from pathlib import Path
from .runtime import require_compute,atomic_json,digest

def main(out):
    require_compute(); out=Path(out)
    v=json.loads((out/'verification.json').read_text()); assert v['complete'] and not v['rejected'] and not v['test_data_included']
    assert digest(out/'paired_sites.parquet')==v['paired_sha256'] and digest(out/'native_state_index.json')==v['index_sha256']
    good=json.loads((out/'replay-aligned/result.json').read_text()); bad=json.loads((out/'replay-complex/result.json').read_text()); small=json.loads((out/'replay/result.json').read_text())
    assert small['passed'] and good['passed'] and good['native_site_order_restored']
    assert (good['complex_id'],good['state'])==(bad['complex_id'],bad['state'])
    assert good['source_hashes']==bad['source_hashes'] and good['max_curve_error']==0 and good['max_pka_error']==0
    package=Path(os.environ['PKABENCH_RUNTIME'])/'envs/pypka/lib/python3.10/site-packages/pypka'
    report={'passed':True,'native_data_export_ready':True,'native_order_mc_replay_passed':True,'unmodified_pypka_reload_safe':False,
       'binary_scalar_projection_validated':False,'new_model_trained':False,'test_data_included':False,
       'counts':v['counts'],'states':v['state_statuses'],'unchanged_prediction_rows_checked':v['unchanged_prediction_rows_checked'],
       'limitation':'Replay checked a two-site state and a 16-site histidine-containing complex, not every exported structure. Native site order must be retained.',
       'artifacts_sha256':{n:digest(out/n) for n in ('verification.json','manifest.json','replay/result.json','replay-complex/result.json','replay-aligned/result.json')},
       'installed_source_sha256':{n:digest(package/n) for n in ('main.py','molecule.py','titsite.py','tautomer.py','mc/run_mc.py','mc/mc.pyx')},'code_sha256':digest(Path(__file__))}
    atomic_json(out/'readiness.json',report); print(json.dumps(report,indent=2))
if __name__=='__main__': main(Path(sys.argv[1]))
