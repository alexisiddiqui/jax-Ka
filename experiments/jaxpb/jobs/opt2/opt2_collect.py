"""Download OPT2 evidence and summarize JSON locally; no numerical solver work."""
import json
import subprocess
import tarfile
from pathlib import Path
from opt1_results import summarize

T = '/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
archive = Path('/tmp/opt2-evidence.tar.gz')
with archive.open('wb') as stream:
    run = subprocess.run(['ssh', 'Coulson-Chemistry', f'cd {T}/opt2 && tar czf - -- *.json *.out'], stdout=stream, check=False)
    if run.returncode not in (0,1): raise RuntimeError('evidence transfer failed')
folder = Path('/tmp/opt2-evidence'); folder.mkdir(exist_ok=True)
with tarfile.open(archive) as tf: tf.extractall(folder, filter='data')
runs = {}
for file in folder.glob('*.json'):
    data = json.loads(file.read_text())
    if data.get('passes') and data.get('compiles'):
        runs[file.stem] = summarize(data)
    elif 'passes' in data:
        rows = data.get('current', {}).get('rows', {})
        runs[file.stem] = dict(status='running', passes=len(data['passes']), states=len(rows),
            exact=all(all(all(check.values()) for check in row['checks'].values()) for row in rows.values()))
    else:
        runs[file.stem] = data
previous=json.loads(Path('/tmp/opt2-summary.json').read_text()) if Path('/tmp/opt2-summary.json').exists() else {}
for tag, row in runs.items():
    if row == previous.get(tag): continue
    if 'seconds_per_state' in row:
        print(tag, row['passed'], row['states'], {k:round(v,3) for k,v in row['seconds_per_state'].items() if k != 'stages'},
              'GB', round(row['peak_gpu_bytes']/1e9,3), round(row['peak_host_rss_bytes']/1e9,3), 'comp', row['compiles_per_pass'])
    elif 'tune' in tag:
        print(tag, row)
    elif tag.startswith('maps-'):
        print(tag, 'classes',len(row['rows']),'passed',row['passed'],'failures',[(r['size_class'],[k for k,v in r['checks'].items() if not v]) for r in row['rows']])
    elif tag.startswith('memory-'): print(tag, 'programs', len(row.get('programs', [])), 'job', row.get('job'))
    elif tag.startswith('rejected-numerics-'): print(tag, 'sweeps exact', row.get('sweeps_bitwise'), 'energy max abs', max(v['max_abs'] for v in row['energies'].values()))
    elif 'rows' not in row: print(tag, row)
Path('/tmp/opt2-summary.json').write_text(json.dumps(runs, indent=2))
