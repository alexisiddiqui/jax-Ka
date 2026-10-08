import json
from pathlib import Path
from pkabench import expanded


def test_dispatch_cap_and_resume_avoid_duplicate_jobs(tmp_path,monkeypatch):
    monkeypatch.setattr(expanded,'require_compute',lambda:None)
    monkeypatch.setenv('PKABENCH_RUNTIME',str(tmp_path)); monkeypatch.setenv('USER','test')
    tasks=[]
    for name in ('baseline','repeat','N1'):
        root=tmp_path/'case'/name; root.mkdir(parents=True)
        tasks.append((str(root),'case'))
    (tmp_path/'manifest.json').write_text(json.dumps({'cases':[{}],'tasks':tasks,'production_allowed':False}))
    used=[398]; submitted=[]
    def scheduler(command,**kwargs):
        if command[0]=='squeue': return str(used[0])+'\n'
        if command[0]=='sinfo': return 'comp1400 idle\ncomp1234 drain\ncomp1235 idle\n'
        assert command[0]=='sbatch'
        assert '--exclude=comp1234,comp1400' in command
        submitted.append(command); used[0]+=2
        return str(100+len(submitted))+'\n'
    monkeypatch.setattr('subprocess.check_output',scheduler)
    assert expanded.dispatch(tmp_path)==2 and len(submitted)==1
    assert expanded.dispatch(tmp_path)==2 and len(submitted)==1
    used[0]=396
    assert expanded.dispatch(tmp_path)==0 and len(submitted)==3
    for root,cid in tasks:
        receipt=json.loads((Path(root)/'submission.json').read_text())[f'{cid}/pypka']
        assert receipt['queued_cores_before']+receipt['requested_cpus']<=400
        assert receipt['requested_memory_gib']==2*receipt['requested_cpus']
