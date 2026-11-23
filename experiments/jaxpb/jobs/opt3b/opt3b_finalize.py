"""Local transport, metadata audit and report formatting. Scientific statistics come from Slurm summary.json."""
import datetime
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'

def ssh(cmd):
    return subprocess.check_output(['ssh','Coulson-Chemistry',cmd],text=True)

def main():
    r=json.loads(ssh('cat '+T+'/opt3b/summary.json'))
    assert r['passed'], 'Inspect failed gates before finalizing'
    rows=json.loads(Path('results/opt3b-jobs.json').read_text())
    assert all('--mem-per-cpu=2G' in x['command'] and x['cpus']<=12 for x in rows)
    cpu_data=json.loads(ssh('cat '+T+'/opt3b/cpu-final2.json'))
    gpu_data=json.loads(ssh('cat '+T+'/opt3b/guard2.json'))
    ref_data=json.loads(ssh('cat '+T+'/opt3b/ref-final2.json'))
    assert all(gpu_data['energy_hashes'][sid]==value for sid,value in cpu_data['energy_hashes'].items())
    r['inputs']=dict(energy_paths=gpu_data['energy_paths'],energy_hashes=gpu_data['energy_hashes'],seed=1,noise_seed=2,
                     key_scheme='chain_keys(seed, chain_id_of(alias)*73+pH_index)',baseline_source=cpu_data['baseline_source'],baseline_sha256=cpu_data['baseline_sha256'])
    r['environments']=dict(cpu=cpu_data['env'],cpu_ref=ref_data['env'],gpu=gpu_data['env'])
    r['timings']['pallas_per_bucket']={label:{arm:next(x for x in gpu_data['runs'] if x['label']==arm+'-'+label)['buckets'] for arm in ('old','new','noise')} for label in ('bench200k','l3')}
    r['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    diff=subprocess.check_output(['git','diff','a5d5c1d','--','src/jaxpb/titrate.py','src/jaxpb/api.py'])
    r['source_diff_sha256']=hashlib.sha256(diff).hexdigest()
    ids=','.join(x['job'] for x in rows)
    cmd='sacct -X -P -j '+ids+' --format=JobIDRaw,JobName,Partition,State,ExitCode,AllocCPUS,ReqCPUS,ReqMem,ReqTRES,AllocTRES,Elapsed,NodeList,Timelimit'
    accounting=ssh(cmd)
    records=[]
    lines=accounting.splitlines(); header=lines[0].split('|')
    for line in lines[1:]:
        fields=dict(zip(header,line.split('|')))
        if not fields.get('JobIDRaw'):continue
        assert fields['Partition'] in ('amd96','oc'),fields
        assert int(fields['AllocCPUS'])<=12,fields
        if int(fields['AllocCPUS']) and fields['Partition']=='oc':
            assert fields['NodeList']=='comp1400',fields
        records.append(fields)
    selected=['809582','809625','809628','809629','809630','810116','810117','810443','810444']
    for jid in selected:
        q=next(x for x in records if x['JobIDRaw']==jid)
        assert q['State']=='COMPLETED' and q['ExitCode']=='0:0',q
    # Check both selected immutable source snapshots against their submission manifests.
    remote_checks={}
    local_checks={}
    for jid in ('809582','809625','810443','810444'):
        row=next(x for x in rows if x['job']==jid)
        paths=list(row['hashes'])
        sums=ssh('cd '+row['frozen']+' && sha256sum '+shlex.join(paths))
        got={line.split()[1]:line.split()[0] for line in sums.splitlines()}
        assert got=={p:row['hashes'][p] for p in paths}
        remote_checks[jid]=True
        if jid=='809625':
            local_checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==row['hashes'][p] for p in paths if p.startswith(('src/','tests/'))}
            assert all(local_checks.values()),'Local source/tests changed since validation'
    audit=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,records=records,
               selected_completed=selected,remote_source_checks=remote_checks,local_source_checks=local_checks,
               policy='Coulson SSH only; amd96 or oc/comp1400/A40; <=12 CPUs; --mem-per-cpu=2G; existing env and A40 shim; no numerical work on login/local host, PB reruns, installs, downloads, commits or pushes.')
    Path('results/opt3b-compute-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    r.update(status='complete',completed_utc=audit['utc'],design='Explicit sum of scalar site differences; CPU scalar indexing / shared ref-Pallas two-nonzero masked differences. No version, ISA, flags, bucket or cardinality eligibility gates. Scalar CPU eager site acceptance; shared ref/Pallas skips already-rejected unchanged site proposals.',
             default='auto: Pallas on a supported GPU, cpu on CPU, ref otherwise',
             portability={
                 'pair_energy':'Round each site subtraction first, then add the two scalar differences. Each masked difference has at most two nonzeros, so every reduction tree equals the scalar subtraction; equal indices cancel in their cell.',
                 'current_state':'cur_of selects one int32 index among zeros; no overflow or floating arithmetic. CPU direct indexing is identical.',
                 'rebuild_updates':'Rebuild is a sequential loop in site order. E/X row updates are elementwise with unchanged parentheses. No reduction across sites or moved rows.',
                 'counts':'X and flag are 0/1; counts are sequential exact integer f32 additions below 2**24, enforced by public sweep limit. Host f64 counts_to_prot reduces bounded integer counts exactly; curves/pKa use common host routines.',
                 'remaining':'invexp: x*x, x2*x, x2*x2 and numerator/denominator multiply-add polynomials may contract to FMAs differently; GPU division can use different approximations. Nonstandard fast-math settings can additionally reassociate scalar pair/W or E-update arithmetic; subnormal handling can differ. No runtime eligibility gates assume otherwise.',
                 'unchanged_proposals':'The shared-chain shortcut rejects exactly the proposals already rejected by nw==oc. RNG draws, counters and accepted E/X/current/count updates are unchanged. CPU eager evaluation has the identical logical rule.'},
             input_policy='Saved bench/gpu/energies_gpu.npz and M6 runA labels only; seed=1 versus old seed=2; alias chain ids; 20 bench/50k + CPU5/200k on CPU; 104 L3 + 20 bench/200k Pallas.',
             source_audit='results/opt3b-source-audit.json',compute_audit='results/opt3b-compute-audit.json',
             decisions='results/opt3b-decisions.json',jobs='results/opt3b-jobs.json',polls='results/opt3b-squeue.jsonl',
             evidence='results/opt3b-evidence.tar.gz',frozen_paths=sorted({x['frozen'] for x in rows}),
             selected_jobs=selected,limitations=['Timings are non-exclusive Slurm allocations. The one-requested-CPU timing allocation exposes two SMT CPUs (affinity 2), as in phase 3; the CPU12 timing has affinity 12. Cross-backend agreement is measured, not a gate.'])
    Path('results/opt3b.json').write_text(json.dumps(r,indent=2)+'\n')
    archive = Path('results/opt3b-evidence.tar.gz')
    partial = archive.with_suffix(archive.suffix+'.part')
    with partial.open('wb') as out:
        subprocess.run(['ssh','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',
                        'Coulson-Chemistry','tar -C '+T+' -czf - opt3b'],stdout=out,check=True)
    partial.replace(archive)
    table=[]
    for key,label in [('cpu-50k','CPU 20/50k'),('cpu-200k','CPU 5/200k'),('pallas-bench200k','A40 20/200k'),('pallas-l3','A40 L3 104/200k')]:
        v=r['rebaseline'][key];a,b=v['change'],v['seed_noise']
        table.append(f"| {label} | {a['differing_curve_counts']} / {b['differing_curve_counts']} | {a['max_abs_delta_pka']:.6g} / {b['max_abs_delta_pka']:.6g} |")
    cross=r['cross_backend']
    same=all(all(cross[lab][k].values()) for lab in ('50k','200k') for k in ('pallas_vs_gpu_ref','pallas_vs_cpu','gpu_ref_vs_cpu')) and all(cross['bench200k']['pallas_vs_cpu'].values())
    if same:
        cross_text='Pallas, GPU ref and CPU cpu/ref agree bitwise on 20/50k and 5/200k; CPU/Pallas also agree on 20/200k.'
    else:
        a=cross['50k']['differences']['pallas_vs_cpu'];b=cross['200k']['differences']['pallas_vs_cpu'];c=cross['bench200k']['differences']
        cross_text=f"Pallas/GPU ref agree bitwise on 20/50k + 5/200k. Pallas/CPU differs in {a['differing_raw_counts']} raw/{a['differing_curve_counts']} curve counts at 50k; {b['differing_raw_counts']}/{b['differing_curve_counts']} on 5/200k; {c['differing_raw_counts']}/{c['differing_curve_counts']} on 20/200k. Locations are in JSON."
    l=r['l3_teacher'];t=r['timings'];p=t['pallas']
    report=f'''# OPT3B

Pair energy is `(E[na]-E[oa])+(E[nb]-E[ob])`: ordered site differences, exact unchanged-site cancellation. Ref/Pallas uses two-nonzero masked differences; CPU uses scalar picks and ≤12 host threads. Deleted runtime/domain eligibility gates. Auto: supported GPU→Pallas; CPU→cpu; otherwise ref.

Ref/Pallas skips unchanged-site evaluation; CPU evaluates eagerly. RNG, moves, BLOCK, accept rule and E/X/count updates are unchanged.

| Gate | Result |
|---|---|
| CPU/new ref | Bitwise: 20/50k + 5/200k (counts/curves/pKa/shard fields) |
| Determinism/API | Repeats, fresh process and API pass |
| Compiles | Five warm executables; zero later |
| Tests | Pass: nst11, Mp23/2048, single-site, partial pairs, brute/interpreter/API |
| Speed/rebaseline | Pass |

Change / old-code seed noise:

| Arm | Differing curve counts | max ΔpKa |
|---|---:|---:|
{chr(10).join(table)}

A40 raw-count changes: bench {r['rebaseline']['pallas-bench200k']['change']['differing_raw_counts']}, L3 {r['rebaseline']['pallas-l3']['change']['differing_raw_counts']}.

L3 teacher MAE: old **{l['old']['mae']:.8f}**, new **{l['new']['mae']:.8f}**, second seed **{l['noise']['mae']:.8f}**. Drift lies within seed noise.

{cross_text}

Remaining portability limits: invexp FMA/division. Other reductions are exact or sequential; arguments are in JSON.

200k MC seconds/state: CPU1 **{t['cpu1']['s_per_state']:.2f}**, CPU12 **{t['cpu12']['s_per_state']:.2f}**, versus phase 3 **238.5 / 41.7**. A40 bench **{p['bench200k']['old']:.3f}→{p['bench200k']['new']:.3f}**; L3 **{p['l3']['old']:.3f}→{p['l3']['new']:.3f}**. Non-exclusive allocations; CPU1 exposes two SMT threads, matching phase 3.

Direct-indexing lowering, four-pick overhead and scalar CPU branching were rejected. Saved energies; audits/evidence accompany `results/opt3b.json`. Uncommitted; stopped after OPT3B.
'''
    assert len(report.split())<=300,len(report.split())
    Path('OPT3B-REPORT.md').write_text(report)
    print('Finalized OPT3B; report words:',len(report.split()))

if __name__=='__main__':main()
