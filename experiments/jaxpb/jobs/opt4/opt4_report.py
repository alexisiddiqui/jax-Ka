"""Write the <=400-word report from fully gated OPT4 evidence (metadata only)."""
import json
from pathlib import Path
r=json.loads(Path('results/opt4.json').read_text())
b1=r['step0']['cpu1'];b12=r['step0']['cpu12'];g=r['gates']
a1=g['cpu1']['passes'][0];a12=g[r['decisions'].get('benchmark_cpu12_gate','cpu12')]['passes'][1]
p1=a1['target_cpu5_mean_pb_stage_sum_seconds'];p12=a12['target_cpu5_mean_pb_stage_sum_seconds']
other12=g['cpu12']['passes'][1]['target_cpu5_mean_pb_stage_sum_seconds']
s1=a1['target_cpu5_mean_stages'];s12=a12['target_cpu5_mean_stages']
peak=max(max(p['peak_host_hwm_kb'],g[k]['accounting_peak_rss_kb']) for k in ('cpu1','cpu12','cpu12_fresh') for p in g[k]['passes'])/1048576
aggregate=max(v['peak_aggregate_rss_kb'] for v in r['compute']['allocation_control']['memory'].values())/1048576
matched=[row for rows in r['additional_evidence']['matched12']['rows'].values() for row in rows]
assert len(matched)==15
m12=sum(row['baseline_s'] for row in matched)/sum(row['selected_s'] for row in matched)
d=r['decisions'];m=d['maps_arms']['5']['speedup'];geo=d['geometry_arms']['2']['speedup'];site=d['model_site_tile_arms']['65536']['speedup']
def stage(s,k):return s.get(k,0)
def sor(s):return stage(s,'sor1')+stage(s,'sor2')
text=f'''# OPT4 report

Five-state mean PB stage sums: **{p1:.1f}s at 1 CPU; {p12:.1f}s at 12 CPUs** (compilation excluded; 12-CPU fresh warm pass, one L3 group), versus Step-0 {b1['mean_pb_stage_sum_seconds']:.1f}/{b12['mean_pb_stage_sum_seconds']:.1f}s. The 12-CPU result is {464/p12:.2f}× faster than 464s and {p12/8.8:.1f}× PyPKA’s 12-CPU 8.8s PB (its total was 25–29s); the 1-CPU ratio is {p1/8.8:.1f}×. The separate two-L3 allocation measured {other12:.1f}s.

| Stage, seconds/state | Before 1/12 | After 1/12 |
|---|---:|---:|
| Maps | 1255.1 / 249.0 | {stage(s1,'maps'):.1f} / {stage(s12,'maps'):.1f} |
| Global geometry | 78.6 / 41.6 | {stage(s1,'geometry'):.1f} / {stage(s12,'geometry'):.1f} |
| Model geometry | 9.0 / 3.5 | {stage(s1,'site_probes'):.1f} / {stage(s12,'site_probes'):.1f} |
| SOR | 473.7 / 164.9 | {sor(s1):.1f} / {sor(s12):.1f} |

Step-0 isolated maps atom/reach/projection/cells: 894/219/100/15.4s (1 CPU), 179/48.4/18.3/4.23s (12). These separately compiled diagnostics are nonadditive; cells overlap stage0. From-scratch geometry search/burial: 25.6/67.0s and 18.5/24.3s; production reuses AB geometry. SOR: .01514/.00527s per useful lane-sweep. Diagnostic splitting alone changed float32 surf (audit max 1.53e-5); masks/counters matched. Original maps fed reference solves.

Keep occupied-slot queries plus wider tiles ({m:.2f}× at five cores; {m12:.2f}× at 12 CPUs; slots alone 1.03× at five cores), direct geometry spans plus compact candidates ({geo:.2f}×; spans alone 1.57×), and 65536-point model tiles ({site:.2f}×). Geometry/site arm studies used five physical cores. Model temporary: 8.52GB → 152MB. Revert inverse reach (0.62–0.76×), 128-slot slabs, smaller geometry chunks, and two pinned three-core workers (0.91×). Defer setup/padding/post: combined setup/post <0.7% of Step-0 PB.

| Gate | Result |
|---|---|
| Five CPU states: maps/geometry/φ/g/energies, full label shards | Bitwise |
| CPU20; A40 Pallas20 and XLA20 | Bitwise |
| CPU12/A40 repeats + fresh processes; warm compiles | Pass; zero growth |
| Five unit groups; peak selected CPU RSS | Pass; {peak:.2f}GiB |

Parent allocation sampled aggregate peaked at {aggregate:.2f}GiB including its held reference process (<24GiB).

GCC8.5/FFI headers existed; clang absent. Native off/fast costs .001117/.001030s per lane-sweep at 1 thread (15.2/16.5× XLA), .000338/.000349s at 12 (13.4/13.0×); XLA costs .017031/.004539s. Neither matches after 1/10/100 sweeps. Full-state φ errors are 5.34e-5/7.78e-3; g errors 7.15e-6/3.89e-4. Fast shifts one stop by ten sweeps despite unchanged checks. Prototype only; no backend flag.

Unchanged SOR sets an approximately {sor(s12):.0f}s/state floor. Observed non-SOR work is {p12-sor(s12):.0f}s/state. Matching native rounding would suggest ~{r['performance']['cpu12']['conditional_native_off_estimate_seconds_per_state']:.0f}s PB, optimistically excluding callback overhead. Phase 5 should match per-operation FMA choices before revisiting remaining queries. MC unchanged; no commits.
'''
assert len(text.split())<=400,len(text.split())
Path('OPT4-REPORT.md').write_text(text);print('Report words:',len(text.split()))
