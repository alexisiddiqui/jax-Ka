"""Write the <=350-word phase-2 report from verified results."""
import json
from pathlib import Path
r=json.loads(Path('results/opt2.json').read_text());assert r['status']=='pass',r['gates']
f=r['final_tag'];runs=r['runs']
lines=['# OPT2 — phase 2','',
'Kept `sor_chunk_packed`/`_packed_block`: GPU red/black packing and byte-code/LUT operators; `sor_chunk`: CPU conditional convergence reduction. '
'`device_solve` compacts at stop blocks without host synchronization. `records` fixes GPU widths at 16/64/128/256 and CPU tiles at 4/8/16. '
'`setup1`/`setup2` select compact GPU operators. Pallas retains its original device solver.','',
'| Gate | Result |','|---|---|',
'| Pallas energies, potentials, sweeps, maxchange | Bitwise: 104 L3 + 20 bench |',
'| XLA / CPU | Bitwise: 12 / 5 |',
'| A40 determinism | Two passes + fresh process, both backends |',
'| CPU / GPU SOR tests | Pass; rejected CPU packing excluded |',
'| A40 warm compiles / audits | Zero / pass |','',
'Seconds/state: geometry/maps/setup/SOR/PB. CPU compilation excluded; GPU warm passes.','',
'| Arm | Baseline | OPT1B | OPT2 |','|---|---|---|---|']
for arm in ('pallas','xla','cpu'):
 def t(v):return '/'.join(f'{v[k]:.2f}' for k in ('geometry','maps','setup','sor','total_pb'))
 lines.append(f"| {arm} | {t(r['profile2'][arm])} | {t(r['opt1b'][arm]['seconds_per_state'])} | {t(runs[f+'-'+arm]['seconds_per_state'])} |")
lines+=['','Peak GPU/host GB; Pallas includes all 104 states.','','| Arm | Baseline | OPT1B | OPT2 |','|---|---|---|---|']
for arm in ('pallas','xla','cpu'):
 before=r['opt1b'][arm]
 after=runs[f+'-all']['all_states'] if arm=='pallas' else runs[f+'-'+arm]
 def mem(x):return ('—' if arm=='cpu' else f"{x['peak_gpu_bytes']/1e9:.2f}")+f"/{x['peak_host_rss_bytes']/1e9:.2f}"
 # OPT1B Pallas bench has the same peak as the full stored reference in opt1b.json below.
 if arm=='pallas':
  prior=json.loads(Path('results/opt1b.json').read_text())['runs']['s5-all']['all_states'];before=prior
 lines.append(f"| {arm} | {mem(r['baseline_memory_control'][arm])} | {mem(before)} | {mem(after)} |")
counts=[runs[f+'-all']['all_states']['compiles'],runs[f+'-xla']['compiles'],runs[f+'-cpu']['compiles']]
lines+=['','Compiled SOR temporary GB: GPU 4.90→3.13 (W256); CPU 8.16→0.58 (W256→16).','',f"Cold compiles (104/12/5): {'/'.join(map(str,counts))} versus 35/36/34. SOR executables: one per width (Pallas 256; XLA 16/64/128/256; CPU 4/8/16).",'',
'Reverted Pallas compaction (4.69 versus 4.15 s/state), CPU packed colours (localised energy error 3.89e-4; changed sweeps) and a split checked sweep (5.25e-6; sweeps equal), both CPU trials beyond tolerance. '
'CPU keeps the original stencil; geometry-site buffers now limit host memory. CPU tile 16 trades 1.4% pool time for smaller buffers than 32.','',
'Per-change measurements, throughput, occupancy, buffers, job IDs and frozen paths: `results/opt2.json`. Phase 3: CPU MC remains untouched; preserve move ordering and measure it separately. No commits; stopped after phase 2.']
report='\n'.join(lines)+'\n';assert len(report.split())<=350,len(report.split())
Path('OPT2-REPORT.md').write_text(report)
print('Report words:',len(report.split()))
