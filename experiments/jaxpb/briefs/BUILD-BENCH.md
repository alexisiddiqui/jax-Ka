# jaxpb-pipeline: speed + memory benchmark, native PyPKA vs jaxpb on CPU and GPU — 2026-11-21 (user request)

Goal: a like-for-like table of time and memory for the SAME states through
  A. native PyPKA (DelPhi PB + its Cython MC, campaign settings: 200k sweeps, 73 pH points, seed 1234567, couple_min 2.0) on CPU;
  B. jaxpb shipped config (L12347F, src/ as of results/speed.json) on CPU (JAX CPU backend: XLA reference SOR + XLA MC; Pallas is
     GPU-only — say so; if Pallas interpret mode is usable for MC on CPU, report it only as an aside);
  C. jaxpb shipped config on the Coulson A40 (Pallas SOR + MC).

States: the 20 profile states (results/profile.json; classes 0-3, >= 3 complete AB/A/B triples). Use the same PQR/records inputs
PyPKA used in the campaign (M0 shards / mf-surrogate inputs) so A and B/C start from identical structures.

## Measure, per state and as totals
- Wall time and CPU-seconds, split into: structure prep (PyPKA's own prep vs jaxpb prep_pypka), PB/energies, (jaxpb) assembly,
  titration/MC. Report MC at 200k for all arms; jaxpb also at 50k. If jaxpb-CPU MC at 200k is too slow for 20 states, run it on
  >= 5 states and say so explicitly.
- Peak memory: host peak RSS of the whole process tree (PyPKA spawns workers — sum them; use /usr/bin/time -v, cgroup
  memory.peak / sacct MaxRSS, or psutil sampling, and say which), GPU peak bytes for C. Per-state peak where the code allows.
- Cold vs warm: JAX compile time and compile counts reported separately; warm numbers exclude compiles; give cold totals too.
- Core scaling: A and B at 1 core and 12 cores (PyPKA ncpus=12, pinned one core per worker — see pkabench CPU pinning note; JAX
  CPU with XLA intra-op threads = 12). Same CPU node type for A and B; record the CPU model.
- Derived: states per node-hour (12-core node vs 1 A40), core-hours per state, projected cost for 30,000 complexes (90k states)
  for each arm, labelled as projections.
- Sanity: jaxpb CPU outputs vs jaxpb GPU (energies max abs diff, pKa MAE) and vs PyPKA (pKa MAE) on these states, so speed is
  not bought with a broken run.

## Rules
Coulson only (Isambard credits out). A40: partition oc, comp1400, 1 GPU, <= 12 CPUs, --mem-per-cpu=2G, env
~/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9 + jobs/a40site shim. CPU arms: a Coulson CPU partition, <= 12 CPUs,
--mem-per-cpu=2G (raise per-CPU memory only if a run OOMs, and report it). PyPKA from the env the campaign used. Never touch other
sessions' jobs (pkai-scale-train 747977, masif-*, anything not yours) or ~/jax-Ka's working tree. Frozen code copy for kept runs
under ~/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/bench-<tag>/; outputs under tierB/jaxpb-pipeline/bench/. No src/ edits (timing
wrappers in jobs/). No installs/downloads/envs/git commits; few files. Results -> results/bench.json (harness blocks .md). If you
are handed back while jobs run, write numbers so far to results/bench.json (IN PROGRESS) with job ids.

## Output
results/bench.json and a final message (<= 300 words): one table (arm x {prep, PB, MC 200k, total} s/state warm, cold total,
core-h/state, peak host RSS, peak GPU), the scaling numbers, the 30k projection per arm, sanity diffs, and caveats.
