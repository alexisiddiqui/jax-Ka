# jaxpb-pipeline: profile of the CURRENT code, Pallas vs XLA vs CPU — 2026-11-21 (user request)

Code: the shipped L12347F src (identical to jax-Ka branch `jaxpb` @ 8c7435e and the frozen Coulson copy
tierB/jaxpb-pipeline/bench-2026-11-21; sha-check). Previous profile (pre-speed-ups): results/profile.json, method in
jobs/prof_geomaps.py (update it for the current code in a COPY; numerics must not change). Current stage split on the A40 (bench):
geometry 2.06, maps 3.61, solves 4.35 (SOR 4.15), post 0.10 = 10.53 s/state.

## Arms (same states, same code)
1. A40, Pallas backend (SOR + MC Pallas kernels) — production config.
2. A40, XLA backend (backend="xla": XLA reference SOR + XLA MC; everything else identical).
3. CPU, XLA (JAX CPU, 12 Slurm CPUs, --mem-per-cpu=4G as the bench needed; AMD EPYC node as the bench). CPU is ~160x slower on PB:
   use a SUBSET of ~6 states spanning classes 0-2 incl. one complete AB/A/B triple (not the largest d6004 AB); state this.
GPU arms: the bench's 20 states. Warm numbers (second pass); compile counts/time reported separately.

## Measure (per arm, s/state and share)
- Stage split: prep(load), geometry, maps (coarse/focused/model), solve setup, SOR, post, assembly, MC 200k (and 50k).
- Inside SOR: sweeps per lane, lane occupancy / waste (padded or finished lanes), launches, achieved time per sweep per grid point;
  for Pallas vs XLA the per-sweep cost ratio. Inside maps and geometry: the per-op split as in profile.json
  (jax.profiler trace or block_until_ready-timed sub-jits on the copy).
- GPU idle share per stage (arms 1-2), GPU peak bytes per stage.
- CPU arm: per-stage peak RSS and which buffers dominate the 21-25 GB (maps B=16 batch? XLA temporaries?) — measure, e.g. by
  running maps with B=4/8/16 on one state (timing-only, results discarded) and reading memory stats.
- MC: Pallas vs XLA vs CPU per-sweep cost at each Mp bucket at production occupancy (S=32).

## Rules
Coulson only. A40: partition oc, comp1400, 1 GPU, <= 12 CPUs, --mem-per-cpu=2G, env ogqt-main-7c3e3e9 + jobs/a40site shim.
CPU: Coulson CPU partition as the bench, <= 12 CPUs. Never touch other sessions' jobs (pkai-scale-*, masif-*, pool5-*) or
~/jax-Ka. No src/ edits (profiling wrappers in jobs/ or a profiling copy). New folders: frozen copy
tierB/jaxpb-pipeline/a40-2026-11-21-prof2/, outputs tierB/jaxpb-pipeline/prof2/. Keep at most a couple of compressed profiler
traces. No installs/downloads/envs/git commits; few files. Results -> results/profile2.json (harness blocks .md). If handed back
while jobs run, write numbers so far (IN PROGRESS) with job ids; wait with a background until-loop, not repeated polling.

## Output
results/profile2.json and a final message (<= 350 words): one table per arm (stage, s/state, share), Pallas-vs-XLA speed-up per
stage, CPU-vs-A40 per stage, SOR lane waste, GPU idle, memory breakdown (GPU per stage; CPU what drives 21-25 GB), and a ranked
list of the next levers with estimated gains (labelled estimates).
