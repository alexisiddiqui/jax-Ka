# jaxpb-pipeline: geometry + maps profile — 2026-11-20 (user approved: profile first, no optimisation yet)

Context: M6 (results/M6.json) on the A40: PB 13.3 s/state warm = geometry 4.9 s + maps 4.2 s + SOR 3.5 s (+ setup/post);
titration 1.5 s/state; GPU peak 15.9 GB of 48. Geometry (src/jaxpb/geometry.py) runs entirely in float64 (A40 FP64 ≈ 1/64 of FP32;
GH200 ≈ 1/2) and builds the overlapping-pair list on the host with scipy cKDTree. Maps (src/jaxpb/maps.py, maps_batch) are float32,
one vmapped program per class for coarse + focused + model maps of B = 16 sites.

## Questions to answer (measure, don't optimise)
1. Per-operation GPU time inside geometry and maps (use jax.profiler / XLA trace, or block_until_ready-timed sub-jits on a COPY of
   the code; numerics must not change): geometry = host pair list, geo_pairs, geo_vertex (burial), geo_arc, finish/pack, site probes;
   maps = cells/hash, region_probes, reach_mask, query_min / brute_min, boundary projection; for coarse vs focused vs model.
2. Host/GPU overlap: fraction of wall where the GPU is idle (host prep, cKDTree, transfers, Python dispatch) per stage.
3. FP64 cost: run the geometry kernels with float32 inputs as a timing-only experiment (results discarded) to bound what an FP32
   path could save on the A40; also report the FP64 op count / share if the profiler gives it. State clearly this is A40-specific
   and how it should change on GH200 (FP64 at half rate).
4. Map culling potential: on the focused, coarse and model grids of ≥ 20 sites (several classes), the fraction of 4³ (and 8³)
   blocks that are entirely inside (all midpoints eps_in), entirely outside (eps_out and no ion-boundary change), or mixed — from
   the CURRENT maps, so it is exact. Also the fraction of points that need the probe (re-entrant) test.
5. Cross-state reuse potential for AB/A/B triples: fraction of A's and B's probe centres that also appear in AB's set (exact
   coordinate match), and how much of geometry time is spent on probes common to the triple.
6. Concurrency headroom: GPU utilisation (SM busy) during maps and SOR; would two states per GPU overlap usefully (estimate from
   idle fractions; optionally one 2-process test on the same A40 if cheap).

## Rules
Coulson partition oc, comp1400, 1 A40, ≤ 12 CPUs, --mem-per-cpu=2G, env ogqt-main-7c3e3e9 + a40site shim; never touch the
pkai-scale-train array 747977, masif-*, ~/jax-Ka; frozen copy for kept runs under tierB/jaxpb-pipeline/a40-<tag>/; outputs
tierB/jaxpb-pipeline/a40/profile/; no edits to src/ (profiling wrappers live in jobs/ or a profiling copy); no installs/downloads/
envs/git commits; few files (profiler traces: keep at most a couple, compressed). Use ~20 states spanning classes 0–3 from the M0
shards (include ≥ 3 complete AB/A/B triples). Results → results/profile.json (harness blocks .md).

## Output
results/profile.json and a final message (≤ 350 words): a ranked table of where geometry + maps time goes (op, seconds/state,
share), GPU idle share, FP32-vs-FP64 bound, block-culling fractions, triple-reuse fraction, concurrency estimate, and a ranked list
of levers with an estimated gain each (labelled as estimates), noting which need parity-preserving checks.
