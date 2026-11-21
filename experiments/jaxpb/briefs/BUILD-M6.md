# jaxpb-pipeline M6: driver + end-to-end validation — 2026-11-19 (user approved, Coulson A40)

Spec: DESIGN.md §1 (API, driver) and §5 (M6 row); rules as BUILD-M0-M4.md / BUILD-M5.md (A40 on Coulson partition oc,
comp1400, 1 GPU, ≤ 12 CPUs, `--mem-per-cpu=2G`, env ogqt-main-7c3e3e9 + a40site shim; never touch pkai-scale-train 747977,
masif-*, ~/jax-Ka; frozen copies for kept runs; no installs/downloads/envs/git commits; few files; results in JSON, harness blocks
.md). Status: results/M0..M5.json, determinism.json (M0–M5 pass on A40; deterministic; Pallas SOR 10.2×; compiles 33 for PB,
titration 1 per Mp bucket).

## Build (api.py + driver, owning only new files plus minimal wiring changes in existing modules — list every change)
- `label(records, cfg) -> Labels{curves[S,73,n], pKa[S,n], Energies}` and the lower-level `energies(state_batch, cfg)` /
  `titrate(energies_batch, cfg)` per DESIGN §1. `cfg` frozen dataclass (pb_mode="delphi", backend="auto", mc_sweeps=200000 default,
  ph_grid = PyPKA's 73 points −2..16, seed); cfg hash + code sha in every output shard.
- Driver: reads StateRecord shards, runs PB per state (size-class grouping to keep executables warm), queues Energies per Mp bucket
  and flushes titration at S = 32 (and at end), overflow → automatic XL/next-class retry (logged, never silent), resume from
  completed shard outputs (idempotent; a killed job restarts without recomputing finished states), output shards (~256 states/npz:
  g, occ, W (compact), start/nst, pairs, curves, pKa, diagnostics, provenance). Compile cache tarball open/close as in M0/M5.
- Process-wide x64 with explicit dtypes (assembly needs f64 accumulation; PB stays f32 except probe generation) — the energy track
  asked for this; verify nothing in PB changes bitwise.

## Closing tests (DESIGN M6)
1. **L3 end to end** on all 104 recorded states (5 pilot + 99 M0; 33 AB+A+B triples): JAX PB → JAX assembly → GPU MC (200k and 50k)
   vs the saved PyPKA teacher (mc-energies.json and result.json): pKint MAE ≤ 0.05 pK, pair interactions p99 ≤ 0.05 kT, final pKa MAE
   ≤ 0.05, binding shift (AB − free) MAE ≤ 0.05 on paired interface sites of the 33 triples; report also against the MC seed floor
   (PyPKA MC on saved energies vs teacher, from mf-surrogate v1 R1 where states overlap, else rerun G at 200k on the SAVED energies as
   the floor) so pipeline error is separated from MC noise. Missing-midpoint mismatches counted.
2. **Throughput**: states per A40-hour (warm, excl. compile) split PB / assembly / titration; projected GH200 rate stated as an
   estimate only.
3. **Compile flatness**: compile count for 100 vs ~1,000 states (cycle the 104 records with a different seed/order to reach ~1,000;
   it must not grow), and 0 in a fresh process served from the cache tarball.
4. **Resume**: kill the driver mid-run (e.g. timeout after N shards), restart, outputs bitwise identical to an uninterrupted run.
5. **Determinism**: two full runs bitwise identical (labels and energies).

## Output
results/M6.json (all numbers + paths); final message ≤ 350 words with each test vs its criterion, throughput, files changed, open
risks. If L3 fails, localise it (pKint vs W vs MC) before reporting.
