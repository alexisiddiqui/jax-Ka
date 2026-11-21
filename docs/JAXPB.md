# jaxpb-pipeline: batched JAX PB + titration teacher (PyPKA replacement on GH200) — design, 2026-11-19

Code: `src/jaxpb/`; milestone briefs, cluster jobs and closing-test results: `experiments/jaxpb/` (README.md there).

Advisor: Fable (read-only review of `work/jaxpb-pilot/{code,v2/code}`, `work/mf-surrogate/{code,v2/code}` and reports). Saved by
the controller; claims spot-checked (prof.py: `pallas_gpu_trivial` 256.0 and `jax_triton: true` in the Isambard venv; gmc.py:217/248
re-jit per call; jaxrun2.py:89-98 closure over `pos_m`). "verify" = not established by the prototypes.

## 0. Premise corrections
- Pallas is MORE portable than jax-triton: it ships in jax (Triton GPU, Mosaic GPU, Mosaic TPU backends, `interpret=True` on CPU) and
  its kernels are JAX-traced, so the same source is the XLA reference. jax-triton is NVIDIA-only, has no interpret mode, needs a
  second implementation and a version pin (it already imports in the jax-Ka venv, so no download would be needed to try it).
- Only pure XLA runs everywhere → every hot op keeps an XLA implementation as reference + fallback; Pallas is an optional accelerator.
- The compile blow-up is a shape-discipline problem (data-dependent static shapes + closures), not a kernel problem.

## 1. Architecture
CPU producer (PyPKA preprocessing only; `pbrec.py:90` hook in a prep-only variant that returns zeros, adds pKmod, site type,
chargediff, occ, PyPKA site order) → StateRecord shards (npz, ~256 states/file) → GPU `geometry → maps → solve(lanes) → post →
assemble → Energies` → GPU `titrate(Energies bucket) → curves, pKa`.

API: `label(records, cfg)` driver (queues, overflow retry, resume); `energies(state_batch, cfg) -> Energies{g, occ, W, start/nst,
pairs, diag}`; `titrate(energies_batch, cfg) -> curves, pKa`. `cfg` frozen dataclass (pb_mode, backend, mc_sweeps, ph_grid, seed);
its hash goes into every output shard.

Padded layouts:
| level | arrays | shape / bucket |
|---|---|---|
| state context | pos, rad, q_bg | Na_c = 2048·2^c, c = 0..4 (pad radius 0 at FAR=1e4) |
| | whole-molecule probes | P_c = 192·Na_c |
| site batch | acent, valid | B = 16 fixed |
| | fine-region atoms / probes | [B,2048,3] / [B,131072,3] fixed (gathered per site) |
| | model compound | atoms [B,32], probes [B,8192,3] |
| | tautomer charges | [B,T,QM], T = 8, QM = 24 (verify T max over the pool) |
| | NS/NB compaction | `jaxrun2.py:36` SZ fixed + one "XL" overflow variant |
| lanes | phi, rhs [L,81³], op_idx, omega, done | L = 256 flat |
| operators | uint8 code [E,81³] (3 eps bits + ion bit), E = 3B | replaces four f32 coefficient maps |
| energies | Phi [M_c,Na_c], dQ, g, W | M_c = Na_c/8 |
| titration | W [S,Mp,Mp], u [S,73,Mp], start/nst [S,Np], pairs [S,Pp] | Mp ∈ 64..2048, Np = Mp/4, Pp = Mp/16 |

Fused, argument-pure jits: geometry_state[c], geometry_sites (sorted-cell-id + searchsorted instead of dense nc³ tables);
maps_coarse[c], maps_fine, maps_model (build_maps2 → uint8 code, bidx, surf); setup, post[c] (Coulomb BC on 38,402 face points only,
reaction-field kernel G[QM,NB] once per site, drop diagnostic rf_energy calls); sor_block; assemble[c]; mc[Mp]; curves_to_pka.
Batching: energies = sites within a state (B is a memory knob; GPU saturated at B = 8); titrate = states within an Mp bucket
(flush at S = 32; padded states cost nothing).

## 2. Compile budget
Recompile sources to remove: jaxrun2.py:42 (SB), :156 (TM), :193 (PBc/PBf), :89-98 (closure over pos_m); maps2.py:252-258
(SB_BUCKETS), :376-379 (nc, K), :402 (nonzero size), :448 (MX); gmc.py:217,248 (jax.jit per call), :203 (Np/Pp).
After redesign: ~36–40 executables worst case (≈25 for 3 size classes), 10–15 min cold, ~0 warm, independent of state count.
Padding: atoms/probes < 5% time; fine NS ~25% slots; site tail ~8% map time at 100 sites (35% under 32 sites; later fill the tail
from the next state of the same class); SOR waste 1.2× → ~1.0× with early-exit lanes; MC ≤ 2× on the accept update only.
Overflow: raise Ka 16→24, Kp 576→768; any overflow flag → automatic re-run in XL / next class, never silent.
Cache: jax_compilation_cache_dir on node-local /tmp, untar from one $SCRATCH tarball at job start, re-tar at end (1 inode);
min_compile_time ≥ 1 s; verify Pallas executables are cached (M0).

## 3. Kernels
| hot op | GEMM/conv? | choice |
|---|---|---|
| W assembly | yes: W_raw = dQ·dPhiᵀ, one sgemm [M,Na]×[Na,M] + PyPKA sign/symmetrisation | XLA |
| background energy | GEMV ((1−S)∘Phi)@q_bg | XLA |
| reaction-field energy | ½ q_tᵀ G σ_t, batched over T | XLA |
| mean-field titration (cmf, M) | W @ P[M,73] | XLA (deterministic fallback) |
| MC | no (sequential chains) | keep Pallas Triton kernel (gmc.chain); run_ref fallback |
| SOR | no (per-point coefficients, red-black, summation order matters for the 0.01 stop) | new Pallas kernel: one program per lane, whole solve in-kernel, self-checked stop; XLA colour-compressed red/black fallback (~2× vs maps2.py:160-161) |
| map construction | no (cell gathers/scatters) | XLA; revisit after the SOR kernel |
Estimated SOR kernel 2–4× (≈1.4–1.6× end to end) — estimate. XLA-only fallback: PB ~52 GPU-h/30k, MC ~16 GPU-h at 50k sweeps.
`backend="auto"` probes Pallas at import, else XLA; tests run kernels with interpret=True on CPU.

## 4. PB modes
`PBSolver(mode)`: "delphi" (zero start, red-black even first, fixed ω from ρ, max-change < 0.01 at 10-sweep checks, 500 cap; ρ 0.75
coarse, ported DelPhi estimator for focused/model — prototypes read ρ from logs, jaxrun2.py:176) and "converged" (tight tolerance or
jaxpb.pcg). Mode in provenance; only "delphi" has ground truth.

## 5. Modules and milestones (~3 weeks)
src/jaxpb/{records,geometry,maps,pb,sor,assemble,titrate,api,prep_pypka}.py, tests/test_jaxpb_*.py
(closing-test scripts in experiments/jaxpb/closing/).
| M | work | closing test | days |
|---|---|---|---|
| M0 | goldens (5 v2 outputs + ~100 val states via pbrec on Coulson), compile counter, cache-tar check | reproduces v2 numbers | 1 |
| M1 | records, size classes, argument-pure maps + PB with XLA lane pool | ≤ 1e-6 vs v2, sweeps 100% equal, compiles ≤ budget, 0 on 2nd pass | 3 |
| M2 | spectral-radius estimator | sweep counts vs DelPhi logs (87–91% focused); L2 passes without logged ρ | 2 |
| M3 | assembly in JAX (calcBackEnergy, calcpKint, calcInteractionWith, f32 cast, pair rule) | g, W vs saved mc-energies ≤ 1e-5 | 2 |
| M4 | titration (gmc.chain unchanged, fixed buckets, module-level jits, pkhalf) | 426 states reproduce v2 rows; ref vs Pallas bitwise | 1–2 |
| M5 | SOR kernel | sweeps ≥ 99.9% equal to XLA ref; energies ≤ 1e-5; ≥ 2× on SOR | 3–5 |
| M6 | driver (queues, overflow retry, resume, shards) | L3 on ~100 states (pKint/pKa/shift MAE ≤ 0.05); states/GPU-h; compile count flat 100→1,000 states | 3 |
| M7 | converged mode + write-up | converged vs PCG f64 | 1–2 |
Risks: ρ estimator (no prototype); kernel vs XLA FP contraction flipping a stop (accept ≥ 99.9% equal sweeps); producer on aarch64
(PyPKA/pdb2pqr may only run on Coulson); thin overflow margins / atoms outside the box at scale; PyPKA W sign/symmetrisation quirks.

## 6. Decisions (advisor recommendation)
1. Kernels: Pallas Triton + XLA references; no jax-triton.
2. PB default "delphi" until M6 passes; converged relabel later as a separately tagged label set.
3. Producer on Coulson; port to Isambard only if transfer bottlenecks.
4. MC sweeps 200k (0.47 GPU-s/state, ~4 GPU-h/30k, = PyPKA seed noise); 50k saves ~3 GPU-h.
5. Order: shape discipline M1–M4 first (removes the ~200 h compile cost), SOR kernel M5 after (saves ~15–20 of 52 GPU-h).
6. Parity bar tolerance-based (≤ 1e-6 vs prototype, statistical sweep equality), not bitwise.
7. Compile cache as one tarball (1 inode).
8. Store g and W per state in shards (~20 GB / 30k states) so re-titration needs no PB.
9. f32 throughout, f64 only inside probe generation; process-wide x64 with explicit dtypes instead of maps2.py:285 context manager.
