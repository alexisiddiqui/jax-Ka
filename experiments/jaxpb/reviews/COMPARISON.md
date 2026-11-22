# jaxpb non-Pallas path: Codex gpt-6-astra vs Gemini 4 Neon review (2026-11-21)

Same prompt (PROMPT.md), both read-only. Full texts: codex-astra.md, gemini-neon.md (stream: gemini-neon.jsonl).
Claims marked [checked] were verified against src/ by Claude.

## Agree (both rank these top-5)
| # | Issue | Where | Codex gain (est.) | Gemini gain (est.) |
|---|---|---|---|---|
| 1 | Maps: dense per-site cell table [ncell^3, K] + 1024-row min query chunk -> 20 GB CPU temporaries [checked] | maps.py:59, :90 | 2-5x maps; B=4 measured 20 -> 5 GB | 5-15x CPU maps; temp 20 -> <0.4 GB; GPU 11.3 -> ~1 GB |
| 2 | XLA SOR: full-grid stencil computed for both colours then masked (2x work), 4 f32 coef fields gathered per block via exE[op], eager max check [checked] | sor.py:52-79 | 1.5-3x kernel | 6-9x A40 XLA, 12-25x CPU |
| 3 | Lane waste: `done` applied after the 10-sweep block, widths only 16/64/256 [checked] | sor.py:67, records.py:148 | 1.3-2x more | (folded into #2) |
| 4 | MC: vmapped lax.cond = select (rejected moves still pay), O(Mp)/O(Np) masked reductions per move; needs a scalar CPU chain | titrate.py:119-195 | 2-6x CPU MC | 30-80x JAX, 60-150x C/Cython |
| 5 | Setup: coulomb_bc / interp over all 531k points, only 38,402 face points used -> 14 GB setup1 temp | pb.py:99, :183, :205 | 3-8x setup | 10-13x setup |
| 6 | Geometry: GEO_CHUNK 131072 x 432 candidate slots (1.36 GB f64 gather); model geometry expands invalid pairs (8.5 GB) | geometry.py:87-113, :442 | 1.5-3x CPU geometry | 4-6x CPU, 2.5-3.5x A40 |
| 7 | specrad host f32 sum: 84% GPU idle, 0.3-0.45 s/state; must keep summation order | pb.py:277 | overlap only | cache constants + overlap/device scan |
| 8 | Both coefs (4xf32) and codes built regardless of backend; dead batch buffers kept alive | pb.py:177-211 | few GB | 0.5-1 GB |

Bitwise risk: Codex flags moderate risk for SOR restructuring (fusion/FMA) and pair-move reductions; Gemini calls nearly
everything bitwise. Codex is the more cautious and is likely right: any XLA re-fusion can change f32 rounding, so every
change needs the existing bitwise gates.

## Only Codex
- Validation gates are weaker than the stated contract: M5 accepts 99.9% sweep equality / 1e-5 energies; native-DelPhi
  sweep agreement is 95.6/90.5/86.5% (prototype-relative energy max 1.4e-6). Separate "regression vs current backend"
  (exact) from "agreement with DelPhi".
- Padded sites repeat the last real site in _site_layout, so maps recomputes it; give padding zero work.
- Do not hoist cell tables out of the site vmap naively (origins depend on acent).

## Only Gemini
- [checked] Class-bump loop has no upper bound (pb.py:378-384): overflow past class 4 raises IndexError, not
  OverflowError, so the driver crashes instead of logging failed_pb.
- [checked] Redundant D2H sync `spec_h = np.asarray(j["spec"])` (pb.py:432) and timing-only block_until_ready calls.
- [checked] Driver keeps every state's timing dict in summary["per_state"] (api.py:583): ~18 GB at 30k states (known open item).
- MEMTRACK = MemTrack() at import calls jax.devices() (api.py:174).
- post/rf_energy: 3x spread, full-grid Laplacian where only boundary points are read, both energy branches computed.
- Unchecked rmax = 2.0 assumption in maps (atoms with rad > 2.01 break the 27-cell guarantee); site_acent_S ignored.
- Dead code: build_maps, maps_batch_ref, PROBE_HASH path, curves_to_pka_jax.

## Gain estimates
Gemini's are consistently 3-10x larger than Codex's and its combined claim (CPU 50-135 s/state, A40 XLA ~Pallas) is
optimistic. Codex's are anchored to the measured ns/point and lane-waste numbers. Treat Codex's as the planning numbers.

## Merged plan
1. Maps: CSR cell ranges instead of the dense table, byte-budgeted query tiles, CPU B=4, zero-work padding.
2. Setup: face-only boundary evaluation; drop unused coefs/codes per backend; release dead buffers.
3. XLA SOR: colour-compressed sweeps with compact operator codes, 9 unchecked + 1 checked sweep; finer lane widths.
4. CPU MC: scalar-indexed chain with real branch skipping; keep RNG and pair-move reduction order.
5. Geometry: smaller CPU chunks, compacted model candidates.
Quick fixes alongside: class-bump bound, spec_h sync, per_state summary trim, import-time MemTrack.
Gate: exact sweeps + bitwise energies vs the current backend on the 104 L3 states, plus determinism.
