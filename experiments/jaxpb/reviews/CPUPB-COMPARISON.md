# CPU PB review: Astra 6.2 (xhigh) vs Gemini 4 Argon (high), 2026-11-24
Same prompt (CPUPB-PROMPT.md), read-only. Full texts: cpupb-codex-astra.md, cpupb-gemini-argon.md. All gains are estimates.

## Target correction (both reviewers; checked against bench.json)
jaxpb CPU PB 464 s/state is the 5 OPT2 states; PyPKA 85.7 s/state is the 20-state whole-job mean. On the same 5 states
PyPKA is 8.8 s PB, ~25-29 s total. Matched gap ~17x, not ~6x.

## Where the time goes (agree)
| Stage | s/state | Astra | Argon |
|---|---:|---|---|
| Maps atom queries (27x24=648 slots/row, f32 norms, first-min) | | ~105 | 95-115 |
| Maps probe reach masks (1,064 offsets/probe at 0.25 A, scatter) | | ~65 | 55-70 |
| Maps probe projection (CSR 128-slot slabs, 3,456 slots/row) | | ~40 | 20-30 |
| Maps per-site cell builds / region selection | | ~25 | 40-55 |
| SOR: two masked full-grid passes/sweep, W16 578 MB temp >> L3 | 164 | lane filling only ~13 s | rolls+gathers spill L3 |
| Geometry: geo_probes candidate search (searchsorted x54, 432 slots f64) | 44 | | |

## Agreed plan items
1. Maps: stream only occupied cell slots in original (offset, slot) order with a running min (exact tie rule). Astra 50-105 s, Argon top item.
2. Maps: invert reach to query `need` midpoints against nearby probes (early exit). Astra 25-55 s; Argon same idea.
3. SOR: native C kernel via CPU FFI/pure_callback, in-place red/black, same f32 expression. Bitwise possible but must
   reproduce XLA's exact FMA contraction per colour. DISAGREE on gain: Astra 2-4x (80-120 s), Argon ~15-25x (6-12 s).
   Argon's pure-XLA fallback: W=1 lanes on host threads + pre-rolled coefficients.
4. Geometry: precomputed neighbour spans / direct cell lookup, compacted candidates; keep f64 burial. 12-28 s.
5. Threading/processes: maps groups of 4 underfill 6 cores; consider 2 pinned 3-core workers (Astra: 0-30%, memory limits).

## Floor (disagree)
Astra: 165-235 s PB realistic, ~125 optimistic. Argon: pure XLA 135-190 s; with two C kernels 32-51 s.
Claude: the SOR gap is the crux. 81^3 f32 x (phi, rhs, dinv, 3 face fields) ~ 12 MB/lane fits L3 per CCD, so an in-place
native loop could plausibly reach 5-10x; Astra's 2-4x is the safe planning number. Settle by prototype, not estimate.

## Unique / correctness notes
- Astra: phase-2 split-check already changed g by 5.25e-6 -> any check-schedule change is not bitwise. geometry_sites has an 8.52 GB temporary. Measure per-op CPU split first (current profile is A40 pre-1b).
- Argon: rf_energy/post use 6 full-grid rolls where a boundary-only Laplacian suffices; setup2 pads 128 dummy grids on XLA.
