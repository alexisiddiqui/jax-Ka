# Maps slowdown review: Codex gpt-6.2-sol (high) vs Gemini 4 Neon (xhigh), 2026-11-22

Same prompt (MAPS-PROMPT.md), both read-only. Full texts: maps-codex-sol.md, maps-gemini-neon.md.
[checked] = verified by Claude against src/ or arithmetic.

## Agreed causes of the A40 regression (maps 3.40 -> 5.73 s/state)
| Cause | Where | Codex est. | Gemini est. |
|---|---|---|---|
| Byte budgets divide by B=16 and assume 48-64 B/candidate. As a result, atom chunks shrink 6,472 -> 1,078 rows, cull chunks 1,398 -> 233, and reach chunks about 16x [checked arithmetic] | maps.py:95-114, :563 | rank 1 | 0.7-1.2 s (atoms + reach) |
| The probe 32-slot slab fori_loop runs to the slowest lane, with an argmin and 2 take_along_axis per slab | maps.py:134-158 | rank 2 | 0.6-1.0 s |
| CSR double gather on the small atom tables, where the old dense table was 1-21 MB | maps.py:159-162 | part of 1/2 | 0.25-0.45 s |
| One inactive lane sends the whole GPU tail batch through a serial lax.map of B=1 map builds [checked] | maps.py:622-635 | rank 3 | 0.4-0.75 s |
| Redundant all(active) cond around stage0 blocks fusion | maps.py:450 | minor | 0.05-0.15 s |

## CPU: why maps stayed flat
Both reviewers agree:
- Grouping into B=4 saved memory but no time, because the CSR reconstruction adds gather passes to the hottest loop (atom queries).
- The tiles are now tiny (539 atom rows, reach 32x smaller), so each CPU loop iteration does little work.
- Each tile runs an all(cnt>0) cond, and a group with one empty site falls back to per-site B=1.
Gemini adds that the 64 MiB tiles still exceed the 32 MB L3, so intermediates spill to DRAM.

## Fix plan (merged; gains are estimates, and Gemini's run ~2x Codex's as before)
1. GPU tile sizes: restore the baseline row counts with separate budgets for atom, cull, reach and brute. Use GPU-specific bytes/candidate and hoist grid_coords out of the reach loop. Codex 0.6-1.3 s, Gemini 0.7-1.2 s; GPU memory cost a few hundred MB; zero parity risk if the exact baseline chunks are restored.
2. Budget-gated dense tables: build the dense table when B*ncell^3*K*4 fits a static budget. That covers all atom tables on both backends (about 21 MB on the GPU) and, on the GPU, the focused/model probe tables (287 MB each). Coarse probes stay CSR (that table was 11.16 GB, the real memory win). About 0.5-1 s on the A40, zero parity risk (the baseline code path).
3. Drop the GPU B=1 partial_group path: run padded sites at B=16 with zero counts, or as 2 groups of 8. Hoist the CPU activity checks out of the inner loops. 0.1-0.85 s. This also removes a B=1-vs-B=16 rounding hazard, since B=1 on CPU already flipped esolvS bits.
4. Probe queries (Gemini): replace the 32-slot slabs with compacted CSR streaming (a prefix sum over the 27 cells, strict < merge keeps the tie order), or as an interim one larger slab (128-256). Codex prefers a fixed 64/128-slot one-shot with a ranked fallback. Use L3-sized CPU tiles (8-16 MiB). This is the main CPU lever: Gemini claims 376 -> ~140-215 s; Codex says 10-25%.
5. Small wins: one searchsorted in build_cells with an explicit stable argsort, and build the focused probe cells from preg instead of all probes (Gemini). Defer query sorting (both).

Expected outcome: A40 maps back to <= 3.4 s, possibly ~2.6-3.1 s, at roughly 4.5-5.2 GB GPU peak.

## Correctness items found
- [checked] A site is treated as padding when its model radii are all 0 (active = any(srad>0)). Pass the validity mask from _site_layout instead. (both)
- [checked] In partial CPU groups, the empty stage0 sets pcls=0 and flat=0, so padded lanes output ex=ey=ez=True and ion[0]=True. Valid-masking downstream hides this; it is harmless today but should be masked. (Gemini)
- maps_batch raises an error for a CPU B that is not a multiple of 4; pad internally. (Codex)
- The slab carry is int32, but argsort/argmin promote to int64 under x64. It is safe inside GEO.x32() (pb.py:384) but crashes on a direct call; cast to int32. (Gemini)
- XL_KA=48 takes the slab path, while the tests cover only K=64. jax.default_backend() inside jit ignores jax.default_device. (Gemini)
