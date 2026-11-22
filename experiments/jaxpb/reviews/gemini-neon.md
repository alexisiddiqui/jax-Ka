Below is a read-only code review of the non-Pallas ("main JAX") path—XLA backend on GPU and the full pipeline on CPU—grounded in [bench.json](results/bench.json), [profile2.json](results/profile2.json), [profile.json](results/profile.json), [speed.json](results/speed.json), [M5.json](results/M5.json), and [M6.json](results/M6.json).

---

## 1. SOR on XLA (GPU and CPU)

**Evidence:** On A40, XLA SOR is **10.35× slower** than Pallas (43.03 vs 4.16 s/state; 0.460 vs 0.125 ns/point × 2.58–3.13× lane waste). On 12-CPU (6 physical cores), SOR is **373.3 s/state** (38.4% of total CPU time; 8.26 ns/executed point, 171–200× slower than A40 Pallas).

### 1.1 Full-grid double stencil + `jnp.roll` + 4×f32 coefficients + per-sweep `max` check
* **Location:** [sor.py:52-79](src/jaxpb/sor.py#L52-L79), [pb.py:156-164](src/jaxpb/pb.py#L156-L164)
* **What is wrong:**
  1. `sor_chunk` ([sor.py:62-63](src/jaxpb/sor.py#L62-L63)) evaluates `gs(p)` over all $81^3 = 531{,}441$ points **twice per sweep** and discards half each time via `jnp.where(cA/cB, ...)`.
  2. `gs(p)` ([sor.py:55-57](src/jaxpb/sor.py#L55-L57)) uses **9 `jnp.roll` ops** per half-sweep (6 on `p`, 3 on `ex/ey/ez`) even though faces never update (`~fm`). Wraparound semantics inhibit contiguous SIMD vectorization on CPU and add index overhead on GPU.
  3. `sor_block` ([sor.py:79](src/jaxpb/sor.py#L79)) indexes `exE[op], eyE[op], ezE[op], dvE[op]` (4×f32 = **16 B/pt**, 272 MB table at $E=32$) instead of reusing the existing **1 B/pt** `uint8` `op_code` + 16-entry `op_lut` ([sor.py:168-181](src/jaxpb/sor.py#L168-L181)).
  4. `jnp.where(k == check - 1, jnp.max(jnp.abs(new - p)), mc_)` ([sor.py:64](src/jaxpb/sor.py#L64)) is non-short-circuiting: XLA computes `jnp.max(jnp.abs(new - p))` across all $W \times 81^3$ elements on **all 10 sweeps**, discarding 9 of 10 reductions and forcing both `p` and `new` (544 MB each at $W=256$) to stay live across both half-sweeps.
* **Why it costs time/memory:** Explains the **3.7×** per-executed-point gap on A40 XLA (0.460 vs 0.125 ns/pt) and large share of CPU's 8.26 ns/pt. At $W=256$, streaming `phi` (544 MB) + `rhs` (544 MB) + 4×f32 coeffs (272 MB) = **1.36 GB** per half-sweep thrashes CPU L3 cache completely (EPYC 9224 has 32–64 MB L3/CCD).
* **Proposed change:**
  1. **Colour-split red/black layout** (as noted in [DESIGN.md:62](DESIGN.md#L62)): split `phi`, `rhs`, and `code` into `(even, odd)` half-arrays of shape `[W, 81, 81, 41]` (packing `k` at stride 2 by `(i+j+k)%2`). Every `i±1, j±1` neighbor sits at the **same** compressed `m = k // 2` index in the opposite-colour array; only `k±1` shifts `m` by `0/-1` or `0/+1` depending on `(i+j)%2`. Use interior slices `[1:-1, 1:-1, ...]`—zero `jnp.roll`, zero `cA/cB` masks, half the FLOPs and half the memory traffic, and disjoint read/write buffers for true in-place updates.
  2. Switch XLA SOR from `(ex, ey, ez, dinv)` to `op_code` (`uint8`, 1 B/pt) + `op_lut` (`[E, 16]` f32), indexing `code[op]` / `lut[op]` per lane (or grouping lanes by operator; see 1.2).
  3. Split `fori_loop(0, check - 1, ...)` (no `max`, no old `p` kept live) followed by 1 final sweep that computes `jnp.max(jnp.abs(new - p))`.
* **Expected gain (estimate):** **3–4.5×** on A40 XLA SOR (per-executed-point 0.46 → ~0.12–0.15 ns/pt); **4–6×** on CPU SOR before lane-tiling (1.2).
* **Parity risk:** **Bitwise identical** if the 6-neighbor addition order (`+x, -x, +y, -y, +z, -z`) and `jnp.abs(new - p)` on sweep `check - 1` are preserved (already proven bitwise for `op_code`/`op_lut` in [M5.json:4](results/M5.json#L4)).

### 1.2 Coarse `LANE_WIDTHS`, host-driven compaction, and $W=256$ cache thrashing
* **Location:** [sor.py:116-151](src/jaxpb/sor.py#L116-L151), [records.py:148](src/jaxpb/records.py#L148), [pb.py:437-446](src/jaxpb/pb.py#L437-L446)
* **What is wrong:**
  1. `LANE_WIDTHS = (16, 64, 256)` has 4× jumps and lacks `128` (and `32`). Phase 2 has at most $B \cdot T_{\max} = 128$ valid lanes (mean occupancy 24.1% = ~61 lanes), yet whenever `n_valid > 64`, `width_for` allocates **$W=256$** (324 of 575 blocks in phase 2 run at $W=256$, [profile2.json:434](results/profile2.json#L434)), driving `waste_executed_over_useful` to **2.58× (ph1) / 3.13× (ph2)**.
  2. `run_pool` syncs `done/it/mc` to host every 10 sweeps, calls `lanes_store`, and calls `lanes_take` (which lacks `donate_argnums`, [sor.py:100](src/jaxpb/sor.py#L100), temporarily doubling pool memory).
  3. On CPU, running 256 lanes inside `fori_loop` sweeps 1.36 GB through DRAM on every half-sweep. A single lane with `uint8` codes needs only **4.8 MB** (`phi` 2.13 + `rhs` 2.13 + `code` 0.53 MB), and up to 8 tautomers of the same site already share one operator (`op1/op2` are contiguous by site, [pb.py:437,443](src/jaxpb/pb.py#L437-L443)).
* **Proposed change:** Replace `run_pool`'s host-loop + 3-width compaction (12 executables in [compile_budget.py:183](src/jaxpb/compile_budget.py#L183)) with a **single** device `lax.while_loop` solver `sor_solve_xla` over fixed-width chunks $W_{\text{chunk}}$ (e.g. $W_{\text{chunk}}=32$ on GPU—still $17\text{M}$ pts/sweep, saturating 84 SMs—and $W_{\text{chunk}}=8$ on CPU, ~38 MB working set fitting L3 cache). Compact valid lanes on host into $\lceil n_{\text{valid}} / W_{\text{chunk}} \rceil$ chunks (grouped by operator/grid type so coarse/model/focused don't mix sweep counts in one chunk).
* **Expected gain (estimate):** Eliminates ~1.7–2.2× of the 2.6–3.1× lockstep/padding waste on both GPU and CPU, removes all inter-block host syncs, drops XLA SOR compiles from **12 → 1**, and adds an estimated **2–3×** CPU L3-cache locality gain. Combined with 1.1: **~6–9×** on A40 XLA SOR (43 s → ~5–7 s/state, within ~1.5× of Pallas) and **~12–25×** on CPU SOR (373 s → ~15–30 s/state).
* **Parity risk:** **Bitwise identical** (per-lane iteration and stop rule unchanged; zero cross-lane interaction).

---

## 2. Maps and Geometry on CPU

**Evidence:** On the 5-state CPU profile ([profile2.json:220-238,344-348](results/profile2.json#L220-L238)), `maps_batch` is **370.1 s/state** (**230.5×** slower than A40; 38.1% of CPU wall), `geometry` is **45.5 s/state** (32.6×), `solve_setup` is **12.8 s/state** (**116.9×**), and `site_probes` is **3.75 s/state** (31.6×).

### 2.1 Dense per-site probe cell tables + 20,736-candidate padded probe queries (`maps_batch`)
* **Location:** [maps.py:59-70](src/jaxpb/maps.py#L59-L70), [maps.py:90-110](src/jaxpb/maps.py#L90-L110), [maps.py:338-346](src/jaxpb/maps.py#L338-L346), [maps.py:459-460](src/jaxpb/maps.py#L459-L460)
* **What is wrong:**
  1. `stage0` (vmapped over `B=16`) builds a dense probe table `tab_p` of shape `[16, 61^3, 768]` `int32` = **11.16 GB** for the coarse grid (and `[16, 18^3, 768]` for focused) by sorting all `NP_C[c]` probes (393k–3.1M) **32 times per site batch**—even though `probes` is identical across all sites of a state, focused only needs `preg` (`131k`, already gathered on line 346), and average cell occupancy is ~2–7 vs `Kp=768` (>99% `-1` padding).
  2. `_q_one` ([maps.py:103-110](src/jaxpb/maps.py#L103-L110)) gathers all `27 × Kp = 20,736` slots at once for `chunk = 1024` rows × `B = 16` sites: materializing `pp` (`[16, 1024, 20736, 3]` f32 = **4.08 GB**) and `q - pp` (**4.08 GB**) and running `sqrt` on 99% `-1` padding slots (~4B wasted `sqrt`s per batch!).
  3. Similarly, atom queries (`Ka=24`, `27×24=648` slots, `chunk=6472`) and `reach_mask_b` (`budget=1<<22` per site, [maps.py:478](src/jaxpb/maps.py#L478)) were sized for 1 GPU site, then vmapped ×16 (**805 MB** per `pp`/`M` tensor), blowing CPU L3 cache and running all 16 sites at the batch maximum (`max(nmix)`, `max(nsh)`, `max(npreg)`, [maps.py:311,505](src/jaxpb/maps.py#L311)).
* **Why it costs time/memory:** `profile2.json:539-550` shows `cell_tables` (20.5%) + `probe_projection_query` (9.4%) + `midpoint_query` (15.8%) + `reach_mask` (24.6%) dominate maps on GPU; on CPU the unfused multi-GB `tab_p` and `pp` buffers spill to DRAM on every chunk, causing the **230.5×** slowdown and **19.99 GB** temp. (Note: `PROBE_HASH=True` failed on A40 in [speed.json:57](results/speed.json#L57) only because `_q_one_hashed` ran 54 `searchsorted`s per row *and still* expanded to 20,736 slots.)
* **Proposed change:**
  1. Replace dense `tab_p` `[B, ncell^3, Kp]` with **CSR arrays** `(start, stop, order)` from the same `argsort` (`start/stop` is `[ncell^3]` i32 = **1.82 MB** vs **697 MB/site**—384× smaller, O(1) cell lookup with **zero** `searchsorted` during queries!).
  2. Inside `_q_one`, loop `lax.fori_loop(0, 27)` over neighbor offsets (cuts `pp`/`q-pp` buffer 27×: 4.08 GB → 151 MB) and cap inner slot loop at dynamic `kmax = jnp.max(en - st)` of the chunk (cuts probe-query FLOPs **~10–30×** by skipping empty tail slots).
  3. Build whole-molecule atom and probe CSR indices **once per state** (not `2×n_sites` times per state); for focused grids, if keeping per-site `lop`, bin `preg` instead of all `NP_C[c]` probes.
  4. Make `chunk`/`budget` scale inversely with `B` (e.g. `budget_total // B`, smaller chunks on CPU to fit L2/L3) and reduce `B` on CPU (see Section 3).
* **Expected gain (estimate):** **5–15×** on CPU `maps_batch` (370 s → ~25–75 s/state) and **~1.3–1.8×** on A40 `maps_batch` (3.4 s → ~1.9–2.6 s/state), plus **~19 GB** CPU temp / **~10.5 GB** GPU temp reduction.
* **Parity risk:** **Bitwise identical** if CSR stays per-site (exact same `order[start[cid] + k]` sequence and `argmin` tie-breaking). Moving to a single **state-level** global origin `lo_state` preserves all neighbors within `cp = 1.41 > rp = 1.40 Å`, but can alter tie-breaking order or borderline probes at distance `1.41..2.82 Å` → **requires parity verification** (expected $\ll 10^{-6}$ rel energy, not strictly bitwise).

### 2.2 Full-volume `coulomb_bc` and `interp` on interior points (`setup1`, `setup2`)
* **Location:** [pb.py:99-106](src/jaxpb/pb.py#L99-L106), [pb.py:183-189](src/jaxpb/pb.py#L183-L189), [pb.py:205-209](src/jaxpb/pb.py#L205-L209)
* **What is wrong:** `coulomb_bc` computes `r = norm(G - q_pos)` and `exp(-r/deblen)/r` over all **531,441** grid points for all 256 coarse+model slots, then zeroes 92.8% of them with `jnp.where(fm, val, 0.0)`. `setup2` similarly calls 8-corner `interp` over all 531,441 points for 128 focused slots and discards 92.8%. ([DESIGN.md:39](DESIGN.md#L39) specified *"38,402 face points only"*, never implemented.)
* **Why it costs time/memory:** Wastes **13.8×** compute (~3 billion wasted `exp()`/`sqrt()` calls per batch) and allocates **13.98 GB** XLA-CPU temp (`[128, 81^3, 24]` unfused intermediates, [profile2.json:623,646](results/profile2.json#L623)), making `solve_setup` **116.9×** slower on CPU (12.83 s vs 0.11 s).
* **Proposed change:** Precompute the 38,402 face coordinates (or 6 face 2D slices), evaluate `coulomb_bc` / `interp` only on those 38,402 points (and `fori_loop` over `QM=24` or scatter into the 6 faces of `jnp.zeros((N, N, N))`), skipping `q == 0` work or chunking on CPU.
* **Expected gain (estimate):** **~10–13×** on `solve_setup` on CPU (12.8 s → ~1.0–1.3 s/state) and **13.98 GB → <0.3 GB** CPU temp; modest ~0.05–0.10 s/state on A40.
* **Parity risk:** **Bitwise identical** (exact same floating-point ops on the 38,402 face points; interior is identically 0.0).

### 2.3 Per-point binary search + 432-slot padded burial in `geometry.py` and uncompacted `_site_probes`
* **Location:** [geometry.py:87-123](src/jaxpb/geometry.py#L87-L123), [geometry.py:442-475](src/jaxpb/geometry.py#L442-L475)
* **What is wrong:**
  1. In `geo_probes`, every single candidate point (tens of millions per state) runs `2 × 27 = 54` `searchsorted` calls in `_cand` (41.1% of GPU kernel time, [profile2.json:521](results/profile2.json#L521)) and evaluates f64 `sqrt` burial against `27 × 16 = 432` slots (57.2% of time, ~90% `-1` padding), using **int64** for `cand`/`order` and `CH = 131,072` (2.30 GB CPU temp, blowing L3 cache).
  2. In `_site_probes` ([geometry.py:450-463](src/jaxpb/geometry.py#L450-L463)), all `496` pairs of `MA=32` atoms × `MX=720` arc points + `48,000` vertices = **405,120** points (95%+ invalid padding!) run f64 `_burial` against all 32 atoms vmapped across 16 sites simultaneously (`[16, 405120, 32, 3]` f64 = **4.98 GB** + `dd`/`cand` = **8.52 GB** CPU temp, [profile2.json:624](results/profile2.json#L624)).
* **Proposed change:**
  1. **Per-atom / per-pair neighbor lists for `_burial`:** By the triangle inequality, any atom within `bound` of a vertex on atom $a$ is within $R_a + \text{bound}$ of `pos[a]`, and any atom within `bound` of an arc point on pair $p$ is within $\rho_p + \text{bound}$ of `cen[p]`. Precompute a fixed-width neighbor list ($K_{\text{nbr}} \approx 96\text{–}128$, with overflow retry like `CellOverflow`) **once per atom and once per pair** (in `int32`), eliminating per-point `_cand` (41% of time) and cutting `_burial` width 3.5–4.5× (`432 → 96–128`). On CPU, also shrink `CH` from `131,072` to `4,096–8,192` so chunks fit in L3.
  2. **Compact valid pairs in `_site_probes`:** Compact `pok` pairs (typically ≤40 of 496) and loop/chunk over valid pairs (or sites) instead of materializing `[16, 405120, 32, 3]` f64.
* **Expected gain (estimate):** **4–6×** on `geometry` on CPU (45.5 s → ~8–11 s/state) and **~2.5–3.5×** on A40 (2.08 s → ~0.6–0.8 s/state); **~10×** on `site_probes` on CPU (3.75 s → ~0.4 s) and **8.52 GB → <0.2 GB** CPU temp (**3.69 GB → <0.3 GB** GPU temp).
* **Parity risk:** **Bitwise identical**—`_burial` ([geometry.py:90-97](src/jaxpb/geometry.py#L90-L97)) uses `min`, integer `sum`, and `any` over the exact set of distinct atoms within `bound`, which is strictly order-independent. (If additionally replacing `sqrt(dd2)` with squared-distance comparisons, flag as near-bitwise/1-ULP risk and check.)

---

## 3. Memory: What Drives the 21–25 GB Size-Independent CPU RSS, and How to Bound It

**Evidence:** Every state on CPU hits **21.1–25.3 GB** RSS regardless of size class ([bench.json:78,270](results/bench.json#L78)), OOMing 24.6 GB cgroups. [profile2.json:621-646](results/profile2.json#L621-L646) exposes **three separate class-independent multi-GB cliffs** (all linear in `B=16`, independent of `c`):

| Stage | Compiled CPU Temp | Compiled GPU Temp | Exact Buffer Culprit (`file:line`) | Fix & Target CPU Temp (est.) |
|---|---:|---:|---|---|
| `maps_batch` | **19.99 GB** (`1.25 GB × B`) | **11.25–11.43 GB** | Coarse `tab_p` `[B, 61^3, 768]` i32 (**11.16 GB**) + unfused `idx`/gather in `build_cells` ([maps.py:68-69](src/jaxpb/maps.py#L68-L69)) + `_q_one` `pp`/`q-pp` `[B, 1024, 20736, 3]` (**4.08 GB × 2**, [maps.py:96-97](src/jaxpb/maps.py#L96-L97)) | CSR `(start, stop, order)` + 27-offset loop + `B_cpu ≤ 4` → **<0.4 GB** |
| `setup1` | **13.98 GB** | 0.37 GB | Full-grid `coulomb_bc` `[B, 8, 81^3, 24]` f32 unfused on CPU (**6.53 GB × 2**, [pb.py:104-105,189](src/jaxpb/pb.py#L104-L105)) | 38,402 face points only + `QM` loop → **<0.2 GB** |
| `geometry_sites` | **8.52 GB** | 3.69 GB | `[B, 405120, 32, 3]` f64 diffs (**4.98 GB**) + `dd`/`cand` (**3.32 GB**, [geometry.py:458-463](src/jaxpb/geometry.py#L458-L463)) | Compact `pok` pairs + chunk/loop over sites → **<0.2 GB** |
| `geo_probes` | **2.30 GB** | 0.82 GB | `[CH=131072, 432, 3]` f64 `cpos[ii]` (**1.36 GB**) + `i64` `cand` (**0.45 GB**, [geometry.py:91,119](src/jaxpb/geometry.py#L91)) | `i32` indices + per-atom/pair `K_nbr` + `CH_cpu=4096` → **<0.15 GB** |
| `post` | **1.67 GB** | 0.84 GB | `[B, 8, 24, 48000, 3]` f32 `q_pos - surf` + 3× `[B, 8, 81^3]` `rho` grids ([pb.py:141,224](src/jaxpb/pb.py#L141)) | Compute `G` per site/chunk + derive `rho` from `src` or boundary-only → **<0.25 GB** |
| Live Python refs in `_solve_state` | **~4–6 GB** | **~4.5 GB** | `items1/cf1/kf1/st1/items2/cf2/kf2/st2/mc/mf/ms` all held alive until next batch ([pb.py:434-450](src/jaxpb/pb.py#L434-L450)); `preg` coarse `[B, NP_C[c], 3]` (**0.60 GB** c3 / **1.21 GB** c4, [maps.py:346,518](src/jaxpb/maps.py#L346)) | Drop `cf1/cf2` (use `op_code`), `del` dead stage outputs promptly, `B_cpu ≤ 4` → **<1.0 GB** |

**Key takeaway:** All top 3 cliffs (`maps_batch`, `setup1`, `geometry_sites`) must be fixed together to bring CPU RSS below 14 GB; fixing all 5 + setting a CPU default `B = 2` or `B = 4` (which already ran 18% faster/site on CPU in [profile2.json:630-644](results/profile2.json#L630-L644) and keeps compile counts flat per process) bounds CPU RSS to an estimated **1.5–2.5 GB** (~10× reduction) and cuts A40 peak VRAM from **16.1 GB → ~3–4 GB**.

---

## 4. MC on CPU (and XLA GPU) vs PyPKA's Cython MC

**Evidence:** On 12 CPUs, `mc_ref` takes **588.2 s/state** (20-state mean at 200k sweeps; **2,643 s/state** at `Mp=1024`)—**13.5× slower** than 12-CPU PyPKA (43.5 s) and **3–9× slower** than **1-CPU** PyPKA on class-0 states ([bench.json:148-165](results/bench.json#L148-L165), [profile2.json:648-765](results/profile2.json#L648-L765)). On A40, `mc_ref` is **35× slower** than `mc_pallas` (62.7 vs 1.80 s/state).

### 4.1 `vmap` over `lax.cond` executes `upd` (full $W$-row updates) on 100% of trial moves + lockstep padding
* **Location:** [titrate.py:135-186](src/jaxpb/titrate.py#L135-L186), [titrate.py:195-207](src/jaxpb/titrate.py#L195-L207), [titrate.py:285](src/jaxpb/titrate.py#L285)
* **What is wrong:**
  1. `mc_ref` applies `jax.vmap(jax.vmap(one))` over `(S=32, NPH=73) = 2,336` chains. In JAX, `vmap(lax.cond)` lowers to `lax.select`—**both branches execute unconditionally**. On every rejected move (~90–98% of moves, especially away from each site's pKa), `mc_ref` still loads two rows of `W` (`W[nw]`, `W[oc]`), computes `E + (W[nw] - W[oc])`, updates `X`, and masks the result.
  2. `vmap` over `lax.fori_loop` forces all `S=32` states—including **padding states** whose `meta[:, 2] = -1` comment on [titrate.py:285](src/jaxpb/titrate.py#L285) ("0 blocks -> no work") is **false under `mc_ref`**—to execute `max(mcblocks) × (max(n) + max(P))` iterations in lockstep.
  3. At `Mp=1024`, the vmapped state arrays (`E`, `X`, `cnt` each `[32, 73, 1024]` f32 = 9.6 MB, plus `W` = 134 MB) thrash CPU cache on every move, whereas PyPKA's Cython MC keeps 1 chain/thread (`E` = 4 KB) in L1 cache and skips 90–98% of updates with a real `if (ok)` branch.
* **Proposed change:**
  * **On CPU:** Do **not** `vmap` chains. Compile a **single-chain** jitted `mc_single[Mp]` (so `lax.cond(ok, upd, ...)` compiles to a **real CPU branch**, loop bounds are that state's actual `(n, P, mcblocks)` with zero padding waste, and working set fits L1/L2) and dispatch the 73 pH chains (× states) across CPU cores via `ThreadPoolExecutor` (JAX releases the GIL during XLA execution). Alternatively (and even faster for CPU parity with PyPKA), provide a ~70-line Cython/Numba/C FFI CPU kernel implementing the exact same `hash32`/`invexp`/move loop with zero XLA loop overhead.
  * **On XLA GPU:** XLA cannot branch per-lane inside `vmap(lax.cond)`, so Pallas remains essential on GPU for the full 35×; however, fixing 4.2 + packing `S` smaller / removing dead padding launches still cuts `mc_ref` GPU cost materially.
* **Expected gain (estimate):** On CPU, non-vmapped `mc_single` + `ThreadPoolExecutor` + 4.2: **~30–80×** (588 s → ~7–20 s/state on 12 CPUs, beating PyPKA's 43.5 s); a thin C/Cython CPU backend for the same RNG/moves: **~60–150×** (~4–10 s/state).
* **Parity risk:** **Bitwise identical** (same RNG stream, same float32 ops per chain, same `E` rebuild every `BLOCK=1000` sweeps).

### 4.2 Triton register-workaround $O(M_p)$ / $O(N_p)$ reductions inside `chain()` run on XLA
* **Location:** [titrate.py:119-126](src/jaxpb/titrate.py#L119-L126), [titrate.py:138-146](src/jaxpb/titrate.py#L138-L146), [titrate.py:153-167](src/jaxpb/titrate.py#L153-L167), [titrate.py:181](src/jaxpb/titrate.py#L181), [titrate.py:91-107](src/jaxpb/titrate.py#L91-L107)
* **What is wrong:** `chain()` was copied verbatim from the Triton kernel (`gmc.chain`), where Triton lacks dynamic register indexing:
  1. `cur_of(curv, i)` scans all `Np` elements (`jnp.sum(jnp.where(sidx == i, curv, 0))`) instead of `curv[i]`.
  2. `dU` scans all `Mp` elements (`jnp.sum(jnp.where(idx == nw, E, 0) - jnp.where(idx == oc, E, 0))`) instead of `E[nw] - E[oc]` (and 4 scans in `pair_move`).
  3. `X` and `curv` updates scan all `Mp`/`Np` elements via `onehot`/`where` instead of `X.at[nw].add(1)` / `curv.at[i].set(nw)`.
  4. `cnt + flag * X` ([titrate.py:181](src/jaxpb/titrate.py#L181)) adds all `Mp` floats every sweep instead of `O(n)` increments `cnt.at[curv].add(1)` (only `n ≈ Mp/4` active microstates).
  5. `accept` ([titrate.py:107](src/jaxpb/titrate.py#L107)) eagerly computes all 3 rational branches of `invexp(dU)` (2 float divisions) even when `dU <= 0`, `dU > 50`, or `nw == oc`.
* **Proposed change:** Split an XLA-native `chain_xla()` (keeping `chain()` for `_mc_kernel`) using `O(1)` indexing (`curv[i]`, `E[nw] - E[oc]`, `curv.at[i].set(nw)`), `O(n)` `cnt` accumulation (dropping `X`), and nested `lax.cond` around `invexp` when non-vmapped on CPU.
* **Expected gain (estimate):** **~2.5–4×** on XLA GPU `mc_ref` (62.7 s → ~16–25 s/state), and is required alongside 4.1 for the 30–80× CPU MC speedup.
* **Parity risk:** **Bitwise identical** for `site_move` (`E[nw] - E[oc]` equals the masked sum of 2 non-zeros). **Low (1-ULP) risk** in `pair_move`: [titrate.py:158-159](src/jaxpb/titrate.py#L158-L159) sums the 4 `E` terms in `idx` order rather than `(E[na] - E[oa]) + (E[nb] - E[ob])`, which can differ by 1 ULP and flip a borderline acceptance; preserve `idx`-ordered sum of the 4 indexed scalars if strict bitwise parity vs `mc_pallas` is required.

---

## 5. Other Findings: Syncs, Redundant Work, Dead Code, Correctness Risks

1. **Host sequential `specrad` causes 83.8% GPU idle (0.455 s/state) + redundant D2H syncs**
   * **Location:** [pb.py:277-281](src/jaxpb/pb.py#L277-L281), [pb.py:432,436,442](src/jaxpb/pb.py#L432)
   * **What is wrong:** `estimate_specrad` recomputes `_relfac_vec()` (2.13 MB) + `_ODD_C` (1.06 MB) and H2D-transfers them every batch, copies 34 MB `P` D2H, runs a single-threaded `np.cumsum` (allocating another 34 MB just to read `[..., -1]`), and transfers `spec` back H2D. Then [pb.py:432](src/jaxpb/pb.py#L432) does a pointless `spec_h = np.asarray(j["spec"])` D2H sync right after uploading `sp` from host! Lines [pb.py:436,442](src/jaxpb/pb.py#L436) also call `jax.block_until_ready` twice per batch purely for internal `tsor` timing.
   * **Proposed change:** Cache static device constants (`_relfac_vec`, `_ODD_C`, `colour_masks`, `fib_sphere`, `op1/op2`, `PH_TERM`) once per device; run the sequential f32 scan on-device (or overlap host reduction of batch $k$ with GPU work of batch $k-1$, or use a tight C/Numba loop without allocating the 34 MB `cumsum` output); use host `sp` directly instead of `np.asarray(j["spec"])`; gate `block_until_ready` behind a timing flag.
   * **Expected gain (estimate):** **0.3–0.45 s/state** on A40 (both Pallas and XLA); eliminates 68 MB transient host allocs/batch. **Bitwise identical.**

2. **Both `_coefs` (4×f32) and `_codes` (`uint8`) computed unconditionally in `setup1`/`setup2` + phase-2 zero-padding waste**
   * **Location:** [pb.py:180-181](src/jaxpb/pb.py#L180-L181), [pb.py:200-202,211](src/jaxpb/pb.py#L200-L202), [pb.py:439,445](src/jaxpb/pb.py#L439)
   * **What is wrong:** `setup1` and `setup2` build 544 MB of 4×f32 `_coefs` that Pallas throws away (and XLA won't need after 1.1), and `setup2` pads `items2` with 544 MB of zeros to `2BT=256` that XLA `lanes_init` immediately slices away.
   * **Proposed change:** Drop `_coefs` from `setup1`/`setup2` once XLA SOR uses `op_code`/`op_lut` (or gate by backend); don't zero-pad `items2` for XLA.
   * **Expected gain (estimate):** Saves ~544–1,088 MB live memory and ~20–30% of `setup1`/`setup2` time. **Bitwise identical.**

3. **`post` / `rf_energy` recomputes `spread` 3×, full `81^3` Laplacian, `G[QM, NB]` per tautomer, and discarded energy branches**
   * **Location:** [pb.py:133-148](src/jaxpb/pb.py#L133-L148), [pb.py:224-229](src/jaxpb/pb.py#L224-L229)
   * **What is wrong:** `post` re-runs `spread` (allocating 3× `[B, 8, 81^3]` = 816 MB), computes 6 `jnp.roll`s over all 531k points when only `NB` (~40k) boundary points are read, computes `G = EPKT / norm(q_pos - surf)` per tautomer (`[16, 8, 24, 48000, 3]` = 1.77 GB CPU temp), and computes both `E(sv)` and `E(sv * scale)` while discarding one on every call.
   * **Proposed change:** Evaluate the 6-neighbor Laplacian only at `bidx` (via flat offsets `±n^2, ±n, ±1`), compute only the requested energy (`raw` vs `corrected`), and chunk `G` over `QM` or share per site.
   * **Expected gain (estimate):** Cuts `post` CPU temp **1.67 GB → <0.3 GB** and CPU `post` time ~3–4× (1.28 s → ~0.3–0.4 s). **Bitwise identical** (keep `tsum`).

4. **Correctness risks & edge cases**
   * **Uncaught `IndexError` on class-4 geometry overflow:** [pb.py:378-384](src/jaxpb/pb.py#L378-L384) increments `c += 1` without checking `c >= len(RC.NA_C)`. `surface_atoms` then raises `IndexError`, bypassing `api.py:230`'s `except OverflowError:` and crashing the driver instead of logging `failed_pb`.
   * **Unchecked `rmax = 2.0` assumption in `maps.py`:** [maps.py:183,326](src/jaxpb/maps.py#L183) hardcodes `ca = 2.0 + exrad + 0.01 = 4.01`. Any atom with `rad > 2.01` silently violates the 27-cell radius guarantee in `query_min` and block culling. Add an explicit assertion/guard on `rad.max()`.
   * **`site_acent_S` silently ignored:** [pb.py:314](src/jaxpb/pb.py#L314) uses `rec.site_acent_M` for coarse, focused, *and* model grids without asserting `site_acent_S == site_acent_M` (even though `prep_pypka.py:253` stores them separately).
   * **Import-time JAX init + unbounded summary memory:** [api.py:174](src/jaxpb/api.py#L174) instantiates `MEMTRACK = MemTrack()`, calling `jax.devices()[0]` on import (locking JAX platform/flags and leaking state across runs). [api.py:583](src/jaxpb/api.py#L583) appends every state's full timing dict to `summary["per_state"]` in memory across all shards (+0.6 GB per 1k states → ~18 GB at 30k states, [M6.json:367](results/M6.json#L367)).

5. **Dead / redundant code (cleanup)**
   * Unused legacy/rejected code: `maps.build_maps` & `maps.maps_batch_ref` ([maps.py:180-245,525-541](src/jaxpb/maps.py#L180-L245)), `PROBE_HASH` / `probe_cells_hashed` / `_q_one_hashed` ([maps.py:113-124,279,286-293](src/jaxpb/maps.py#L113-L124)), `titrate.curves_to_pka_jax` ([titrate.py:327-339](src/jaxpb/titrate.py#L327-L339)), `titrate.chain.pick` ([titrate.py:122-123](src/jaxpb/titrate.py#L122-L123)), dead `omega = -1.0` branch ([pb.py:171-172](src/jaxpb/pb.py#L171-L172)), dead `- jnp.zeros((), dt)` ([maps.py:164,495](src/jaxpb/maps.py#L164)), unused parameters `w` in `build_cells` ([maps.py:59](src/jaxpb/maps.py#L59)) and `NP` in `_cell_hash` ([geometry.py:304](src/jaxpb/geometry.py#L304)).

---

## Prioritised Top-5 Plan

| Rank | Change | Targets | Estimated Impact | Parity |
|---:|---|---|---|---|
| **1** | **CSR cell tables + 27-offset / dynamic-`kmax` `_q_one` in `maps.py` (plus CPU `B=2..4`)** | `maps_batch` CPU time (370 s) & CPU/GPU peak RAM (20 / 11.3 GB) | CPU maps **5–15× faster** (370 → 25–75 s); `maps_batch` temp **20 → <0.4 GB** (CPU) & **11.3 → ~1 GB** (GPU); A40 maps **1.3–1.8×** | **Bitwise** (per-site CSR); check if sharing state-level `lo` |
| **2** | **Rewrite XLA SOR: colour-split red/black slices + `op_code`/`op_lut` + 1-of-10 max check + chunked device `while_loop`** | A40 XLA SOR (43 s) & CPU SOR (373 s) | A40 XLA SOR **6–9× faster** (43 → 5–7 s, ~1.5× of Pallas); CPU SOR **12–25× faster** (373 → 15–30 s); XLA SOR compiles **12 → 1**; drops 544 MB `_coefs` | **Bitwise** |
| **3** | **Non-vmapped scalar-indexed CPU MC (`mc_single` + `ThreadPoolExecutor`, or thin C/Cython kernel using exact `hash32`/`invexp`) + `O(1)` indexing in `mc_ref`** | CPU MC (588 s/state; 2,643 s at Mp1024) & A40 XLA MC (62.7 s) | CPU MC **30–80× faster** in pure JAX (588 → ~7–20 s, beating PyPKA 12-CPU) or **60–150×** in C/Cython; A40 XLA MC **2.5–4×** | **Bitwise** (preserve `idx`-order 4-term sum in `pair_move`) |
| **4** | **Eliminate the other two CPU memory cliffs: face-only (38,402 pts) `coulomb_bc`/`interp` in `setup1`/`setup2` + pair-compacted/chunked `geometry_sites` + `del` dead batch buffers** | CPU RSS (14 GB `setup1` + 8.5 GB `geometry_sites`) & CPU `solve_setup` (12.8 s) | Unlocks **<2.5 GB total CPU RSS** (no more 24.6 GB OOMs); CPU `solve_setup` **10–13× faster** (12.8 → ~1.1 s); GPU peak VRAM **16.1 → ~3–4 GB** | **Bitwise** |
| **5** | **Per-atom/per-pair neighbor lists (`K_nbr ≈ 96–128`, `int32`) in `geometry.py` + device-side/overlapped `estimate_specrad` & cached constants** | CPU `geometry` (45.5 s) & A40 `geometry` (2.08 s) + `specrad` GPU idle (0.455 s) | CPU `geometry` **4–6× faster** (45.5 → ~8–11 s); A40 `geometry` **2.5–3.5×** + `specrad` **-0.3–0.45 s/state** (helps Pallas path too) | **Bitwise** |

**Combined effect (estimate):**
* **12-CPU full pipeline (PB + MC 200k):** ~1,500–2,300 s/state & 21–25 GB RSS → **~50–135 s/state & <2.5 GB RSS** (competitive with 12-CPU native PyPKA at 85.7 s / 6.8 GB, with zero OOMs).
* **A40 XLA-only PB:** 49.5 s/state & 16.1 GB VRAM → **~9–12 s/state & ~3–4 GB VRAM** (nearly matching current A40 Pallas PB at 10.6 s, while also speeding up A40 Pallas PB by ~2–3 s/state via items 1, 4, and 5).
