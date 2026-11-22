The main problems are **oversized intermediate arrays, full-grid work that is later masked out, and vectorized control flow that still executes inactive work**. Compile time and host dispatch are secondary.

I only read sources and evidence; I did not run the pipeline, compile kernels, or change files. Estimates below are engineering estimates, not measured speed-ups, and should not be multiplied indiscriminately.

The matched comparisons in [profile2.json](results/profile2.json) establish:

| Stage | Current evidence |
|---|---|
| SOR, 20 states | A40 XLA **43.03 s/state**, Pallas **4.16 s/state** |
| CPU maps, matched five states | **370.08 vs 1.61 s/state: 230.5×** |
| CPU geometry, same states | **45.54 vs 1.40 s/state: 32.6×**, rather than 100× |
| CPU temporary allocations | Maps **19.99 GB**, setup1 **13.98 GB**, model geometry **8.52 GB** |
| MC, full S=32, Mp1024 | CPU **2643 s/state**, A40 XLA **224.7**, Pallas **6.31** |

Ranked findings follow. Maps ranks first because it combines a dominant CPU cost with the OOM problem; SOR is first for GPU-only work.

1. **Maps materializes enormous dense cell tables and candidate tensors.**

   Locations: [maps.py:59](src/jaxpb/maps.py:59), [maps.py:90](src/jaxpb/maps.py:90), [maps.py:333](src/jaxpb/maps.py:333), [maps.py:478](src/jaxpb/maps.py:478).

   `build_cells` expands sorted probe indices into `[ncell³,Kp]`. For the coarse grid, `ncell=61`, `Kp=768`: **697 MB per site, 11.16 GB at B=16**, just for int32 indices. This is largely independent of molecular size.

   Query chunking compounds that cost. `_q_chunk` has a minimum of 1024 rows, so a probe query considers `1024 × 27 × 768 = 21.2 million` candidate slots **per site**. The logical gathered XYZ array alone is **4.08 GB across B=16**, before distances, indices and masks. CPU fusion does not eliminate enough of these intermediates. `reach_mask_b` likewise sets its chunk budget without accounting for B or the multiple arrays per candidate.

   This explains the memory floor much better than “large fine grids”: all three sets of four boolean grids occupy about **102 MB**, and `[16,131072,3]` fine probe coordinates occupy **25 MB**. The measured 19.99 GB is compiled temporary storage, not those outputs or compilation itself.

   **Change:** retain `order`, per-cell `start`, and `stop/count` instead of expanding the dense K table. Those ranges already exist in `build_cells`. Reconstruct candidates only for the current query tile, with direct cell-range lookup. This avoids the repeated per-query binary searches that made `PROBE_HASH=True` slower on A40. Tile both query rows and candidate slots, carrying the current minimum and its original candidate rank. Make CPU chunk limits byte-based and account for B.

   Use a fixed CPU B=4 initially, or process fixed B=16 inputs through four internal B=4 groups. Give padded sites zero work counts: `_site_layout` currently repeats the last real site, so maps computes its work again.

   Do **not** simply hoist current tables outside the site `vmap`: their origins depend on `acent`. Sharing a global spatial index requires preserving candidate coverage and tie ordering.

   **Gain:** measured maps-only B=4 reduces temporary storage **20→5 GB**, RSS **21→5.84 GB**, and time/site **4.36→3.55 s**. **Estimate:** compressed tables plus smaller CPU tiles could deliver **2–5× maps speed-up**; eliminating the dense coarse table removes an 11.16 GB logical allocation at B=16. Whole-pipeline RSS still needs validation.

   **Parity:** low risk for microbatching; moderate for candidate streaming. Preserve the existing distance expression, first-candidate tie rule, padding behavior and overflow checks. Keep the current culling and dynamic reach optimizations—they already shipped.

2. **XLA SOR is already red–black, but its representation computes both colours over the entire lattice.**

   Locations: [sor.py:52](src/jaxpb/sor.py:52), [sor.py:76](src/jaxpb/sor.py:76).

   Each half-sweep forms a six-neighbour stencil over `[W,81,81,81]`, then `where` retains only one colour. It repeats this for the other colour. Coefficients are four f32 arrays gathered through `exE[op]`, etc.; the Pallas implementation instead reads compact operator codes and updates in place.

   The profile separates the causes: XLA costs approximately **0.46 ns/executed lane-sweep-point**, versus Pallas **0.125 ns/useful lane-sweep-point**, and XLA additionally executes **2.6–3.1×** unnecessary lane work. Together these explain approximately 10×. The source identifies redundant work, but the supplied evidence lacks an XLA GPU trace; exact fusion and kernel-count attribution remains unverified.

   **Change:** implement the colour-compressed fallback promised in DESIGN: two contiguous colour arrays, fixed neighbour offsets/halos, and boundary masks. Because 81 is odd, flattened-grid parity matches checkerboard parity, simplifying packing. Update only the target colour and keep each half-sweep’s stencil arithmetic together. Avoid reconstructing full-grid potentials between sweeps.

   Keep operators shared by site. Compare compact codes/LUTs against prepacked coefficients; avoid repeatedly expanding four operator fields to every tautomer lane. Keep spatial elements contiguous within a lane and use smaller lane tiles on CPU.

   Also split each block into **nine unchecked sweeps plus one checked sweep**. Currently `jnp.where(k == check-1, max(abs(new-p)), mc)` expresses an eager reduction rather than a conditional branch; source-level separation makes its elimination explicit. Carry only the current potentials, not scan histories.

   **Gain — estimate:** **1.5–3× SOR kernel improvement**, before reducing lane waste. A `fori_loop`→`scan` substitution alone is unlikely to matter.

   **Parity:** moderate risk from changed fusion/FMA behavior. Preserve six-neighbour addition order, even-first updates, f32 arithmetic, omega and boundary values. The check must remain **last-sweep change**, every **10 sweeps**, with the same threshold and cap. Changing check frequency, precision or solver changes the requested result.

3. **The XLA lane pool freezes finished outputs but continues calculating finished lanes.**

   Locations: [sor.py:67](src/jaxpb/sor.py:67), [sor.py:116](src/jaxpb/sor.py:116), [records.py:148](src/jaxpb/records.py:148).

   `done` is applied **after all ten sweeps**. Compaction happens only when live lanes cross the widely spaced widths `(16,64,256)`: 65 live lanes still execute 256 lanes. These are real stencil calculations, unlike Pallas’s early-return programs.

   **Change:** first add a bounded width ladder such as `(16,32,64,128,256)` and compact at the existing ten-sweep boundaries. For CPU, evaluate smaller independently scheduled lane groups with scalar control over each group. Subsequently consider a bounded refill pool across site batches, scheduling focused solves only after their coarse dependencies finish.

   A single device `while_loop` is useful only if it retains effective work avoidance. Moving everything into a fixed-width loop with masks can remove synchronization while making lane waste worse. Likewise, a per-lane `cond` under `vmap` does not provide the desired branch skipping.

   Host synchronization is not the main explanation: on the twelve GPU states, roughly **476 of 486 s** inside the two pools is spent in `sor_block`. On CPU, approximately **1864 of 1866 s** is there.

   **Gain — estimate:** **1.3–2× additional SOR speed-up** from practical packing improvements. Perfect elimination of current waste gives an arithmetic upper bound of roughly **2.6–3.1×**, not a realistic promise. Removing pool overhead alone offers only about **2% GPU / 0.1% CPU** on these measurements.

   **Parity:** low risk when rescheduling independent lanes. Preserve per-lane iteration state, first passing check, output identity and focused/coarse dependencies. Extra fixed widths increase the finite executable budget; they do not introduce per-state recompilation.

4. **MC’s shared chain implementation is shaped for vector registers, not CPU scalar access and branches.**

   Locations: [titrate.py:119](src/jaxpb/titrate.py:119), [titrate.py:135](src/jaxpb/titrate.py:135), [titrate.py:195](src/jaxpb/titrate.py:195).

   Every attempted move obtains current states through masked **Np-element reductions** and energy differences through **Mp-element reductions**. Updates construct whole one-hot vectors. Nested `vmap` batches states and all 73 pH chains, producing large lockstep working sets and random W-row accesses.

   Crucially, the acceptance `lax.cond` becomes a select when its predicate is vmapped, so rejected moves do not retain ordinary scalar branch-skipping semantics. This is documented by [JAX](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html). Padding states and shorter chains also participate in the batched computation until the longest active loop finishes; “padding states cost nothing” is incorrect for this backend.

   **Change:** provide a CPU-specific pure-XLA chain using indexed integer current-state access, scalar acceptance control, and indexed occupancy/count updates. Schedule independent chains through bounded CPU tasks or a memory-bounded mapping arrangement; benchmark small chain tiles against independent chains. Replacing everything with a sequential `lax.map` would bound memory but could sacrifice CPU parallelism.

   Preserve W-row field updates on acceptance, the sequential field rebuild every 1000 sweeps, and exact RNG counter advancement. Use fixed state/chain batch sizes, including a small tail bucket, rather than compiling each actual batch size.

   **Gain — estimate:** **2–6× CPU MC improvement**, workload dependent. Partial batches should benefit particularly. The measured full-batch Mp256 cost is 99 s/state, versus 437 s/state in the different, sparsely populated benchmark configuration; those are not interchangeable throughput measurements. The same control-flow issue contributes to GPU XLA MC’s roughly 35× gap, but a scalar CPU formulation is not automatically a GPU fix.

   **Parity:** substantial risk if energy arithmetic is simplified casually. In particular, the pair-move masked reduction is not generally bitwise equivalent to four scalar additions in a chosen order. Preserve its reduction tree—or retain that reduction initially. Integer indexing and exact count updates are safer first changes. Do not replace the maintained field with direct energies or change RNG, sweep count or acceptance approximation.

5. **Boundary setup evaluates expensive expressions at interior points that are immediately discarded.**

   Locations: [pb.py:99](src/jaxpb/pb.py:99), [pb.py:183](src/jaxpb/pb.py:183), [pb.py:205](src/jaxpb/pb.py:205).

   `coulomb_bc` computes distances, exponentials and charge sums over **531,441 grid points × QM**, then masks to faces. There are only **38,402 unique face points**: **13.84× fewer**. CPU setup1’s measured **13.98 GB** temporary allocation is consistent with this unfused point–charge expansion. Focused setup similarly interpolates the entire grid before retaining only faces.

   **Change:** precompute a fixed unique face-index list. Generate those coordinates using the existing coordinate arithmetic; evaluate boundary values in fixed point tiles; scatter to an initially zero grid. Apply the same approach to focused interpolation. Tile points or lanes first, leaving the 24-charge reduction intact.

   **Gain — estimate:** **3–8× setup-stage speed-up**, with a large reduction in its 14 GB temporary allocation. The affected boundary arithmetic has a 13.84× work reduction. On the matched CPU subset, setup is only **12.83 s/state**, so this ranks higher for memory than for total runtime.

   **Parity:** low-to-moderate risk. Preserve coordinate construction and the existing charge reduction; repartitioning that reduction can change rounding. A global CPU B=4 also reduces setup’s lane dimension; the existing B-test measured maps alone, not full-pipeline RSS.

6. **Geometry uses very large candidate tiles; model geometry expands invalid candidates before masking.**

   Locations: [geometry.py:87](src/jaxpb/geometry.py:87), [geometry.py:113](src/jaxpb/geometry.py:113), [geometry.py:442](src/jaxpb/geometry.py:442), [records.py:144](src/jaxpb/records.py:144).

   `GEO_CHUNK=131072` and 27×16 candidate slots produce **56.6 million candidate pairs per chunk**. Their f64 XYZ gather alone is **1.36 GB**. `_cand` performs two searches for each of 27 neighbouring cells per point; `_burial` computes distances and several reductions over the expanded arrays.

   Model geometry constructs **405,120 vertex/arc candidates per site** from padded MA=32 and MX=720, then tests all against 32 atoms. Many invalid pairs/arc slots are rejected only after this computation. Its CPU temporary allocation reaches **8.52 GB**.

   **Change:** use a fixed smaller CPU geometry chunk, initially 8K–16K, and stream model candidates through the same ordered compaction pattern already used for whole-molecule geometry. Compact valid model pairs before expensive burial work, preserving their original order. Cache cell ranges where practical and use int32 for proven-bounded candidate indices; retain int64 where global candidate numbering/sentinels require it.

   **Gain — estimate:** **1.5–3× CPU geometry speed-up** and **8–16× smaller chunk-proportional intermediates**. Geometry is about 45.5 s/state on the matched subset, much less than maps or SOR.

   **Parity:** moderate. Keep f64 geometry, exact burial/rank semantics, probe order and overflow detection. The earlier profile already demonstrated changed probe counts with f32. Removing per-chunk host syncs, reducing K to 16 and AB reuse are already implemented; they are not new opportunities.

7. **Buffer lifetimes and redundant setup outputs add memory; most remaining synchronization is lower priority.**

   Locations: [pb.py:177](src/jaxpb/pb.py:177), [pb.py:277](src/jaxpb/pb.py:277), [pb.py:407](src/jaxpb/pb.py:407).

   Setup returns both XLA coefficients and Pallas codes regardless of backend. Previous-batch `items1/items2`, coefficients and solution stores remain referenced while the next batch begins allocating maps. These cannot explain the 20 GB maps temporary, but they raise the baseline beneath it.

   **Change:** select setup outputs through a finite static backend variant, release consumed batch buffers after postprocessing, and donate buffers only where no later consumer needs them. Keep post/assembly data on-device when useful instead of downloading potentials and immediately uploading padded assembly inputs.

   The spectral-radius estimator is a separate GPU synchronization cost: **0.455 s/state, 84% GPU-idle**. Its sequential NumPy f32 sum is deliberate numerical behavior. Overlap that existing operation where dependencies allow; an ordinary device reduction or parallel cumsum is not equivalent.

   **Gain — estimate:** several hundred MB to a few GB less live storage, depending on retained buffers; modest runtime improvement. Spectral-radius overlap has at most roughly **0.3–0.4 s/state** opportunity on the measured GPU workload.

   **Parity:** low for lifetime changes, high if the spectral-radius accumulation order changes.

8. **The existing validation gates do not enforce the stated hard parity contract.**

   Locations: [tests/test_pb_m5.py:104](tests/test_pb_m5.py:104), [compile_budget.py:180](src/jaxpb/compile_budget.py:180).

   M5 accepts **99.9% sweep equality and 1e-5 energy error**, whereas this request requires equal sweeps and 1e-6. Moreover, [M5.json](results/M5.json) reports native-DelPhi sweep agreement of **95.58%/90.52%/86.5%**, and one prototype-relative energy error of **1.407e-6**. Its A40 Pallas/XLA equality is excellent, but that is a different comparison. CPU/GPU absolute differences in bench.json likewise do not establish the requested relative bound.

   **Change:** distinguish optimization regression against the current backend from absolute agreement with DelPhi. Require exact sweeps and deterministic reruns for every optimization; apply the requested energy bound explicitly. For MC, compare counts with identical keys, not just pKa averages.

   Update the compile-budget inventory for any fixed widths/microbatches and existing reuse variants. There is no evidence that warm recompilation causes the present slowdown: measured compile costs are small beside CPU execution, and M6 demonstrated flat scaling. Retain the old XLA/reference routines as validation oracles.

   **Gain:** no direct runtime gain; prevents accepting a numerically invalid “speed-up.” The supplied evidence does not support claiming that strict native-DelPhi parity is already satisfied.

The prioritised top-five implementation plan is:

1. **Bound CPU memory:** fixed B=4 execution, compressed probe-cell ranges, and byte-budgeted query/reach tiles. Measure the whole pipeline, including XL retries.
2. **Replace full-grid XLA SOR with colour-compressed sweeps:** preserve arithmetic order and the exact ten-sweep stopping rule; measure compiled buffers and fusion.
3. **Reduce XLA lane waste:** add bounded intermediate widths, then evaluate smaller CPU groups and bounded refill scheduling.
4. **Implement CPU-specific XLA MC:** indexed state access, effective scalar acceptance branches and bounded chain scheduling, with exact count/RNG regression checks.
5. **Remove discarded setup work and oversized geometry intermediates:** face-only boundary calculations first, then streamed model geometry and smaller f64 CPU tiles.

All five can retain optional Pallas acceleration, portable XLA references and a finite, state-count-independent compile budget.