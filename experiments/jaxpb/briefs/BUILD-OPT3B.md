# jaxpb phase 3b: replace the gated CPU MC fast path with an explicit summation order (2026-11-23, user decision)

User verdict on phase 3: "The gated version is stupid." The phase-3 `cpu` backend (uncommitted diff vs a5d5c1d in
src/jaxpb/titrate.py + api.py; read OPT3-REPORT.md, results/opt3.json) reproduces the XLA-CPU reduction tree of the ref
backend's pair-move dU by hand and is gated to Linux/x86_64/AVX2/JAX 0.11.1, Mp 64-1024, <= 8 microstates/site. Remove
that design. Do NOT commit.

## Change
1. In `chain` (titrate.py ~:155, shared by ref and the Pallas kernel), the pair-move energy is today
   `dU = jnp.sum(where(idx==na,E) - where(idx==oa,E) + where(idx==nb,E) - where(idx==ob,E))`: four nonzeros reduced in a
   backend-defined tree. Replace it with an explicit, documented order on scalar picks, e.g.
   `dU = (E[na] - E[oa]) + (E[nb] - E[ob])` then the existing W terms, keeping exact cancellation when na==oa / nb==ob.
   Picks may stay as masked sums of a single nonzero (exact) or become direct indexing; site moves (two nonzeros) are
   already order-free - leave them unless the same cleanup is free. Any other reduction whose order is backend-defined
   and affects the chain (check cur_of, rebuild, X/E updates, count accumulation) gets the same treatment, or a written
   argument why it is exact.
2. The `cpu` backend uses exactly the same expressions. Delete the sorting network, the bit-reversal rank, cpu_available()'s
   version/platform/AVX2/XLA_FLAGS checks and the microstate/bucket domain limits that existed only for the tree trick.
   Keep a plain capability check (CPU platform) and the threads/chain design if it stays fast. `auto`: Pallas on a
   supported GPU, cpu on CPU, ref otherwise.
3. Keep RNG, move order, BLOCK structure, accept rule, E/X/count accumulation unchanged otherwise.

## Gates
1. CPU: cpu backend bitwise == new ref backend on CPU (20 bench states at 50k, the 5 CPU states at 200k; reuse saved
   energies, no PB rerun), determinism (two runs + fresh process), flat compiles, unit tests (rewrite tests/test_opt3.py
   for the new contract: no domain limits; include > 8 microstates/site, buckets outside 64-1024 if supported, 1-site
   states, pair moves with na==oa / nb==ob).
2. Portability: explain why the new order is backend-independent for the chain's control flow; if any remaining
   difference between backends is possible (e.g. GPU FMA contraction in invexp or W terms), say exactly where.
3. Rebaseline (this deliberately changes bits once): new vs a5d5c1d outputs for Pallas on the A40 (104 L3 states + 20
   bench) and ref/cpu on CPU (20 + 5): number of differing curve counts, max |delta pKa|, pKa MAE vs the PyPKA teacher
   (L3 metric as in results/M6.json) old vs new, and an MC-noise reference (same states, a different seed, old code) so
   the change can be judged against noise. Pass = new-vs-old differences well inside seed noise and L3 pKa MAE unchanged
   within noise.
4. Cross-backend: report whether Pallas, GPU ref and CPU cpu/ref now agree bitwise for the same chains (desired; report,
   not a gate) and where not.
5. Speed: Pallas MC s/state not slower than a5d5c1d; CPU 200k MC s/state at 1 and 12 CPUs vs phase 3 (238.5 / 41.7).

## Compute rules (strict, unchanged from BUILD-OPT3.md)
Coulson only via `ssh Coulson-Chemistry`; CPU partition amd96 and A40 (oc, comp1400, gpu:a40:1), <= 12 CPUs,
--mem-per-cpu=2G, env envs/ogqt-main-7c3e3e9 + jobs/a40site shim. New frozen folders tierB/jaxpb-pipeline/opt3b-<tag>/,
outputs tierB/jaxpb-pipeline/opt3b/. Never modify existing folders or touch others' jobs or ~/jax-Ka; no login-node compute,
installs, downloads, commits or pushes. Keep file counts low.

## Output
src/tests edits (uncommitted), jobs/opt3b*, results/opt3b.json, OPT3B-REPORT.md (<= 300 words): design, gate table,
rebaseline numbers vs seed noise, cross-backend agreement, timings. Stop after this.
