# Tier 3 — gradient sanity on set-2b complexes

Experiment 03, tier 3 (`03_dataset_and_model.md`). No experimental data, no training, no
labels: this asks only whether `d(binding linkage)/dP` points somewhere the discrete model
agrees with. It is the cheapest check that catches the failure that matters most for
design — a gradient that points where the hard-sequence rescore disagrees.

Code: `benchmarks/gradient_sanity.py`. Submission:
`_HPC/submission/jax-Ka/pkabench/gradient-sanity.sbatch`.
Results: `_runtime/jax-Ka/pkabench/diagnostics/gradient-sanity/{v1,diag-1}/`.

## Target and sign convention

The same soft sequence `P[N,20]` is sliced into the AB, A and B states, so every state
sees matched identities and the fixed deprotonated charges cancel site by site:

    dQ(pH) = Q_AB(pH) - Q_A(pH) - Q_B(pH)        protons taken up on binding

Wyman linkage for association is `d(ln Ka)/d(ln[H+]) = dQ`, hence
`d(dG_bind)/dpH = +1.364 dQ` and

    L = 1.364 * integral_5.5^7.5 dQ dpH = dG_bind(7.5) - dG_bind(5.5)   [kcal/mol]

**Positive L means binding is tighter at the LOW pH.** This matches `pkabench.linkage`,
whose `delta_g(pH) = 1.364 * integral_7^pH dQ` is `dG_bind(pH)` referenced to pH 7.
Endpoints are grid points of `PH = linspace(-2, 16, 73)`.

The first-order score of substituting identity `a` at position `i` is the directional
derivative along the simplex edge `native -> a`:

    s[i,a] = dL/dP[i,a] - dL/dP[i,native_i]

which is exactly the linear prediction of `L(mutant) - L(native)`. Candidates are then
discretised to a hard one-hot `P` and rescored. The rank agreement between `s` and the
exact rescore is the test.

## Method

- **Design set.** Interface positions (minimum heavy-atom distance to the partner
  <= 5 A), excluding disulfide-frozen positions and Gly/Pro, capped at the 48 closest.
  Alphabet `ADEHKNQRSY` on top of the native identity — titratable (DEHKRY) *and*
  non-titratable (ANQS). Offering the non-titratable ones is what makes "titratable
  substitutions rank top" a claim rather than a tautology.
- **Gradient path.** optimistix implicit adjoint (`optx_solver`), not reverse mode
  through the 1024-step damped loop: unrolling 1024 steps across 73 pH points would store
  several GB of activations. This is the opt-in design-gradient use the solver study
  (`25_jaxka_v2_report.md`) left in place.
- **Rescore path.** Every candidate is rescored with *both* the optx solver and the
  production 1024-step damped solver, so the discrete ranking never depends on the solver
  the gradient came from.
- **float64 throughout** (caches built at `dtype=np.float64`), so the small exact deltas
  and the finite-difference check are meaningful.

**Caveat, stated once.** Mutant side chains use ONE frozen CCD conformation in the
backbone-local frame. No repacking, no rotamer search. Every number here is for that
specific placement, not a relaxed mutant. For buried positions this is a real limitation;
the rotamer-averaging argument in 04 is the principled answer.

## Results (run `v1`, 50 rescored candidates per system)

| | 1brs | 1fcc | 1frt |
|---|---|---|---|
| partners | barnase A / barstar D | Fc A / protein G C | FcRn+b2m A,B / Fc C |
| residues / design positions | 195 / 38 | 262 / 34 | 573 / 43 |
| candidates | 353 | 318 | 405 |
| base L (kcal/mol) | +1.09 | -0.34 | **+3.74** |
| spearman (score vs exact) | 0.983 | 0.995 | 0.993 |
| pearson | 0.996 | 0.991 | 0.999 |
| sign agreement | 1.00 | 1.00 | 1.00 |
| spearman within top 20 | 0.923 | 0.943 | 0.944 |
| top-20 categories | 12 introduce, 8 swap | 10 introduce, 10 swap | 14 introduce, 6 swap |
| top-20 `neutral` (failure signal) | **0** | **0** | **0** |
| top-20 / all interface distance (A) | 3.25 / 3.57 | 3.06 / 3.41 | **2.20 / 3.28** |
| max relative error | 0.58 | 0.69 | 0.67 |
| wall / peak RSS | 72 min / 3.0 GB | 25 min / 3.8 GB | 64 min / 9.8 GB |

Solvers are not a confound anywhere: base optx-vs-damped gap `2e-12` to `5e-12` kcal/mol,
`max_optx_damped_gap` over rescored candidates `<= 4e-11`, `spearman_optx_vs_damped = 1.0`,
`active_set_leak = 0` in all states.

One solver defect exists and is confined: on 1frt the optx continuation sweep fails to
converge at exactly one pH point, **pH 15.0** (residual 3.2e-3 against the 2e-5
tolerance), in the AB and A states. The damped solver converges everywhere (1e-16). The
linkage window is 5.5-7.5, so pH 15 carries zero quadrature weight and cannot enter L or
its gradient. `base_residual_by_ph` in each record localises this; an aggregate maximum
only reports that it happened.

### The chemistry lands on the known residues

- **1brs** `L = +1.09`: tighter at low pH, the protonated His102 (barnase) to Asp39
  (barstar) salt bridge. The ranking's most positive and most negative candidates are the
  two sides of that pair — `A102 H->D/E` at +3.29 and `D39 D->R/K` at -3.97/-3.65.
- **1frt** `L = +3.74`: tighter at pH 5.5 than 7.5, which is the FcRn recycling switch,
  recovered from structure alone with no fitting. Top 2 are `C310 H->D` (+4.13) and
  `C310 H->E` (+3.76); rank 9 is `C435 H->D`. Fc His310 and His435 are the canonical
  FcRn pH-switch histidines. Most negative are `A117 E->R` (-4.04) and `A117 E->Y`
  (-3.57) — FcRn Glu117, the acidic partner that raises His310's pKa on binding.
- **1fcc** `L = -0.34`: a weak, slightly high-pH-preferring linkage. Top candidates are
  `C28 K->D/E` and Glu/Asp introductions at Fc 253/254/434/436.

1frt is the most selective: its top-20 candidates average 2.20 A from the interface
against 3.28 A over all candidates. That addresses the obvious worry that the gradient is
merely counting "add a charge near the partner".

### Where the first-order score fails

1. **It does not reliably pick the single best substitution.** 1brs ranks `A27 K->E`
   first (score 3.517) but its exact value is 3.103, below rank 1 `A102 H->D` at 3.297.
   1fcc ranks `C28 K->D` first (exact 3.712) but its rank-8 `C31 K->D` is exact 3.866.
   `spearman_within_top` is 0.92-0.94 on all three: the top block is reordered everywhere.
2. **Charge reversals at Lys are the consistent failure mode, in both directions.**
   1brs `A27 K->E` overestimates by 13%; 1fcc `C31 K->D` *under*estimates by 61%
   (2.408 vs 3.866). Worst single case `1brs A103 Y->K`, -2.59 predicted vs -1.64 exact.
   Removing a positive charge while adding a negative one is a large perturbation and the
   linear term does not capture it.

Neither failure breaks the tier-3 claim — signs never disagree and the global Spearman is
0.98-0.99 — but both say the same thing: **the gradient is a screen, not a scorer.** A
design loop must rescore discretely, which is precisely what this tier was meant to settle.

## Gradient verification (run `diag-1`)

Analytic directional derivatives against a one-sided finite difference at two step sizes,
over both near-zero and top-scoring directions (`--fd-top 5`).

- **1brs: verified.** `max_relative_gap = 1.3e-4` over derivatives spanning 0.07 to 3.5;
  `gap_vs_magnitude_correlation = 0.16`. Example: analytic 2.5643174 vs FD 2.5643169.
- **1frt: verified.** `max_absolute_gap = 3.4e-4`, `max_relative_gap = 7.9e-3` — and that
  worst relative figure is on the smallest derivative in the set (0.021). On the
  top-ranked directions the agreement is ~5e-5 relative: rank 0 `H->D` analytic 4.1302828
  vs FD 4.1305190.
- **1fcc: a bounded absolute offset, immaterial to ranking.** The gap is capped at
  **2.47e-3 kcal/mol** across derivatives from 0.025 to 3.05, and
  `gap_vs_magnitude_correlation = -0.71` — it does not grow with the signal. On the
  top-ranked candidates this is <= 0.05% relative; the 9.9% figure that first drew
  attention is 0.00247 divided by the smallest derivative in the set.
  Ruled out as causes, with evidence:
  - *Non-convergence* — all residuals 5e-10 (optx), 1e-16 (damped), zero pH points over
    the 2e-5 tolerance.
  - *Burial-clip saturation* — saturation is pervasive and does not track the error.
    1fcc has 770/1810 AB channels at the lower clip and shows the offset; 1brs has
    527/1369 (plus 110 at the upper clip) and 1frt has 1304/3947 (plus 206 upper), and
    both are clean. 1frt is the most saturated and the most accurate.
  - *A solution-branch flip* — would make the FD scale as `delta/eps` and differ tenfold
    between the two step sizes; the two agree to 1e-5.

  **Mechanism unresolved.** A fixed 0.00247 recurring across unrelated directions looks
  like one omitted term of fixed size, but no candidate survives the 1brs comparison. It
  is recorded as a characterised, bounded discrepancy rather than attributed to a cause
  that has not been demonstrated.

## Status

Tier 3 passes. Signs never disagree, global Spearman is 0.98-0.99, and no top-20 candidate
on any system fails to touch a titratable group. The chemistry lands on His102-Asp39
and His310/His435-Glu117 without being told about either.

Not claimed: that these substitutions would survive repacking, that the magnitudes are
calibrated against experiment (no set-2b labels are admitted — see
`curation/set2_leads.json`), or that first-order ranking can replace discrete rescoring.

## Reproduction

```
# full run: 50 rescored candidates per system, 1024-step damped rescore
sbatch --array=0-2 --cpus-per-task=6 gradient-sanity.sbatch v1
# verification: FD probes on top-scoring directions, per-pH residuals, clip counts
sbatch --array=0-2 --cpus-per-task=8 --time=04:00:00 gradient-sanity.sbatch \
    diag-1 --top 5 --controls 0 --fd-top 5
```

Jobs 733560 (v1) and 735853 (diag-1), all six array tasks COMPLETED. Peak RSS 3.0-9.8 GB;
1frt is the memory driver and needs more than the 4 cpus x 2G the smaller systems take.
Note that `diag-1` reports a lower `spearman_all` (0.92-0.93) purely because
`--top 5 --controls 0` rescores only 10 candidates over a restricted range; the `v1`
figures over 50 candidates are the ones to quote.
