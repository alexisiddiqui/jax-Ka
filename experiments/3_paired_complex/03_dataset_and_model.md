# 03 — Generate the paired dataset and train the model

**Timebox:** 3 days (0.5 pilot, 1.5 generation on CPU in background, 1 training)
**Hardware:** CPU jobs capped at 400 queued/running requested cores across the user account, 2 GB/core; accelerator allocation is checked when needed.
**Depends on:** curation pipeline from 01

**2026-10-05 entry gate:** existing labels and native Monte Carlo intermediates are
exported and verified for the frozen 778 train / 151 validation complexes. See the
[native data contract](04_native_data_contract.md). Native interactions are between
tautomer states; the scalar intrinsic/pair design below is a proposal requiring a
validated reduction or a change to the output representation. Do not launch new
large-scale label generation or scalar-pair training before that gate passes.
The 20,000-complex generation target and timing estimates below remain aspirational.

---

## Part A — Pilot (500 complexes, half a day)

Run the full chain end to end before spending the big compute:

```
select → prep → two states → PypKa → dump intermediates → pair → CatBoost on intrinsics
```

Every failure mode shows up in the first few hundred: interface glycans, metals, missing
CDR side chains, altloc ambiguity, assemblies that aren't really assemblies. Each one
changes what you store, which is why the pilot comes before the schema freezes.

**Exit criteria:** ≥70% accept rate, intermediates parse, CatBoost trains, schema stable.
If accept rate is under 50%, fix curation before scaling — you'll otherwise burn
1.5 days producing a biased subset.

---

## Part B — Dump the PB intermediates

**This is the highest-leverage decision in the project.**

PypKa computes per-site intrinsic pKa's and a site–site interaction matrix, then runs
Monte Carlo over them. Those intermediates (`.pkint` / `.g`-style files feeding the MC
step) are direct supervision for exactly the two tensors the network predicts.

For a 450-residue complex with ~100 titratable sites:
- midpoints only: ~100 numbers
- intrinsics + pairs: ~100 + ~5,000 numbers

**~50× the signal for the same PB solve.** And the gradient path is one layer deep
instead of backpropagating a scalar through the whole fixed-point solve.

PypKa exposes and the existing runs retain `mc-energies.json` and per-tautomer
intrinsics. Their units, state mapping and downstream solver compatibility are
checked in the native export; a direct scalar-per-site reduction is not assumed.

---

## Part C — Full generation

### Scale and budget

| | |
|---|---|
| Complexes | ~20,000 after curation losses |
| States each | 2 (complex, rigid-separated) |
| Core-hours | ~10,000 (≈0.5 core-h per Fv-scale triple) |
| Current admission cap | At most 200 two-CPU jobs, fewer while other user jobs are queued/running |
| Ionic-strength subset | 1/3 of complexes × 3 values (0.05 / 0.15 / 0.5 M) → +~7,000 core-h |
| **Total legacy estimate** | **~17,000 numerical core-hours; not a validated wall-time estimate** |

Re-estimate after measuring the admitted population. Allocated cores and numerical
worker cores differ, and the pilot showed a substantial long-running tail. The
earlier 28-hour claim must not be used for scheduling this generation target.

### Scheduling

Default teacher jobs request **2 CPUs and 4 GB**, with one numerical worker. Count
all queued/running requests against the user-wide **400-core cap**; exclude comp1400.
Use measured size/runtime data before estimating throughput. The pilot's statewise
recovery required up to three hours per state and still retained seven failures;
the original one-hour assumption is not a demonstrated production budget.

Checkpoint per job to its own file. No shared database during generation; merge after.

### Conditions

Use an explicitly versioned current teacher. Exact historical pKPDB equivalence
is unresolved and must not be claimed. The current settings are:
internal dielectric 15, solvent 80, ionic strength 0.1 M, 81-point grid,
`pbc_dimensions=0`, GROMOS 54a7, PDB2PQR with H optimisation, no ions.

> Historical entries require per-entry settings for exact reconciliation. Note that
> εᵢₙ = 15 is high and implicitly absorbs conformational relaxation — it compresses
> shifts toward model values, so the teacher systematically under-predicts large shifts.

### Schema

Per site, per state, per condition:

```
chain, resnum, icode, restype,
intrinsic_pka, pair_row (sparse), midpoint_pka,
charge_curve[pH −2:16 step 0.25],
dsasa, interface_flag, convergence_flag, valid_flag
```

Per structure: coordinates, prep provenance, teacher version, content hash, and
**covariates stored but unused in v1** — crystallization pH from PDB metadata,
resolution, method. These let you test later whether predictions drift with the pH the
structure was solved at.

Do **not** precompute heavy features. You have the CPUs and it's tempting, but a
coordinate model wants raw coordinates plus local frames, and frozen features will
constrain architecture choices you haven't made. Featurise on the fly; precompute only
for the CatBoost baseline.

Shard by sequence cluster so splits are file-level.

Disk: pair matrices ~2–3 GB; coordinates dominate at a few hundred GB.

### Splits

Inherit the already frozen sequence-component and antigen-family assignments from
01, including separate antibody novelty annotations. Do not resplit by cluster pair
or PDB ID. Additional data must be checked against existing experimental/test family
reservations before admission.

Held out and never touched until the end:
- tier-B clusters
- PKAD-3, cluster-decontaminated against pKPDB
- experimental ΔpKa and pH-dependent affinity from 01's set 2

---

## Part D — Model

### Shape

Siamese: one encoder, two passes, subtract. Same weights. (Rationale in 02.)

Sanity invariant, as a diagnostic not a loss: ΔpKa(A→AB) must equal −ΔpKa(AB→A).
It's automatic if the architecture is right, so drift means a bug.

### Encoder

- Invariant features in local N–CA–C frames (not equivariant vector channels — simpler,
  and gradients still reach coordinates through frame construction)
- 4–6 layers, d=128, 8 heads → ~1–3M params
- Sparse radius-graph attention; **query tokens only at titratable sites** (~25% of residues)
- Guard Gram–Schmidt frame construction against degenerate triples
- Soft `P[N,20]` sequence representation end to end — no argmax, no rotamer search

### Output head — residuals on physics, bounded

```python
intrinsic = model_pka[restype] + SCALE * tanh(head_i(h))      # starts at "no shift"
W_raw     = debye_huckel_baseline(coords, I) * (1 + head_w(h_pair))
W         = 0.5 * (W_raw + W_raw.T); W = W.at[diag].set(0)
```

The DH baseline supplies the correct distance decay for free; the network learns only
the deviation. Keeps gradients small and well-scaled, and an untrained net starts in a
physically sane place.

### Solver

Keep jax-Ka's relaxation. Three changes:

1. **Mean-field + exact small clusters.** Mean-field is genuinely wrong for strongly
   coupled dyads (the classic Asp/Glu pairs). Partition sites by coupling strength,
   enumerate exactly within clusters of ≤10–12 sites via `logsumexp` over 2^k states
   (perfectly smooth), mean-field between clusters. Compute the partition from a
   **detached** coupling matrix so the assignment itself carries no gradient.

2. **Implicit differentiation of the fixed point.** `lax.custom_root` or custom VJP:
   `dx/dθ = (I − ∂F/∂x)⁻¹ ∂F/∂θ`, solved with CG. Constant memory, no pathology from
   variable iteration counts. Add small Tikhonov regularisation — the system is
   ill-conditioned near strongly coupled transitions.

3. **Midpoints by implicit root**, not grid interpolation. The current grid-midpoint
   approach has gradient kinks when the crossing moves between intervals.

### Smoothness checklist (gradients flow to coordinates)

- Cosine/sigmoid switching over a 2 Å window — no hard distance cutoffs
- Fixed-k neighbours from a **detached** index computation; smooth weights carry the gradient
- `sqrt(r² + ε)` everywhere; never `norm()` at zero
- SiLU/GELU, not ReLU, if you want second derivatives for design later
- **fp32.** FlashABB had to run fp32 because the expanded-form distance trick cancels
  catastrophically in bf16; your pair matrix is distance-dependent and inherits this.

### Losses, in priority order

1. intrinsics + pair matrix (direct, tier B)
2. charge curves through the solver — **primary solver-level loss**, since the curve is
   smooth in the parameters while the midpoint is a root and is badly conditioned where
   curves are flat (buried, weakly titrating sites)
3. midpoints
4. ΔpKa between states — **ramp this up over training**; it's the smallest-magnitude
   signal and will otherwise be ignored

Train on **shifts from model values** (3.7 Asp, 4.2 Glu, 6.5 His, 8.5 Cys, 9.5 Tyr,
10.4 Lys), not absolutes. The null model already gets RMSE ~1.36 acid / 1.04 base, so
absolute regression spends capacity on residue identity and inflates R². Normalise per
residue type.

### Schedule

- Tier A pretrain (pKPDB, output-level only — no intermediates available) → ~12 h on 3090
- Tier B fine-tune (paired, intermediate-level) → ~12 h
- Calibrate on PKAD-3
- HPO on the 4× A40s: layers, d, loss weights, cluster size cap, DH baseline screening length

### Evaluation

Everything from 01, run identically: sets 1, 2a and 2b (linkage) with the shared scoring
module, component bootstrap, same pH grid.

**Must beat the delta-learning CatBoost from 02.** If it doesn't, report that honestly;
the differentiability is still a contribution but the accuracy claim isn't.

### Mutation ranking — the design use case

Per-site RMSE does not test what the model is for. Design asks a different question: given
an interface, **which mutations shift its pH-dependence, and in which direction?** A model
can be badly calibrated per site and still rank mutants correctly, and a model with good
site-level RMSE can rank badly if its errors are correlated with the mutation. Score it
directly.

This is the one evaluation that exercises the soft-sequence path (`P[N,20]`) rather than a
fixed structure, so it is also the integration test for that path.

#### Tier 1 — pH-dependence ranking (the claim)

Mutant series where affinity was measured at ≥2 pH values for ≥3 point mutants of the same
interface. Predict `dΔG_bind/dpH` per mutant, rank within series.

- within-series Spearman, then aggregate across series (n = series)
- sign accuracy on "does this mutation increase or decrease pH-dependence"
- top-1 / top-3 enrichment: is the strongest measured switch in the predicted top 3

Honest about supply: these series are **rare**. Budget half a day of curation during 01's
set-2 pass, since it is the same literature. **Fewer than ~8 series → report as case
studies, not a benchmark**, matching 01's rule for set 2b.

#### Tier 2 — charge-reversal subset at fixed pH (the fallback)

Where tier 1 is too thin, use interface mutations of titratable residues with measured
ΔΔG_bind at a single pH (SKEMPI is the obvious source).

> **This is not a test of ΔΔG_bind.** The model predicts only the protonation-linked
> component; SKEMPI measures the total, including packing and desolvation the model has no
> representation for. Restrict to charge-reversal and charge-deletion mutations at the
> interface, where the electrostatic component is expected to dominate, expect weak
> correlation, and label the axis "protonation-linked component vs. total measured ΔΔG".
> A null result here is close to uninformative — do not let it stand in for tier 1.

#### Tier 3 — gradient sanity (cheap, do it first)

No experimental data needed. On a handful of set-2b systems, take `∂(target linkage)/∂P` and
check the top-ranked single substitutions are chemically sensible: titratable introductions
near the interface, correct acid/base direction for the requested shift. Then discretise the
top candidates, rescore them as hard sequences, and confirm the gradient's ranking survives
discretisation.

This catches the failure that matters most for design — a gradient that points somewhere the
discrete model disagrees with — and it needs nothing but the trained checkpoint.

**Caveat to state once, up front:** mutant side chains use one frozen candidate conformation
in the backbone-local frame (no repacking, no rotamer search). The prediction is therefore
for that specific placement, not a relaxed mutant. For buried positions this is a real
limitation; 04's rotamer-averaging argument applies here and is the principled answer.

---

## Outputs

```
data/tierB/            # sharded by cluster
results/model/
  checkpoints/
  eval_vs_benchmark.csv
  linkage_validation.png
  mutation_ranking.csv       # tier 1 per-series Spearman; tier 2 component correlation
  gradient_sanity.md         # tier 3: top substitutions, discretised rescore agreement
  ablations.csv
```

## Scope of the claim

pH-dependent titration and binding linkage **from a fixed conformation**. Not
pH-dependent conformational change. Histidine-switch and endosomal-release cases are
expected failure modes — say so in the paper rather than letting a reviewer find it.

For mutations, the claim is **ranking and direction** of change in pH-dependence, not
calibrated ΔΔG_bind(pH), and it is conditioned on one frozen candidate conformation per
substitution. The model is not a binding free-energy predictor and must not be presented
beside ΔΔG_bind methods as though it were one.
