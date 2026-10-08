# 01 — Benchmark: are existing pKa models tuned for ΔpKa on binding?

**Timebox:** 5 days of pipeline work — day 1 gates + smoke set; day 2 universe, split, prep;
day 3 runs; day 4 scoring + figures; day 5 slack — **plus 1 person-day of set-2 literature
curation**, which needs no compute and can be scheduled anywhere from day 1. If that day
cannot run in parallel, this is 6 days.
**Hardware:** CPU node (~700 core-hours: ~1000 complexes × 3 states through the teacher;
other methods are cheap)
**Deliverable:** `results/benchmark/` + a standalone writeup. This is publishable on its own.
**Inherits:** [`00_shared.md`](../00_shared.md) — prep, states, schema, split, teacher
config, scoring, jobs. Change those there, not here.

---

## Hypothesis

Every existing pKa predictor was fitted on single-structure absolute pKa of mostly
monomeric proteins. None has been trained or validated on a bound/free pair.

ΔpKa is a *difference of two predictions*. It survives only where a model's errors cancel
between the two states. Errors driven by features that binding leaves unchanged cancel for
free. Errors in how the model *responds to the partner* do not, and no published training
objective constrains that response.

**H0:** existing methods have no skill on ΔpKa-on-binding relative to a zero-shift null.

A model can have excellent absolute pKa accuracy and zero ΔpKa skill. That dissociation is
worth reporting if it occurs. It may not: methods with an explicit desolvation term
(PROPKA3, jax-Ka) could track PB on rigid-separation ΔpKa, which is also a reportable
result and changes what 03 claims.

Three questions, kept separate throughout:

- **Q1 (set 1):** do fast methods reproduce the PB teacher's ΔpKa? Agreement, not accuracy.
- **Q2 (set 2a):** does anything, *including the teacher*, predict experimental ΔpKa? This is
  the only real test of H0, and it is the ceiling for anything distilled from PypKa in 03.
- **Q3 (set 2b):** does the linkage integral predict measured pH-dependence of affinity?
  This is the application claim, and it is a weaker claim than Q2 — a method can get
  individual sites wrong and still get `ΔQ = Q_AB − Q_A − Q_B` roughly right, because the
  integral sums over sites and per-site errors partly cancel. Report Q3 on its own terms,
  never as a corollary of Q2.

---

## Part A — Build the shared pipeline

Specified in `00_shared.md`. Step 1 builds it, in this order:

1. **Day-1 gates G1–G5** on the smoke set (~50 FoldBench protein–protein pairs). Run the
   whole chain — prep → states → annotate → teacher + methods → score — before writing
   PDB-wide selection.
2. **Universe + split.** Metadata selection over PDB + SAbDab, MMseqs2, component split,
   antibody component-size check. Freeze `split.parquet`.
3. **Sample** set 1 (~500 test complexes) and the pilot (~500 train complexes). Prep,
   states, annotations.
4. **Run.** Teacher + PROPKA3 on both pools (02's delta-learning baseline needs PROPKA on
   the pilot). All scored methods on set 1 only.
5. **Score** with the shared module.

Expect to lose a large fraction at prep. pKPDB covered only ~67% of available protein
structures with a comparable pipeline. Antibody–antigen interfaces are worse than average
(interface glycans, disordered CDR side chains, metals), and the interface-gap rule will
bite hardest on CDR loops.

**Log every rejection with its code.** The rejection histogram is a figure in the paper and
tells you what to fix for v2.

No independently relaxed state in v1. ΔpKa is defined as the rigid-separation difference.
State this explicitly in the writeup.

---

## Part B — Methods

**Reference (set 1):** PypKa, teacher config from 00. It defines the axis in set 1, so it is
not scored there.

**Scored (core tier):**

- **PROPKA3** 3.5.1 — the empirical baseline; adapter already exists (`reference.py`)
- **jax-Ka** — current PROPKA surrogate (`gap_policy="cap"`, `freeze_disulfides=True`)
- **pKAI** and **pKAI+** — fast ML, distilled from pKPDB
- **KaML-CBtree** — released end-to-end predictor
- **Null** — ΔpKa = 0 everywhere

In set 2, **PypKa is also scored** as a method.

**Not in step 1:** DeepKa, H++, DelPhiPKa, MCCE2, KaML-GAT. Add one only if the writeup
needs it. A benchmark of 6 methods run correctly beats 11 run badly.

### Pre-check: can each method even represent the task?

Report as a table before any accuracy metric:

1. **Multi-chain handling** — does it treat an N-chain input as one system, or silently
   per-chain? Test on a smoke-set complex and diff against manually merged chains (gate G3).
2. **Coverage** — per method, the fraction of reference-valid sites it returns `ok` in both
   states. jax-Ka's strict topology is a known risk here.
3. **Structural zeros** — per distance shell (0–5 / 5–10 / 10–15 / 15–20 Å), the fraction of
   sites with `|ΔpKa_pred| < 0.01` where `|ΔpKa_ref| ≥ 0.1`. A hard neighbourhood cutoff
   returns exact zero beyond its radius. That is an *inability to represent the task*,
   categorically different from being wrong. It only shows up **outside** the interface, so
   this check must include non-interface sites.

---

## Part C — Evaluation sets

### Set 1 — PB agreement (~500 complexes, test split)

PypKa rigid separation as reference. Label the axis **"agreement with PB"**, never
"accuracy". Headline on interface residues; all sites in the 20 Å scoring shell also scored,
by distance shell.

Contamination: pKAI and pKAI+ were distilled from pKPDB, which is PypKa output. Against a
PypKa reference they measure self-consistency. Flag prominently; do not present as a win.

### Set 2 — Experimental (the only ground truth; budget 1 person-day)

Everything else in this benchmark is teacher agreement. Curation is literature work with no
compute dependency, so it can start on day 1 and run beside the pipeline; only the prep and
scoring of its systems wait on Part A. Their split components are forced to test.

Two sub-sets, scored separately because they test different things.

#### Set 2a — site-level ΔpKa on binding (~20–60 sites)

Hand-curated literature ΔpKa-on-binding. Candidate systems: barnase–barstar,
protease–inhibitor complexes, antibody–antigen pH switches. Tests the per-site quantity
every method in Part B emits directly.

#### Set 2b — linkage: ΔΔG_bind(pH) (target ~10–25 systems)

This is the set that decides whether any of this predicts **pH-dependent binding**, which is
the application claim. Inclusion needs, per system:

- a structure of the complex that survives prep
- affinity at **≥3 pH values** (SPR K_D or k_off series, ITC, pH-dependent competition), or
- a **direct proton-uptake measurement** — ITC in buffers of differing ionization enthalpy
  gives Δn(H⁺) on binding at one pH

Record buffer, ionic strength, temperature and construct per entry. Mismatched ionic
strength against the teacher's 0.1 M is a caveat, not a rejection; flag it.

**Prefer direct Δn(H⁺) entries.** They compare against `ΔQ = Q_AB − Q_A − Q_B` with no
integration, no reference-pH choice and no accumulated curve error. That is the cleanest
single test of the whole linkage path.

Metrics, all per system (n = systems, not sites):

| Metric | Why |
|---|---|
| Sign of ΔQ at pH 7 | Does binding take up or release protons. The weakest claim, and the one most likely to hold. |
| Δn(H⁺) vs. measured | Direct, where ITC linkage data exists |
| Slope dΔG/dpH over the measured range | Magnitude of the pH-dependence |
| Spearman of ΔG(pH) across measured pH points | Shape, within a system |

Bootstrap over systems. At n ≈ 15 the CIs will be wide; report them and do not round a wide
interval into a conclusion.

**Expected failure mode, stated in advance.** ΔpKa here is the rigid-separation difference,
so the predicted linkage is the protonation-linked component only. Where pH-dependence comes
partly from conformational change — endosomal-release antibodies, histidine switches — expect
correct sign and ordering with **underestimated magnitude**. Report observed-vs-predicted
slope, not just correlation, so the compression is visible.

**If 2b comes in under ~8 systems**, report it as case studies and drop the word "benchmark"
for this set. Do not back a claim about predicting pH-dependent binding with 4 systems.

### Set 3 — Noise floor → step 1b

Moved to [`1b_noise_floor/01b_noise_floor.md`](../1b_noise_floor/01b_noise_floor.md). It does
not depend on this pipeline beyond the cluster definitions.

### Set 4 — Competence baseline (stretch)

Absolute pKa on PKAD-3. Decontaminate by **sequence cluster**, not PDB ID — KaML trained on
PKAD-3, pKAI on pKPDB. Separates "bad at pKa" from "bad at ΔpKa specifically". Run only if
days 1–4 land on time.

---

## Part D — Metrics

Definitions live in 00 (one scoring module, reused by 02–04).

Primary: **skill score vs. zero-shift null** (headline), **Spearman ρ**, **sign accuracy**
on `|ΔpKa_ref| ≥ 0.5`.

Stratified by: |ΔpKa_ref|, residue type, ΔSASA, distance shell, antibody vs. other.

**Error-cancellation diagnostic (the mechanistic figure):**
scatter each method's error in the complex state against its error in the free state.
High correlation → errors cancel → good ΔpKa despite poor absolutes. Low correlation →
two independent errors compound. This explains *why* each method succeeds or fails and is
probably the most interesting plot in the paper.

**Linkage readout:** from curves (native for the teacher, HH-reconstructed for
midpoint-only methods, labelled as such):

```
ΔG_bind(pH) − ΔG_bind(7) = +RT·ln10 · ∫_7^pH [Q_AB − Q_A − Q_B] dpH′
```

Compare to experimental pH-dependent affinity in set 2. Lives in the shared module; it is
the quantity that justifies the whole project and 03 reuses it.

### Pre-registered calls — freeze before the first scoring run

| Call | Rule (proposed; edit, then freeze) |
|---|---|
| No skill | skill 95% CI upper bound < 0.1 |
| Tracks PB | skill 95% CI lower bound ≥ 0.5 |
| Teacher has experimental site skill | set 2a skill 95% CI lower bound > 0 |
| Linkage sign is predictable | set 2b ΔQ sign accuracy 95% CI lower bound > 0.5 |
| Linkage magnitude is predictable | set 2b observed-vs-predicted slope CI excludes 0 **and** the regression slope is within [0.5, 2.0] |

Anything between "no skill" and "tracks PB" is reported as partial skill with its CI, not
rounded to either side.

---

## Outputs

```
results/benchmark/
  rejections.parquet
  curation_report.json      # accept/reject counts by code
  split.parquet
  structures.parquet
  sites.parquet
  predictions.parquet
  pairs.parquet             # if G1b passes
  representability.csv      # multi-chain, coverage, structural zeros by shell
  scores_set1.csv
  scores_set2a.csv
  set2b_systems.csv         # one row per system: conditions, source, caveat flags
  scores_set2b_linkage.csv  # sign, Delta n(H+), slope, within-system Spearman
  scores_set4.csv           # if run
  figures/
```

## Success criteria

- Gates G1–G5 resolved on day 1 (pass or explicit fallback)
- Teacher valid on ≥80% of sampled set 1 complexes
- Skill, Spearman and sign accuracy with component-bootstrap CIs for every scored method on
  sets 1 and 2a
- Set 2b: ≥8 systems, of which ≥3 carry a direct Δn(H⁺) measurement; linkage computed for
  every method that emits curves

## Kill criteria

- **G1 fails** (PypKa won't install or run) → stop and re-plan. There is no fallback
  teacher; set 1, 02 and 03 all depend on it.
- Any other core method that cannot install or ingest multi-chain input by end of day 1 →
  drop it, record why, proceed. Do not debug installs into day 2.
- Smoke-set accept rate < 50% → fix prep before sampling the universe.

## Reading the outcome

| Result | Implication |
|---|---|
| All methods ≈ null on set 1 | Nobody reproduces PB ΔpKa. Project justified; benchmark is the paper. |
| Fast methods track PB on set 1 | Matching the teacher isn't the gap. 03's contribution becomes differentiability, speed and curves; rewrite its claim before generating 20k complexes. |
| PypKa has no skill on set 2a | Distilling PypKa can't be justified on accuracy. Revisit the teacher (εᵢₙ, relaxed states) before 03's full run. |
| PypKa skilled on set 2a, fast methods not | The cleanest case for 03. |
| Set 2b sign right, magnitude compressed | Expected. 03's claim is ordering and direction of pH-dependence, not calibrated ΔΔG(pH). Write it that way from the start. |
| Set 2b sign no better than chance | The linkage claim does not survive at fixed conformation. Either the teacher or the rigid-separation definition is the problem — both are upstream of the model, so fix before 03's full run, not after. |
| Set 2b works but only on non-switch systems | Say so explicitly and define the applicability domain by mechanism, not by accuracy threshold. |
| Good absolutes (set 4), no ΔpKa skill | Failure is two-state sensitivity → paired training is the fix, not better features. |
| PROPKA3 holds up | Its explicit desolvation/burial term transfers → borrow that functional form for the intrinsic head in 03. |
