# 00 — Shared contract

Decisions every stage inherits. **Change them here, not in a stage doc.** A stage doc that
needs a different rule adds a row to the decision log at the bottom and says why.

Step 1 builds everything in this file. Steps 02–04 consume it.

**Latest component policy (2026-10-04):**
[Stripped protein-only reference](1_benchmark/07_stripped_reference_policy.md)
uses **15 Å training / 20 Å evaluation for eligible buffer/additive candidates**,
following the [corrected buffer pilot](1_benchmark/14_buffer_sensitivity.md) and
explicit user approval. Resolved neutral glycans with checked attachments retain
**20/25 Å**; other eligible noncovalent ligands retain **15/25 Å**. Exposed,
uncoordinated monatomic metals/ions retain **25/25 Å**; buried/coordinated metals
remain excluded. Buffer exposure and bridging remain annotations, not new entry
rejection gates. Existing chemical eligibility is preserved.

The buffer experiment supports neutral alcohol/ether additives; application to
other eligible buffer species remains an operational extrapolation. Individual
buffer deletions can produce outliers beyond 15 Å. These masks are uncertainty
flags, not proven physical error bounds. Source components and removal provenance
are retained. Full ligand-aware modeling remains in section 05.

The `buffer-15-20-v3-pool1` and `buffer-15-20-v3-pool5000` campaigns supply the
[verified structural freeze v1](1_benchmark/16_structural_split_freeze.md):
**778 train / 151 validation / 523 test** usable interface pairs. The protein G–Fc
reference groups are reserved in test; the entire 41-candidate training group
moved together, costing one usable training interface pair. Structural inputs,
current masks, assignments and refreshed antibody novelty are frozen at
`universe/structural-freeze-v1`. Consumers must use its split-aware
`training_eligible` / `evaluation_eligible` site masks and verify input hashes.

The [preceding buffer recount](1_benchmark/15_buffer_mask_revision.md) is preserved
as historical evidence. [The frozen-input gate/scorer and 50-pair smoke](1_benchmark/17_frozen_smoke_and_scoring.md)
are complete and verified. The [completed numerical follow-up report](1_benchmark/21_smoke_validation_report.md)
records 50/50 completed teacher calculations, 47/50 usable teacher interface pairs,
and convergence in all 150 JAX states at 1024 iterations. Remaining invalid sites
stay flagged. A separately versioned production wrapper passed its equivalence
gate; [production labeling is now running](1_benchmark/22_production_launch.md)
on all 1452 usable complexes, prioritizing a fixed 500-complex training pilot.
Full production predictions and experimental Set 2 curation remain unfinished. New experimental
references require an overlap audit before independent claims. Production remains
subject to explicit coverage/failure review before training; the current-preparation
gates passed without claiming historical pKPDB equivalence.

The first dataset-generation round covers protein–protein systems through the
backbone-only stage. [Section 05](5_ligands/05_ligand_complexes.md) defers small-molecule,
drug and ligand complexes, including bound/unbound pairs, to a later extension.
Retain their source inventory for that extension; initial exclusion is task scope,
not a claim that those systems are unusable for training.

---

## Glossary

| Term | Meaning |
|---|---|
| Partner | One side of a binary interface. May be several chains (antibody H+L is one partner). |
| States | `AB` complex; `A`, `B` each partner alone, **bound conformation retained**. No relaxation. |
| Site | One titratable group: `(complex_id, chain, resnum, icode, group)` |
| Group | jax-Ka channel name: `ASP GLU HIS CYS TYR LYS ARG NTERM CTERM` |
| ΔpKa | `pKa(AB) − pKa(free state containing the site)`. Positive = binding raises the pKa. |
| Teacher | PypKa with the locked config below |
| Tier A | pKPDB: single-structure PypKa midpoints. Output-level labels only. |
| Tier B | This project's paired teacher runs: midpoints + curves + intermediates (if exposed) |
| Interface residue | Residue heavy-atom ΔSASA > 10 Å² |
| Interface zone | Any heavy atom within 10 Å of a partner heavy atom |
| Scoring shell | Sites within 20 Å of the partner. Interface residues are the headline subset. |

---

## Selection

Two passes. Metadata first (cheap, whole universe), coordinates second (only what gets sampled).

**Metadata:** PDB biological assembly 1 + SAbDab. X-ray or cryo-EM. Total ≤1500 residues
(provisional — reset from smoke-set timings, see Jobs).

**Partner rule (revised pilot):** exactly two explicitly defined partners.
- two-chain assemblies → one chain each; homomers kept and flagged `homomeric`
- SAbDab → H+L vs a single antigen chain
- manageable larger assemblies may supply declared partner sets; preserve omitted
  protein context and identify labels as the selected prepared pair's response.
  Do not infer an assignment of extra chains. The full observed assembly retains
  the 1,500-residue cap. The historical `prepare` command retains its strict
  binary baseline; use `curation-pilot` for this revision.

**Coordinates:** half-sum buried area (per side) ≥ 500 Å²; ≥10 interface residues on the complex.

---

## Prep policy

Every method reads the **same heavy-atom file** per state. Hydrogen placement and protonation
are method-internal. Provenance dict stored with each structure.

| Decision | Policy |
|---|---|
| Altlocs | First deposited positive-occupancy alt ID per residue, shared atoms retained. Same selected geometry in AB/A/B. Alternatives are provenance only: no extra masks or conformer teacher campaign. Zero-occupancy atoms are not observed inputs |
| Missing backbone atoms in a residue | Attempt PDB2PQR repair before topology checks; retain successful, identity-preserving preparations. Do not manually remove the residue to force success |
| Chain gaps | Retain when PDB2PQR preparation succeeds, including interface gaps. Record physical segments and artificial termini; do not score artificial termini as native targets |
| Missing side-chain atoms | Attempt PDB2PQR repair regardless of interface distance; retain successful preparations with `was_completed` provenance. Missing-coordinate proximity is metadata, not a rejection radius |
| Hydrogens | Strip |
| Waters | Remove |
| Ions | Remove (PypKa default `keep_ions=False`) |
| Ligands / glycans / metals | Reject `ligand` / `glycan` / `metal` |
| Nonstandard residues | Reject `nonstandard_residue` |
| Disulfides within a partner | Keep; jax-Ka `freeze_disulfides=True` |
| Disulfides across partners | Reject `interpartner_disulfide` |
| Multiple models | Revised pilot refuses silent model-1 selection; explicit model-specific paired examples required. Initial metadata scope excludes NMR |

Why distal defects are candidates for tolerance: their representation is identical
in both states, so some error may cancel in ΔpKa. This is a hypothesis to validate,
not a guarantee: electrostatic coupling can propagate beyond a local defect.
The current inclusive experiment measures this error instead of pre-rejecting
interface-zone defects.

jax-Ka then runs with `missing_sidechain="error"` as a check that completion worked.

### Revised supervision policy for incomplete structures

The user selected retaining usable examples with distal missing coordinates and
excluding unreliable **site targets**, rather than copying neighbouring shifts or
requiring a label for every residue. The latest decision supersedes radius-based
admission: **include structures PDB2PQR can prepare within the protein–protein
scope, then measure error versus distance using controlled whole-residue deletions.**
No 5/8/10/15/20 Å missing-coordinate exclusion is active in the inclusive policy.
Repaired observed sites can receive conditional teacher targets; absent residues
cannot. Artificial termini are not native supervision targets. Preparation and
teacher failures remain explicit coverage outcomes.

Start from complete/near-complete references. Delete 1 or 3 residues at either
terminus and a buried internal residue or three-residue segment. Determine burial
and deleted-atom positions from the intact reference, not from the damaged model.
Compare AB pKa, free-state pKa, and paired ΔpKa errors in 0–5, 5–10, 10–15,
15–20, 20–30 and >=30 Å bands, plus cumulative outside-radius summaries. Distance
is the minimum from a retained target's functional atoms to any deleted reference
heavy atom. Include repeated intact calculations to check reproducibility; fixed-
seed repeats do not estimate independent-seed Monte Carlo variance.
Select any future masking radius from these measurements, separately for exposed
terminal and buried deletions; the initial small pilot cannot establish a
population-wide safe radius.

The following describe retained **historical diagnostic mask comparisons**, not
current admission/training exclusions:

- Separate `input_atom_mask` (coordinates available to the student) from
  `supervision_mask` (targets included in loss/evaluation). Preserve full sequence
  identity and distinguish observed, reconstructed and absent coordinates.
- For real unresolved residues/atoms, do not invent pKa or ΔpKa labels. In
  particular, neither sequence neighbours nor spatial neighbours provide valid
  replacement labels. Mask affected sites and nearby observed targets.
- Compare 10, 15 and 20 Å exclusion radii before freezing a default. Measure from
  target functional-group atoms to the **uncertain region**, and use the same
  union mask for AB and its matching free-state site. Include artificial break/
  truncation termini in that uncertain region; never score them as native termini.
- Compare smaller **5/8/10 Å** radii for exposed-tail candidates against the
  corresponding **10/15/20 Å** general-defect radii. The current candidate screen
  is deliberately narrow: terminal gap of at most five residues, three complete
  observed attachment residues each with exposure fraction >=0.4 in both AB and
  the matching free partner, all >20 Å from the other partner, no alternate
  uncertainty in those flanks, and no missing ionisable residue. These numerical
  thresholds are pilot choices, not established physical cutoffs. Exposure is
  residue SASA divided by SASA of that same isolated residue; <=0.1 is recorded
  as a buried-anchor proxy, with intermediate/unknown cases kept conservative.
  Missing-residue burial is not observed. Absent neighbours may inflate flank
  exposure, and the current exposure measurement is for the selected partners.
  Longer or charged tails need explicit sensitivity evidence rather than being
  assumed safe or permanently excluded from future work.
- A missing side chain occupies a region beyond its remaining atoms. Expand its
  uncertain region using plausible side-chain extent or reconstruction variability.
  For a whole missing segment, flanking coordinates alone do not locate it. Use
  an explicitly modelled envelope, or keep its localisation unresolved rather
  than asserting that it is distal or buried. Long unconstrained segments may
  leave no defensible supervised sites in that example.
- Keep the possible-location envelope when testing the smaller exposed-tail
  radius: 3.8 Å per missing backbone step plus a provisional 8 Å heavy-atom
  extent. This avoids disguising a localisation assumption as a radius change.
- Alternate conformations contribute the union of their deposited heavy-atom
  positions to uncertainty masks; incomplete alternatives receive an additional
  8 Å extent. Targets on ambiguous residues are masked pending conformer-specific
  teacher validation. Full alternatives are retained in `conformers.json` so this
  is not irreversible data removal. Local alt IDs do not imply a global A/B
  ensemble or equilibrium weights. Test local alternatives with matched AB/A/B
  inputs and retain conformer-specific responses; never occupancy-average pKas.
  Distinct deposited structures remain separate examples in the same sequence
  split component, not independently assigned train/test conformations.
- Backbone-only reconstruction may supply approximate input geometry and help
  localise this envelope. It is not a restored atomistic electrostatic model and
  does not make a missing side-chain target valid. The PB teacher still needs a
  chemically parameterisable input: validated completion or an explicitly defined
  capped/truncated representation, identical across paired states. Record that
  representation and label provenance; teacher failures remain coverage failures.
- Before adopting a radius, use a small repair/truncation sensitivity pilot to
  measure how much paired shifts change outside it. Report retained sites and
  complexes by radius, residue type, size and interface distance. A distance mask
  is a locality approximation, not proof of unchanged reference values.
- Synthetic coordinate dropout is different: when a reliable complete-structure
  teacher target exists, hide student coordinates while retaining that target.
  Real unresolved structure has no such known complete-structure label.

These partial labels can support site-level ΔpKa training. They cannot by themselves
support whole-system charge/linkage evaluation, which requires complete accounting.
Being buried is not by itself evidence that a defect is harmless or distal to the
binding interface.

The assembly-chain restriction is independent of this policy. The audit found
123/139 excluded multi-chain assemblies within the existing 1,500-residue cap;
replacing a two-chain restriction requires defining two interacting partners and
their retained/omitted context, not treating extra chains as missing-coordinate
fragments.

### Rejection codes

Machine-readable, one row per rejected candidate: `(candidate_id, stage, code, detail)`.

```
selection:  multi_partner, size_cap, buried_area, interface_residues
prep:       missing_backbone, interface_gap, interface_missing_sidechain,
            missing_titratable_sidechain,
            ligand, glycan, metal, nonstandard_residue, interpartner_disulfide,
            ambiguous_disulfide, ambiguous_residue_key, cyclic_peptide, covalent_crosslink
teacher:    teacher_timeout, teacher_failed
```

`load_topology` currently raises free-text `ValueError`s; map each to a code (split the
single noncanonical error into ligand / glycan / metal / nonstandard by CCD type).

A method failing on an accepted structure is **not** a rejection. It is recorded as a
method status and shows up as coverage.

---

## States

Each state is its own file: `AB.cif`, `A.cif`, `B.cif`. Partners run in their own PB box,
never translated inside the complex box. The sites of `A` are the A-chain sites of `AB`.

---

## Teacher config (locked)

Current generated labels use the explicit configuration below. Historical pKPDB labels retain their original provenance; equivalence to this current teacher is not assumed:

- internal dielectric 15, solvent 80, ionic strength 0.1 M
- 81-point grid, `pbc_dimensions=0`
- GROMOS 54a7, PDB2PQR with H optimisation, no ions
- temperature 298.15 K
- explicitly disable SER/THR titration (`ser_thr_titration=False`) for the
  nine-group benchmark site schema. Shared server metadata also says false,
  but does not establish historical per-entry settings; see the
  [historical evidence amendment](1_benchmark/18_historical_pkpdb_settings.md).

> G2_current verifies our frozen preparation, inputs, configuration and runtime evidence. G2_historical remains unresolved and applies only to claims of reproducing deposited pKPDB labels. See [the gate revision](1_benchmark/17_frozen_smoke_and_scoring.md).

**pH grid for every curve, every method:** −2 to 16, step 0.25 (73 points). Same grid as
jax-Ka's `pka_from_grid` default usage.

Store curves and, if PypKa exposes them, intrinsic pKa + site–site pair terms on **every**
teacher run from day one. Re-running the teacher later because the schema grew is the
double-back this file exists to prevent.

---

## Split

**2026-10-04 revision approved for feasibility:** separate antigen-family split
assignment from antibody CDR novelty reporting. Target 500 test and 150 validation
pairs with usable interface sites, retaining the rest for training; keep sequence
groups intact even when they exceed the original 5% size target. The former
80/10/10-by-candidate-count rule below is historical. The historical proposal is in [08_usable_split_proposal.md](1_benchmark/08_usable_split_proposal.md).
[Structural freeze v1](1_benchmark/16_structural_split_freeze.md) is verified; method coverage and production readiness remain separate.

**Representation decision (2026-10-04):** retain the current usable-size proposal
without further balancing or data removal. Primary aggregation computes each
complex's score, averages complexes within each sequence component, then weights
components equally. Report pooled scores and antibody/general breakdowns as
secondary results; bootstrap sequence components, not sites or complexes
independently. Missing/undefined scores and method coverage must be reported,
not converted to zero. The new frozen-input scorer implements this aggregation and reports engineering-smoke results separately from production estimates; the legacy scorer remains historical.

**Approved experimental exception (2026-10-04):** retain the antigen-held-out
assignments and exclude the overlapping 1AXT H/P01865 antibody reference family
from independent PKAD evaluation claims. The runtime
`usable-proposal-v2/independent-experimental-scope-v2.json` records reference-level
eligibility; experimental scoring must use it and review unknown references
before inclusion. Excluded references may be shown only as non-independent
diagnostics. This exception supersedes the blanket PKAD reservation below for
this family; all other experimental reservations remain required.

The official downloaded PKAD-3 release has now been reconciled and its additional
reference-chain matches are all in test. See [10_experimental_inventory.md](1_benchmark/10_experimental_inventory.md)
for release hashes, parent-sequence proxy coverage and model-peptide exclusions.
Experimental set-2 literature curation and reservations remain separate work.

Frozen once in step 1, on the whole candidate universe after metadata selection, before
any prep or teacher run. Later stages never re-split.

1. MMseqs2 on every chain: `--min-seq-id 0.3 -c 0.8`
2. Graph: nodes are clusters, each complex is an edge between its two partners' clusters
3. Split unit = connected component. Assign components to train / val / test at 80 / 10 / 10
   by complex count
4. Forced to test: components touching any set-2 experimental system or PKAD-3 protein

**Antibody problem (check before freezing).** At 30% identity, VH and VL frameworks collapse
into a few clusters. That puts every antibody complex into one giant component, so all
antibodies would land in one split. Default fix: the antibody partner's node is the cluster
of its concatenated CDR sequence, not of its chains. Check on the real universe:
**no component may hold > 5% of complexes.** If one does, adjust and log it below.

Outputs: `split.parquet` (`complex_id, component_id, split`) plus the MMseqs2 inputs and
version.

### Pools

One generation campaign feeds every stage:

| Pool | Split | Size | Used by |
|---|---|---|---|
| Smoke set | — | ~50 FoldBench protein–protein pairs | day 1 of step 1 only |
| Set 1 | test | ~500 complexes | 01 scoring; final eval of 02/03/04 |
| Set 2a / 2b | test (forced) | ~20–60 sites / ~10–25 systems | 01's only experimental ground truth; 03 linkage + mutation ranking |
| Pilot | train | ~500 complexes | 02 training; satisfies 03 Part A |
| Full | train/val | ~20k | 03 Part C |

Set 2 systems and any mutant series built on them share one curation pass and one forced-test
component list. Curating them twice is how the same literature gets read twice.

---

## Tables

Long format, Parquet. Every method writes the same `predictions` rows.

```
structures   complex_id, pdb_id, assembly, partner_A_chains, partner_B_chains,
             n_residues, homomeric, antibody, resolution, exp_method,
             crystallization_ph, provenance (json), content_sha256, split, component_id

sites        complex_id, chain, resnum, icode, group, restype, partner,
             residue_delta_sasa, functional_delta_sasa, functional_atoms_complete,
             min_partner_distance, in_interface_zone, is_break_terminus, was_completed

predictions  complex_id, state, chain, resnum, icode, group,
             method, method_version, config_sha256,
             pka, status, curve (float32[73] | null), curve_source, intrinsic_pka (null ok)

pairs        complex_id, state, site_i, site_j, w        # teacher only, sparse

rejections   candidate_id, stage, code, detail
```

`status`: `ok | out_of_range | not_titrating | not_reported | failed`.
`curve` is the protonated fraction. `curve_source`: `native | hh` (Henderson–Hasselbalch
from the midpoint, for methods that report only pKa values).

Raw ΔSASA and distances are stored. Thresholds are applied at analysis time, never baked in.

---

## Scoring (one module, used by 01–04)

- **Common site set.** Primary metrics use sites where the reference and every scored
  method return `ok` in both states. Per-method coverage is reported separately.
- **Skill** = `1 − MSE_model / MSE_null`, with `MSE_null = mean(ΔpKa_ref²)`
- **Spearman ρ** on ΔpKa
- **Sign accuracy**, only on sites with `|ΔpKa_ref| ≥ 0.5`
- **Bootstrap**: resample split components, 1000 reps, percentile 95% CI. Never resample
  sites; sites in one complex are correlated.
- **Error cancellation**: `corr(e_AB, e_free)`, with `e = pred − ref` per state
- **Structural zeros**: fraction with `|ΔpKa_pred| < 0.01` where `|ΔpKa_ref| ≥ 0.1`, per
  distance shell 0–5 / 5–10 / 10–15 / 15–20 Å

**Linkage** (Wyman):

```
ΔG_bind(pH) − ΔG_bind(pH₀) = +RT·ln10 · ∫_{pH₀}^{pH} ΔQ(pH′) dpH′
ΔQ = Q_AB − Q_A − Q_B          # total charge; equals protons taken up on binding
pH₀ = 7.0,  T = 298.15 K,  RT·ln10 = 1.364 kcal/mol
```

Trapezoid rule on the shared grid. Sign check: if binding takes up protons (ΔQ > 0), binding
must weaken as pH rises. Unit-test that before using the function.

`ΔQ(pH)` is also returned unintegrated. It *is* the proton uptake on binding, so it compares
directly against ITC buffer-mismatch Δn(H⁺) with no integration and no reference-pH choice.
Prefer that comparison where the data exists (01 set 2b).

## Ranking metrics (shared)

Used by 01 set 2b and 03's mutation ranking, so they live in `score.py` too:

- within-group Spearman, then aggregate across groups; **n = groups, not members**
- sign accuracy on a signed quantity, with the near-zero band excluded and the threshold
  recorded
- top-k enrichment
- observed-vs-predicted regression slope, reported alongside correlation — correlation alone
  hides systematic compression, which is the expected failure mode for anything computed at
  fixed conformation

---

## Jobs

Pattern already in `benchmarks/regress_interfaces.py` + `merge_foldbench.py`:

- one job = one complex × one method, all three states inside
- 1 core per job, many concurrent
- per-job timeout set from smoke-set timings so ≤2% time out; timeouts are logged as
  `teacher_timeout`, and accepted-vs-candidate size distributions are reported so the size
  bias is visible
- each job writes `<out>/<method>/<complex_id>.parquet` + a JSON sidecar (method version,
  config sha256, input sha256, wall time, status)
- merge validates hashes; no shared database during generation

External methods run in their own environments as subprocesses. Adapter contract:
`run(state_files: dict[str, Path], workdir: Path) -> predictions rows`.

---

## Code

New package `src/pkabench/` (provisional name), kept separate so the `jaxpropka` library
doesn't pick up PypKa / MMseqs2 / CatBoost dependencies.

| Module | Starts from |
|---|---|
| `prep.py` | `jaxpropka.topology.load_topology` (add rejection codes) |
| `annotate.py` | `benchmarks/interface_exposure.py` (generalise two chains → two partners) |
| `split.py` | new |
| `adapters/propka.py` | `jaxpropka/reference.py` |
| `adapters/{pypka,pkai,kaml,jaxka,null}.py` | new |
| `jobs.py` | `benchmarks/regress_interfaces.py`, `merge_foldbench.py` |
| `score.py`, `linkage.py` | new |

Smoke set: FoldBench protein–protein manifest (upstream URL in `regress_interfaces.py`)
over `ground_truth_1522.tar`.

---

## Day-1 gates

| Gate | Check | If it fails |
|---|---|---|
| G1 | PypKa installs (DelPhi licence), runs one smoke pair in all three states, returns curves | Stop and re-plan. There is no fallback teacher. |
| G1b | PypKa exposes intrinsic pKa + pair terms | Not fatal. Decide whether 03 calls DelPhi directly. Tier B schema keeps the columns nullable either way. |
| G2_current | Verified frozen preparation/masks, explicit current teacher config, hashed runtime and successful native teacher states | Fix reproducibility evidence before current-label runs |
| G2_historical | Exact historical pKPDB configuration/preparation equivalence | Do not claim historical reproduction; does not block independently generated current labels |
| G3 | Each core method treats a multi-chain file as one system | Drop that method from the core tier |
| G4 | pKAI weights + training code usable for fine-tuning | 02 picks another fine-tune target |
| G5 | PDB2PQR completion keeps residue identity and heavy-atom naming | Fall back to rejecting any titratable residue with missing atoms |

---

## Decision log

Latest missing-residue policy: [10 Å training / 20 Å evaluation and coverage audit](1_benchmark/06_missing_residue_policy.md).
This supersedes the historical no-radius and adaptive-radius decisions below for
downstream selection; frozen teacher campaigns remain unchanged.

| Date | Decision | Why |
|---|---|---|
| 2026-10-03 | Reject gaps / missing side chains in the interface zone; tolerate distal ones | Distal defects are identical in both states and largely cancel in ΔpKa |
| 2026-10-03 | Noise floor (old set 3) moved to step 1b | Independent of the shared pipeline; not on the critical path |
| 2026-10-03 | Set 1 drawn from the frozen test split; one teacher campaign feeds 01/02/03 | Avoids re-running PypKa and makes 01 numbers comparable with the final model |
| 2026-10-03 | Buried-area cutoff read as half-sum (per side) ≥ 500 Å² | The original "BSA ≥ 500 Å²" was ambiguous between total and per-side |
| 2026-10-03 | Set 2 split into 2a (site ΔpKa) and 2b (linkage ΔΔG(pH)); 1 person-day, curated once for both 01 and 03 | Linkage is the application claim and is a weaker, separately-testable claim than per-site ΔpKa |
| 2026-10-03 | `ΔQ` exposed unintegrated for direct Δn(H⁺) comparison | Avoids integration error and reference-pH choice; cleanest test of the linkage path |
| 2026-10-03 | Mutation ranking added to 03 as a first-class evaluation, with the gradient-sanity tier requiring no experimental data | Per-site RMSE does not test the design use case; the soft-sequence path is otherwise unexercised |
| 2026-10-03 | Prep removes standalone Na/K/Cl only; other metal-containing components are rejected | User-selected resolution of the ions/metals ambiguity |
| 2026-10-03 | G5 fallback rejects incomplete titratable residues; distal incomplete nontitratable residues remain eligible and strict-method failures count against coverage | User explicitly chose the titratable-only fallback; interface defects remain excluded |
| 2026-10-03 | All workloads, installs, tests, prep and analysis run in Slurm allocations excluding comp1400; pending plus running user CPU requests capped at 400; 2 GB per requested CPU | User's resource constraint. Eligible partitions are generalaccess and amd96; oc contains only comp1400 |
| 2026-10-03 | Runtime work requests 2 CPUs/4 GB while predictors use one thread; installations request 4 CPUs/8 GB | Initial prep exceeded 2 GB and some nodes allocate CPUs in pairs; reserve and count both CPUs |
| 2026-10-03 | Parquet curves use nullable variable-length float32 lists, validated to exactly 73 elements | Arrow 20 fixed-size-list nulls failed Parquet round-trip for failed predictions |
| 2026-10-03 | Retain native teacher SER/THR outputs in raw artifacts and flag linkage incomplete until the teacher configuration/schema is reconciled | PypKa 2.10 emits additional titratable groups by default; silently omitting their charge is invalid |
| 2026-10-03 | Set current teacher to explicit `ser_thr_titration=False`; preserve previous smoke artifacts | Shared server constants and responses say false, but are not independent per-entry evidence. Historical SER/THR remains unresolved; the 2026-10-04 evidence amendment qualifies the original rationale. Current benchmark configuration remains explicit and unchanged. |
| 2026-10-03 | Enforce the existing binary-assembly rule in FoldBench prep | Audit found two of seven original accepted pairs came from larger assemblies. Same 50 candidates yield five conforming accepted pairs; no chemical exclusions were relaxed |
| 2026-10-03 | Keep ARG in the shared schema and record missing teacher titration as coverage; do not fabricate midpoints/curves | PypKa 2.10 `TITRABLETAUTOMERS` excludes ARG. Corrected smoke has two ARG sites without curves; full linkage stays incomplete pending explicit fixed-charge accounting |
| 2026-10-03 | Audit all 279 local general protein–protein FoldBench pairs and cleanup/partner-scope scenarios without changing production policy | User requested measuring combined curation and sequence-similarity effects. Strict prep retains 28 pairs/27 components; broader additive cleanup 35/34 at 30% identity. The 30% rule groups split units rather than deleting examples |
| 2026-10-03 | Defer small-molecule/drug/ligand training and bound/unbound pairs to section 05, after protein–protein backbone-only work | User explicitly requested this later extension; no ligand label generation is included in the first round |
| 2026-10-03 | Adopt site-level supervision masking as the next prep/training revision for real distal missing coordinates; no neighbour-label imputation | User proposed retaining examples and excluding labels within a radius of defects. Candidate radii 10/15/20 Å require sensitivity validation; backbone-only reconstruction supplies geometry, not complete PB chemistry. Existing executable audits remain strict baselines |
| 2026-10-03 | Implement provisional exposure-dependent radii and explicit alternate-conformer accounting | User requested smaller masks for exposed tails than buried defects and proper alternate-conformation coverage. Preserve uniform-mask comparisons, coherent paired conformers, full alternative coordinates/occupancies, and separate locality/teacher sensitivity evidence |
| 2026-10-03 | Use the first deposited alternate conformation consistently; alternatives are provenance only | User simplified the conformer policy. No additional exclusion or teacher ensemble for altlocs |
| 2026-10-03 | Admit missing-coordinate structures on successful PDB2PQR preparation and measure radial error before choosing masks | User requested inclusive admission and controlled whole-residue deletions at termini and buried positions, starting from complete/near-complete references. Supersedes interface-defect rejection and provisional radius exclusions; retains scope, geometry and identity checks |
| 2026-10-03 | Audit a bounded 1,000-assembly PDB/SAbDab sample and expand controlled deletions across size, interface type and charge | 296 pairs survive protein-only inclusive preparation. Reference search yields 23 distinct complete/near-complete entries against a target of 30; shortfall and source defects remain explicit. Perturbations bypass only interface admission thresholds to avoid censoring area-reducing deletions; intact references retain all gates. No distance mask or production split is chosen |
| 2026-10-05 | Experiment 02 fits the fixed pilot's training-interface sites, selects neural checkpoints on frozen validation groups, and leaves test untouched | Verified handoff and frozen pKAI gate passed. Three seeds per trained arm; diagnostic CPU jobs reserve 8 cores/16 GB and use two numerical threads. Existing prediction guards and frozen benchmark artifacts remain unchanged. See 2_finetune/04_diagnostic_protocol.md. |
| 2026-10-06 | Pretraining/augmentation datasets (pKPDB, extracted PINDER pairs) use component policy `mask-all-v1` (`src/pkabench/component_mask_policy.py`): component chemistry never rejects; strip and mask with train/eval radii ligand (incl. covalent) 15/25, buffer 15/25, glycan 20/25, exposed ion 25/25, bound/buried metal or metal complex 30/30 Å. Noncanonical residues and missing coordinates keep their gap/defect fallbacks. The frozen antibody benchmark keeps `buffer-15-20-v3` | User decision. Threshold test on 19,277 pilot-screened pKPDB entries: entries with ≥1 training site 5,222→9,312, training sites 216,111→290,778; 25 Å non-metal training masks would cost 37% of sites. Glycans need 20 Å (shielding); metals never use 15 Å. pKAI's own environment cutoff is 15 Å, but masks guard physical label error, not the predictor's view |
| 2026-10-07 | PINDER (extracted pairs) supervision: training uses pKPDB anchor-tier gap tiers (clean + uncertain), excludes sites on PDB2PQR-rebuilt residues, no rebuilt-neighbour mask; evaluation adds a 15 Å exclusion around rebuilt atoms of other residues. Component radii per `mask-all-v1` | User decision after sample scoring (`experiments/4_backbone/05_pinder_dataset.md`): training usable sites 117,534→165,201, interface 18,475→29,080 vs the previous worst-case gap + rebuilt-neighbour mask |
| 2026-10-08 | PINDER long-gap rule (pKAI teachers): a site near a long gap is usable if its functional atoms are at least D Å from the gap's flank CA, D = strictest over pKAI/pKAI+ × absolute/ΔpKa: terminal ≤10 → 30 (replaces 20 Å for 6–10), ≤20 → 35, ≤30 → 40, ≤50 → 45; internal 4–10 → 20, 11–20 → 25. Longer gaps stay uncalibrated/excluded; terminal 1–5 and internal 1–3 rules unchanged | User decision after deletion calibration on 516 PINDER references (`experiments/4_backbone/05_pinder_dataset.md`, `audits/pinder-longgap-v1`): approximate training rescore 1.80M→2.66M sites, interface 445k→575k, complexes with a usable interface site 25.5k→39.6k (17.8k clusters). Strictest column chosen while the teacher is undecided; a per-teacher column can replace it later |
| 2026-10-08 | pKPDB pretraining uses the PINDER held-out leakage rule: exclude an entry when any protein entity matches a held-out chain (`pinder-prefilter-v1/ref_heldout_all.fasta`: test + validation + reserved + set-2) at ≥ 70% identity and ≥ 80% coverage of both sequences (replaces the 5k pilot's ≥ 90% / 80%-of-shorter rule for frozen val/test; the stricter 30% experimental reservations are unchanged). Exclusion list: `audits/seq-overlap-v1/pkpdb_heldout_exclusions_70.tsv` | User decision so both pretraining sources share one threshold. Excludes 20,302 of 121,283 pKPDB entries (16.7%); 211 of the 19,277-entry test slice (1.1%), where training sites under the adopted long-gap rule go 382,352 → 377,067 and entries 13,753 → 13,583 |
| 2026-10-08 | Structure-defined site loss weights for pretraining (`src/pkabench/site_weights.py`): absolute-pKa loss × `burial_weight(RSA_free)` = 0.4 + 0.6·(1 − clip(RSA, 0, 1)); Siamese (bound − free) loss × `interface_weight(d)` = 1 for residue-level partner distance ≤ 4 Å, else exp(−(d − 4)/3 Å), floor 0.05. Weights depend on structure only, never on labels (preferred over target-shift-bin weighting). Auxiliary burial/interface prediction terms to follow | User decision. PypKa 2.10 benchmark (`audits/shift-by-environment-v1`): mean |free shift| 2.70 at RSA < 0.05 → 0.33 at RSA > 0.6 (r = 0.58 with 1 − RSA); |ΔpKa| 1.13 within 4 Å of the partner, 0.34 at 4–6 Å, 0.09 at 8–10 Å, ~0 beyond 15 Å, and near the interface independent of burial. Fields added to PINDER `sites.json` (rsa_free, rsa_bound, partner_distance_A, w_burial, w_interface) and pkpdb-5k-v3 `environment.json` (rsa, w_burial) |
| 2026-10-09 | Full pKPDB build `pretraining/pkpdb-full-v1` (`pkabench.pkpdb_mask_all --full`): every pKPDB entry under mask-all-v1, long-gap-v1 and the 70% held-out rule, with graph.npz; pipeline errors recorded in audit.json rather than aborting. Supersedes pkpdb-5k-v3 for new work; pkpdb-5k-v3 is left untouched because it is in use | User decision. Lifts the 2026-10-06 5k pilot scope. Expected ~60k accepted structures (~200 GB with graph.npz); burial weights by `audits/pinder-label-v1/pkpdb_env.py` (`PKPDB_BUILD=pkpdb-full-v1`) |
| 2026-10-09 | PINDER held-out exclusions harmonised with pKPDB: `pinder_heldout_exclusions_v2.tsv` (experimental 30% / 80%-both, benchmark fragment 90% / 80%-shorter, reserved PDB IDs; Ab/Ag dimers via the prefilter's antibody path: antigen chain whole-chain, antibody chain CDRs ≥ 70% vs experimental antibody CDRs; 4,017 complexes) is a filter on `pretraining/pinder-pkai-v1`; counts in `summary_clean_v2.json`. v1 (whole-chain on antibodies, 5,605 complexes) superseded the same day | User decisions. Same leakage rules for both pretraining sets; whole-chain rules on antibodies match any Fab to 1igc/1axt via conserved domains, so antibodies are checked by CDR as in the prefilter and split design. Leaves 36,134 complexes with labelled usable interface sites (Ab/Ag 1,211). Existing `summary.json` and the factorial cohort are not modified; the same antibody path is applied to pkpdb-full-v1 (860 sequence-overlap entries released, 760 accepted: 64,000 structures, 1,802,108 clean sites; `pkpdb_mask_all --revise`, previous build kept in `revisions/00/`) |
| 2026-10-09 | `training/ogqt-pinder-factorial-v1` is left as registered (262 of its 5,400 complexes are on `pinder_heldout_exclusions_v2.tsv`: 250 train, 12 val; `audits/pinder-exp-leak-v1/factorial_cohort_flagged_v2.tsv`). Future PINDER training subsets are drawn from the full pool after the v2 exclusions | User decision. Work moves to the full dataset; between-arm comparisons within the factorial run remain valid, but its absolute scores on benchmark/experimental sets carry this leakage caveat and are not comparable to clean-pool runs |

# Status update — 2026-10-05, full structural benchmark

The initial non-JAX structural benchmark is complete and verified: 1,452 frozen
complexes, 7,260 method receipts, and 1,358 complexes with usable teacher interface
labels. See [the consolidated report](1_benchmark/23_full_benchmark_report.md)
for group-bootstrap scores, coverage, antibody/general strata and limitations.
Experimental Set 2 remains outstanding; JAX-Ka remains deferred.

Experiment 02 handoff verification and the 13-run diagnostic are complete;
targeted pilot timeout recovery is complete and versioned separately.
The fixed 500 training IDs and 151 frozen validation IDs are retained, with no test
inputs in the handoff. See [the handoff contract](2_finetune/03_handoff.md).
The frozen-model gate, twelve trained runs and independent 468-metric audit passed.
Fine-tuning gains are small and their paired confidence intervals include zero;
see [the diagnostic results](2_finetune/05_diagnostic_results.md). Recovery
labels are versioned separately. Experimental Set 2 remains pending primary
measurement and construct checks; see [the follow-up](1_benchmark/24_set2_followup.md).

Experiment 03's [native teacher export](3_paired_complex/05_native_export_results.md)
contains 44,202 training paired labels and 8,365 unchanged validation pairs, plus
2,738 validated native-state energy exports. Saved-energy replay requires retaining
the original site order; the verified wrapper reproduces a 16-site complex exactly.
Native tautomer interactions must not be treated as an already validated scalar
binary-site coupling matrix. Test data remains excluded from this training export.
