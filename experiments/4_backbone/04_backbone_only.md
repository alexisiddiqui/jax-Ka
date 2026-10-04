# 04 — Backbone-only mode

**Timebox:** 1 day, 1 day slack
**Hardware:** 3090 (+ A40s if an ensemble-label run is triggered)
**Depends on:** a trained all-atom model from 03

**AFDB scope (2026-10-03):** Consider AFDB structures and associated pKa data at
this backbone-only tuning stage. The current pKPDB coverage audit uses AFDB only
as sequence/coverage evidence; it does not predict structures, replace PDB
coordinates, or transfer pKPDB labels onto AFDB geometry. See
[the missing-residue policy](../1_benchmark/06_missing_residue_policy.md).

**Later extension:** [05 — small-molecule and ligand complexes](../5_ligands/05_ligand_complexes.md).
The first dataset-generation round remains protein–protein; ligand-containing
training examples and bound/unbound pairs are deferred until after this stage.

---

## The thing that makes this cheap

The worry was that backbone-only labels are ill-posed: a pKa shift depends on H-bonds and
salt bridges made by side chains, so a backbone-only input cannot recover a label
computed from one specific rotamer.

But **MSE under random side-chain masking converges to the conditional mean over the
implied rotamer distribution.** That's what MSE minimises. So masking augmentation gets
you the right target automatically, provided the training set contains enough diverse
side-chain configurations for each backbone context — which 20k complexes does.

Repacked ensemble labels are therefore a **variance-reduction technique, not a
correctness requirement**. That collapses this section from a data-generation campaign
into a training-time augmentation plus a measurement.

Your instinct was right. The earlier tier-C spec was overbuilt.

---

## Step 1 — Measure before fixing (half a day)

Take the trained all-atom model from 03 and mask side chains at inference. No retraining.

Report, on held-out clusters:

- ΔpKa RMSE: all-atom vs. backbone-only
- Same, stratified by ΔSASA, burial, residue type
- Fraction of sites where backbone-only flips the sign

**Decision gate:**

| Backbone-only degradation | Action |
|---|---|
| < 0.2 RMSE | Done. Ship it. The backbone already pins down most of the environment via geometry + neighbour identity. Document and move on. |
| 0.2 – 0.5 | Step 2 (augmentation retrain) is enough. |
| > 0.5 | Step 2 + step 3 (ensemble labels for the hard subset). |

Do not skip this gate. Spending two days fixing a problem you haven't sized is how the
week slips.

---

## Step 2 — Augmentation retrain (half a day)

Retrain (or fine-tune) with side-chain dropout as a training-time augmentation.

### Masking schedule

Sample per-example, not per-batch:

- 40% full all-atom
- 30% full backbone-only
- 30% partial (Bernoulli per residue, p ~ U(0.1, 0.9))

Partial masking matters more than it looks: it's what teaches the model to use
side-chain information *when present* rather than learning two disjoint modes. It also
covers the realistic case of structures with some missing side chains, which your
curation currently rejects outright.

### Representation

Mask by zeroing the side-chain channel **plus a learned "absent" embedding** — not by
zeroing alone, which is ambiguous with a genuinely zero-valued feature. One shared
encoder, one set of weights, both modes.

### Real missing coordinates versus synthetic dropout

For synthetic dropout, calculate trustworthy teacher labels from the complete
structure first, then hide student coordinates without discarding those labels.
For genuinely unresolved experimental atoms/residues, there is no corresponding
complete-structure label to inherit. Retain usable examples but apply the shared
[defect supervision policy](../00_shared.md#revised-supervision-policy-for-incomplete-structures):
mask the affected and nearby site targets identically in the bound/free pair.

Backbone-only reconstruction can provide approximate context or locate a missing
segment for masking. It does not restore missing side-chain charges or justify
assigning the neighbouring residue's pKa shift. The teacher must still operate on
an explicit chemically valid completed or capped/truncated representation.

Compare exclusion radii of 10/15/20 Å in a repair-sensitivity pilot. Freeze the
radius only after measuring target stability and retained-site bias. A prediction
may be emitted at a masked site, but it has no supervised loss or benchmark score.
Report site-level and whole-complex coverage separately; incomplete charge labels
are insufficient for full linkage supervision.

### Heteroscedastic head

Add a variance output and train with Gaussian NLL on the shift targets:

```
L = 0.5 * (exp(-logvar) * (pred - target)^2 + logvar)
```

Under masking this makes the model report its own uncertainty, and the predicted
variance tells you which sites are rotamer-determined vs. backbone-determined. That's
a useful output in its own right for design work — it flags where a backbone-only
prediction can be trusted.

Watch for the usual NLL failure mode: variance collapse early in training. Warm up with
plain MSE for the first few thousand steps, then switch.

---

## Step 3 — Ensemble labels, only if the gate fires

Variance reduction for the hard subset.

- Repack side chains **5 times**, not 20
- **Only** for sites with |Δintrinsic| above threshold or low SASA — roughly 15–20% of sites
- Label = mean; also store std as the heteroscedastic target instead of learning it

Overhead: ~1.5× tier B, so ~5,000 core-h, half a day on the node. No new pipeline —
your prep already builds frozen CCD candidates in the local frame, so sampling
alternatives is a loop around existing code.

Surface sites keep their single-structure labels and a per-residue-type variance prior.

---

## Step 4 — Validate against the actual use case

The point of backbone-only is composing with a structure predictor. So test it that way:

1. Predict structures for a held-out set of complexes with FlashABB / Boltz
2. Run the pKa model on predicted coordinates, backbone-only and all-atom
3. Compare against labels computed from the crystal structure

This is the number that matters for the end-to-end pipeline, and it's different from
masking a crystal structure — predicted backbones have their own error distribution.
If performance collapses here but not under masking, the fix is adding predicted
structures to training (cheap: re-run the teacher on predicted coords for a subset),
not more ensemble labels.

---

## Outputs

```
results/backbone/
  gate_measurement.csv       # the step-1 decision table, filled in
  masking_ablation.csv
  calibration_plot.png       # predicted variance vs. actual error
  predicted_structure_eval.csv
```

## Success criteria

- Backbone-only within 0.3 RMSE of all-atom on held-out clusters
- Predicted variance is calibrated (reliability diagram roughly on the diagonal)
- Single checkpoint serves both modes — no separate backbone model

## Honest limitation to document

Backbone-only returns a rotamer-averaged pKa. For design that is arguably the *better*
target — you want the expected pKa over side-chain configurations, not one crystal's
rotamer. But it means the model cannot resolve cases where a specific rotamer is
mechanistically essential, and buried histidines in particular may be systematically
smoothed. State this.
