# Scratch oGQT paired structural-weight factorial

The first paired oGQT experiment uses pKAI-labelled PINDER complexes and starts
from scratch.  This keeps the structural-weight comparison separate from
transfer between the pKPDB/PypKa and PINDER/pKAI teachers.

The model predicts the signed state shift
`pKa(state) - PK_MOD(site type)`.  The paired prediction is the bound state
shift minus the free state shift; the fixed `PK_MOD` cancels exactly.

The matched one-seed factorial is:

| Arm | State-shift weight | Binding-shift weight |
|---|---|---|
| vanilla | uniform | uniform |
| burial | `w_burial` | uniform |
| interface | uniform | `w_interface` |
| both | `w_burial` | `w_interface` |

All arms use identical seed-17 initialization, sampled structures, batch order,
optimizer, schedule, and validation selection.  The deterministic pilot contains
5,000 training and 400 validation complexes, at most 768 residues, with one
complex per PINDER cluster and at least one masked, paired interface label.

`w_burial` is normalized to mean one over complex-level training means and is
applied only to the two state-shift losses.  `w_interface` is normalized
independently and is applied only to the AB-minus-free loss.  Validation and
checkpoint selection remain unweighted so an arm cannot improve merely by
changing the reporting distribution.

After selecting the scratch arm, score the completed pKPDB oGQT checkpoint
zero-shot and run two transfer experiments: pretrained paired vanilla and
pretrained with the winning structural weights.  This tests pretraining without
repeating the full factorial under a second initialization.

Implementation: `src/pkatrain/gqt_paired_pinder.py`. Runtime artifacts:
`$PKABENCH_RUNTIME/training/ogqt-pinder-factorial-v1/`. No test data are used.
