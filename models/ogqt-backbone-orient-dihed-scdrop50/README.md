# oGQT production model, backbone-only with orientation, dihedrals and side-chain dropout (width 64)

Selected checkpoint of `prod-100pct-w64-drop10-bur1-orient-scdrop50-dihed-e40` (gqt-production-v2, Isambard, 2026-10-12): scratch oGQT trained jointly on the
full pool-v4 PINDER (27,933 complexes) and pKPDB (56,191 structures) training sets for 40 epochs. Seed 17 was chosen
for export before the runs finished; it is one of three seeds (17/29/43). **148,002 parameters**, including the
65-parameter burial head.

| | |
|---|---|
| Architecture | `pkanet.ogqt` site model, width 64, ff 128, 2 residue blocks; inputs: 24 backbone node features + 128 side-chain columns (always zero at inference) + sin/cos phi/psi/omega (6), residue edges + 13 orientation features (production_inputs) |
| Objective | pKPDB state-shift MSE + PINDER AB/free state-shift MSE + binding-shift (Siamese) MSE, plus 1.0 x burial cross-entropy (soft target 1 - clip(RSA), PINDER free branch and pKPDB sites) |
| Regularisation | residual dropout 0.1 (one key per complex shared by its AB and free branches); side-chain features shown in training with probability 0.5 per structure (AB/free share the draw), hidden in validation and inference; AdamW weight decay 1e-4 |
| Training | batch 16, seed 17, 40 epochs (stretched_learning_rate x 1.4142 (1e-3 hold to epoch 20, cosine to 1e-5 by epoch 40); patience 16, min delta 0.001); selected epoch 39 of 40 |
| Selection | minimum PINDER validation state MAE + interface paired MAE |
| Seed spread | 3-seed mean selection 0.6035 +/- 0.0023 (20 epochs: 0.6080); 3-seed ensemble (averaged predictions) 0.5788 (benchmark MAE 0.4783; 20-epoch ensemble 0.5872) |

## Validation (absolute values at the selected epoch)

| Set | Metric | Value |
|---|---|---:|
| PINDER pool-v4 validation (800 complexes, 29,906 sites, 7,703 interface) | state MAE / MSE | 0.3826 / 0.3507 |
| | paired (AB - free) MAE / MSE | 0.0636 / 0.0451 |
| | interface paired MAE / MSE | 0.2185 / 0.1698 |
| | selection | 0.6011 |
| pKPDB pool-v4 validation (800 structures, 21,666 sites) | site MAE / MSE | 0.4637 / 0.5027 |
| | structure-macro MAE | 0.3710 |
| Benchmark validation (142 complexes, PypKa labels, 76 groups) | MAE / RMSE / Spearman | 0.4900 / 0.7113 / 0.8181 |
| | site MSE / group-macro MSE | 0.6550 / 0.5811 |
| Burial head, AUROC for RSA < 0.1 / 0.25 / 0.5 | PINDER free branch | 0.984 / 0.977 / 0.962 |
| | pKPDB | 0.981 / 0.975 / 0.964 |

No test data were read. The full validation record, the training protocol and the code hashes are in `model.json`.

## Files

- `params.npz`: the parameters only (no optimizer state), one array per parameter keyed by its tree path; sha256 in
  `model.json` (`params_sha256`).
- `model.json`: architecture, parameter count, selected epoch, validation metrics, training protocol (including the
  backbone-v1 and sidechains-v1 table verification hashes), code hashes, and the sha256 of the source checkpoint.

## Loading

```python
import jax
from pkatrain.production_train import exported_functions, load_exported
from pkanet.model import PKPDB_PK_MOD

path = "models/ogqt-backbone-orient-dihed-scdrop50"
params = load_exported(path)
_, predict_shift = exported_functions(path)          # input transform rebuilt from model.json
shifts = jax.vmap(predict_shift, in_axes=(None, 0))(params, graphs)
pka = shifts + PKPDB_PK_MOD[graphs["query_group"]]
```

`graphs` are production oGQT graph batches (`pkatrain.production_loading`) with the sidechains-v1 (128 columns) then backbone-v1 (14 columns) tables appended to the node features,
as the training loader built them (`sidechains=` `ConcatTable(SidechainTable(root, dataset), BackboneTable(root, dataset))`). The transform strips the table, adds
the 13 orientation edge features and appends sin/cos phi/psi/omega on the device. Call `predict_shift` without a key:
the side-chain columns are then multiplied by zero, so the model is backbone-only and the 128 columns can be zeros
(no side chains needed). `aux_burial` is the training-time burial head; `predict_shift` ignores it.

Exported with `python -m pkatrain.production_train export prod-100pct-w64-drop10-bur1-orient-scdrop50-dihed-e40 <dir>`; reload checked against the run's selected
checkpoint (all parameters identical). Training notes, the 20- vs 40-epoch comparison and the seed ensembles:
`experiments/4_backbone/07_gqt_production_stores.md`.
