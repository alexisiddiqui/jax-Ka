# oGQT production model, "50K" tier (width 44)

Selected checkpoint of `prod-100pct-w44-drop10-bur1` (gqt-production-v2, Isambard, 2026-10-11): scratch oGQT trained
jointly on the full pool-v4 PINDER (27,933 complexes) and pKPDB (56,191 structures) training sets. The tier is named
50K after the historical GQT count; with the site-attention block this model has **67,770 parameters**
(67,725 in the pKa model plus a 45-parameter burial head).

| | |
|---|---|
| Architecture | `pkanet.ogqt`, width 44, ff 88, 24 backbone node features (no side chains) |
| Objective | pKPDB state-shift MSE + PINDER AB/free state-shift MSE + binding-shift (Siamese) MSE, plus 1.0 x burial cross-entropy (soft target 1 - clip(RSA), PINDER free branch and pKPDB sites) |
| Regularisation | residual dropout 0.1 (one key per complex shared by its AB and free branches), AdamW weight decay 1e-4 |
| Training | batch 16, seed 17, 20 epochs; selected epoch 20 of 20 (selection still improving) |
| Selection | minimum PINDER validation state MAE + interface paired MAE |

## Validation (absolute values at the selected epoch)

| Set | Metric | Value |
|---|---|---:|
| PINDER pool-v4 validation (800 complexes, 29,906 sites, 7,703 interface) | state MAE / MSE | 0.4219 / 0.4265 |
| | paired (AB - free) MAE / MSE | 0.0709 / 0.0575 |
| | interface paired MAE / MSE | 0.2413 / 0.2174 |
| | selection | 0.6632 |
| pKPDB pool-v4 validation (800 structures, 21,666 sites) | site MAE / MSE | 0.5072 / 0.5999 |
| | structure-macro MAE | 0.4045 |
| Benchmark validation (142 complexes, PypKa labels, 76 groups) | MAE / RMSE / Spearman | 0.5328 / 0.7669 / 0.7698 |
| | site MSE / group-macro MSE | 0.7484 / 0.6648 |
| Burial head, AUROC for RSA < 0.1 / 0.25 / 0.5 | PINDER free branch | 0.983 / 0.973 / 0.957 |
| | pKPDB | 0.978 / 0.971 / 0.960 |

No test data were read. The full validation record, the training protocol and the code hashes are in `model.json`.

## Files

- `params.npz`: the parameters only (no optimizer state), one array per parameter keyed by its tree path
  (`blocks/0/q/w`, `head/w`, `aux_burial/w`, ...); sha256 in `model.json` (`params_sha256`).
- `model.json`: architecture, parameter count, selected epoch, validation metrics, training protocol, code hashes,
  and the sha256 of the source checkpoint (`checkpoint_sha256`).

## Loading

```python
import jax
from pkatrain.production_train import load_exported
from pkanet.ogqt import predict_shift
from pkanet.model import PKPDB_PK_MOD

params = load_exported("models/ogqt-production-50k")
shifts = jax.vmap(predict_shift, in_axes=(None, 0))(params, graphs)   # production graph batch (pkatrain.production_loading)
pka = shifts + PKPDB_PK_MOD[graphs["query_group"]]
```

The model predicts the shift from `PKPDB_PK_MOD[group]` for each query site. Inputs are the production oGQT graphs
(`pkatrain.production_graphs`, 20 A residue graph plus site graph, backbone features only). `aux_burial` is the
training-time burial head; `predict_shift` ignores it.

Exported with `python -m pkatrain.production_train export prod-100pct-w44-drop10-bur1 <dir>`. Size sweep and
training notes: `experiments/4_backbone/07_gqt_production_stores.md`.
