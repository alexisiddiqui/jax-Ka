# All-residue GQT with decoy-partner ranking

Alternative to the default `ogqt-allres15-siamese-50` (not the default). Training run `decoy-v1-50-both-w064-s17`,
selected epoch 17 of 20, 50% training fraction, seed 17. Source checkpoint
`checkpoints/epoch-017/state.npz`, sha256 `b94b5b477011d495b6583e30244b132a509071c86c9db3ac9230c9a3f9b5bdb2`.
180,000 parameters, width 64. Inputs are backbone and sequence only.

The architecture, heads and parameter tree are identical to the default (see `../ogqt-allres15-siamese-50/README.md`);
only training differs. Recipe (the default's run-4 flags plus the two decoy weights):

```
pkatrain.production_train train decoy-v1-50-both-w064-s17 --fraction 0.5 --seed 17 --shift-weight none --batch 16
  --width 64 --dropout 0.1 --aux-weight 1 --aux-heads burial --orientation --dihedrals --variant relations
  --relation-maps 4,1 --all-residue-tokens --token-cutoff 15 --drop-bonded-cys --gate-weight 0.1 --gate-cys-weight 10
  --residue-edge-weight 0.1 --residue-burial-siamese --decoy-retype-weight 0.1 --decoy-batch-weight 0.1
```

Both decoy terms are margin losses on the residue edge head (`aux_redge`), softplus(1 − (logit_true − logit_decoy)) over
sampled true contacts, weight 0.1 each:

- Re-typed decoys: a third free-branch pass in which half of one chain's interface residues are re-typed to another
  chemistry class (D/E ↔ K/R; others to an interface-composition type outside their class; G/P/C never re-typed).
- In-batch decoys: a true contact (a_i, b_i) is scored against (a_i, b'_j), b'_j taken from the interface contacts of
  other complexes in the same batch (4 partners per complex).

## Validation (PINDER pool-v4 validation, 800 complexes, free branch for the decoy metrics; seed 17 only)

Pairwise accuracies count ties as half.

| metric | this model | default (run 4) |
|---|---:|---:|
| re-typed decoy pairwise accuracy | 0.925 | 0.844 |
| swapped-partner pairwise accuracy | 0.851 | 0.733 |
| surface-only all-pairs accuracy | 0.816 | 0.539 |
| residue-pair contacts, free branch: AUROC (AP) | 0.667 (0.037) | 0.669 (0.029) |
| selection without bonded-Cys sites (state + interface MAE) | 0.6121 | 0.6047 |
| selection, all sites | 0.6331 | 0.6251 |
| state MAE / interface binding-shift MAE | 0.3998 / 0.2333 | 0.3994 / 0.2257 |
| pKPDB validation site MAE / benchmark-val MAE (PypKa) | 0.4822 / 0.5058 | 0.4814 / 0.5029 |

Run 4's 3-seed mean of the clean selection metric is 0.6152 (sd 0.012), so the pKa difference is within seed noise.
Burial AUROC is unchanged (0.967). A wider or deeper pair head and extra binding-only layers gave nothing beyond noise, so this is the plain edge head.
Reports: Isambard `audits/decoy-rank-v1/`.

Caveats:

- Single seed.
- Motivated by the POISK antibody binder/decoy benchmark, where the default's blind surface-only scores rank below
  chance; POISK has not yet been run with this model.
- The swapped-partner decoy metric uses the same family of decoys (other complexes' partners) that the in-batch loss
  trains on, so part of that gain is in-distribution.

## Use

```bash
ogqt predict complex.pdb --model ogqt-allres15-siamese-decoy-50 --cache prepared/ --output predictions.jsonl
GQT_DEFAULT_MODEL=ogqt-allres15-siamese-decoy-50 ogqt predict complex.pdb --cache prepared/ --output predictions.jsonl
```

```python
from pkatrain.production_defaults import load_default
model = load_default("ogqt-allres15-siamese-decoy-50")   # same outputs as the default model
```

`--model`, `load_default(path)`, `TorchStructureReadout.from_default(..., model_path=...)` and `GQT_DEFAULT_MODEL` take
either the name under `models/` or the export directory.

`model.json` records the training protocol (including the `decoy_retype` and `decoy_batch` settings), code hashes and
the source checkpoint checksum. The export was written by `pkatrain.production_train export` from the training code,
the Isambard snapshot `_submission/decoy-rank-v1` (`residue-siamese-v1` plus a width guard and the two decoy losses).
