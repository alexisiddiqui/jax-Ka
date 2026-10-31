# All-residue GQT with within-complex ranking and hotspot masking

Alternative to the default `ogqt-allres15-siamese-50` (not the default). Training run
`hmask-v2-50-within03-asym50-w064-s17`, selected epoch 18 of 20, 50% training fraction, seed 17. Source checkpoint
`checkpoints/epoch-018/state.npz`, sha256 `9fe46802253020c1691fc0e88d1725e5507d6855424753a1df7718009bfde1da`.
180,000 parameters, width 64. Inputs are backbone and sequence only.

The architecture, heads and parameter tree are identical to the default (see `../ogqt-allres15-siamese-50/README.md`);
only training differs. Recipe (the default's run-4 flags plus the within-complex term and the hotspot mask):

```
pkatrain.production_train train hmask-v2-50-within03-asym50-w064-s17 --fraction 0.5 --seed 17 --shift-weight none
  --batch 16 --width 64 --dropout 0.1 --aux-weight 1 --aux-heads burial --orientation --dihedrals --variant relations
  --relation-maps 4,1 --all-residue-tokens --token-cutoff 15 --drop-bonded-cys --gate-weight 0.1 --gate-cys-weight 10
  --residue-edge-weight 0.1 --residue-burial-siamese --within-weight 0.3 --decoy-retype-weight 0
  --mask-oracle pesto --mask-store <store-hotspot-v1> --mask-sides one --mask-prob 0.5 --mask-keep 0.5
```

- Within-complex listwise negatives (weight 0.3, free branch): for each interface anchor residue, a softmax
  cross-entropy over the partner chain's interface residues (true contacts with a soft distance target, the rest as
  negatives) on the residue edge head (`aux_redge`). Of the negative-sampling terms tried, this is the only one that
  survived the shortcut checks (matched source and construction).
- Hotspot-mask augmentation: on half the visits, one randomly drawn chain has its residue identities zeroed except the
  top half of its surface by PeSTo protein-protein interface probability (buried residues always masked). The free
  edge BCE, the free-to-bound distillation and the within term are computed on that masked pass; the bound branch,
  Siamese burial, gate and pKa losses are unchanged.

Neither term adds parameters. PeSTo is a training-time dependency only: the exported model runs unmasked and needs no
PeSTo scores at inference.

## Validation (PINDER pool-v4 validation, 800 complexes; free branch, unmasked)

Three seeds of this recipe (s17 / s29 / s43; this export is s17), against the within-0.3 term alone (same three seeds)
and the default (run 4, seed 17). Pairwise accuracies count ties as half.

| metric | this recipe (s17 / s29 / s43) | within 0.3 alone | default (run 4) |
|---|---:|---:|---:|
| same-stratum swapped-partner accuracy | 0.765 / 0.751 / 0.753 | 0.746 / 0.731 / 0.738 | 0.731 |
| antibody/antigen vs antibody/antigen accuracy | 0.671 / 0.667 / 0.659 | 0.640 / 0.628 / 0.630 | 0.634 |
| wrong-partner (different protein) accuracy | 0.582 / 0.580 / 0.579 | | 0.557 |
| residue-pair contacts: AUROC | 0.730 / 0.724 / 0.722 | | 0.669 |
| residue-pair contacts: AP | 0.042 / 0.040 / 0.040 | | 0.029 |
| within-complex contact AUROC | 0.661 / 0.657 / 0.654 | | 0.605 |
| selection without bonded-Cys sites (state + interface MAE) | 0.6118 / 0.6154 / 0.6109 | | 0.6047 (3-seed mean 0.6152) |

Seed 17 (this export), other pKa metrics: selection with all sites 0.6315 (default 0.6251); state MAE / interface
binding-shift MAE 0.3986 / 0.2329 (0.3994 / 0.2257); pKPDB validation site MAE / benchmark-val MAE (PypKa)
0.4823 / 0.5069 (0.4814 / 0.5029). Burial AUROC is unchanged (about 0.967). The pKa differences are within seed noise.
Reports: Isambard `audits/hotspot-mask-v2/`.

Caveats:

- Not yet run on the POISK antibody binder/decoy benchmark.
- Training used masked passes; every number above, and the exported model, is unmasked.
- This export replaces the decoy-ranking export `ogqt-allres15-siamese-decoy-50` (2026-10-29), withdrawn because its
  swapped-partner gain was a family-similarity shortcut.

## Use

```bash
ogqt predict complex.pdb --model ogqt-allres15-within-hmask-50 --cache prepared/ --output predictions.jsonl
GQT_DEFAULT_MODEL=ogqt-allres15-within-hmask-50 ogqt predict complex.pdb --cache prepared/ --output predictions.jsonl
```

```python
from pkatrain.production_defaults import load_default
model = load_default("ogqt-allres15-within-hmask-50")   # same outputs as the default model
```

`--model`, `load_default(path)`, `TorchStructureReadout.from_default(..., model_path=...)` and `GQT_DEFAULT_MODEL` take
either the name under `models/` or the export directory.

`model.json` records the training protocol (including the `within_listwise` and `hotspot_mask` settings), code hashes
and the source checkpoint checksum. The export was written by `pkatrain.production_train export` from the training
code, the Isambard snapshot `_submission/hotspot-mask-v2`.
