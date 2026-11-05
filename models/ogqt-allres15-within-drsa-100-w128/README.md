# All-residue GQT with dRSA within-ranking: 100% training pool, width 128

Selectable alternative to the default `ogqt-allres15-siamese-50` (not the default). Training run
`drsa100-v1-asym50-w128-s17`, selected epoch 20 of 20, seed 17, the full (100%) training pool. Source checkpoint
`checkpoints/epoch-020/state.npz`, sha256 `50d53402c43c7e330e9db9e3e0ee86f00450236b04247f9f82e8a51da45ab25b`.
693,856 parameters: width 128, feed-forward 256, 4 attention heads, relation maps 4,1, every residue a token,
15 A token cutoff. The heads, inputs and parameter tree are those of the other all-residue exports; only the widths
differ (the width-64 exports have 180,000 parameters).

## Training and inputs

The recipe is the dRSA-weighted within-ranking export (`../ogqt-allres15-within-drsa-50/README.md`) with two changes:
`--fraction 1.0` (27,933 PINDER and 56,191 pKPDB training structures instead of 13,948 and 27,778) and `--width 128`.
The frozen Isambard training snapshot is `_submission/datascale-v1` (the drsa-within-v1 code plus a width allow-list
that the public `pkatrain.production_position` now carries for this variant only):

```
pkatrain.production_train train drsa100-v1-asym50-w128-s17 --fraction 1.0 --seed 17 --shift-weight none --batch 16
  --width 128 --dropout 0.1 --aux-weight 1 --aux-heads burial --orientation --dihedrals --variant relations
  --relation-maps 4,1 --all-residue-tokens --token-cutoff 15 --drop-bonded-cys --gate-weight 0.1 --gate-cys-weight 10
  --residue-edge-weight 0.1 --residue-burial-siamese --decoy-retype-weight 0 --within-weight 0.3
  --mask-oracle pesto --mask-store <store-hotspot-v2> --mask-sides one --mask-prob 0.5 --mask-keep 0.5 --within-target drsa
```

The hotspot-mask store is `store-hotspot-v2`: the `store-hotspot-v1` records (50% pool and validation, copied
bitwise) plus PeSTo records for the other 13,985 training complexes, so the full pool trains with no random-oracle
fallbacks (0 in every epoch). Native RSA (burial and dRSA targets) and PeSTo (the one-side hotspot mask) are used in
training only. Inference is unmasked and uses backbone and sequence only, like the other all-residue exports. All
heads (pKa, gate, burial, contact edge) are exported; optimizer states are omitted. See `model.json` for the full
protocol, code hashes and checksums.

## Validation

Selected checkpoint (seed 17, epoch 20), PINDER pool-v4 validation (800 complexes, all sites) and the pKPDB /
benchmark-val PypKa teacher panels, against the seed-17 export of the 50% width-64 recipe
(`ogqt-allres15-within-drsa-50`, epoch 18):

| metric (selection.json) | this export | drsa-50 (s17) |
|---|---:|---:|
| selection (state + interface binding-shift MAE) | 0.6079 | 0.6297 |
| PINDER state MAE | 0.3820 | 0.3984 |
| PINDER interface binding-shift MAE (7,703 sites) | 0.2258 | 0.2314 |
| PINDER interface binding-shift MAE, abs. true shift 2-4 (205 sites) / >= 4 (38 sites) | 1.045 / 1.785 | 1.127 / 2.068 |
| pKPDB validation site MAE | 0.4576 | 0.4822 |
| benchmark-val MAE / Spearman | 0.4852 / 0.818 | 0.5022 / 0.806 |
| burial AUROC at RSA 0.1 / 0.25 / 0.5 | 0.987 / 0.979 / 0.966 | 0.985 / 0.976 / 0.962 |

Three seeds of this recipe (s17 / s29 / s43, selected epochs 20 / 18 / 18) against the three seeds of the 50%
width-64 export (`ogqt-allres15-within-drsa-50{,-s29,-s43}`). Free branch, unmasked inputs, selected checkpoints; pKa
columns exclude bonded-Cys sites. Mean (sample SD); "seeds" counts the seeds whose difference has the sign of the mean.

| metric | this recipe, 100% w128 | drsa-50, 50% w64 | seeds |
|---|---:|---:|---:|
| antibody/antigen swapped-partner accuracy | 0.704 (0.010) | 0.704 (0.003) | equal |
| same-stratum swapped-partner accuracy | 0.835 (0.009) | 0.799 (0.004) | 3/3 |
| wrong-partner accuracy, different protein | 0.622 (0.009) | 0.590 (0.012) | 3/3 |
| within-complex contact AUROC | 0.670 (0.004) | 0.655 (0.002) | 3/3 |
| contact-edge AUROC / AP | 0.746 / 0.056 | 0.729 / 0.046 | 3/3 |
| surface-residue interface AUROC (from the free state) | 0.721 (0.007) | 0.700 (0.003) | 3/3 |
| pKa selection (state + interface MAE) | 0.5934 (0.0041) | 0.6157 (0.0064) | 3/3 |
| PINDER state MAE | 0.3721 (0.0031) | 0.3880 (0.0046) | 3/3 |
| PINDER interface binding-shift MAE | 0.2213 (0.0017) | 0.2277 (0.0019) | 3/3 |
| pKPDB validation site MAE | 0.4621 (0.0039) | 0.4819 (0.0019) | 3/3 |
| benchmark-val MAE | 0.4861 (0.0012) | 0.5054 (0.0038) | 3/3 |

Most of the gain comes from the larger training pool: the same recipe at width 64 and 100% (three seeds) reaches the
same pKa (selection 0.5917 (0.0018)). Width 128 adds seed-consistent gains on same-stratum swaps (+0.024),
wrong-partner accuracy (+0.020) and within-complex AUROC (+0.006) over that width-64 100% model, and gives a smaller
seed spread on antibody/antigen swaps (SD 0.010 vs 0.023); its antibody/antigen lead (+0.019) holds at 2 of 3 seeds only.
Numerical sources, per seed, are in `validation-summary.json` (Isambard audits `audits/datascale-v1/`,
`audits/drsa-within-v1/`, `audits/drsa-cap-v1/`).

Caveats:

- Antibody/antigen swapped-partner accuracy does not improve on the 50% export; wrong-antibody discrimination remains
  weak. These are ranking checks, not affinity measurements. POISK has not been evaluated.
- The 100% runs select epochs 17-20 of 20, so the 20-epoch schedule may still be short.
- 3.9 times the parameters of the width-64 exports; training wall time at 100% is about the same as width 64.
- Reused development validation (the same PINDER validation split used for checkpoint selection); no new uncertainty
  analysis.

## Use

```bash
ogqt predict complex.pdb --model ogqt-allres15-within-drsa-100-w128 --cache prepared/ --output predictions.jsonl
```

```python
from pkatrain.production_defaults import load_default
model = load_default("ogqt-allres15-within-drsa-100-w128")
```

`GQT_DEFAULT_MODEL`, `load_default`, `ogqt predict --model` and the Torch endpoint's `model_path` accept the export
name or directory. The repository default is unchanged.

The Isambard release check (`experiments/4_backbone/scripts/check_allres_drsa100_w128.sbatch`) exports with the
original training snapshot and verifies restored parameter equality, then compares bound/free predictions of the
`ogqt predict` feature path against the run's stored validation predictions on the first 20 validation complexes with
both JAX and Triton backends (`verification-jax.json`, `verification-triton.json`); the model-loading and CLI
regression suite also runs on Isambard (`verification-tests.txt`).

Release check completed on Isambard (2026-11-05): the parameter export matches the restored checkpoint bitwise
(693,856 parameters; params.npz sha256 `e82c7bb426b1714ae9f2bc872d6b99a0bbd49d36811ea8cbb82473160d31c0ce`). Both
backends matched 628 sites across 20 complexes, with 25 sites unmatched by the existing site-key alignment. Maximum
absolute pKa difference over the matched sites, both states, was 0.001159 for Triton and 0.001368 for JAX; mean
absolute differences were at most 0.000148. The all-residue/default, selectable-export CLI, width-128 registration
and relational structure regression tests reported **16 passed, 2 skipped**. Slurm export job: 8104005;
verification job: 8104007.
