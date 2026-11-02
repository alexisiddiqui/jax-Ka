# All-residue GQT with dRSA-weighted within-complex ranking

Selectable alternative to `ogqt-allres15-siamese-50`. This is the seed-17 export of
`drsa-v1-50-within03-asym50-w064-s17`, selected epoch 18 of 20, 50% training fraction.
The seed-29 and seed-43 exports are `ogqt-allres15-within-drsa-50-s29` (epoch 20) and
`ogqt-allres15-within-drsa-50-s43` (epoch 16). Each has 180,000 parameters and width 64;
the architecture and heads match the default and the within-ranking hotspot-mask export.

## Training and inputs

The training recipe is the asymmetric 50% hotspot-mask control
(`../ogqt-allres15-within-hmask-50/README.md`), with `--within-target drsa` in the
frozen Isambard training snapshot `_submission/drsa-within-v1`. The within-complex
listwise loss keeps coefficient 0.3. For contact candidate c and anchor a, its target is
`soft(distance_ac) * min(dRSA_a, dRSA_c)`, where
`dRSA = clip(RSA_free - RSA_bound, 0, 1)`. Noncontacts have zero target; targets are
normalised per anchor. Anchors with target sum below 1e-6 are dropped (about 48% of
sampled anchors across training visits). Anchor eligibility and candidate sampling
are unchanged from the distance-target control. See `model.json` for the full original
protocol, code hashes and selected-checkpoint SHA256.

Native RSA is a training target and PeSTo drives training-time masking. Inference is
unmasked and uses backbone and sequence only; neither native RSA nor PeSTo is required.
No new PypKa calculations were used. The existing native bonded-Cys annotation policy
is shared with the default model. All heads, including pKa, gate, burial and contact-edge
outputs, are exported; optimizer states are omitted.

## Validation

PINDER pool-v4 validation: 800 complexes. The following are descriptive means over
seeds 17, 29 and 43, each compared with its matched asymmetric hotspot-mask control.
Binding metrics use the free branch, unmasked inputs and the selected checkpoint.
Selection remains minimum PINDER state MAE plus interface binding-shift MAE; neither
interface-ranking results nor the large-shift strata selected a checkpoint.

| Metric | Matched control | dRSA |
|---|---:|---:|
| Surface-residue interface AUROC | 0.6598 | 0.7004 |
| Same-stratum swapped-partner accuracy | 0.7563 | 0.7986 |
| Antibody/antigen swapped-partner accuracy | 0.6656 | 0.7039 |
| Wrong-partner accuracy, different protein | 0.5804 | 0.5901 |
| Contact-edge average precision | 0.0406 | 0.0458 |
| dRSA-weighted NDCG | 0.5195 | 0.5338 |
| Within-complex contact AUROC | 0.6575 | 0.6546 |

Surface interface AUROC and swapped-partner accuracy improve in each of the three
seeds. This is not an affinity measurement or established general binding specificity;
wrong-antibody discrimination remains weak. POISK has not been evaluated.

Binding-shift MAE on the fixed clean-site panel (bonded Cys excluded), means over the
same seeds; native clusters use the existing max bound/free 4 A titration-centre
connected-component annotation, including Arg and termini, for reporting only:

| Sites | Number | Matched control | dRSA |
|---|---:|---:|---:|
| Overall | 29305 | 0.06561 | 0.06608 |
| Absolute true shift >= 2 | 222 | 1.24151 | 1.24596 |
| Absolute true shift >= 4 | 34 | 2.17511 | 2.18434 |
| Absolute true shift >= 2, native clusters 3-4 | 59 | 1.47864 | 1.45523 |

Large-shift gains are inconsistent across seeds; this is not a replacement for the
default pKa model. These are reused development validation results, with no new
uncertainty analysis. Numerical source results are in `validation-summary.json`.

## Use

```bash
ogqt predict complex.pdb --model ogqt-allres15-within-drsa-50 --cache prepared/ --output predictions.jsonl
# Replicas: --model ogqt-allres15-within-drsa-50-s29 or --model ogqt-allres15-within-drsa-50-s43
```

```python
from pkatrain.production_defaults import load_default
model = load_default("ogqt-allres15-within-drsa-50")
```

`GQT_DEFAULT_MODEL`, `load_default`, `ogqt predict --model` and the Torch endpoint's
`model_path` accept the export name or directory. The repository default is unchanged.

The Isambard release check (`experiments/4_backbone/scripts/check_allres_drsa.sbatch`)
exports with the original training snapshot and verifies restored parameter equality,
then compares bound/free predictions against stored validation predictions on the
first 20 validation complexes with both JAX and Triton backends. Each export's
`verification-jax.json` and `verification-triton.json` report matching coverage and
numerical differences; the model-loading and CLI regression suite also runs on Isambard.

Release check completed on Isambard (2026-11-02): all three parameter exports match
the restored checkpoint bitwise. Both backends matched 628 sites across 20 complexes
per export, with 25 sites unmatched by the existing site-key alignment. Maximum
absolute pKa difference over the matched sites, all three seeds and both states,
was 0.001772 for Triton and 0.002002 for JAX. Mean absolute differences were at most
0.000133. The all-residue/default, selectable-export CLI, and relational structure
regression tests reported **14 passed, 2 skipped**. Slurm export job: 7960629;
verification array: 7960630 (tasks 0-2).
