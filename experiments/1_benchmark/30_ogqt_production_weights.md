# Production oGQT weights on the frozen PINDER/PypKa benchmark

## Question

Do the production oGQT weights published on the latest `main` branch improve on the earlier oGQT checkpoints when every estimator is evaluated against the same frozen PINDER/PypKa cohort?

## Frozen inputs

- Source: `alexisiddiqui/jax-Ka` `main` at `7c3e3e97582f7f3162525add8b265824b0af29a8`.
- Weights: `models/ogqt-production-50k/params.npz`, SHA-256 `29eb740008244e5069f54ff81ebfbeb13e8fb1bf5df0fba9abe367ae255c5fcc`.
- Model: oGQT width 44, feed-forward width 88, 67,770 parameters.
- Training recorded by the checkpoint: scratch initialization, joint pKPDB and PINDER supervision, 20 epochs, batch 16, residual dropout 0.1, AdamW weight decay `1e-4`, and a burial auxiliary task.
- Evaluation: the original frozen 400-complex PINDER validation cohort. No final-test examples were used.

The source was fetched into an immutable runtime worktree rather than switching the dirty development checkout. Before inference, one validation complex was rebuilt with the production graph builder. All 25 stored graph fields, including residue/site features, branch masks, geometry, targets, RSA, and weights, were exactly equal to the frozen mmap graph.

Evaluation used an isolated environment matching the current source declaration: Python 3.12.15, JAX/JAXlib 0.11.1, `jax-triton` 0.4.1, and Triton 3.8.0. An earlier compatibility probe under JAX 0.6.2 and `jax-triton` 0.3.0 differed from native attention by at most 0.00042 pKa on the smoke complex, so it was discarded in favor of this exact-environment result.

## Frozen-cohort accuracy

Intervals are 2,000-replicate cluster-bootstrap intervals.

| Model | State MAE | Paired-shift MAE | Interface paired MAE |
|---|---:|---:|---:|
| Joint-trained pKAI (`pKAI-joint-aa20`, scratch initialization) | 0.273 | 0.082 | 0.168 |
| Regular pKAI | 0.286 | 0.087 | 0.176 |
| **Production oGQT 50K** | **0.474** [0.455, 0.492] | **0.138** [0.124, 0.152] | **0.285** [0.264, 0.307] |
| Earlier oGQT joint + dropout | 0.521 | 0.148 | 0.306 |
| Earlier oGQT joint | 0.524 | 0.154 | 0.312 |
| Earlier oGQT pKPDB-pretrained | 0.546 | 0.156 | 0.317 |

The production model was evaluated on 21,807 state observations, 10,744 matched AB/free sites, and 4,827 interface sites. It reduces error relative to the strongest earlier oGQT by 8.9% for states, 6.7% for paired shifts, and 6.7% at interfaces. It does not close the gap to regular pKAI: its errors are 66%, 58%, and 62% larger on the three metrics, respectively.

## Binding-shift range

| Reference `|ΔpKa|` | Sites | MAE | Predicted shift SD | Reference shift SD |
|---|---:|---:|---:|---:|
| `<0.5` | 9,634 | 0.073 | 0.203 | 0.111 |
| `0.5–1` | 520 | 0.407 | 0.791 | 0.718 |
| `1–2` | 364 | 0.726 | 1.320 | 1.408 |
| `≥2` | 226 | 1.377 | 2.099 | 3.220 |

The predicted-versus-reference shift slope is 0.639. The model represents more shift variation than earlier compressed GQT runs, but it still compresses the largest PypKa shifts substantially.

On the same 10,744 sites, the comparable per-bin errors are:

| Reference `|ΔpKa|` | Production oGQT BB | pKAI BB 15 Å | pKAI BB 20 Å | BB-pretrained pKAI | Regular pKAI all-atom | Joint-trained pKAI all-atom |
|---|---:|---:|---:|---:|---:|---:|
| `<0.5` | **0.073** | 0.097 | 0.107 | 0.083 | 0.047 | **0.043** |
| `0.5–1` | **0.407** | 0.479 | 0.449 | 0.506 | **0.284** | 0.288 |
| `1–2` | **0.726** | 0.839 | 0.809 | 0.959 | 0.438 | **0.416** |
| `≥2` | **1.377** | 1.958 | 1.935 | 2.367 | 0.788 | **0.757** |

Bold values identify the best backbone-only and best all-atom result separately. Production oGQT is the best backbone-only model in every bin. All-atom pKAI retains a substantial advantage for large shifts.

## pH-linkage proxy

For each common AB/free site, scalar pKa midpoints were converted to Henderson–Hasselbalch curves and integrated analytically to obtain `log10[Kd(7.4)/Kd(6.1)]`. All models use the same 378 complexes and 10,744 site pairs.

| Model | Mean PypKa ratio | Mean predicted ratio | Mean reference − prediction [95% CI] | Mean absolute difference |
|---|---:|---:|---:|---:|
| Joint-trained pKAI (`pKAI-joint-aa20`, scratch initialization) | -0.0209 | -0.0318 | +0.0109 [-0.0060, +0.0276] | 0.0539 |
| Regular pKAI | -0.0209 | -0.0336 | +0.0127 [-0.0024, +0.0298] | 0.0537 |
| **Production oGQT 50K** | **-0.0209** | **-0.0531** | **+0.0323 [+0.0112, +0.0544]** | **0.0774** |
| Earlier oGQT joint + dropout | -0.0209 | -0.0619 | +0.0410 [+0.0181, +0.0655] | 0.0940 |
| Backbone pKAI 15 Å | -0.0209 | -0.0615 | +0.0406 [+0.0142, +0.0685] | 0.1023 |

The production checkpoint improves the older oGQT linkage mean absolute difference by 17.7%. Its mean ratio remains too negative; the signed bias excludes zero. A mean absolute log10-ratio error of 0.0774 corresponds to a typical multiplicative ratio error of `10^0.0774 = 1.20×`, while the signed mean bias of 0.0323 corresponds to `1.08×`.

## Decision

Use the production oGQT weights as the default oGQT checkpoint. They are a clear improvement over the previous oGQT variants on all frozen metrics. Regular and fine-tuned pKAI remain the stronger estimators on this cohort, especially for the largest binding shifts and the pH-linkage proxy.

## Artifacts

- Evaluator: `benchmarks/ogqt_production_frozen_eval.py`
- Bootstrap report: `benchmarks/ogqt_production_report.py`
- Common-support linkage: `benchmarks/kd_ratio_comparison_production_ogqt.py`
- Evaluation outputs: `_runtime/jax-Ka/pkabench/audits/ogqt-production-frozen-v2-exact/`
- Linkage outputs: `_runtime/jax-Ka/pkabench/audits/kd-ratio-comparison-v5-ogqt-production-exact/`
- Exact-environment Slurm jobs: install `749722`, smoke `749723`, full evaluation `749724`, linkage `749725`, bootstrap report `749726`.
