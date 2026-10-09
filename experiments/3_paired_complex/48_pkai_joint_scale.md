# Joint pKAI data-scale experiment

The experiment trains the native 3,608,001-parameter pKAI architecture on both the revised pKPDB state dataset and the leakage-filtered PINDER paired dataset. The registered arms are backbone-only at 10%, 50% and 75%, plus native all-atom pKAI at 10% and 50%. Every arm uses seed 17 and its matching nested fraction from both datasets. Validation and test memberships remain fixed across fractions.

Each update combines three task-normalized losses: pKPDB signed shift from the native fixed `PK_MOD`, PINDER AB/free signed state shifts, and the PINDER Siamese binding shift. The two branches use one shared pKAI model. Losses are normalized by task rather than concatenating rows, preventing the larger state dataset and abundant near-zero paired sites from setting the objective implicitly.

Production registration waits for the final pKPDB pool-v2 and environment manifests. The completed PINDER pool-v2 counts are:

| Fraction | Complexes | Clusters | Labelled sites | Interface sites |
|---:|---:|---:|---:|---:|
| 10% | 3,592 | 1,586 | 184,096 | 48,178 |
| 50% | 17,347 | 7,937 | 908,299 | 230,004 |
| 75% | 26,087 | 11,905 | 1,361,481 | 343,327 |

## Smoke gate

Job `747150` exercised both representations on real frozen pilot records. Each mode used 64 pKPDB sites and 51 matched PINDER AB/free sites, evaluated all three losses through one shared model, and performed one Adam update.

| Mode | pKPDB loss | PINDER state loss | Siamese loss | Peak allocated VRAM | Result |
|---|---:|---:|---:|---:|---|
| Backbone-only | 0.97077 | 0.85195 | 0.000118 | 124 MiB | pass |
| All atom | 0.97075 | 0.85202 | 0.000118 | 124 MiB | pass |

Every component gradient was finite and nonzero. The shared parameters changed, input arrays remained bit-identical, and swapping both Siamese branches left the paired loss unchanged. No production pool or test data was read. The machine-readable receipt is `training/pkai-joint-scale-v1/smoke.json` in the benchmark runtime.
