# Frozen relational GQT interface probes

50% checkpoints, seeds 17/29/43. Free tokens only; GQT unchanged. Targets are contacts between existing pKa-labelled surface sites, not all protein residues.

| Head | Seed | Epoch | Site macro AP | Site pooled AP | Site AUROC | Pair AP |
|---|---:|---:|---:|---:|---:|---:|
| site | 17 | 1 | 0.4738522118619113 | 0.2753381113572882 | 0.6196711140355561 | — |
| pair | 17 | 2 | 0.45418341789441846 | 0.27742223326683896 | 0.623171821263374 | 0.030380688444098847 |
| site | 29 | 1 | 0.4636771354201246 | 0.2707470793327934 | 0.615570809120801 | — |
| pair | 29 | 6 | 0.44363851874521926 | 0.2580872935258164 | 0.6100109391069959 | 0.025655444002236443 |
| site | 43 | 3 | 0.47562724133713347 | 0.27705501340942484 | 0.621995684963546 | — |
| pair | 43 | 13 | 0.4522507876689109 | 0.26120869831965193 | 0.6106040744635285 | 0.024753876150808627 |

Selection: best validation site macro AP. Macro AP excludes complexes with no positive site; pooled metrics retain them. No test set was used.

Pair head uses maximum pair probability for site ranking. Balanced training and maximum aggregation mean scores should not be interpreted as calibrated probabilities. Pair and site heads differ in supervision and parameter count; this measures practical probe performance, not a parameter-matched attention ablation.

Free means partner-removed bound conformation; generalization to independently measured unbound structures remains untested.
