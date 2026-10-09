# smooth_propka on the frozen PINDER validation cohort

The external [`smooth_propka`](https://github.com/ovavourakis/smooth_propka) repository was frozen at revision `1f3a96bd348c54a8b27406a37c43040bb8ddbd37`. Its `ph-loss` package is a differentiable smooth approximation to PROPKA 3.5.1, rather than a model trained against experimental pKa values.

The benchmark used the same 400-cluster PINDER validation cohort, AB/A/B coordinates, evaluation masks and current-PypKa reference as the pKAI audits. Each state was evaluated independently with the package's generic defaults: CPU float64, `protein_only=True`, inferred disulfides and strict rejection of incomplete heavy-atom residues. No smoothing calibration or training was performed.

The upstream package declares Python 3.12–3.13. The compute image provides Python 3.11, under which this revision compiled and passed the representative smoke set without source changes. The benchmark therefore uses an unsupported interpreter version; package, PROPKA, PyTorch, NumPy and Gemmi versions remain frozen in every receipt.

| Estimator | State MAE | Paired-shift MAE | Interface paired-shift MAE |
|---|---:|---:|---:|
| smooth_propka | 0.656 | 0.211 | 0.447 |
| Regular pKAI | **0.286** | **0.087** | **0.176** |
| pKAI+ | 0.502 | 0.121 | 0.253 |
| Backbone-pretrained pKAI | 0.660 | 0.181 | 0.387 |

All 400 structures completed. Coverage was 22,384/22,384 state sites and 11,032/11,032 paired sites. Median state runtime was 0.84 seconds, p95 was 5.35 seconds, and the 1,200 state calls used 1,742.7 aggregate CPU seconds.

The state error has a positive bias of 0.248 pKa. The largest residue-level biases against PypKa are GLU (+0.743), ASP (+0.575) and LYS (-0.437). Paired errors are smaller because part of the absolute-state bias cancels between AB and free states, but interface paired-shift MAE remains 0.447 pKa.

These values measure agreement with current PypKa on the cleaned structural cohort. They do not measure experimental accuracy. The complete report, matched site tables, failure inventory and strata are under `audits/smooth-propka-pinder-v1` in the benchmark runtime.

## Frozen oGQT rescore

The existing oGQT prediction files were subsequently joined to the completed current-PypKa audit by complex, chain, residue number, insertion code and site type. No inference, training or checkpoint selection was repeated. To prevent the small oGQT coverage difference from changing the comparison, every method below is rescored on the exact same 21,807 state sites and 10,744 paired sites.

| Estimator | State MAE | Paired-shift MAE | Interface paired-shift MAE |
|---|---:|---:|---:|
| oGQT pKPDB-pretrained | 0.546 | 0.156 | 0.317 |
| oGQT joint | 0.524 | 0.154 | 0.312 |
| oGQT joint + dropout | 0.521 | 0.148 | 0.306 |
| smooth_propka | 0.663 | 0.212 | 0.451 |
| Regular pKAI | **0.286** | **0.087** | **0.176** |
| pKAI+ | 0.504 | 0.121 | 0.253 |
| Backbone-pretrained pKAI | 0.660 | 0.181 | 0.387 |

oGQT covers 97.4% of the current-PypKa audit sites. The pKPDB-pretrained row is the frozen epoch-zero parent; the two joint checkpoints also saw PINDER/pKAI supervision. Joint training improves oGQT's PypKa agreement, and dropout gives the best oGQT result, but regular pKAI and pKAI+ remain better on all three metrics. The complete rescore and immutable input hashes are under `audits/gqt-pinder-pypka-v1`.
