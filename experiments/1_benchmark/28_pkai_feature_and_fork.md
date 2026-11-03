# smooth_propka pKAI feature branch and pKAI fork benchmark

Date: 2026-10-10.

## Scope

This evaluation uses the frozen 400-cluster PINDER validation cohort and the current
PypKa teacher audit. No parameters were updated and no final test data were used.

- `smooth_propka` feature branch: revision
  `dd5790879d901e5b1a04addb05200e3a3c7ed3fb`.
- pKAI fork `joint-retrained-models`: revision
  `1ced84288cf0dfd75b52a038af7f9dc14bdd55e4`.
- `pKAI-joint`: original 4,008-input atom16 encoding.
- `pKAI-joint-aa20`: 9,008-input atom16+amino-acid encoding.

The smooth variants use `PKAIEstimator` in smooth mode with the branch defaults:
0.5 Å cutoff window, 0.01 Å sorting temperature and CPU float64. The fork variants
use the released Python inference path, 15 Å cutoff, 250 neighbour slots, CPU
float32 TorchScript and the public API's 0.01-pKa rounding.

All four methods have exactly 21,807 matched state sites, 10,744 matched AB/free
site pairs and 4,827 interface pairs. These occur in 388 of the 400 complexes; the
remaining complexes have no shared eligible supported sites, rather than inference
failures. All 400 inference receipts completed without an error.

## PypKa agreement

| Estimator | State MAE | Paired-shift MAE | Interface paired MAE |
|---|---:|---:|---:|
| Regular pKAI | 0.2856 | 0.0874 | 0.1765 |
| smooth-pKAI | 0.2855 | 0.0851 | 0.1715 |
| pKAI-joint | 0.2748 | 0.0829 | 0.1691 |
| **pKAI-joint-aa20** | **0.2731** | **0.0824** | **0.1679** |
| pKAI+ | 0.5043 | 0.1210 | 0.2531 |
| smooth-pKAI+ | 0.5091 | 0.1209 | 0.2534 |

Paired cluster bootstrap against the matching released baseline gives:

| Candidate | State MAE change | Paired MAE change | Interface MAE change |
|---|---:|---:|---:|
| smooth-pKAI vs pKAI | -0.0001 [-0.0015, +0.0014] | -0.0023 [-0.0034, -0.0013] | -0.0050 [-0.0074, -0.0029] |
| pKAI-joint vs pKAI | -0.0108 [-0.0136, -0.0081] | -0.0044 [-0.0063, -0.0025] | -0.0074 [-0.0118, -0.0030] |
| pKAI-joint-aa20 vs pKAI | -0.0125 [-0.0156, -0.0095] | -0.0050 [-0.0070, -0.0030] | -0.0085 [-0.0129, -0.0039] |
| smooth-pKAI+ vs pKAI+ | +0.0048 [+0.0039, +0.0057] | -0.0000 [-0.0007, +0.0005] | +0.0003 [-0.0009, +0.0015] |

Changes are candidate minus baseline; negative values favour the candidate. The
fine-tuned models improve every metric on the held-out clusters, and the cluster
bootstrap intervals exclude zero. The amino-acid-aware model is numerically best,
although its advantage over `pKAI-joint` is small. Smooth features leave stock pKAI's
state error unchanged while slightly improving paired errors. Smooth pKAI+ slightly
worsens state MAE and does not materially change paired error.

## pH linkage proxy

The common-support linkage calculation reconstructs independent Henderson--Hasselbalch
curves from scalar pKas and evaluates
`log10[Kd(7.4)/Kd(6.1)]`. The signed difference is PypKa minus prediction.

| Estimator | Mean predicted ratio | Mean signed difference | 95% cluster-bootstrap CI | Mean absolute difference |
|---|---:|---:|---:|---:|
| PypKa reference | -0.0209 | — | — | — |
| Regular pKAI | -0.0336 | +0.0127 | [-0.0024, +0.0298] | 0.0537 |
| smooth-pKAI | -0.0300 | +0.0091 | [-0.0059, +0.0261] | **0.0524** |
| smooth-pKAI+ | -0.0316 | +0.0107 | [-0.0075, +0.0303] | 0.0643 |
| pKAI-joint | -0.0330 | +0.0121 | [-0.0039, +0.0288] | 0.0538 |
| pKAI-joint-aa20 | -0.0318 | +0.0109 | [-0.0060, +0.0276] | 0.0539 |

This linkage comparison contains 378 complexes and 10,744 common matched AB/free
sites. The signed-error intervals include zero for every pKAI variant. The fine-tunes'
clear site-level improvement therefore does not produce a measurable improvement in
the complex-level linkage proxy.

## Interpretation and limits

The joint checkpoints were trained with pKPDB/PypKa-derived and PINDER/pKAI-derived
labels. Their improvement against PypKa on held-out PINDER clusters measures teacher
retention and transfer, not independent experimental accuracy. Experimental pKa and
affinity validation remains necessary.

The fetched smooth_propka branch accepts only the two stock model names and pinned
stock weight hashes. The two fork fine-tunes were therefore benchmarked through their
released inference API. Testing the fine-tuned weights inside smooth_propka's
differentiable feature path requires an explicit compatible adapter; the aa20 model
also requires a differentiable 9,008-input feature implementation.

Artifacts:

- `_runtime/jax-Ka/pkabench/audits/pkai-feature-pinder-v1/`
- `_runtime/jax-Ka/pkabench/audits/kd-ratio-comparison-v2-pkai-feature/`
