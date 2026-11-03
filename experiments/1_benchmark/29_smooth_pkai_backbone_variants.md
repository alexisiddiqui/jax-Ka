# smooth_propka backbone-only pKAI variants

## Question

Test the two backbone-only pKAI estimators added on smooth_propka
`feature/pkai-loss` at `616900a4aa89d8ed99883b05faac4c30d6b5420f`:

| Label used here | Fork checkpoint | Context |
|---|---|---|
| `pKAI-backbone-15A` | `pKAI-joint-aa20-backbone` | backbone N/O atoms within 15 Å of the site Cα; 250 slots |
| `pKAI-backbone-20A` | `pKAI-joint-aa20-backbone-r20s540` | backbone N/O atoms within 20 Å of the site Cα; 540 slots |

The pinned pKAI fork is `49ab002684450c55f1e2364dd1b1e9a6e09d6071`.
Checkpoint SHA-256 hashes are frozen in the run manifest. Both models retain amino-acid
identity channels, but no side-chain coordinates enter their features.

## Gates and protocol

Ten focused tests passed before the benchmark. They covered agreement with the fork's
native hard-feature oracle, smooth inference, rigid-motion invariance, batching, feature
checkpointing, the default variant, and the requirement that moving side-chain atoms
changes neither backbone features nor their gradients.

One frozen PINDER complex then passed an end-to-end AB/A/B smoke test for both variants.
The full evaluation processed all 400 frozen validation complexes with no pipeline
failures. Metrics use the same current-PypKa teacher, site masks and exact residue keys as
the other estimator comparisons. There are 21,807 matched state observations and 10,744
matched AB/free site pairs from 388 complexes; 4,827 pairs are interface sites. The other
12 processed complexes contain no supported matched sites after the common filters.

smooth_propka ran in smooth mode with a 0.5 Å cutoff window, temperature 0.01 Å and CPU
float64. No parameters were updated. The 399 non-smoke Slurm tasks used two CPUs each;
median task time was 8 seconds (range 4–17 seconds), and median peak RSS was 479 MiB.

## Results

| Estimator | State MAE | Paired-shift MAE | Interface paired MAE |
|---|---:|---:|---:|
| Regular all-atom pKAI | 0.286 | 0.087 | 0.176 |
| Fine-tuned pKAI-joint | 0.275 | 0.083 | 0.169 |
| **Fine-tuned pKAI-joint-aa20** | **0.273** | **0.082** | **0.168** |
| pKAI-backbone-15A | 0.587 | 0.180 | 0.381 |
| pKAI-backbone-20A | 0.589 | 0.185 | 0.388 |
| Earlier backbone-pretrained pKAI ensemble | 0.660 | 0.181 | 0.387 |

The fine-tuned all-atom `pKAI-joint-aa20` checkpoint is the best pKAI model on all
three metrics. Against regular all-atom pKAI, it reduces state, paired and interface
MAE by 0.0125, 0.0050 and 0.0085 pKa respectively; the previously computed paired
cluster-bootstrap intervals exclude zero.

Against regular all-atom pKAI, the 15 Å backbone variant increases MAE by 0.302 pKa for
states, 0.092 for paired shifts, and 0.205 at interfaces. All three 95% cluster-bootstrap
intervals exclude zero. It therefore does not recover the accuracy supplied by side-chain
geometry.

Against the fine-tuned all-atom `pKAI-joint-aa20`, the 15 Å backbone variant increases
state, paired and interface MAE by 0.314, 0.097 and 0.213 pKa. Their 95% intervals are
`[+0.299, +0.328]`, `[+0.086, +0.109]` and `[+0.197, +0.231]`, respectively.

Against the earlier backbone-pretrained ensemble, the 15 Å variant improves state MAE by
0.073 pKa (95% CI `[-0.082, -0.064]`). Its paired improvement is only 0.002 pKa
(`[-0.006, +0.003]`) and its interface improvement is 0.006 (`[-0.015, +0.004]`), so those
paired differences are not resolved.

The 20 Å model is worse than the 15 Å model for paired shifts by 0.0058 pKa (95% CI
`[+0.0040, +0.0076]`) and at interfaces by 0.0066 (`[+0.0025, +0.0108]`). Their state MAEs
are unresolved: 20 Å minus 15 Å is +0.0019 (`[-0.0007, +0.0046]`). The extra radius and
slots do not help on this cohort.

## pH-linkage proxy

For scalar pKa outputs, Henderson–Hasselbalch site curves were integrated analytically to
obtain `log10[Kd(7.4)/Kd(6.1)]`. Every estimator uses the same 378 complexes and 10,744
complete AB/free site pairs.

| Estimator | Mean PypKa ratio | Mean predicted ratio | Mean reference − prediction | Mean absolute difference |
|---|---:|---:|---:|---:|
| Regular all-atom pKAI | -0.0209 | -0.0336 | +0.0127 | 0.0537 |
| Fine-tuned pKAI-joint | -0.0209 | -0.0330 | +0.0121 | 0.0538 |
| Fine-tuned pKAI-joint-aa20 | -0.0209 | -0.0318 | +0.0109 | 0.0539 |
| pKAI-backbone-15A | -0.0209 | -0.0615 | +0.0406 | 0.1023 |
| pKAI-backbone-20A | -0.0209 | -0.0705 | +0.0497 | 0.1081 |

The signed 95% cluster-bootstrap intervals are `[+0.0142, +0.0685]` for 15 Å and
`[+0.0225, +0.0782]` for 20 Å. Both backbone variants overstate the magnitude of the
negative ratio relative to PypKa on average. This is midpoint-derived linkage, not native
PypKa Monte Carlo curve linkage.

## Decision

Use `pKAI-joint-aa20-backbone` (15 Å/250 slots) as smooth_propka's backbone-only default.
It is consistently at least as good as the 20 Å/540-slot model and is smaller. Treat its
accuracy as a backbone-only compromise. The fine-tuned all-atom `pKAI-joint-aa20` is the
best pKAI estimator for state and binding-shift accuracy; its complex-level linkage error
is similar to regular pKAI. The backbone default remains useful when side-chain-independent
features and gradients are required.

## Artifacts

- Frozen run manifest and summary: `audits/pkai-backbone-pinder-v1/`
- Paired cluster-bootstrap comparison: `audits/pkai-backbone-pinder-v1/paired-comparison.md`
- Common-support linkage report: `audits/kd-ratio-comparison-v3-pkai-backbone/report.md`
