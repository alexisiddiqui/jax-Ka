# oGQT auxiliary-loss pilot

## Registered question

Does direct supervision of free-state burial and bound-state interface proximity improve held-out pKa prediction? Auxiliary accuracy is diagnostic; checkpoints are selected only by unweighted state MAE plus interface paired-shift MAE.

## Arms

| Arm | Auxiliary multiplier |
|---|---:|
| baseline | 0 |
| low | 0.1 |
| standard | 1 |

All arms share seed 17, initial parameters, a common pKa-only first epoch, optimizer state, batch plans, and the 10-epoch compressed cosine schedule. The warm-up checkpoint remains eligible for selection.

## Data

- Training: the nested 10% PINDER `pool-v2` subset, capped at 768 residues and excluding `chain_70_to_validation` rows.
- Validation: the frozen 400-complex PINDER cohort.
- Training and validation clusters must be disjoint.
- Supervision remains restricted to the existing valid-site mask.

## Targets and loss

The two sigmoid heads use the final site embedding and predict

- normalized burial: `(w_burial - 0.4) / 0.6 = 1 - RSA_free` from the free branch;
- normalized interface proximity: `(w_interface - 0.05) / 0.95` from the bound branch.

The primary loss remains unweighted state MSE plus paired-shift MSE. After the common warm-up, 32 fixed training-only batches calibrate fixed auxiliary coefficients from shared-parameter gradient norms. Calibration excludes every output head and targets 10% of the primary shared-gradient norm per auxiliary in the standard arm.

## Decision

Run all three arms through epoch 10. An auxiliary improvement of at least 0.001 triggers confirmation against baseline with seeds 29 and 43; if low and standard differ by less than 0.001, prefer low. This is a ten-epoch screen; a longer optimization study is a separate experiment.

The report includes primary distance and RSA strata, paired-complex bootstrap intervals, auxiliary correlations and errors, interface ranking/overlap against the stored binary interface flag, structural-reference rankings, runtime, and peak memory. A later audit established that this stored flag is partner heavy-atom distance <=10 A, rather than residue delta-SASA >10 A2 as the original plan stated.

The full 93,293-complex definition audit is in `audits/interface-definition-v1/summary.json`. Among teacher-labelled training sites, the stored 10 A zone marks 497,872 sites as interface. Residue delta-SASA >10 A2 marks 171,481 sites whose residue type is known from the stored tokens; 11,878 interface terminal sites have no inferable residue type, giving a possible total of 171,481-183,359. Thus delta-SASA would retain about 35-37% of the current interface-labelled sites. It is a strict subset on every site for which both definitions can be evaluated. This changes interface-stratified reporting, not the number of sites in the all-site state or paired losses.

## Runtime location

`_runtime/jax-Ka/pkabench/training/ogqt-auxiliary-pilot-v1`

## Result

The registered cohort contained 2,968 training structures from 1,349 clusters and the frozen 400-complex validation cohort. It contributed 132,613 supervised sites in total, including 41,393 interface sites. All data, model, gradient, masking, checkpoint, and matched-batch-plan gates passed.

| Arm | State MAE | Interface paired MAE | Selection | Burial MAE | Burial Spearman | Interface AP |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.4635 | 0.2967 | **0.7602** | 0.1907 | 0.0942 | 0.6221 |
| low | 0.4651 | 0.2991 | 0.7642 | 0.1307 | 0.6485 | 0.6607 |
| standard | 0.4637 | 0.2965 | 0.7603 | **0.1059** | **0.7784** | **0.6760** |

The standard auxiliaries learned the intended structural quantities, but did not improve the primary pKa selection score. Its paired-bootstrap delta relative to baseline was +0.00009 with a 95% interval of [-0.00337, 0.00348] and P(improved)=0.480. Low strength was slightly worse: +0.00394 [-0.00002, 0.00807], P(improved)=0.026. The registered 0.001 improvement gate was not met, so no extra-seed confirmation is triggered and the pKa-only baseline remains selected.

Each arm selected epoch 9, used 0.845 GiB peak GPU memory, and took about 28 minutes including its cold first epoch and final predictions. The common warm-up and calibration took 35.4 minutes once. Calibration fixed `lambda_burial=0.09783` and `lambda_interface=0.02920`.

### Interface discrimination and counterfactual errors

AP is complemented by ROC-AUC because the validation set has a high interface-residue prevalence. Standard supervision improved equal-complex ROC-AUC from 0.520 to 0.617 (95% complex-bootstrap interval 0.588–0.643); pooled ROC-AUC improved from 0.507 to 0.577. Discrimination therefore improved but remains modest.

At the standard arm's descriptive validation Youden threshold of 0.197, the native equal-complex false-positive and false-negative rates were 0.313 and 0.594. Removing every cross-chain residue and site edge supplies a separated-structure counterfactual equivalent to moving partners beyond the model's 20 Å cutoff. Treating every separated residue as negative gave an equal-complex false-positive rate of 0.252. Native positive residues moved from mean score 0.200 bound to 0.173 separated, while native negatives barely moved (0.187 to 0.186). Thus the head uses partner context preferentially at true interfaces, but the positive and negative score distributions overlap substantially.

The 0.5 cutoff is not meaningful for this regression head: only one standard-arm residue exceeded it. The Youden counts are descriptive because their threshold was selected and evaluated on the same validation cohort; AP and ROC-AUC remain the threshold-free headline metrics.

### Pair-geometry noise audit

The standard auxiliary checkpoint was evaluated on all 400 validation complexes after perturbing the graph's pairwise geometry directly. Labels and adjacency were fixed. Distance noise was symmetric Gaussian error on residue and site-pair distances, followed by recomputing the 16-bin RBF and cutoff switch. Angle noise was a reciprocal Gaussian SO(3) perturbation of the site-pair relative-orientation matrices; direction features were left unchanged. This is a feature-sensitivity test, not a physically realizable coordinate ensemble.

| Condition | State MAE | Interface paired MAE | Interface AP | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Native | **0.46375** | 0.29654 | 0.67605 | 0.57684 | 2,294 | 2,643 |
| distance sigma=0.5 A | 0.48324 | 0.29304 | 0.67768 | **0.58006** | 1,915 | 2,954 |
| distance sigma=1.0 A | 0.48833 | 0.29352 | 0.67352 | 0.57760 | 1,923 | 2,954 |
| angle sigma=5 degrees | 0.46379 | 0.29667 | 0.67515 | 0.57687 | 2,294 | 2,645 |
| angle sigma=10 degrees | 0.46396 | 0.29640 | **0.67620** | 0.57726 | 2,289 | 2,649 |
| combined 0.5 A / 5 degrees | 0.48355 | **0.29170** | 0.68023 | 0.57930 | 1,958 | 2,964 |
| combined 1.0 A / 10 degrees | 0.48744 | 0.29589 | 0.67912 | 0.57769 | 1,935 | 2,949 |

Distance error materially worsened absolute-state prediction by 0.0195-0.0246 pKa. Orientation error through 10 degrees changed state MAE by only 0.00021 and left interface AP and ROC-AUC effectively unchanged. The present checkpoint is therefore sensitive to pair distances but barely uses the explicit relative-orientation matrix. Small apparent improvements in paired MAE under random noise are not evidence of a better model; they accompany worse state MAE and were measured for one deterministic noise draw.

FP/FN counts use the descriptive native Youden threshold 0.197. Distance noise reduced false positives while increasing false negatives, consistent with a downward or compressed interface response rather than improved discrimination; threshold-free AP and ROC-AUC barely changed.

### Coordinate-level rotation invariance

The selected standard checkpoint was tested on 50 held-out complexes with three deterministic random rotations each. Graphs were rebuilt from transformed atomic coordinates. Joint AB rotations tested both heads in both branches; independent A/B rotations were evaluated only through the free branch after compacting it to intrachain edges. The latter is the relevant invariance because independent rotations change the bound interface.

| Transform and output | Mean absolute difference | 99th percentile | Maximum |
|---|---:|---:|---:|
| Joint rotation, bound interface | 7.07e-8 | 3.58e-7 | 1.25e-6 |
| Joint rotation, free interface | 6.51e-8 | 3.13e-7 | 8.49e-7 |
| Joint rotation, bound burial | 1.53e-7 | 8.34e-7 | 2.44e-6 |
| Joint rotation, free burial | 1.44e-7 | 7.75e-7 | 2.50e-6 |
| Independent partner rotations, free interface | 6.45e-8 | 3.15e-7 | 1.10e-6 |
| Independent partner rotations, free burial | 1.41e-7 | 7.75e-7 | 2.80e-6 |

All 28,134 comparisons passed the registered 1e-5 maximum-difference gate. The residual differences are consistent with float32 coordinate/frame reconstruction and GPU reduction order. No rotation augmentation is required to repair invariance in the current auxiliary heads.

### Auxiliary-head feature attribution

The two auxiliary heads share the complete residue and site-attention stack, so raw attention weights are not head-specific. Two complementary analyses were therefore run: deterministic within-complex feature permutation on all 400 validation complexes, and three Rademacher randomized-Jacobian probes on 50 held-out complexes. Permutation estimates causal dependence under a distribution-preserving disruption; gradient-times-input energy measures local sensitivity around native structures. Neither should be read as mechanistic proof in isolation.

#### Permutation: increase in target MAE

| Feature disrupted | Burial delta MAE | Interface delta MAE |
|---|---:|---:|
| Site-pair distance RBF | **+0.12677** | **+0.03782** |
| Residue-pair distance RBF | +0.12385 | +0.01657 |
| Residue identity | +0.07651 | +0.02822 |
| Site directions | +0.04396 | +0.00965 |
| Same-residue site flag | +0.04167 | +0.00709 |
| Site relative orientation | +0.03615 | +0.00789 |
| Site type | +0.02330 | +0.00039 |
| Residue directions | +0.01881 | +0.00125 |
| Remove cross-chain context | +0.00000 | +0.01040 |
| Residue same-chain flag | +0.00019 | +0.00036 |
| Site same-chain flag | -0.00006 | +0.00030 |
| Sequence separation | -0.00021 | +0.00007 |

Native normalized-target MAE was 0.10595 for burial and 0.28719 for interface. As expected, removing cross-chain context has exactly zero effect on the free-branch burial prediction. It degrades the bound interface head, although less than disrupting distance or residue identity.

#### Local gradient-times-input energy share

| Feature family | Burial share | Interface share |
|---|---:|---:|
| Residue-pair distance | **31.55%** | 14.50% |
| Residue identity | 19.57% | **37.60%** |
| Frame-valid channel | 18.88% | 23.82% |
| Site-pair distance | 11.11% | 9.76% |
| Same-residue site flag | 10.43% | 6.06% |
| Site direction | 3.20% | 3.28% |
| Site orientation | 3.19% | 3.28% |
| Residue direction | 1.07% | 1.06% |
| All remaining differentiable features | 1.01% | 0.65% |

The frame-valid input is nearly constant over usable residues, so its large gradient-times-input share chiefly indicates a sensitive constant channel and is not evidence that frame failures discriminate the targets. Site type and hard cross-chain masks are integer/boolean and are covered only by the permutation/intervention analysis.

Together, the results show that burial is predominantly a distance-and-residue-chemistry prediction. Interface also uses these features, but cross-chain removal has a measurable causal effect. Orientation is used: fully permuting it degrades both heads, while the separate 5-10 degree noise audit has negligible effect. Thus oGQT is locally robust to small angular errors but does not ignore orientation entirely.

![Interface ROC curves](../../../_runtime/jax-Ka/pkabench/training/ogqt-auxiliary-pilot-v1/plots/interface_roc.png)

![Native and separated-structure errors](../../../_runtime/jax-Ka/pkabench/training/ogqt-auxiliary-pilot-v1/plots/interface_fpfn_augmentation.png)
