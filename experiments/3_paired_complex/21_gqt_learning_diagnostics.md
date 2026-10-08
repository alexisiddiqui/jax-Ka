# Backbone-only GQT learning diagnostics

This is the parsimonious follow-up to the data-alignment audit. It uses the completed 49,709-parameter backbone-only batch-8 checkpoint and performs no architecture selection.

1. Compare shift calibration on the 5k training cohort and frozen validation split. On validation, measure prediction changes after retaining only self edges, removing titratable context, removing four distance shells, removing non-query residue identities, or removing edge geometry.
2. Measure aggregate parameter-gradient norms and gradient cosine similarity in five native-coupling strata. Coupling annotations only stratify frozen validation results and are never training targets.
3. Train the same architecture from scratch on 64 distinct training components without dropout or masking. Stop at MAE 0.05, after a 100-epoch plateau, or at 500 epochs.

The experiment runs in full float32 on the `comp1400` A40 with eight support CPUs and 16 GiB RAM. It reads no test data. The final atomic artifact is `_runtime/jax-Ka/pkabench/audits/gqt-learning-diagnostics-v1`.

## Result

Job `739868` passed all 13 tests and completed the three stages. The frozen checkpoint has a shift calibration slope of 0.618 on its 207,064 training sites and 0.578 on 7,986 validation sites, with predicted/teacher shift standard-deviation ratios of 0.747 and 0.768. Compression therefore already exists on the training cohort and is not principally a validation-only generalization effect. The output nonlinearity is not saturated: the mean derivative of the `8*tanh` residual head is 7.83/8 on training and 7.84/8 on validation, with zero sites below the registered 0.8 derivative threshold.

The model uses environmental context. Retaining only self edges raises validation site-weighted MAE from 0.647 to 1.158 and changes predictions by 1.018 pKa on average. Removing titratable neighbours raises MAE to 0.896, changes predictions by 0.687 pKa, and reduces the shift slope to 0.093. Removing the 6–10 Å or 10–15 Å shells causes the largest individual distance-shell degradations; removing 15–20 Å has a smaller effect. The zero-geometry ablation is strongly out of distribution and is retained as a sensitivity result rather than a calibrated measure of geometry importance.

Gradients do not vanish for strongly coupled sites. Total and block-level gradient norms increase in the ≥2 kBT stratum; the head/query/first-encoder norms are 5.58/0.875/1.61 there, versus 1.90/0.192/0.280 below 0.25 kBT. However, the ≥2 kBT aggregate gradient opposes every lower-coupling stratum: cosine similarities range from −0.45 to −0.78. This indicates optimization tension between weakly and strongly coupled examples at the frozen solution. Coupling strata are also correlated with residue group and target magnitude, so this is diagnostic rather than proof of a causal coupling-specific conflict.

The 64-complex train-only model recovered the shift range: its final slope was 1.039 and correlation 0.995. It did not reach the pre-registered 0.05 MAE target. The best observed MAE was 0.1118 at epoch 360; the constant-learning-rate run oscillated and stopped by the 100-epoch plateau rule at epoch 460 with MAE 0.1264. Thus the architecture can express much more variation than the full-cohort checkpoint, while constant-rate optimization and/or finite capacity prevents exact memorization.

The parsimonious next experiment is a longer run on the same cleaned 5k backbone-only cohort with a decaying learning rate, plus target-shift-stratified reporting. Explicit site tokens remain justified for the small same-residue collision class, but they are not the first explanation for the broad compression observed here.
