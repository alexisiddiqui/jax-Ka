# GQT and pKAI parameter-size sweep

Before paired siamese training, measure whether the 49,709-parameter backbone
GQT is capacity limited. Train GQT at approximately 50k, 200k, 800k and 3.2M
parameters. Train scratch pKAI controls at approximately 50k, 200k, 800k and
its native 3.608M parameters by preserving its 4:2:1 hidden-width ratio.

All runs use the cleaned 5k pKPDB cohort, the frozen component split, seed 17,
full float32, and the explicit signed shift from historical pKPDB `PK_MOD`.
GQT uses the established batch-8 fixed-20-epoch protocol. pKAI uses the best
measured batch size of 64 and its native validation-shift-MSE early stopping
rule. This keeps data and targets matched while retaining the validated
optimizer recipe for each architecture. Comparisons of size are made within
model family.

Reuse the exact completed 49,709-parameter GQT and 3,608,001-parameter pKAI
controls. GQT uses an adaptive cost gate: compare 50k with 200k, run 800k as a
third point, and run 3.2M only if the lower tiers show a meaningful capacity
trend. Live timing at 200k is used to project the cost of the larger tiers.
Report parameter count, epochs, GPU time, group-macro validation MAE and its
component-bootstrap interval. No test data are read. The selected GQT capacity
becomes the initialization for paired AB/free siamese training.
