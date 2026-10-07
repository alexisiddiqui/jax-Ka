# Cleaned 5k backbone GQT cosine continuation

Resume the 49,709-parameter batch-8 backbone-only model from its pre-registered epoch-20 checkpoint and train through epoch 100. Parameters, Adam moments/count, sampling RNG, cleaned 5k cohort, frozen validation split, graph inputs, batch size and objective are inherited exactly.

The learning rate decays from `1e-3` after epoch 20 to `1e-5` at epoch 100 with a per-update cosine. A runtime scalar multiplies Adam updates so the optimizer state tree remains exactly compatible with the inherited constant-rate checkpoint; Adam is not restarted.

Validation is reporting-only and the final epoch is selected in advance. Report site calibration on train and validation overall and in absolute teacher-shift bins `[0,0.5)`, `[0.5,1)`, `[1,2)` and `>=2` pKa. The bins do not alter sampling or loss weights. No test data are read.
