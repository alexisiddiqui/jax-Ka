# GQT train-shift weighting

Resume the cleaned 5k backbone-only, batch-8 GQT from the same fixed epoch-20 checkpoint used by the unweighted 100-epoch continuation. Preserve the model, data, component split, graph construction, sampler, batch size, optimizer state, RNG state and cosine learning-rate schedule.

Change only the scalar training objective. Place each training site into an absolute teacher-shift bin `[0,0.5)`, `[0.5,1)`, `[1,2)` or `>=2` pKa relative to its model-compound value. Give each bin equal total weight using inverse training-site frequency, normalized to mean site weight one. Normalize the weighted site MSE within each structure, then average structures, preserving the existing structure-level weighting. Validation labels must not determine weights.

Select epoch 100 in advance. Report the same group-macro validation metrics and train/validation calibration by shift bin as the unweighted arm. No test data are read. This arm tests whether target imbalance causes the compressed large-shift predictions; it does not simultaneously change capacity, sampling or graph construction.
