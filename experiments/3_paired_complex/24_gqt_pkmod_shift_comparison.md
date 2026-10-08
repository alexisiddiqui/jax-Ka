# GQT explicit pKPDB-shift comparison

Train two backbone-only GQT arms from the same seed-17 initialization on the same cleaned 5k pKPDB cohort and frozen component split. The network head predicts the signed environmental shift directly:

`target_shift = deposited_pKa - historical_pKPDB_PK_MOD[group]`

Absolute pKa is reconstructed as `PK_MOD[group] + predicted_shift` only for scoring. Historical constants are ASP 3.79, GLU 4.20, HIS 6.74, CYS 8.67, TYR 9.59, LYS 10.46, NTERM 7.99 and CTERM 2.90. Historical pKPDB supplies no ARG reference and this cohort has no ARG labels; registration fails if one appears.

Both arms use 20 epochs, batch size 8, Adam at `1e-3`, identical component-uniform sampling, identical batches and the existing strict backbone 20 Å graphs. The unweighted arm gives all eligible sites equal weight within a structure. The weighted arm gives the four absolute-shift bins `[0,0.5)`, `[0.5,1)`, `[1,2)` and `>=2` equal total training weight using inverse train-only frequency, then normalizes within each structure.

Epoch 20 is fixed in advance. Validation is reporting-only. Compare group-macro validation metrics and train/validation shift calibration by bin. No test data are read.
