# pKAI scratch/pretrained shift-weighting comparison

Train four pKAI arms on the same cleaned 5k pKPDB cohort: random initialization with uniform or inverse-frequency shift loss, and released pKAI initialization with uniform or inverse-frequency shift loss.

pKAI natively predicts the signed shift relative to the historical pKPDB `PK_MOD` constants: ASP 3.79, GLU 4.20, HIS 6.74, CYS 8.67, TYR 9.59 and LYS 10.46. The installed encoder does not provide termini, so all comparisons use its common 7,778-site validation support.

All arms use the native 4008→800→400→200→1 architecture and dropout, batch size 256, Adam at `1e-6`, weight decay `1e-4`, full float32, and validation-shift-MSE early stopping with `min_delta=0.001`, patience 5 and a 200-epoch cap. The pretrained arms include the released checkpoint as selectable epoch 0. Within each initialization, only training-loss weighting differs.

Weights are derived from clean training labels only for absolute-shift bins `[0,0.5)`, `[0.5,1)`, `[1,2)` and `>=2`. Report overall group-macro validation MAE with group-bootstrap intervals and, for each bin, pooled site MAE plus group-macro MAE and intervals. No test data are read.
