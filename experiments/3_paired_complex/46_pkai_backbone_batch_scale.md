# Backbone pKAI critical-batch audit

This mirrors the five-epoch oGQT batch audit for the native 3,608,001-parameter
pKAI MLP using the already prepared strict-backbone feature tensor.  Two
initializations are tested: PyTorch/native random initialization (scratch) and
the released pretrained `pKAI_model.pt` checkpoint.

Each initialization uses batch sizes 32, 64, 128 and 256.  Learning rates obey
`1e-6 * sqrt(batch/64)`; Adam beta and epsilon, weight decay `1e-4`, native
dropout, features, train/validation sites and shuffle seed are held fixed.  The
primary comparison is fixed epoch 5.  No test data are read.

Gradient noise is measured separately at each initialization from 32 gradients
of 32 randomly sampled training sites.  Native dropout remains active, so the
estimate includes both site-sampling and dropout noise.  The batch-gradient
covariance is rescaled to per-site covariance before computing
`B_noise = tr(Sigma) / |G|^2`.
