# pKAI batch-size sweep

The scratch pKAI control is trained on the cleaned 5k cohort at batch sizes 64,
128, 256, 512, and 1024. Batch 256 is the native recipe control. Every run uses
seed 17, Adam at 1e-6 with 1e-4 weight decay, the native dropout schedule, full
float32, and validation-MSE early stopping with min-delta 0.001 and patience 5.

The report uses identical matched validation sites and provides group-bootstrap
MAE intervals, paired differences against batch 256, epoch timing, selected
epoch, and measured PyTorch peak allocated/reserved VRAM. No test data are used.
