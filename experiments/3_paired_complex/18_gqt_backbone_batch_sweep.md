# Backbone GQT batch-size sweep

The cleaned 5k backbone-only GQT is trained from scratch at batch sizes 4, 8,
16, 32, and 64. Every run uses seed 17, 20 epochs, learning rate 0.001, the
same group/complex sampling rule, full float32, and the same unaugmented clean
validation set. Final epoch 20 is selected in advance.

This is an operational speed–accuracy sweep. Fixing the epoch count and learning
rate means larger batches make fewer optimizer updates; the result therefore
captures the actual consequence of changing the configured batch size. The
report includes epoch time, total updates, measured peak VRAM, matched-site MAE,
and paired group-bootstrap differences against batch 8. No test data are used.
