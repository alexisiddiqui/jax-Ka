# Converged explicit-shift GQT size sweep

The fixed-20-epoch screen left larger GQTs less optimized: final training MSE
was 0.552, 0.584 and 0.614 for the 50k, 200k and 800k models. Continue all
three exact checkpoints through epoch 100 before drawing a capacity conclusion.

Preserve parameters, Adam moments and count, sampling RNG, cleaned 5k cohort,
frozen component split, batch membership, batch size 8, strict backbone input,
explicit historical-pKPDB-shift objective and validation reporting. Decay the
learning rate per update from `1e-3` after epoch 20 to `1e-5` at epoch 100.
Activate the verified capped `N=128/K=64/Q=current` capacity policy and the
read-only mmap graph store; these change padded execution only.

Epoch 100 is selected in advance. Report training MSE, group-macro validation
MAE with component-bootstrap intervals, learning curves and measured GPU time.
No test data are read. Select the capacity knee only after all three runs pass.

## Post-hoc futility stop

The three runs were stopped after every size had passed epoch 54. At that common
epoch, training MSE improved strongly with capacity while validation MAE was
worse than the earlier epoch-13/20 minima for every model. This abandons the
fixed-epoch-100 endpoint and must not be reported as its result. All epochs are
checkpointed and resumable. A separate futility report compares the models at
their latest common epoch and records the resource decision without selecting
an epoch-100 checkpoint.
