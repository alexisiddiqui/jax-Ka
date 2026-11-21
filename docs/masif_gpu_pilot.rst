Frozen MaSIF GPU inference pilot
================================

``benchmarks/masif_gpu_pilot.py`` reconstructs 256 real patches from each of six
native pilot surfaces and exports the original TensorFlow CPU descriptors and
all 66,080 trained parameters. The reference covers small/medium/large pKPDB,
separated PINDER and bound PINDER surfaces. Patch geometry and normalized
chemistry come from the saved native meshes; identical float32 input arrays are
used for both inference backends. No model training is performed.

The JAX GPU implementation retains the original five feature branches, 80
Gaussian filters, 16 rotations, rotation maximum, ReLU and final 400-to-80
linear map. Full float32 matrix-product precision is explicit. Checkpoint and
reference-bundle hashes are verified before testing every reference patch at
batches 32, 128, 256, 512 and 1024. Timings separate resident inference from host
conversion, transfer, inference and descriptor return.

One A40 measured approximately 41,400 resident and 33,800 transfer-inclusive
patches/s at batch 1024. Maximum absolute difference from the frozen CPU
reference was below 5e-6. Combined with the prior CPU preprocessing measurements,
the isolated native-unit set projects to about 7.4 GPU-hours plus 212 CPU
preprocessing wall-hours at ten preprocessing workers. Including bound PINDER
gives about 9.5 GPU-hours and 272 CPU preprocessing hours. Four GPUs reduce only
the neural phase; CPU preparation still determines the approximately 9-day or
11--12-day capacity estimate. These are projections; the complete mixed
producer-consumer pipeline and four-GPU throughput are not measured.

Checkpoint finding
------------------

The original bundled checkpoint contains NaNs in all six shape-index branch
parameter arrays. The CPU descriptors nevertheless remain finite because
TensorFlow's maximum reduction ignores NaNs: this branch has zero post-ReLU
activations. An independent TensorFlow probe confirms that modifying shape-index
inputs changes no descriptors. The GPU implementation explicitly matches the
same maximum-reduction semantics; weights are not sanitized or retrained.

Numerical equivalence therefore establishes compatibility with the measured
legacy checkpoint, not a valid five-feature model. Before full annotation,
obtain or validate a checkpoint without this defect. The native CPU timing
report is annotated with the finding. ``masif_checkpoint_probe.py`` preserves
the operator tests and per-branch activation evidence.

The entry points now reject that checkpoint. A separately versioned clean
original-paper model and strict validation procedure are described in
``masif_checkpoint.rst``. The throughput figures above remain historical
measurements of the damaged model; new checkpoint results are stored separately.
