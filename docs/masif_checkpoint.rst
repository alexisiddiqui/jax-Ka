MaSIF-search checkpoint selection
================================

The Docker ``sc05/all_feat`` checkpoint has nonfinite shape-index branch
parameters. The 2020-02-13 Python-trained revision has the same defect. The
upstream maintainers attribute it to nonfinite examples in the original
training data and retain the damaged weights for compatibility with their
precomputed descriptor databases:

https://github.com/LPDI-EPFL/masif-neosurf/issues/28

Legacy TensorFlow CPU maximum reduction suppresses NaNs; consequently finite
80D descriptors alone cannot validate the checkpoint. ``masif_checkpoint.py``
checks all checkpoint floating tensors, expected model names and shapes before
restoring. Native annotation and GPU reference extraction require an explicit
checkpoint. GPU inference also checks the exported weights and uses ordinary
maximum reduction. No weights are zero-filled, transplanted or initialized.

Replacement profile
-------------------

Use the author's complete original paper all-features MaSIF-search checkpoint,
from ``pablogainza/masif_paper`` commit
``05d15e6bdcee6dc8eff0047735c2db59edcf70fc``:

https://github.com/pablogainza/masif_paper/tree/05d15e6bdcee6dc8eff0047735c2db59edcf70fc/data/masif_ppi_search/nn_models/sc05/all_feat/model_data

SHA256:

* ``model.data-00000-of-00001``:
  ``4541e79570cdefe0e741359a0601e9b8391a136544c7c26070cde1ec59fa7133``
* ``model.index``:
  ``98c00fbe7b9b46efdee00179f2f996be7cdda8ef30985a0dfed1a187f896dcc0``
* ``model.meta``:
  ``49126028c5c8552ad0d18336c6f7b19855e88942406799d18dbca663d4a61a92``

This is a distinct model profile, not a repair of the six damaged arrays.
Retain it as a separately versioned checkpoint and regenerate all fingerprints;
old descriptors and original precomputed databases are not interchangeable.
The paper weights were trained using the older MATLAB geodesic preprocessing.
Our numerical checks of the current Python patches do not establish equivalent
task accuracy or reproduce the paper's benchmark scores.

``masif_checkpoint_validate.py`` runs strict tensor validation, verifies that
the damaged checkpoint is rejected, computes CPU references for 1,536 saved
real patches from six surfaces, checks all feature branches are active and
tests each feature's effect on descriptors. The resulting bundle is checked
against JAX GPU inference at several batch sizes. No training is involved.

Deployment
----------

The submission scripts select
``$root/masif-checkpoints/active/model`` by default, or the explicit
``MASIF_CHECKPOINT_PREFIX`` override. The active directory is selected only
after checkpoint and CPU/GPU validation. Direct Python entry points require
``--checkpoint /path/to/model``. Per-structure receipts and packed-release
provenance record the actual checkpoint hashes so mixed-model output is visible.

Historical pilot outputs remain diagnostic artifacts of the damaged checkpoint;
their preparation timings are reusable, while their fingerprints must be
regenerated. Full dataset annotation is a separate operation.

Validation result (2026-11-15)
------------------------------

All 98 checkpoint tensors and 66,080 model parameters are finite. All five
branches have nonzero activations on the 1,536 real-patch panel; reversing
shape-index signs changes descriptors by up to 2.408, whereas the damaged
model was insensitive. CPU and A40 descriptors agree to within 2.4e-6 across
batches 32, 128, 256, 512 and 1024. Transfer-inclusive throughput remains about
33,850 patches/s. The damaged checkpoint is rejected by the new guard.

An independent native pKPDB structure-to-packed-output smoke test generated
2,682 finite 80D fingerprints. The three reference-center checks matched
exactly, and re-reading the compressed release verified the selected checkpoint
hashes. CPU, GPU and integration jobs were 799784, 799788 and 799791.
The clean profile is selected by the runtime ``active`` symlink. No full data
annotation or model training was started.

Original preprocessing verification
-----------------------------------

The original published MATLAB preprocessing was subsequently run on the same
saved meshes as the Python pilot. An unmodified full run produced coordinates
for 2,682 vertices in 17.84 seconds after startup. A selected-center variant
matched it exactly at the paper loader's float32 precision. Across three pKPDB
surfaces (96 matched centers), the original feature pipeline produced finite
fingerprints, but differed from the newer Python pipeline by 15.7--17.3 percent
in relative descriptor L2 norm. Mean shape-index absolute differences were
approximately 0.37; classical full-patch MDS and radius-shrinking patch caps
also differ from the newer Python construction.

The comparison additionally exposed an inherited normalization defect: the
container's Python implementation clips positive electrostatics twice and
never clips negative values. The original paper function clips both bounds.
The local adapter now uses the original two-sided clipping with a regression
test. Native receipts identify this as Python geometry with corrected paper
electrostatics; it is not a complete original-preprocessing backend.

Use the original curvature, coordinates, patch extraction and feature assembly
with the original paper weights. At the time of this initial verification the
production loader had not yet been switched. The comparison holds surfaces and chemical
fields fixed; it does not reproduce every component of the 2019 PDB-to-surface
environment. The earlier CPU/GPU agreement test establishes forward-pass
equivalence only. The runtime report is
``masif-paper-preprocessing-v1/verification.json`` (MATLAB job 799801,
comparison job 799825). No task training or full annotation was launched.

A separate native pKPDB PDB-to-fingerprint smoke (job 799835) also passed using
the original surface/protonation/APBS functions, original FLANN 1.6.14 binary,
MATLAB geometry and original feature assembly. It produced 2,680 finite 80D
fingerprints with electrostatics features bounded to [-1,1]. Python 2 source
was syntax-migrated into an isolated compatibility directory; no scientific
algorithms were substituted. Existing frozen author-container binaries were
used, so this validates the published source pipeline rather than bitwise
reproduction of the complete 2019 environment.

The batch loader now defaults to the verified original-paper backend and uses
the matching runtime/checkpoint manifest. CPU and persistent GPU inference,
training-chain selection, bounded scratch and packed publication are integrated;
see ``masif_paper_batch.rst``. No full dataset annotation has been launched.
