GPU construction from compact augmentation inputs
=================================================

``pkatrain.augmentation_online_features`` compares the existing expanded cache
with a compact cache retaining backbone coordinates, node/site chemistry,
canonical maps and exact precomputed residue/site neighbour lists. It removes
the 20-/34-column edge tensors and both distance switches. A JAX function
reconstructs those fields inside the forward/backward program. Neighbour search
is excluded. NumPy defines the independent correctness reference.

The compact host loader uses the same production size capacities, views and
branch identities as the expanded loader. Native PINDER bound inputs remain
native; only the selected free partner's backbone and topology change. Existing
native stores are read unchanged, including their expanded arrays, which are
discarded before the compact host-to-device copy. All branch geometry is then
constructed on the GPU. Masks and topology must match exactly, and active
features, objective and parameter gradients are compared numerically.

The isolated pilot uses one GH200, batch 16, workers 16, prefetch 2, the frozen
base-A width64/ff128 orientation+dihedrals/dropout0.1 protocol and production
Triton attention. Each arm has two alternating-order repeats of 20 synchronised
batches. Separate probes measure geometry materialisation and resident fused
forward/backward compute. No optimiser or parameter update is performed.

The final comparison enables device prefetch for both arms, matching production
whole-batch loading. Across eight pKPDB/PINDER size cases, construction alone
takes 0.30--2.73 ms per 16-structure batch, and the fused resident forward/backward
step costs 1.8--5.7 percent more. End-to-end throughput is nevertheless
1.14--1.82 times the expanded-cache path because host assembly and transfer
move fewer bytes. Active-feature discrepancies are at most 2.56e-6; worst relative
loss and gradient-L2 differences are 3.08e-5 and 1.90e-5. All checks passed.
The final job was 7810748 and took 2 minutes 53 seconds; all sequential diagnostic
and follow-up GPU jobs together consumed 9 minutes 15 seconds, one GPU at a time.

Geometry dot products explicitly request full float32 precision. Default GPU
dot precision produced approximately 7e-4 discrepancies in directions and
relative rotations; full precision removes this discrepancy without changing
model arithmetic. The initial diagnostic and subsequent corrected runs are
preserved separately. Extraction uses a unique directory beneath ``/tmp``;
shared storage receives only one overlay archive and aggregate output files.

``_HPC/submission/augmentation-online-isambard.sbatch`` runs standalone CPU
regressions and the GPU pilot. ``benchmarks/augmentation_online_report.py``
generates the timing table from the JSON report and a separate stratum-calibrated
storage projection. The 1,272-view pilot packs to about 57.7 MB instead of
4.13 GB; the full compact reader is projected at about 21 GB including metadata.
This is a projection, not a built or activated compact release. The working set
is warm, and cold full-dataset random I/O remains unmeasured.
