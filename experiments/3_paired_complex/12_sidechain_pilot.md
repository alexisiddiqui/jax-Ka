# Side-chain input ablation

The initial GQT pilots used backbone local frames and CA radius graphs, residue
identity, terminal flags and a disulfide flag. Their geometry otherwise omitted
side chains. The user requested adding side chains on 2026-10-06.

Append 32 named side-chain heavy-atom slots to each residue's input. Each slot
contains local-frame xyz divided by 10 Å and a presence bit: 128 additional
features. Atom identity is fixed by slot; missing atoms are zero with presence
false. Coordinates come from the same prepared AB structures as before; no new
reconstruction is performed and no pKa, PB energy, teacher feature or label enters
the inputs. Prepared atoms are not necessarily all experimentally observed.

The original backbone edges, query-site keys, labels, eligibility masks and
train/validation assignment are copied exactly and content-addressed. The network
can now use side-chain conformations through its residue embeddings; this remains
a residue graph, not atom-to-atom attention. The graph radius stays 20 Å in CA
distance. This experiment does not enable coordinate differentiation.

Both arms have 50,001 parameters: width 44, FF width 68, four heads, two encoder
layers, one site-query attention block, 152 input features. The side-chain arm uses
all inputs. The control zeroes the 128 side-chain columns before every training
and evaluation call; those input weights therefore receive zero gradient in the
control. Both have identical initialization and downstream architecture. Retain
the earlier 49,709-parameter backbone pilot as a separate reference.

Both use seed 17, 20 epochs, Adam 1e-3, gradient clipping 1, accumulation 8,
group-uniform sampling, full float32, final-epoch reporting, and 2,000 sequence
component bootstrap replicates. Tests check atom-feature rigid-motion invariance,
presence masks, padding, exact parameter count and nonzero gradients through the
new inputs, along with the existing model tests.

Runtime directories:
- `pretraining/graph-sidechains-50k-v1`
- `pretraining/graph-sidechains-control-50k-v1`

Preparation job 737369; side-chain GPU job 737370; control GPU job 737371.
Each GPU job requests one A40, 8 CPUs and 16 GB. Preparation and final scoring
run off comp1400. All submissions share the user-wide 400-core cap.
