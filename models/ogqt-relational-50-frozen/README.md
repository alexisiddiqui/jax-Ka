# Default frozen relational GQT

Pinned experiment backbone: `control-v2-50-relations-s17`, selected epoch 18,
50% training fraction, seed 17. 189,730 parameters; width 64, orientation and
backbone dihedrals, relation projections, unweighted pKa objectives and burial
auxiliary supervision. No learned interface head is included.

Use `pkatrain.production_defaults.load_frozen_default()` for frozen embedding and
shift functions. Supply production graphs with the appended backbone table and
appropriate bound/free masks. Set `GQT_FROZEN_MODEL` when loading from an installed
package outside this checkout. `model.json` records source checkpoint identity,
parameter checksum and training provenance. This is not a supported export for
the legacy 24-feature standalone `ogqt predict` CLI.

New experiments start with seed 17 only. GQT stays frozen by default. The existing
three-seed relational runs remain the confirmation baseline.

For differentiable structural readouts, use
`TorchStructureReadout.from_default(features, state="bound" | "free")`.
This keeps weights fixed while retaining input gradients; see
[the structural adapter](../../docs/TORCH_STRUCTURES.md).
