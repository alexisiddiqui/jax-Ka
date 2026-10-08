# Five-angstrom GQT crop and backbone-only pKAI ablation

Authorized 2026-10-08. The 25 Å GQT crop was dropped because the frozen encoder
graph contains edges only to 20 Å; implementing a genuine 25 Å crop would require
recovering coordinates and would no longer be a direct extension of the completed
crop-radius screen.

The 5 Å GQT arm retains the completed screen's 49,709-parameter model, cleaned 5k
cohort, three seeds, AdamW `1e-4`, 25% query-centred crop draws, full validation,
and learning-rate schedule (`1e-3` through epoch 10, then cosine to `1e-5`). It
records end-to-end run time and JAX allocator peak device memory for every seed.

The pKAI ablation defines a backbone-only input that is usable without side-chain
coordinates: the query origin is the residue Cα, environmental atoms are the
backbone `N` and `O` atoms of other residues within pKAI's native 15 Å cutoff,
and residue identity is retained. Native pKAI discards carbon atoms and has no
`C`/`CA` input class, so Cα is used only as the query origin; representing those
carbon atoms would require changing the architecture. The native
4008→800→400→200→1 network and
dropout remain unchanged. Scratch and released-checkpoint initializations are
each run with full-atom and backbone-only inputs over seeds 17, 29, and 43.
Matched controls use batch 64, Adam at `1e-6`, weight decay `1e-4`, unweighted
signed-shift MSE, and validation-MSE early stopping. Full validation is used and
test data are not read. Each run records wall time and PyTorch peak allocated and
reserved VRAM.

Runtime locations:

- `pretraining/gqt-50k-crop-5a-v1-triton`
- `pretraining/pkai-backbone-ablation-v1`

Submitted Slurm chains:

| Experiment stage | Job |
|---|---:|
| GQT registration | 745177 |
| GQT Triton smoke | 745179 |
| GQT three-seed GPU array | 745180 |
| GQT report | 745181 |
| pKAI one-structure feature gate | 745251 |
| pKAI 5k backbone-feature preparation | 745252 |
| pKAI GPU smoke | 745253 |
| pKAI 12-run matched GPU array | 745254 |
| pKAI report | 745255 |

Two earlier chains were cancelled before execution after their gates were pinned
to saturated 32-core nodes. These final gates and reports use the 96-core nodes;
the 32-core preparation runs on `comp0650`.

The first pKAI feature gate exposed an invalid assumption that native pKAI keeps
carbon atoms. It does not: its parser discards them and the fixed 16-class atom
schema cannot encode `C` or `CA`. The corrected strict-backbone representation
uses Cα only as the query origin and encodes backbone N/O environment atoms.
