# Default all-residue GQT

Pinned default model (since 2026-10-21): training run
`control-v2-50-relations-4-1-allres-cut15-cysmask-gate01-ressiamese-redge01-s17`, selected epoch 17 of 20,
50% training fraction, seed 17. 180,000 parameters, width 64. Inputs are backbone and sequence only.

## Architecture

- Two relational residue layers over the 20 Å CA graph: 4 relation maps in layer 1 (±1, ±2, same chain, other chain),
  1 in layer 2; backbone orientation edges (13 features) and backbone dihedrals (6 node features).
- Every residue is a token (residue embedding + amino-acid embedding). Tokens pass a query block and a token-to-token
  block whose neighbours are restricted to CA–CA ≤ 15 Å (same chain and across chains).
- Each token has three outputs: side chain, N-terminus, C-terminus (terminus maps start as the identity).
- Free state: the same graph with every edge between chains masked; outputs cannot depend on the partner.

## Heads and training targets

| head | output | target |
|---|---|---|
| pKa (`head`) | 8 tanh(·): shift from the group reference pKa (PKPDB_PK_MOD) | PINDER pKAI labels (bound, free, binding-shift Siamese loss) and pKPDB PypKa labels; bonded Cys excluded; Arg never labelled |
| titratable gate (`aux_gate`) | logit per output | D/E/H/C/Y/K/R minus disulfide Cys (SG–SG < 2.3 Å); chain termini; weight 0.1, Cys ×10 |
| burial (`aux_burial`) | logit per residue | 1 − clip(RSA) on every PINDER residue, bound and free branches, plus the bound − free change vs ΔRSA |
| residue edges (`aux_redge`) | symmetric pair logit | cross-chain residue contacts (heavy atoms ≤ 10 Å), soft target; both branches, free distilled toward bound; weight 0.1 |

## Validation (PINDER pool-v4 validation, 800 complexes; seed 17 only)

| metric | value |
|---|---:|
| selection (state MAE + interface binding-shift MAE), all sites | 0.6251 |
| selection without the 601 bonded-Cys sites | 0.6047 |
| state MAE / interface binding-shift MAE | 0.3994 / 0.2257 |
| pKPDB validation site MAE / benchmark-val MAE (PypKa) | 0.4814 / 0.5029 |
| burial on interface residues: free AUROC (RSA < 0.25) / bound AUROC | 0.968 / 0.959 |
| burial change at the interface: Spearman with ΔRSA | 0.855 |
| free-state interface residues (5 Å): edge-score per-chain median AUROC | 0.675 (measured free RSA 0.622) |
| residue-pair contacts, free / bound branch: AUROC (AP) | 0.669 (0.029) / 0.804 (0.141) |

Reports: Isambard `audits/residue-siamese-v1-50/`, `audits/residue-siamese-v1-50-v4/`, `audits/pose-check-v1/`.
Single seed; the seed spread of the selection metric at this scale is about 0.004.

## Use

```python
from pkatrain.production_defaults import load_default
model = load_default()                 # pkanet.allres heads; outputs detached by default
out = model.outputs(graph)             # query_pka / query_shift, per residue pka, gate_logit, burial_logit, edge_embedding
```

- Files: `ogqt predict structure.pdb --cache prepared/ --output predictions.jsonl` (bound, free, binding shift and
  per-residue records; see `docs/INFERENCE.md`).
- Gradients: `TorchStructureReadout.from_default(features, state="bound" | "free", output="pka" | "shift" | "gate" | "burial")`.
- Graphs: `pkanet.allres.AllResidueStructureFeatures` (differentiable template) or `pkanet.allres.graph_from_features`
  (cached files) append the backbone table; `pkanet.allres.state_graph(graph, "free")` gives the free state.
- Set `GQT_DEFAULT_MODEL` to load this export outside the checkout.

Arg and bonded-Cys side-chain pKa outputs exist but were never supervised; read them together with the gate.
`model.json` records the training protocol, code hashes and the source checkpoint checksum; the training code is the
Isambard snapshot `_submission/residue-siamese-v1`.
