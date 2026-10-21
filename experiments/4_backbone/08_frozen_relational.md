# Frozen relational GQT default and interface experiments

The default frozen checkpoint is `control-v2-50-relations-s17`, epoch 18, trained
on 50% of the pool with unweighted pKa state/shift losses and burial supervision.
Its 189,730 parameters include width-64 relation projections. It does not include
an interface probe head. Frozen interface experiments did not update GQT or its
pKa predictions.

Since 2026-10-21 the repository default is the all-residue GQT
(`models/ogqt-allres15-siamese-50`); this relational checkpoint loads with
`load_relational_frozen` (formerly `load_frozen_default`).

```python
from pkatrain.production_defaults import load_relational_frozen
embeddings, predict_shift, metadata = load_relational_frozen()
# graph: production graph with its 14-column backbone table appended to nodes
site_tokens, query_site = embeddings(graph)
```

The checked-in export is `models/ogqt-relational-50-frozen`. To load it from an
installed package outside this checkout, set `GQT_FROZEN_MODEL` to the export
directory. The loader validates the checkpoint identity and parameter checksum,
and stops gradients through GQT outputs. Training-store code can also use
`selected_model(runtime)` to resolve the default run; that lower-level function
returns parameters and is not a freeze guard. The legacy `ogqt predict --model`
interface remains separate and does not support this representation export.

## Experiment policy

Start new experiments at 50% with **seed 17 only**. The relational baseline's
three seeds already exist. Repeat promising results across seeds only when needed
and approved. Default to frozen GQT; unfreezing requires explicit scope. Run heavy
work on Isambard compute nodes, one GPU and at most one node per job. Independent
jobs may run in parallel. Do not recreate a human-only dataset restriction.

## Completed evidence

The 50% relational model improved mean pKa selection from 0.649432 to 0.632378
across three seeds. Global site-query attention scored 0.649714 and did not give a
consistent improvement. Inverse-square-root shift weighting worsened overall
selection, despite some middle-tail bin improvements.

Frozen interface validation, three-seed mean per-complex average precision:

| Experiment | Free-input AP |
|---|---:|
| Small per-site MLP | 0.471052 |
| Partner free-token pair head | 0.450024 |
| Capacity-matched per-site MLP | 0.465217 |
| Shallow same-chain transformer | 0.461502 |
| Shared bound/free Siamese small MLP | 0.431700 |
| Shared bound/free Siamese matched MLP | 0.428007 |
| Shared bound/free Siamese transformer | 0.445017 |
| Separate fixed teacher, student weight 0.1 | 0.470135 |
| Separate fixed teacher, student weight 1 | 0.460716 |

The independent small-MLP teacher scored 0.624790 with **bound input**; this is not
a free-input result or an established performance ceiling. No independent test
set was used. Macro AP averages positive-containing complexes; pooled metrics
include complexes without positives.

Targets cover existing pKa-labelled surface sites, with free RSA >=0.1 and
contacts within 10 Å to eligible labelled sites on opposite partners. They do not
cover all surface residues. Free conformations are partner-removed bound
structures. The transformer head only reads same-chain cached tokens. Its bound
tokens already contain partner context from GQT. Report code verifies matching
validation support across comparisons.

## Reproduction and artifacts

`results/` holds job mappings, configurations and completed comparison reports.
`scripts/` holds the exact experiment drivers and compute-node launchers. The
launchers retain the Isambard deployment paths used by these runs; deploy each
script directory to its documented `_submission` location to reproduce it.
Older README launch statuses are historical; completed comparison files and the
table above describe the final outcomes.

On Isambard, runtime artifacts are under
`/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench`. Training checkpoints are in
`training/gqt-production-v2/runs`; frozen probe caches and results are in
`audits/frozen-interface-v1`, `frozen-transformer-v1`, `frozen-siamese-v1` and
`frozen-distill-v1`. Source snapshots remain intact under `_submission`.
