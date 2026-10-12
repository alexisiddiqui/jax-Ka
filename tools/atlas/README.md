# ATLAS preparation on Isambard

These scripts record the full ATLAS preparation, sequence audits, frozen split,
nested portions, compressed graph build, and GPU smoke checks completed on
2026-10-12. The Python and Slurm files retain their executed contents; they are
site-specific scripts, not a new installed package or an automatic fresh-download
installer. Large source archives, sequences, labels, graphs, membership TSVs,
and rendered reports remain in the runtime directory.

## Release

The full labelled pool contains 155,677 complexes. Its frozen membership is
153,677 training complexes and 2,000 validation complexes, with no shared
sequence components. Graph eligibility leaves 153,584 usable training complexes
and 1,996 usable validation complexes. The 97 inactive complexes have no eligible
paired interface sites; their original membership is retained.

| Portion | Frozen training membership | Usable graphs |
| --- | ---: | ---: |
| 10% | 15,367 | 15,362 |
| 50% | 76,838 | 76,798 |
| 75% | 115,257 | 115,191 |
| 100% | 153,677 | 153,584 |

All portions are nested and share the same validation set. The sampling seed is
`atlas-full-portions-v1|20261012`; within-stratum SHA256 ordering is interleaved by
normalized rank, then exact floor-percentage prefixes are selected. Training
portions sample complexes; sequence components can span training portions but
cannot span training and validation.

Runtime root (`PKABENCH_RUNTIME`):

```
/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench
```

Release, relative to that root:

```
pretraining/atlas-prepared-v1/graph-v1/release-v1/
```

Use `config-010.json`, `config-050.json`, `config-075.json`, or `config-100.json`
there for the corresponding membership and manifest paths. These JSON files
describe loader inputs; they do not start training. The existing
`pkatrain.production_loading.PinderSource` adapter consumes the ATLAS manifest
with explicit uniform burial/interface norms of 1. The compressed store has
155,580 records and 580,207,045,094 bytes.

## Execution layout

Scripts must be staged into the runtime layout before execution because their
relative paths refer to adjacent data and audit files:

| Repository directory | Runtime audit directory |
| --- | --- |
| `source/` | `audits/atlas-v1/` |
| `prep/` | `audits/atlas-prep-v1/` |
| `split/` | `audits/atlas-split-v1/` |
| `alignment/` | `audits/atlas-alignment-v1/` |
| `phate/` | `audits/atlas-phate-check-v1/` |
| `graphs/` | `audits/atlas-graphs-v1/` |

The `portions/create.py` metadata generator was run locally as
`work/atlas-full-portions/create.py`. It expects `outputs/atlas-train.tsv` and
`outputs/atlas-validation.tsv` under that workspace root and writes
`outputs/atlas-full-portions-v1/`. Those files were transferred to
`audits/atlas-full-portions-v1/` for graph finalization.

The source stage also requires the official CSV, repository file manifests,
download receipts and existing reference membership. Preparation requires the
existing jax-Ka environment, pKAI environment, pdb2pqr and MMseqs executable paths
recorded in the scripts. The PHATE/alignment stage uses the separate audit
environment; its package list is retained with the runtime audit. Scripts that
render HTML locally require the original report workspace layout.

All submitted compute jobs request one node, one GPU and 72 bundled CPUs.
Scheduler submission and transfers can use the login node; processing must run
inside the allocation. The environment setup file is
`/scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh`.

The completed graph run used `graphs/resume.sbatch`: inventory, sharded build,
packed-store finalization and then CUDA smoke tests. It expects the existing
labelled archives, frozen split and portions. `graphs/preflight.py` records the
earlier pilot size check and expects pilot intermediates; it is not part of the
sharded resume command. Before any fresh build, check both project byte and file
quotas. The final builder writes 72 compressed shards instead of per-complex
directories, and deletes only its own derived graph intermediates after flushing
records. It never deletes the labelled source archives.

Graph construction uses the existing backbone/site graph implementations, a
20 Å residue radius, source CA pLDDT >=70, valid complete source sites, finite
paired pKAI AB/free labels and uniform weights. `rsa_free` is unavailable metadata
represented by NaN; it is not an objective input. All model inputs are checked
for finiteness and index bounds. Each compressed record and canonical graph
array set has its own SHA256 digest.

## Verification evidence

[`docs/atlas-release-v1`](../../docs/atlas-release-v1/) contains the source
summary, frozen-portion verification, final graph summary, store integrity
receipt, GPU smoke receipt, readiness receipt and local export verification.
`pipeline-source-hashes.json` records the copied scripts' SHA256 values.

The full graph job was 7269688 and completed with exit code 0. The store verifier
decoded and checked all 155,580 records. GPU tests checked every occupied loader
bucket, optimizer steps, finite gradients in the largest bucket, padded-batch
loss invariance and all 1,996 usable validation complexes. Export verification
checked unchanged membership hashes, nested active portions, each validation ID
exactly once, and zero train/validation ID or component overlap.

These are technical smoke checks at batch size 2, width 44 and feedforward width
88. Full training, convergence, accuracy and production-batch memory limits have
not been tested. PHATE plots describe a sampled visualization and do not replace
the exhaustive sequence split audit.
