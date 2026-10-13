# Global site-query attention — 50%, unweighted

User-approved first experiment: global site-query-to-residue attention only. Residue encoder and site-to-site attention stay local. No relation projections and no inverse-square-root weighting. Base-A width 64, dropout .1, orientation, dihedrals, burial weight 1, batch 16, unchanged 20-epoch schedule/selection; seeds 17, 29, 43.

The site-query block retains its original parameters and 33 geometric edge features. Features are evaluated for every site/residue pair. Its 20 Å neighbour cutoff and smooth radial switch are removed. Thus the change is global query connectivity, not a parameter increase or a fully global residue encoder.

Loader explicitly labels AB as allowing cross-chain attention and free as chain-isolated. Single-state pKPDB/benchmark structures allow all chains within their construct. Batch elements remain separate. Padding is masked. An inference graph must provide the explicit branch flag; omission raises an error.

Query chunks of 32 with checkpointed recomputation bound activation memory. Before training: compare chunked outputs and gradients to original attention on a complete small graph, verify zero free-query partner gradients, perturb partner coordinates/features to test end-to-end free-branch independence, and run a largest-bucket batch-16 finite update plus actual GPU training smoke.

Code is isolated in Isambard `_submission/global-query-v1/src` and locally `work/global-query-model/src`. Existing experiments and their source snapshots are untouched. Training jobs use one GPU and one node each, eligible in parallel after the numerical/GPU gate succeeds. No test data used.

Dense query/key and attention/value contractions explicitly request highest precision rather than reduced-precision tensor-core products. The complete-graph equivalence test checks outputs and gradients.

2026-10-13: GPU preflight 7366994 passed dense output/gradient equivalence, zero partner gradients, end-to-end free-branch isolation, and a finite batch-16 update at padded length 1536. Actual training smoke remains in that same job. Training jobs 7366998/7366999/7367000 (seeds 17/29/43) are queued in parallel with afterok:7366994. Comparison report 7367003 depends on all three training jobs. First numerical preflight 7366987 failed before the precision correction; no full training was launched from it.
