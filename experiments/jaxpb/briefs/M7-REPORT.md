> Note (2026-11-25): the converged mode described here was dropped after these measurements and is not in the code. The default delphi mode is unchanged and the bounded driver-summary change was kept.

# M7 report

Selected f64 Jacobi PCG: on the real-site A40 coarse/model probe, warm PCG took 1.69s versus tight residual-stopped SOR 6.26s (CPU 11.62/63.34s). Four-lane tiles bound memory. The equation residual is max interior |(rhs−Aφ)/diag(A)| in kT/e: internal tolerance 1e-9; separately checked float32 stored-buffer tolerance 1e-4. A separate sliced-interior, unpreconditioned f64 CG reference uses L2 rtol 1e-13/atol 1e-12 and checks true residual ≤1e-10. Existing focusing and postprocessing remain float32.

| Gate | Result |
|---|---|
| Default vs pristine 31b30bc | CPU20 and A40 L3-104 energies bitwise |
| Repeat + bench20 fresh process | Both modes pass; warm compile counts flat |
| Reference, all three solve types | A40 bench20 (including OPT2 five) and CPU five; max Δφ 1.52e-05 kT/e |
| Analytic lattice, one-site, mode isolation, 10 self-contained test files | Pass |
| Residuals, all tested converged solves | Internal max 1.00e-09; stored 1.82e-05 kT/e |
| Driver memory, 4,096 synthetic states | RSS span 11388 KiB over states 1,024–4,096 |

On L3, tightening DelPhi's solve changes pKa by **MAE 0.52656, max 3.94382 pK** with common random numbers; 7073 finite midpoints move and 156 lose a finite midpoint. 6194 shared midpoints move by >0.1 pK. Interface shifts move by MAE 0.52425, max 2.60148 pK. Teacher pKa MAE is 0.00844 (DelPhi; 7229 finite) and 0.52865 (converged; 7073 finite). Two-seed pKa noise MAE is 0.00403/0.00479 pK (DelPhi/converged); paired-shift noise is 0.00548/0.00678. Shift MAE versus teacher is 0.01440/0.52781 pK. M6's pKa/shift MC floors were 0.00406/0.0061 pK. These differences measure stopped-iterate error in this discrete pipeline, including remaining MC sensitivity.

Warm A40 PB costs 213.29s/state versus 8.40s/state for DelPhi mode (25.4×); CPU-five PB costs 1052.08/443.78s/state (converged/DelPhi). Per-type iterations and memory peaks are in m7.json. Summary histories retain 256 rows plus aggregates; complete rows stream to run.log.jsonl. Per-shard arrays/resume behavior pass existing tests. OPT1 had already aggregated state timings; M7 completes the history bound.

Risks: float32 storage/postprocessing, finite grids/boundaries and MC sampling remain. Long physical host-RSS scaling and GH200/30k production remain unverified. Raw per-state reference errors, all job outcomes and frozen-source hashes are retained. No commits.
