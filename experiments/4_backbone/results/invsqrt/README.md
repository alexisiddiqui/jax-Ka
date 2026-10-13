# Inverse-square-root binding-shift experiment series

2026-10-13. User-authorized: 50% and 100%, seeds 17/29/43; architectural ablations at 50%, seed 17.

Base A: width 64, FF 128, orientation, dihedrals, residual dropout .1, burial auxiliary 1, batch 16, 20-epoch schedule and original early stopping. Only the paired binding-shift MSE is weighted. State, pKPDB and burial losses and validation selection remain unchanged.

Weights: magnitude boundaries 0,.1,.25,.5,1,2,4,infinity; raw 1/sqrt(n_bin), fitted on the respective training fraction only. A single scale enforces mean one under equal-complex reduction, with cap 5. The two fractions have separate frozen weight tables. Validation never enters fitting. Unweighted per-bin validation MAE, RMSE and bias are recorded each epoch.

References: six weighted runs. Add three matched unweighted 50% controls because earlier 50% models differ from base A. Reuse the three existing unweighted 100% A runs. No new data sources or altered splits.

Single-seed 50% weighted ablations: Fourier position features; within-chain RoPE; relation-specific width-64 pre-attention maps with deduplicated ±1/±2 edges; a shared-map control on the same graph; an antisymmetric paired-delta correction head. The latter corrects AB/free state predictions by ±delta/2 and retains their supervision. It is not interface classification. A single-state export is explicitly blocked for this paired model until paired inference packaging exists.

Fourier features encode observed within-chain position. Chains use same/different-chain relationships, never numerical chain-order distance. RoPE applies positional dependence only within a chain. Relation types are -2,-1,+1,+2,other-same-chain,cross-chain; local edges require continuous peptide geometry. Cross-chain edges retain the original spatial graph and free-branch mask. Separate maps act before context K/V projection in residue blocks; site attention remains unchanged.

Isolation: source snapshot under Isambard `_submission/invsqrt-v2/src`; shared Git checkout and frozen pkanet source files are unchanged. Run protocols carry source hashes. Local implementation: `work/invsqrt-model/src/pkatrain/production_{train,shift_weight,position}.py`.

Numerical gate: weight-one objective/gradient equivalence, exact weighted paired loss, unchanged state term, padded-complex invariance; architecture initialization controls, chain renaming, RoPE position-offset invariance, free-branch partner exclusion, paired-head identity/swap checks, finite optimizer steps. Actual GPU training smoke precedes references. Large-bucket checks and ablation smoke runs gate architectural training.

Each job requests one node, one GPU and bundled CPUs. References are eligible in parallel. Ablations are eligible in parallel after the weighted and unweighted 50% seed-17 references complete. The report waits for all runs. A failed prerequisite blocks its dependents. Test data remains untouched. No accuracy gain is claimed until completed runs are compared.

## Submitted jobs

Prerequisite GPU smoke: 7364848. Weighted references: 7364977–7364982. Unweighted 50% controls: 7364983–7364985. Single-seed architectural ablations: 7364986–7364990. Final report: 7364991. Exact mappings are in jobs.tsv.

Results will be collated on Isambard under `/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench/audits/invsqrt-experiments-20261013/`. Training artifacts remain in `training/gqt-production-v2/runs/<run>/`.

## Verified launch status

Smoke job 7364848 completed successfully (3m53s). All five architectural numerical checks passed. First weighted 50% reference job 7364977 entered RUNNING. The original serial schedule was replaced at the user’s request with parallel one-GPU jobs. The earlier 6–9 hour serial ETA no longer applies; elapsed time depends on scheduler availability and ablation timings.

## Follow-up: unweighted relation projections

User-authorized follow-up: 50% training fraction, seeds 17/29/43, relation-specific projections with `--shift-weight none`. All other base-A settings and the existing validation selection remain unchanged. Submitted independently, one GPU and one node per job: 7366919, 7366920, 7366921. Compare against the existing `control-v2-50-s17`, `control-v2-50-s29`, and `control-v2-50-s43` runs. Exact names are in unweighted-relations-jobs.tsv. The architecture and largest-bucket finite-update checks already passed in the preceding experiment series.
