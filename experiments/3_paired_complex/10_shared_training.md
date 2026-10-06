# Shared LocalTerms training pilot

## Current production decision

The full campaign is registered under `training/shared-v4-float32`, derived
from the prepared v3 dataset. All 477 training and 142 validation records are
retained. Shape buckets contain 194/217/66 training complexes with maximum
microbatch sizes 8/2/1. Group-uniform sampling is unchanged; sampled records
are reordered by size within each epoch. Unequal microbatch mean gradients
are weighted by complex count before each eight-complex optimizer update.

GPU preflight measures worst-size representatives, checks batched results
against separate solves, and computes an ETA weighted by actual sampling
probabilities. Host packing now pads the immutable prepared AB/free arrays
directly, rather than reconstructing three structural caches on every step;
bitwise equivalence is tested in both precisions. The updated suite has 41
passing tests. Full-dataset loss profiles run on CPU compute nodes under the
400-requested-core cap. `gpu_dispatch` releases baseline/training jobs only
after the GPU preflight and applicable profile gates, handles checkpointed
wall-time continuations, and produces the final comparison report.

The training precision target is **full float32, with x64 disabled**, per the
user's latest instruction. Float64 remains a numerical reference; mixed
precision is no longer a candidate. The distinct-complex GPU smoke uses
float32 parameters, inputs, labels, solver and optimizer. The original strict
1e-6 float64 curve-parity verdict remains visible. Float32 smoke admission is
recorded separately at the solver's 2e-5 occupancy/residual tolerance, 1e-3
relative gradient tolerance and identical validity masks. These criteria do
not relax the existing missing-supervision or nonfinite-gradient limits.

Keep `seed_steps=None`, which uses `ModelConfig.steps=1024`, and use
`lm_steps=32`. The kernel optimization applies without changing
`experiment.py` or shortening the production seed. The user's eight-record
production-seed check reports full convergence and zero active-set leakage;
the reported maximum LM iteration count on a converging record is 14.

`shared-v3-production` registers these settings separately from
`shared-v2-fast`, whose explicit 64-step override is superseded. No 64-step
default or retry policy is being adopted. The measurements below remain
historical performance experiments; their timings and extrapolated training
durations are not estimates for the current 1024-step configuration.

## GPU precision/batching measurements (2026-10-05)

One A40, eight host CPUs, 16 GB host RAM; 200 real residues padded to
N=256, Ke=192, Kc=144, M=64. Batch eight repeats the same complex as dynamic
inputs; it is a throughput probe, not a heterogeneous minibatch validation.
Warm timings include the forward validity audit, mean curve-loss gradient,
and Optax update, with resident inputs and compilation excluded.

| Working / seed precision | Batch 8 seconds | Seconds / complex | Gradient relative difference vs float64 | Peak live device GB |
|---|---:|---:|---:|---:|
| float64 / float64 | 15.933 | 1.992 | reference | 6.58 |
| float64 / float32 | 9.593 | 1.199 | 3.20e-11 | 6.62 |
| float32 / float32 (x64 disabled) | 5.774 | 0.722 | 2.02e-7 | 3.21 |

All used seed_steps=64 and LM max_steps=32. Within each precision, batch-one
and batch-eight curves matched exactly and mean gradients agreed. Against
float64, mixed-precision maximum curve difference was 1.11e-7 and full
float32 was 1.25e-6; validity masks matched. The latter exceeds the previous
strict 1e-6 curve-parity threshold slightly, so this is not an unconditional
full-float32 release gate. Float32 maximum residual was 2.98e-7. Allocator
pool peaks (not live tensor peaks) were 14.77, 15.03 and 8.59 GB respectively.

Artifacts: `training/gpu-batch-check-v1`, `gpu-batch-mixed-v1`, and
`gpu-batch-float32-v1`; jobs 736171, 736181 and 736182. The 37-test suite
passed after wiring the seed dtype through Engine and validation.

Separately, `training/gpu-check-v1/seed_coverage.json` shows the 1460-residue
complex loses convergence at pH 6.5 in both branches with a 64-step seed,
even with 512 LM steps. A 1024-step seed with 512 LM steps converges across
the grid. This supports retaining the production seed; more LM steps alone
did not resolve the shortened-seed failure.
Neither a detached seed nor a lower seed precision guarantees preservation
of the selected mean-field branch on every structure.

## Performance revision under validation

The vectorized seeding implementation and `solve_branches(seed_steps=...)`
handoff are adopted. `Engine` propagates the seed budget; `make_engine(out)`
reads both `seed_steps` and `lm_steps` from the registered manifest. The
original run remains unchanged. `register-fast` creates a separate candidate
with the production 1024-step seed and 32 LM steps, a parent-manifest hash, identical records
and immutable prepared payloads, and no inherited gate/training results.

`benchmarks/validate_training_speed.py` compares 1024/512, 64/512 and 64/32
budgets at initial and perturbed parameters on the full pH grid. It records
curve differences, validity, gradients and cold/warm timing. Comparisons run
with one and eight CPU threads on 61-, 200- and 1460-residue complexes. An
eight-thread run explicitly keeps eight CPUs in its affinity mask; other
benchmark workers retain their existing one-CPU default. A shorter seed may
select another mean-field branch, so detaching it does not establish parity.
These measurements and fresh real-complex gates must precede release.

Implementation: `src/pkatrain/`. Runtime artifacts live under
`_runtime/jax-Ka/pkabench/training/shared-v1/`. This pilot trains JAX-Ka's
three physical scales against current PypKa curves; it does not establish
experimental accuracy or reproduce historical pKPDB settings.

## Shared boundary

`adapters/jaxka.py` maps parameters to `LocalTerms`. The model-independent
`forward.py`, `losses.py` and `trainer.py` perform the solve, paired loss,
implicit differentiation, optimization and checkpointing. A later graph
encoder must return the same terms and reuse this path. It should not add
another trainer, pairing implementation or custom implicit adjoint.

The three scales are `exp(log(4) * tanh(theta))`, initialized at one.
They multiply desolvation, hydrogen-bond and Coulomb contributions. The
hydrogen-bond scale also multiplies carboxylate reorganization and
protonation-dependent hydrogen-bond couplings. `ModelConfig` stays static.

AB and free have identical site indexing and padded shapes. Free is the
block-diagonal union of independently prepared A and B caches, including
their independently computed burial and backbone summaries. Removing AB's
cross-partner edges would not remove those summaries. Within-residue
terminal/side-chain couplings are retained. Dummy active channels are
distinct, have zero physical weight, and use identity residuals.

## Fixed data and objective

The original 500-complex training pilot and 151 validation complexes remain
the starting population. Requiring all three native-v2 states and paired,
eligible interface sites leaves **477 training and 142 validation complexes**.
All 32 exclusions and their reasons are recorded in `manifest.json`.
No test predictions are read by this experiment. Existing sequence-group
splits and label eligibility masks are inherited, not recomputed.

The current objective supervises eligible paired interface occupancy curves
on 73 pH points (-2 to 16, spacing 0.25): absolute branch curve MSE plus
paired curve-difference MSE, ramped from zero to full weight over five
epochs, plus 0.001 times mean squared log scale. These are PypKa occupancy
labels, not invented curves from scalar pKas. Reporting uses the original
teacher scalar pKas instead of rederiving their midpoints on this grid.

Adam uses learning rate 0.001, global gradient clipping at one, and eight
complexes per update. Sampling is uniform over training groups, then uniform
over complexes within a group. Twenty epochs and seeds 17, 29 and 43 are
registered. Seeds change sampling only; all start at theta zero. The final
epoch is the primary checkpoint; validation does not select it.

## Solver and failure contract

Every pH point starts from the detached production damped solution (1024
steps), followed by optimistix LM with its implicit adjoint (512 steps,
tolerance 1e-6). Full conditional residual tolerance is 2e-5. Checkpoint
diagnostics additionally compare upward and downward sweeps.

A failed solve invalidates its entire branch/pH point. Paired loss requires
both branches. The differentiated rerun replaces rejected root systems with
parameter-independent identity systems; masking a NaN gradient after the
solve is insufficient. Loss denominators retain the original eligible
observation counts. Stop above 5% missing observations in a complex or 1%
over an epoch, and stop on any nonfinite objective or gradient.

Checkpoints atomically save parameters, Optax state, RNG state, sampled
order, offset, coverage counters and source/data hashes. A resumed allocation
continues the same update sequence. The persistent JAX compilation cache is
shared across seeds; N/Ke/Kc/M are bucketed.

## Release gates and execution

1. Synthetic and existing solver tests: default LocalTerms equivalence,
   scale gradients, free-union values and gradients, independent mean-field
   root checks, padding, masked differentiation, and exact checkpoint resume.
   Exact enumeration is an oracle for uncoupled systems, not coupled mean
   field or a collapsed PypKa tautomer model.
2. Eight real complexes spanning both roles and residue-count ranks:
   full-grid free union versus separate A/B curves and midpoints, tightened
   finite-difference checks, free-union versus A+B gradients, convergence,
   warm-step runtime and memory.
3. Twenty-update smoke optimization. Then conditional full-training-set loss
   profiles at scale values 0.25, 0.5, 0.75, 1, 1.5, 2 and 4, varying one
   scale at a time. Profiles are not evidence of a global optimum.
4. Three registered training runs and untrained shared-solver validation.
   Existing benchmark aggregation supplies 2,000 group bootstrap replicates.
   Primary comparisons use common valid interface sites; report coverage
   alongside accuracy. Other completed predictors are supplemental.

`_HPC/submission/jax-Ka/pkabench/shared-training.sbatch` runs all computation
on Slurm. `submit-shared-training.sh dispatch <runtime-directory>` starts a
restartable ready-only dispatcher. It journals worker arrays and admits jobs
under the user-wide 400-requested-core cap, including other jobs. Each worker
requests eight CPUs and 16 GB; comp1400 is excluded. A failed task stops
admission rather than silently dropping the complex. Restarting the
dispatcher reuses completed artifacts and tracks existing workers.

## Later phases

Experimental data curation/fine-tuning and graph representation pretraining
are separate phases. Scalar-only labels can later supervise the unsaturated
residual `effective_pka(pH=label) - label`; no curve needs to be fabricated.
Native tautomer energies remain separate. An isolated-site binding-polynomial
reduction must be verified before supervising a binary intrinsic head;
native pair interactions must not be assumed interchangeable with mean-field
couplings. AFDB data belongs to the later backbone-only work.
