# jaxpb: discrete Poisson–Boltzmann energies and titration labels

jaxpb replaces the repeated PB and Monte Carlo work in a PyPKA teacher pipeline. PyPKA still prepares atom coordinates, radii, background charges, site order, tautomer charges and model compounds. Those inputs become portable StateRecord shards. JAX computes molecular geometry, dielectric/ion maps, three PB problems per tautomer, reaction-field and background energies, the interaction matrix, and titration curves. The GPU is the production target; CPU execution is a correct fallback and remains substantially slower than native PyPKA.

## Discretization and execution

Each tautomer uses an 81³ coarse molecular grid at 1 Å spacing, a focused molecular grid at 0.25 Å, and a model-compound grid at 0.25 Å. Coarse/model faces use screened Coulomb potentials; focused faces interpolate the coarse solution. The tests preserve the existing dielectric constants, ionic strength, charge assignment, surface corrections, PyPKA interaction signs and symmetrization. Convergence refers to these discrete equations and boundary conditions, rather than to a continuum or experimentally exact pKa.

Atom and probe arrays use a small size-class ladder. Sites use fixed padded batches; solver lanes and MC state buckets have bounded shapes. Jits are module-level and take array arguments, so additional states reuse executables. GPU DelPhi SOR uses Pallas Triton kernels; XLA supplies a portable implementation. MC has both Pallas and scalar CPU implementations. Assembly uses float64 accumulation before the established float32 energy storage. Probe generation also uses float64; default PB arithmetic stays float32.

The driver processes input shards, retries capacity overflows explicitly, groups MC work by bucket, and writes atomic label shards. Stable per-state chain identifiers make MC independent of ordering, batching and restart boundaries. Configuration hashes and source fingerprints protect resume; PB mode is part of configuration and provenance. Energies are saved so titration can be repeated without PB.

## PB mode and the cost of DelPhi's stopping rule

`Config()` uses `pb_mode="delphi"`, the only supported mode: zero start, even/odd red-black SOR, the existing spectral-radius relaxation estimate, a maximum potential-change threshold of 0.01 kT/e checked every ten sweeps, and a 500-sweep cap. This is the mode with a PyPKA parity target.

M7 also built a tightly converged mode (Jacobi-preconditioned f64 conjugate gradients on the same discrete system) and then dropped it. It matched an independent f64 reference to a maximum potential difference of 1.5e-5 kT/e, but on 104 L3 states it moved pKa by MAE 0.52656 pK (maximum 3.94382 pK; 7,073 shared finite midpoints, 156 lose a finite midpoint) and interface shifts by MAE 0.52425 pK, against two-seed MC noise of about 0.004-0.005 pK. Versus the teacher, pKa MAE went from 0.00844 (DelPhi mode) to 0.52865. Warm A40 PB cost 213.29 versus 8.40 seconds/state. So PyPKA's labels contain a large, reproducible stopped-iterate effect, and the converged solution is further from the teacher and 25 times slower. Whether it is closer to experiment was not tested. The code was not kept; the measurements are in `M7-REPORT.md` and `results/m7.json`.

## Validation history and performance

The evidence is in `results/*.json`; the A40 milestone files supersede earlier reports that were awaiting GPU access.

| Stage | Measured result |
|---|---|
| M0 | A40 reproduced prototype energies within 2e-6 relative error; sweep counts matched. |
| M1 | Fixed-shape pipeline and overflow retry worked; its initial gate recorded one energy outlier and one executable over budget. |
| M2 | On 99 states, focused/model sweep agreement was 90.5%/86.5%; solvation relative-error median was 7.0e-5 and p99 0.0043. |
| M3 | Assembly on saved PB data matched teacher g/W within 1.88e-6 kT; categorical arrays were exact. |
| M4 | MC counts matched prototype results on 426 states at 50k and 200k sweeps; teacher pKa MAE was 0.0063/0.0040. |
| M5 | All 92,691 tautomer-call sweep counts matched XLA; energies/potentials were bitwise equal. Pallas SOR was 10.2× faster. |
| M6 | L3: 104 states, pKa MAE 0.00844 and interface-shift MAE 0.0144 versus teacher at 200k sweeps; compiles stayed at 42 through 1,040 states. |

The subsequent speed work measured 8.83 PB seconds/state on the 104-state mix and reported 346 states/A40-hour, approximately the 344-state/hour planning rate. OPT1/OPT1B bounded memory and reduced map work; OPT2 introduced device lane compaction and reduced five-state CPU SOR to 163.8 seconds/state. OPT3 established scalar CPU MC; OPT3B reduced the 20-state, 200k-sweep mean from 238.5/41.7 to 197.5/33.4 seconds in the CPU1/CPU12 benchmark configurations. OPT4 reduced the five-state CPU PB stage sum to 1,377/351 seconds at one/twelve CPUs. The native PyPKA reference measured 8.8 PB seconds/state (25–29 seconds total). These mixes and timing definitions differ; their rates should not be combined into a synthetic benchmark. The native CPU prototype failed rounding parity and is not a production backend.

## Reproduction and limits

Use `jobs/m7_submit.py` to freeze source and submit the gate, closing-reference, label and memory scripts. Its `--baseline` option packages pristine 31b30bc source. The complete commands, hashes, allocations and raw evidence accompany `results/m7.json`. Coulson jobs use the existing environment, the A40 shim for GPU work, at most twelve CPUs and 2 GiB/CPU; no installation is required. `jobs/m7_poll.py` polls `squeue` with three-minute sleeps. The API driver remains `python -m jaxpb.api`.

Run summaries preserve their established keys as recent histories capped at 256 entries, with complete counts/sums in `history_totals`. Full rows stream to `history_file` (`run.log.jsonl`); per-shard `timing_json` remains available. This bounds summary retention; the input plan and one active shard still occupy memory.

Only DelPhi mode is supported. MC midpoint availability can change near the pH-range limits. M7 does not establish a GH200 throughput measurement or a 30,000-state production run.
