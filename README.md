# jax-propka

Frozen-structure, soft-sequence titration in JAX, with Biotite topology handling.

**Scientific status:** this is a PROPKA-3.0/Nov30-parameterized **mean-field surrogate**, not a numerically faithful port of PROPKA's coupled determinant algorithm. The structural pipeline, differentiable outputs, reference adapters, tests, CLI and benchmarks are implemented. Biotite integration, pinned pip PROPKA primitives, the full reference path and float32 MPS execution are exercised by the test suite. CUDA remains a future channel. See [validation status](docs/VALIDATION.md).

The compiled readouts accept only `P[N,20]`. Geometry, candidate side chains, residue identities/keys, terminal masks, neighbor graphs and geometric kernels are prepared once outside JIT. An output selection never removes the other residues or chains from the physical environment.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,reference]'
pytest -ra
```

The `reference` extra installs the pinned PyPI release `propka==3.5.1`. This is the default external reference used by the CLI, tests and comparison benchmark.

Apple Silicon MPS uses a separate, pinned Python 3.12+ channel:

```bash
python3.13 -m venv .venv-mps
source .venv-mps/bin/activate
python -m pip install -e '.[test,reference,mps]'
JAX_PLATFORMS=cpu pytest -ra -m 'not mps'
JAX_PLATFORMS=mps pytest -ra -m mps
```

The `mps` extra pins JAX/JAXLIB 0.11.1 and `jax-mps` 0.11.0 because the plugin's StableHLO format must match JAXLIB. Setting `JAX_PLATFORMS=mps` prevents a CPU fallback from passing the MPS test. CUDA will use its own optional dependency channel when added.

The core dependency ranges in `pyproject.toml` are compatibility targets, not a claim that every version combination was tested. PROPKA is pinned because its output is used as regression data. The included reports record the exact executed environment.

## API: charge, titration curves and pKa

This example uses the included synthetic two-chain **molecular topology fixture**, not an experimentally measured structure. Replace its path and residue selectors for a real design backbone.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jaxpropka import prepare, TitrationModel, ModelConfig, StructureCache

# Host-only work. Native side-chain coordinates are preserved; substitutions
# use frozen candidate coordinates generated from the backbone local frame.
cache = prepare("tests/data/two_chains.pdb")
cache.save("structure-cache.npz")
# Later: cache = StructureCache.load("structure-cache.npz")
model = TitrationModel(cache, ModelConfig(steps=64, damping=0.35))
# Opt-in packed interaction-field contraction:
# model = TitrationModel(cache, ModelConfig(...), backend="packed")

# Construct these readouts ONCE, not in the optimization loop.
charge_all = model.charge(ph=7.0)                    # P -> [N]
charge_selected = model.charge(7.0, residues=[("A", 2), ("B", 2)])
curves = model.curves(np.linspace(0, 14, 29))         # P -> CurveResult
midpoints_all = model.pka()                         # P -> PkaResult [N,9]
midpoints_selected = model.pka_sites([
    (("A", 4), "HIS"), (("B", 4), "HIS")
])                                                 # P -> PkaResult [2]

# Faster all-site option: shared grid, piecewise-differentiable interpolation.
# These are approximate grid midpoints, NOT direct numerical roots.
grid_midpoints = model.pka_from_grid(np.linspace(-2, 16, 73))

# Alphabet: ACDEFGHIKLMNPQRSTVWY. P is row-stochastic, not an integer sequence.
logits = jnp.log(0.9 * model.native_probabilities + 0.1 / 20)
P = model.probabilities_from_logits(logits)
q = charge_all(P)
titration = curves(P)
pka = midpoints_selected(P)
assert bool(pka.valid.all())
assert bool(titration.converged.all())

# All state channels are retained under the sequence relaxation.
# HIS = index 6 in the 20-AA alphabet. The example also prevents the optimizer
# from evading the task simply by deleting the selected histidine identities.
def loss(z):
    p = model.probabilities_from_logits(z)
    pk = midpoints_selected(p)
    identity_penalty = -jnp.log(p[3, 6] + 1e-8) - jnp.log(p[8, 6] + 1e-8)
    return (
        jnp.mean(charge_selected(p) ** 2)
        + 0.1 * jnp.mean((pk.value - 6.5) ** 2)
        + 0.1 * identity_penalty
    )

value, gradient = jax.jit(jax.value_and_grad(loss))(logits)
# Re-check validity/convergence during optimization, not only at initialization.
# Reject invalid steps rather than relying on a zero pKa sentinel as a loss.
```

`backend="packed"` retains every active hypothetical identity edge and changes
only the repeated interaction-field contraction. The dense backend remains the
default pending a backend-independent branch-stability guard and accelerator validation.
Packing is performed once when the model is constructed and is reused by charge,
curve, grid-pKa, and direct-root readouts.

`P` can be supplied directly to every readout. Use `model.validate_probabilities(P)` outside JIT to validate external probabilities. A static `allowed[N,20]` mask can be passed to `probabilities_from_logits()`. Disulfide-frozen positions are clamped to native cysteine before all calculations. A single structure can be reused across a sequence batch with `jax.vmap(charge_all)`.

### Output conventions

There are nine conditional channels, in this exact order:

```text
ASP GLU HIS CYS TYR LYS ARG NTERM CTERM
```

The seven side-chain channels are alternative sequence identities. NTERM and CTERM are separate physical sites with unit presence at free termini, not alternatives to the side chain. There is deliberately no arithmetic average pKa across acidic/basic identities.

| Readout/field | Shape | Meaning |
|---|---|---|
| `charge(ph_scalar)(P)` | `[R]` | Residue charge, including any terminal site owned by that residue |
| `charge(ph_grid)(P)` | `[H,R]` | Same charge on a pH grid |
| `curves(...).protonated` | `[H,R,9]` | Conditional protonated fractions, not multiplied by sequence probabilities |
| `curves(...).site_charge` | `[H,R,9]` | Probability-weighted physical charge contributions |
| `curves(...).residue_charge` | `[H,R]` | Sum of the site's nine charge channels |
| `curves(...).chain_charge` | `[H,C]` | Full-environment charge per input chain |
| `curves(...).total_charge` | `[H]` | Full-environment charge, even with selected outputs |
| `curves(...).probability` | `[R,9]` | Channel presence probability; free termini have weight 1 |
| `curves(...).effective_pka` | `[H,R,9]` | pH-dependent local field value, **not** a midpoint |
| `pka(residues, groups)(P).value` | `[R,G]` | Direct numerical midpoint of each conditional titration curve |
| `pka_sites(sites)(P).value` | `[Q]` | Arbitrary selected `(residue, group)` midpoints |
| `pka_from_grid(ph,...)(P).value` | `[R,G]` | Faster interpolated grid midpoints |

Here `R=N` when no selection is supplied. Residues may be selected using zero-based cache indices, `ResidueKey(chain,number,insertion)`, or `(chain,number[,insertion])` tuples. The order of an explicit selection is preserved. All sequence positions remain in the solve.

Direct midpoint results include `valid`, `bracketed`, `probability`, `slope`, `crossing_error` and `residual`. An absent terminal or unbracketed root has a finite zero sentinel and `valid=False`; never interpret that zero as a measured pKa. A hypothetical side-chain identity can have a valid conditional pKa even when its sequence probability is zero.

Grid midpoint results additionally expose sampled monotonicity and bracket width. Gradients are through the interpolation, with possible kinks when the crossing changes intervals. Refine the grid and compare against direct roots before using that approximation in a loss. The bracket width is a resolution indicator, not an independently measured error estimate.

## PyTorch integration

Use `from jaxpropka.torch import torch_readout` to wrap a prepared JAX readout
as a PyTorch module. Gradients propagate through its input to an upstream
PyTorch model. Install `.[torch]`; see [usage, supported devices and gradient
validation](docs/TORCH.md).

## Biotite topology and multiple chains

Biotite handles PDB/mmCIF/BCIF coordinates, AtomArray residue annotation, CCD residue templates and intra-residue bond topology. A small PDB `TER` reader supplements Biotite because its AtomArray representation does not retain those records. Peptide bonds are rebuilt using chain identity, explicit segment boundaries and C–N geometry; spatial neighbors across chains still contribute noncovalent interactions.

The implementation retains insertion codes and multi-character mmCIF **author chain IDs**. It supports repeated residue numbers on different chains. Ambiguous repeated full keys are rejected. Each genuine peptide segment gets its own N-/C-terminal sites. Charge aggregation remains by input chain ID, so multiple segments under one chain label contribute to the same chain output.

```python
cache = prepare(
    "complex.cif",
    topology_options={
        "model": 1,
        "altloc": "occupancy",
        "gap_policy": "error",
        "freeze_disulfides": True,
        # "chains": ["AA", "BB"],  # physical environment selection, not readout
        # "break_after": [("AA", 120)],
        # "capped_n": [("AA", 1)],
    },
    geometry_options={
        "preserve_native": True,
        "missing_sidechain": "error",
    },
    kernel_options={
        "dtype": np.float32,
        # "max_env_neighbors": 256,  # overflow raises; NEVER silently truncates
        # "max_pair_neighbors": 128,
    },
)
```

An unexplained intrachain gap raises by default. `gap_policy="cap"` suppresses invented ionizable termini at missing segments; it does **not** build a chemical capping group or restore missing atoms. `"free"` explicitly treats the gap as free ends. `capped_n/capped_c` suppress specified terminal sites but likewise do not provide atomistic cap chemistry. The external-reference exporter rejects such mismatches rather than comparing to a freely terminated PDB.

Water and explicit hydrogen records are removed. Noncanonical residues, ligands/metals, cyclic peptides and unsupported crosslinks are rejected by default, not assigned guessed parameters. `ignore_nonprotein=True` is an explicit omission recorded in cache metadata, not support for those molecules. Disulfides require explicit freezing; cysteine identities are fixed and thiol titration is suppressed. The model uses the provided coordinate assembly: it does not expand biological assemblies, crystal neighbors or periodic boxes.

## Candidate geometry and sparse kernels

A native all-atom structure does not contain coordinates for every possible mutation. For each allowed canonical identity, preprocessing builds one frozen CCD conformation in the residue's N–CA–C frame; native heavy atoms are preserved. This is not sequence-dependent repacking or an optimized rotamer library. Missing native side-chain atoms require `missing_sidechain="template"`; virtual terminal OXT construction and rebuilding are recorded.

Complete candidate-side-chain coordinate overrides are supported:

```python
cache = prepare("protein.pdb", geometry_options={
    "overrides": {
        # (zero_based_position, three_letter_identity): complete side-atom map
        # (17, "LYS"): {"CB": cb, "CG": cg, "CD": cd, "CE": ce, "NZ": nz}
    }
})
```

All distances, distance cutoffs, angular factors and per-candidate atomistic contributions are reduced outside JIT. Neighbor graphs are conservative unions over candidate reach bounds, not arbitrary small K-nearest-neighbor truncations. Environment and interaction graphs have separate capacities. Side-chain atom axes do not survive into evaluation; no global `N×N×atoms²` tensor is built.

See [model equations and limitations](docs/MODEL.md) for the exact relaxation and cost model.

## PROPKA regression

The standard reference path uses the PROPKA version installed by the `reference` extra:

```bash
jaxpropka regress tests/data/two_chains.pdb \
  --steps 128 --output reports/propka.json
```

The adapter exports the **same frozen candidate coordinates** used by the model, runs PROPKA out of process, and restores original chain/number/insertion identities through a bijection. Each PDB segment receives a distinct short chain code. The PDB bridge is limited to 62 segments and PDB atom/residue capacities; those are reference-export limits, not JAX multi-chain array limits.

Reports contain per-site midpoint discrepancies, pKa MAE/RMSE/max error, HH charge-curve discrepancies reconstructed from reference pKas, version/commit, input SHA256, structural-cache fingerprint and complete captured reference output. A categorical comparison requires a hard sequence. `--sequence ADKHE...` evaluates a mutant using the same frozen candidates. Missing required sites and invalid/nonconverged solves fail rather than disappearing from the statistics.

Record a regression baseline only after scientific review:

```bash
jaxpropka regress tests/data/two_chains.pdb \
  --steps 128 --record-baseline tests/data/approved-propka-3.5.1.json

# Subsequent runs compare against that frozen, provenance-matched report.
jaxpropka regress tests/data/two_chains.pdb \
  --steps 128 --baseline tests/data/approved-propka-3.5.1.json
```

Existing baselines are not overwritten. `--max-mae` adds an explicit scientific acceptance threshold; none is invented as a default. An approved external PROPKA baseline is **not included** because accepting one requires scientific review. The included synthetic JSON snapshot is labeled implementation-only and must not be treated as PROPKA ground truth.

The `legacy30` backend remains available for an explicit historical comparison with the original source checkout. It is outside the standard development setup and is never selected by default.

## Tests, benchmarks and CLI

```bash
# No molecular dependencies exercised by this subset:
pytest -ra -m 'not integration and not reference and not mps'

# Full installation: fail, rather than silently skip, when dependencies are absent.
JAXPROPKA_REQUIRE_INTEGRATION=1 JAXPROPKA_REQUIRE_REFERENCE=1 pytest -ra

# Pinned pip PROPKA integration and discrepancy report:
pytest -ra tests/test_external_reference.py

# MPS execution must actually use Metal, not silently fall back to CPU:
JAX_PLATFORMS=mps python benchmarks/bench.py --cache structure-cache.npz --require-mps
JAX_PLATFORMS=mps pytest -ra -m mps

# CPU numerical scaling and separate forward/gradient timings:
python benchmarks/bench.py --synthetic-n 64 --output reports/cpu.json
python benchmarks/bench.py --pdb protein.pdb --output reports/protein.json

# Same-coordinate reference process timing, not a claim of algorithmic equivalence:
python benchmarks/compare.py protein.pdb \
  --output reports/comparison.json

jaxpropka prepare tests/data/two_chains.pdb complex.npz
jaxpropka predict complex.npz --pka --residues 1,3,6,8 --output prediction.npz
```

Benchmarks separate host preprocessing, compilation and synchronized warm device execution. Charge, curves, selected direct pKas, all-grid pKas and sequence batching have distinct timings. `--all-pka` explicitly enables the much more expensive all-channel direct-root benchmark. Compiled temporary-memory estimates are reported where supported; they are not physical peak accelerator allocation measurements.

[Benchmark results](docs/BENCHMARKS.md) and [validation inventory](docs/VALIDATION.md) describe what was actually executed. Included GitHub workflows are configurations to run after pushing this repository, **not a claim of completed CI runs**. The original-3.0 workflow requires an explicit reviewed commit.

For opt-in checkpointing, fully packed local terms, implicit derivatives and
streamed bound/free objectives, see [memory-efficient gradients](docs/GRADIENTS.md).
Existing backend and derivative defaults are unchanged.
`TitrationModel.for_design(cache)` additionally offers a CPU-qualified design
preset with bounded adaptive equilibrium solving and explicit branch audits;
its promotion status is described in the same guide.

## Layout

```text
src/jaxpropka/       topology, frozen candidates, sparse kernels, JAX model, CLI
                    portable cache and external reference adapters
benchmarks/         synchronized kernel and reference-process comparisons
tests/              analytic, gradient, regression, topology, reference, MPS tests
scripts/            explicit reference/data fetch and synthetic baseline recording
examples/           sequence-loss integration example
docs/               equations, approximation boundaries and validation status
reports/            executed test logs and synthetic CPU benchmark data
```

Upstream parameter and API references are recorded in [SOURCES.md](SOURCES.md). This project does not vendor PROPKA or Biotite source code. See each dependency's license separately.

The [ATLAS preparation guide](tools/atlas/README.md) records the full-dataset
Isambard pipeline, frozen validation split, nested 10%/50%/75% training portions,
compressed graph release and completed GPU smoke checks.

## Trained transformer prediction

The repository includes the production oGQT checkpoint at
`models/ogqt-production-50k`. Use `ogqt predict` (or `python -m pkanet.cli predict`)
for PDB/mmCIF/BCIF inputs, cached features and mixed-size batches.
See [standalone inference](docs/INFERENCE.md) for CPU/GPU commands and compilation behaviour.
