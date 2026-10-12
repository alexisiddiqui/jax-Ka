# 6SC6 ABC-context, A–B interface gradients

The forward pass retains all 604 protein residues in chains A, B and C. A 5 Å
heavy-atom A–B contact mask selects 39 residues (18 A, 21 B). The scalar objective
is the mean predicted pKa shift at 14 supported interface query sites. Sequence,
backbone and side-chain input gradients are masked to this interface; chain C
remains part of the forward context. This is a numerical derivative check, not a
binding-energy or experimental-accuracy validation.

The backbone model uses `models/ogqt-production-50k`. The 152-feature side-chain
model has initialized parameters (width 44, feed-forward width 88, seed 4).
Water and nonprotein components are omitted; source hash, topology decisions and
residue identifiers are recorded in `interface.json`.

## Reproduce

Install the package with its Torch and test extras in a compatible environment.
Use a GPU allocation for CUDA runs. Obtain the original PDB from
https://files.rcsb.org/download/6SC6.pdb and save it under `$CHECK_INPUT/6SC6.pdb`.
Run from the repository root:

```bash
export CHECK_INPUT=/path/to/6sc6-input
export CHECK_OUT=/path/to/6sc6-results
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
python benchmarks/prepare_6sc6_gradients.py "$CHECK_INPUT"
for variant in backbone sidechain; do
  for mode in forward backward check; do
    JAX_PLATFORMS=cuda python benchmarks/check_6sc6_gradients.py --variant "$variant" --mode "$mode"
  done
  JAX_PLATFORMS=cuda python benchmarks/check_6sc6_gradients.py --variant "$variant" --mode backward --run 2
done
```

For CPU checks use `JAX_PLATFORMS=cpu` and append `--device cpu`, with a separate
output directory. `CHECK_MODEL` optionally overrides the backbone checkpoint.
Each command uses a fresh process; backward executables and forward executables
share the persistent cache under `$CHECK_OUT/jit-cache`. GPU memory is sampled
for the whole process with nvidia-smi and accompanied by JAX allocator peaks;
Torch-only counters exclude JAX allocations. Sampled peaks may miss transients.

Checks compare central differences at epsilon 0.01 and 0.003 with normalized
gradient-aligned directions, separately for each chain/input group. Sequence
perturbations are zero-sum per residue. Acceptance is absolute error <= 5e-5 +
2% of the analytic derivative. These are directional checks, not a complete
Jacobian check. All gradients must be finite and exactly zero off the interface.

## Results

See [measured memory, gradient checks and cache reuse](RESULTS.md).
