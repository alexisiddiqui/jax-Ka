# Standalone oGQT prediction

## Default all-residue model

Without `--model`, `ogqt predict` runs the default all-residue GQT
(`models/ogqt-allres15-siamese-50`, or `GQT_DEFAULT_MODEL`):

```bash
ogqt predict complex.pdb --cache prepared/ogqt --output predictions.jsonl
```

`--state` selects what is predicted: `bound` (the structure as supplied), `free`
(every chain alone: edges between chains removed at the supplied conformation) or
`both` (default). Each JSONL record has:

- `sites`: the titratable query sites (as below), with `pka`, `shift` and `gate`
  per state (suffixed `_bound` / `_free` for `both`) and, for `both`,
  `binding_shift` = bound − free pKa;
- `residues` (omit with `--no-residues`): every residue with its `side_chain`,
  `n_terminus` and `c_terminus` outputs where present (group, pKa and titratable-gate
  probability per state), `burial` probability per state and, for `both` on
  multi-chain inputs, `interface_logit`: the free-state contact-edge logit, maximised
  over residues on other chains.

Arg and bonded-Cys side-chain pKa are reported but were never supervised; read them
with the gate. `--backend triton` (CUDA) uses the training kernels; `--backend jax`
uses the plain-JAX equivalent (`pkanet.attention_backend`). The feature cache schema
is now `ogqt-backbone-inference-v2` (it also stores backbone coordinates and chain
indices); older v1 caches still serve the legacy model.

## Legacy 24-feature model

The repository ships a trained backbone-only transformer in
`models/ogqt-production-50k/`: `params.npz` plus `model.json`. This is the
67,770-parameter production oGQT, including a training-time burial head.
Inference predicts pKa shifts and adds the checkpoint's pKPDB model-compound
reference for absolute pKa. It is separate from the `jaxpropka` titration solver.

Install the project (`python -m pip install -e .`), then run from the repository:

```bash
ogqt predict protein1.pdb protein2.cif \
  --model models/ogqt-production-50k \
  --cache prepared/ogqt --batch-size 4 --device cpu \
  --output predictions.jsonl
```

`python -m pkanet.cli` is equivalent to `ogqt`. For many files, use
`--input-list structures.txt`; relative paths in the list resolve against the
list file's directory. PDB, mmCIF and BCIF are supported by the existing topology
loader. Compressed input files must first be decompressed.

Prepare features separately, then reuse the same cache during prediction:

```bash
ogqt prepare --input-list structures.txt --cache prepared/ogqt \
  --output prepared/manifest.json
ogqt predict --input-list structures.txt --cache prepared/ogqt \
  --model models/ogqt-production-50k --device gpu --backend triton \
  --compilation-cache .jax-cache --batch-size 4 --output predictions.jsonl
```

Run compute work in a Slurm allocation on Isambard. `--device gpu` requires
GPU-enabled JAX. `--backend triton` uses the existing indexed-attention kernels
and requires the repository's optional Triton dependencies. `--backend jax`
uses the established portable reference forward pass on CPU or GPU; JIT drops
its unused attention diagnostics. The portable path can require more working
memory than indexed Triton. Missing requested accelerators cause an error.

## Feature and output contract

The featuriser reuses the existing topology loader and radius-graph builder.
Its site-graph functions match `pkatrain.site_graph_data` exactly, without
requiring PyArrow or training datasets. Features remain the checkpoint's
24-column backbone representation, 20 A residue neighbours, and 34-column
site edges. No side-chain feature representation is introduced.

The default `--gap-policy cap` matches production preparation: missing-residue
gaps do not create artificial free termini. Nonprotein material is rejected
unless `--ignore-nonprotein` is explicitly passed. ARG is retained in the site
context but excluded from predictions because the checkpoint has no trained
ARG target or reference. Known disulfide cysteines are also excluded from
prediction queries while retaining their context tokens. Each output includes the original chain, residue
number, insertion code and group. Predictions describe the supplied assembly;
no biological-assembly expansion or partner removal is performed automatically.

Each JSONL record represents one input structure, in original input order,
with `shift` and `pka` per query site, and the parameter checksum. A companion
`.summary.json` records preparation time, explicit compilation time, execution
time, the device/backend and compiled shapes. These are operational timings,
not a controlled accelerator benchmark; execution includes result transfer and
synchronisation. Cached features are identified by content, options, feature
code and Biotite version. Metadata and strings never enter the JIT arguments.

## Mixed sizes and compilation

`pkanet.inference.Predictor` loads weights once and reuses its compiled
executables across calls. `pkanet.features.prepare_file` returns a cache path
and metadata; pass a list of these records to `predictor.predict(records)`.
The generator returns `(input_index, shifts, pKas)` in bucket order; the CLI
spools and reorders the records on disk.

Residue, neighbour, query, site and site-neighbour capacities are rounded to
powers of two and grouped by the resulting tuple. Every batch in a bucket has
the same leading batch dimension; the final partial batch uses independent
dummy lanes whose predictions are discarded. One executable is built per
encountered capacity tuple per predictor. Model parameters and graphs are
runtime array inputs. Feature generation does not invoke the model or JIT.

The loader retains one active and one prefetched host batch. `--max-input-mib`
(default 512) caps each batch's input buffers and rejects overflow rather than
truncating neighbours. This is **not a peak GPU-memory estimate**: attention
intermediates can be larger. Lower `--batch-size` if needed. The first release
uses explicit batch sizes, not an unvalidated automatic GPU memory heuristic.
Persistent compilation caching is opt-in and still depends on compatible JAX,
backend and executable settings. It does not eliminate first-time compilation.

### Persistent JIT cache

Prediction enables JAX's disk compilation cache before any feature/model work.
It uses `--compilation-cache PATH`, an existing JAX cache configuration or
`JAX_COMPILATION_CACHE_DIR`, then `$XDG_CACHE_HOME/jax-Ka/ogqt` (falling back to
`~/.cache/jax-Ka/ogqt`). On Isambard, use a stable directory on scratch across jobs.
The cache stores even sub-second compilations. Repeating configuration with the
same resolved path is a no-op; changing directories in one process raises an
error because JAX's cache is process-wide. `--no-compilation-cache` disables disk
caching for a CLI run. For library use, configure before other JAX compilation;
`Predictor(..., compilation_cache=PATH)` configures it automatically, while
`compilation_cache=False` leaves the host application's JAX settings untouched.

JAX keys entries by the compiled computation and its compiler/device context.
Shapes, batch sizes, precision and relevant code changes can require new entries.
Weights are runtime arguments: different compatible weights reuse code without
reusing predictions. Do not include a timestamp, job ID or input filename in the
cache directory. Use a cache writable only by trusted users, since entries contain
executable code. See [JAX's cache documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

`compiled_shapes` counts executable preparations in the current Predictor,
including loads from disk; `compile_seconds` includes lowering/cache lookup/loading,
so neither field alone proves a disk-cache miss. A restart still traces and lowers
before loading an executable. The persistent cache avoids repeated XLA compilation;
it does not eliminate new bucket shapes or padding overhead.
