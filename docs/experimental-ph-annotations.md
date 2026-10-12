# Experimental pH annotations

`pkabench.experimental_ph` adds versioned, structure-level pH targets to the
current pKPDB and PINDER pools. It preserves the input structures, pKa targets,
feature stores, pool membership and existing loss functions. PINDER's frozen
400-entry validation cohort is annotated separately. The frozen pKPDB pilot
(`pkpdb-5k-v3`) is outside this annotation build.

Sidecars live at
`$PKABENCH_RUNTIME/annotations/experimental-ph-v1/{pkpdb,pinder}/`:

- `records.jsonl`: explicit dataset IDs, parent PDB ID, split and loss-ready metadata.
- `pdb-evidence.jsonl`: original deposited metadata and audit findings, once per PDB.
- `manifest.json`: source hashes, policy, counts and hashes of the two tables.

Dataset export/import includes these files in the core bundle when present.
The read-only squashfs dataset images do not need rebuilding.

## Fields and eligibility

`ph_value` retains a single structured candidate where available. `ph_target` is
populated only when `ph_mask` is true. `ph_quality_weight` is **binary eligibility**
(0 or 1), not a calibrated probability or confidence estimate.

Eligibility requires one consistent structured point and no unresolved warnings.
Sources include crystal growth pH, NMR sample pH and EM buffer pH. Equal endpoints
and exact single values recorded in a range field can be normalized to a point.
The source remains in `ph_sources`.

Ranges, note-only candidates, different values in the notes, multiple structured
values, malformed fields, pD/pH* or unspecified NMR units, and the pH 0.0 outlier
are masked. Raw evidence is retained for later manual curation. Experimental
stages can differ; these metadata are not independent residue-specific pKa
measurements or a paper-by-paper experimental verification.

`validation_chain_similarity` preserves the existing pool flag. A consumer can
exclude flagged records without rewriting the annotation. `parent_weight` is
1 divided by the number of entries sharing a parent PDB within the same split.
It is separate from quality weight and is not automatically applied to losses.

## Build and verify (Slurm compute node)

The builder uses the cached `audits/ph-coverage` metadata snapshot, supplemented
with `extra-metadata.json` for validation records and recovered original mmCIF
metadata for retired PDB IDs. It refuses to silently classify an unavailable PDB
record as missing pH.

```sh
python -m pkabench.experimental_ph build pinder
python -m pkabench.experimental_ph build pkpdb
python -m pkabench.experimental_ph verify pinder
```

A repeat build with identical inputs verifies and reuses the sidecar. Changed
inputs require a new annotation version. Loaders verify file hashes, pool hashes,
unique IDs and target/mask consistency. Large annotation files stay in the runtime,
not the source repository.

## Future auxiliary loss

```python
from pkabench.experimental_ph import load_annotations, targets_for_feature_records

annotations = load_annotations(runtime, "pinder")
ph = targets_for_feature_records(
    annotations, selection["pinder-train"]["names"],
    exclude_validation_similar=True,
)
```

This returns float32 `ph_target`, boolean `ph_mask`, binary `ph_quality_weight`
and optional `ph_parent_weight` in exactly the existing feature-record order.
Missing IDs or a split mismatch raise an error. Masked targets use finite zero
fillers in arrays; the mask determines eligibility. JSON stores masked targets
as null.

A future auxiliary head should produce one prediction per complex/structure and
use masked, normalized squared error. Avoid repeating an experimental condition
as independent supervision for every residue. Apply parent weights deliberately
if balancing repeated PDBs. Preserve source categories for source-specific
sensitivity checks, and return no auxiliary contribution when a batch has no
eligible structures. No auxiliary head or loss is activated by this change.

Definitions: [crystal growth pH](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_exptl_crystal_grow.pH.html),
[NMR collection pH](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_pdbx_nmr_exptl_sample_conditions.pH.html),
[EM buffer pH](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_em_buffer.pH.html).

## Pool-v4 training annotations (`experimental-ph-v2`, 2026-10-12)

`experimental-ph-v1` was built for pool-v3. Pool-v4 moved PINDER's retired
400-entry validation cohort into training and changed membership, so v1 covers
27,120 of 27,933 PINDER and 55,374 of 56,191 pKPDB pool-v4 training entries.
`experimental-ph-v2` re-annotates the pool-v4 training tables
(`training/pool-v4/{pinder,pkpdb}.tsv`, split `train`) with the same
`audit_record` / `quality_record` policy. Parent weights are recounted over pool-v4
training membership. The pool-v4 validation sets keep their
`annotations/validation-ph-v1` records.

PDBs absent from the `audits/ph-coverage` snapshot (546) are fetched once into
`audits/ph-coverage/pool-v4-metadata.json` (RCSB GraphQL in the snapshot's record
format; EM buffer pH from the mmCIF header files). A PDB that RCSB does not serve
is listed as unresolved and the build refuses it.

```sh
python -m pkabench.experimental_ph supplement-v4        # network, HTTP only
python -m pkabench.experimental_ph build-v4 pinder      # Slurm
python -m pkabench.experimental_ph build-v4 pkpdb
python -m pkabench.experimental_ph verify-v4 pinder
```

## Training auxiliary (`pkatrain.production_ph`, `production_train --ph-weight`)

`production_train train ... --aux-weight W --ph-weight P` adds a structure-level
pH loss. `PhTable` joins v2 training and validation-ph-v1 records by structure ID
and raises if any manifest record (training fraction or validation) lacks an
annotation, or if its split differs. The head pools the site tokens with gated
attention: LayerNorm, then a sigmoid gate times a value, normalised per channel.
It predicts cumulative logits for [pH > t], t = 4.75 / 5.75 / 6.75 / 7.75 / 8.25,
once per structure, on the PINDER AB branch and on pKPDB. The loss is the mean
BCE over thresholds, weighted by eligibility x parent weight, and is 0 for a batch
without eligible structures.

Validation adds a `ph` section for PINDER val, pKPDB val and the 142-structure
benchmark. The expected pH comes from the threshold-bin probabilities and the
training bin means; the comparison baseline is the training-mean pH. Both are
fitted on training labels only and stored in the run's `ph-annotations.json`.
The section reports MAE / RMSE / Spearman, per-threshold AUROC and mean BCE,
overall and split by source (crystal / EM buffer / NMR), because EM buffer pH is
concentrated at 7-8. ATLAS (predicted structures) stays masked. PDBbind has no
production graph store yet.
