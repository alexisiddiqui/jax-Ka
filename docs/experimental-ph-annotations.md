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
