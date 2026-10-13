"""Packed original-input masks, distinct from atoms present at repair input.

The pilot's source_present flag describes its repair input, which may already
contain upstream completion. This table preserves original observation evidence
for later featurisation. Unknown upstream presence is explicit, never guessed.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from .augmentation_store import AugmentationStore, ReleaseWriter


def upstream_mask(record, pilot, archive):
    meta = record["metadata"]; row = meta["source"]; arrays = record["arrays"]
    if not arrays: return {}, dict(status=meta["status"], source_hashes={})
    if row["dataset"] == "pkpdb":
        return dict(original_input_present=arrays["source_present"].copy(),
                    original_input_known=np.ones(len(arrays["source_present"]), bool)), \
               dict(source_hashes=dict(native_source=meta["source_sha256"]), policy="uncompleted pKPDB native source")
    keys = meta["residue_keys"]; hashes = {}
    if row["dataset"] == "pinder":
        folder = Path(row["path"]).parent; mask_path = folder/"input_atom_mask.json.gz"
        evidence = json.loads(gzip.open(mask_path, "rt").read())
        mapping_path = folder/f"map_{row['state']}.json"
        mapping = json.loads(mapping_path.read_text())
        # map_A/B starts with renumbered teacher keys. A.cif/B.cif retain the
        # original AB destination keys, including long chain IDs and numbering.
        allowed = {(dc, int(dn), ins) for _, _, dc, dn, ins in mapping}
        if any(tuple(k) not in allowed for k in keys):
            raise ValueError("PINDER repair input key is not in its native partner mapping")
        hashes = dict(input_mask=digest(mask_path), partner_mapping=digest(mapping_path))
    else:
        blob = archive.extractfile(row["id"]+".tar.gz").read()
        if hashlib.sha256(blob).hexdigest() != row["archive"]["sha256"]: raise ValueError("ATLAS archive hash changed")
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            evidence = json.load(tf.extractfile("input_atom_mask.json"))
        hashes = dict(native_archive=row["archive"]["sha256"])
    lookup = {}
    for residue in evidence["atoms"]:
        if len(residue["expected_atoms"]) != len(residue["input_atom_mask"]):
            raise ValueError("upstream mask dimensions changed")
        for name, present in zip(residue["expected_atoms"], residue["input_atom_mask"]):
            key = (*residue["key"], name)
            if key in lookup: raise ValueError("ambiguous upstream atom evidence")
            lookup[key] = bool(present)
    observed = []; known = []
    for i, (owner, name) in enumerate(zip(arrays["atom_residue"], meta["atom_names"])):
        key = (*keys[owner], name)
        # Repair-input absence does not prove original-source absence. Keep
        # upstream evidence separate from the already stored repair completion flag.
        observed.append(lookup.get(key, False)); known.append(key in lookup)
    return dict(original_input_present=np.asarray(observed, bool), original_input_known=np.asarray(known, bool)), \
        dict(source_hashes=hashes, policy="original preprocessing input mask; unknown is not observed",
             artificial_terminal_keys=evidence.get("artificial_terminal_keys", []))


def build(coordinates, output, pilot):
    pilot = Path(pilot)
    with AugmentationStore(coordinates) as store, tarfile.open(pilot/"atlas-records.tar", "r:") as archive:
        provenance = dict(kind="original-input-evidence-v2", coordinate_manifest_sha256=digest(Path(coordinates)/"manifest.json"),
                          mask_semantics="upstream observation independent of repair completion; unlisted atoms unknown",
                          atlas_tar_sha256=digest(pilot/"atlas-records.tar"))
        if (Path(output)/"manifest.json").exists():
            with AugmentationStore(output) as old:
                if old.manifest["provenance"] != provenance: raise ValueError("evidence provenance differs")
                old.verify_files()
            return verify(coordinates, output)
        with ReleaseWriter(output, provenance) as writer:
            for row in store.rows:
                if row["id"] in writer.ids: continue
                record = store.raw(row["id"]); arrays, meta = upstream_mask(record, pilot, archive)
                meta.update(coordinate_record_sha256=row["sha256"], atoms=len(record["arrays"].get("atom_residue", [])))
                writer.add(row["id"], meta, arrays)
            writer.finish(expected_ids=[r["id"] for r in store.rows])
    return verify(coordinates, output)


class CoordinateInputs:
    """Coordinates plus original observation masks, pinned to identical atom order."""
    def __init__(self, coordinates, evidence):
        self.coordinates = AugmentationStore(coordinates); self.evidence = AugmentationStore(evidence)
        if self.evidence.manifest["provenance"]["coordinate_manifest_sha256"] != digest(Path(coordinates)/"manifest.json"):
            self.close(); raise ValueError("evidence belongs to a different coordinate release")
        if set(self.coordinates.by_id) != set(self.evidence.by_id):
            self.close(); raise ValueError("evidence parent inventory differs")

    def raw(self, parent):
        record = self.coordinates.raw(parent); evidence = self.evidence.raw(parent)
        if evidence["metadata"]["coordinate_record_sha256"] != self.coordinates.by_id[parent]["sha256"]:
            raise ValueError("atom evidence belongs to another coordinate record")
        n = len(record["arrays"].get("atom_residue", []))
        for name, value in evidence["arrays"].items():
            if value.shape != (n,) or value.dtype != np.bool_: raise ValueError("invalid original-input mask")
        return dict(metadata=dict(record["metadata"], original_input_evidence=evidence["metadata"],
                        source_present_semantics="repair-input presence; original_input_present is upstream observation"),
                    arrays={**record["arrays"], **evidence["arrays"]})

    def close(self): self.coordinates.close(); self.evidence.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def verify(coordinates, output):
    observed = 0; unknown = 0; atoms = 0
    with CoordinateInputs(coordinates, output) as inputs:
        inputs.evidence.verify_files()
        for parent in inputs.coordinates.by_id:
            record = inputs.raw(parent); arrays = record["arrays"]
            if not arrays: continue
            observed += int(arrays["original_input_present"].sum())
            unknown += int((~arrays["original_input_known"]).sum()); atoms += len(arrays["atom_residue"])
        result = dict(passed=True, parents=len(inputs.coordinates.rows), atoms=atoms, observed_atoms=observed,
            unknown_atoms=unknown, coordinate_manifest_sha256=digest(Path(coordinates)/"manifest.json"),
            evidence_manifest_sha256=digest(Path(output)/"manifest.json"))
    atomic_json(Path(output)/"evidence-verification.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("coordinates", type=Path)
    p.add_argument("output", type=Path); p.add_argument("pilot", type=Path)
    args = p.parse_args(argv); require_compute(threads=1)
    print(json.dumps(build(args.coordinates, args.output, args.pilot), indent=2))


if __name__ == "__main__": main()
