"""Import the audited native+min pilot into a packed, independently usable release.

Usage: python -m pkatrain.augmentation_pilot pack PILOT OUTPUT
       python -m pkatrain.augmentation_pilot verify OUTPUT
All dataset processing runs in a Slurm allocation. The pilot is historical input,
not the production generator; ambiguous source mappings block graph publication.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from .augmentation_store import AugmentationStore, ReleaseWriter, fingerprint, verify_records
from .augmentation_quality import verify_coordinate_record


def parent_id(row):
    return f"{row['dataset']}/{row['id']}/{row['state']}"


def inventory(pilot):
    pilot = Path(pilot); rows = json.loads((pilot/"sample.json").read_text())
    parents = [parent_id(row) for row in rows]
    if len(set(parents)) != len(parents): raise ValueError("duplicate sampled parent")
    audits = {r["index"]: r for r in json.loads((pilot/"coordinate-audit.json").read_text())["results"]}
    items = []
    for row, parent in zip(rows, parents):
        unit = pilot/"units"/f"{row['index']:03d}"
        report = json.loads((unit/"report.json").read_text())
        if report["source"] != row or report["status"] == "running": raise ValueError("unfinished or changed pilot")
        if report["status"] == "complete" and audits[row["index"]]["status"] != "passed":
            raise ValueError("pilot coordinate audit not passed")
        items.append(dict(parent=parent, source=row, report_sha256=digest(unit/"report.json"),
                          audit=audits[row["index"]]))
    provenance = dict(kind="native-min-coordinate-pilot-v1", inventory=items,
        sample_sha256=digest(pilot/"sample.json"), audit_sha256=digest(pilot/"coordinate-audit.json"),
        population_sha256=digest(pilot/"population.json"),
        pilot_provenance_sha256=digest(pilot/"provenance.json"),
        storage="selected float32 heavy atoms, source mapping and zero control; zstd level 3",
        training_enabled=False, bound_geometry="native", validation_role="separate robustness only",
        source_mapping="historical pilot node order; canonical production alignment required before graph use")
    return rows, provenance


def pilot_record(pilot, row):
    from openmm import app
    unit = Path(pilot)/"units"/f"{row['index']:03d}"
    report = json.loads((unit/"report.json").read_text())
    zero = next((m for m in report["metrics"] if m["candidate"] == 0), None)
    base = dict(source=row, config=report["config"], selected=[],
                status="preparation_failed", error=report.get("error"),
                failures=report["failures"], report_sha256=digest(unit/"report.json"))
    if report["status"] != "complete": return base, {}
    if zero is None: return dict(base, status="missing_zero_control"), {}
    info = json.loads((unit/"preparation.json").read_text())
    if info != report["preparation"]: raise ValueError("preparation/report mapping differs")
    heavy = np.flatnonzero(np.asarray(info["is_native"]) & np.asarray(info["is_heavy"]))
    remap = {int(a): i for i, a in enumerate(heavy)}
    owner = np.asarray(info["residue"], np.int32)[heavy]
    names = [info["atom_names"][j] for j in heavy]
    lookup = {(int(r), a): j for j, (r, a) in enumerate(zip(owner, names))}
    bbids = np.asarray([[lookup[i, a] for a in ("N", "CA", "C", "O")] for i in range(info["n"])], np.int32)
    inserted = {tuple(a) for a in info["inserted_atoms"]}
    present = np.asarray([(int(r), a) not in inserted for r, a in zip(owner, names)], bool)
    top = app.PDBFile(str(unit/"prepared.pdb")).topology
    bonds = np.asarray([(remap[a.index], remap[b.index]) for a, b in top.bonds()
                        if a.index in remap and b.index in remap], np.int32).reshape(-1, 2)
    selected = report["selected"] if zero["geometry_valid"] else []
    if len(selected) != len(set(selected)): raise ValueError("duplicate selected conformers")
    if any(not m["geometry_valid"] for m in report["metrics"] if m["candidate"] in selected):
        raise ValueError("selected invalid candidate")
    baseline = np.load(unit/"prepared.npz")
    arrays = dict(native_coordinates=baseline["xyz"][heavy].astype(np.float32),
        source_backbone=baseline["source_bb"].astype(np.float32),
        zero_coordinates=np.load(unit/"native/c00/min.npy")[heavy].astype(np.float32),
        coordinates=np.stack([np.load(unit/"native"/f"c{c:02d}/min.npy")[heavy] for c in selected]).astype(np.float32)
            if selected else np.empty((0, len(heavy), 3), np.float32),
        atom_residue=owner, source_present=present, completed=~present, backbone_atoms=bbids,
        segments=np.asarray(info["segments"], np.int32), bonds=bonds)
    baseline.close()
    meta = dict(base, status="usable" if len(selected) >= 3 else
                ("zero_control_failed" if not zero["geometry_valid"] else "insufficient_candidates"),
        selected=selected, atom_names=names, residue_names=info["names"], residue_keys=info["keys"],
        disulfides=info["disulfides"], zero_quality=zero,
        selected_quality={str(m["candidate"]): m for m in report["metrics"] if m["candidate"] in selected},
        selected_pairwise_ca_rmsd_A=report["selected_pairwise_ca_rmsd_A"] if selected else [],
        preparation_sha256=digest(unit/"preparation.json"), prepared_coordinates_sha256=digest(unit/"prepared.npz"),
        source_sha256=digest(row["path"]), capped_singletons=info["capped_singletons"],
        production_s=report["production_s"], minimisation="100-step capped cleanup; convergence separate")
    # Test the persisted float32 baseline, control and selected structures against
    # the independent full-precision audit. No near-threshold classification changes.
    verify_coordinate_record(dict(metadata=meta, arrays=arrays))
    return meta, arrays


def pack(pilot, output):
    rows, provenance = inventory(pilot)
    if (Path(output)/"manifest.json").exists():
        with AugmentationStore(output) as store:
            if fingerprint(store.manifest["provenance"]) != fingerprint(provenance):
                raise ValueError("published release differs from input")
        return verify(output)
    with ReleaseWriter(output, provenance) as writer:
        for number, row in enumerate(rows, 1):
            parent = parent_id(row)
            if parent in writer.ids: continue
            meta, arrays = pilot_record(pilot, row)
            writer.add(parent, meta, arrays, summary=dict(dataset=row["dataset"], split=row["split"],
                state=row["state"], status=meta["status"], selected=len(meta["selected"]), n=row["n"]))
            if number % 25 == 0: print(json.dumps(dict(packed=number, total=len(rows))), flush=True)
        writer.finish(expected_ids=[parent_id(row) for row in rows])
    return verify(output)


def verify(output):
    with AugmentationStore(output) as store:
        counts = dict(frames=0, selected=0); cohorts = {}
        def check(record, row):
            counts["frames"] += verify_coordinate_record(record)
            summary = row["summary"]; dataset = summary["dataset"]
            cohorts.setdefault(dataset, Counter())[summary["status"]] += 1
            counts["selected"] += summary["selected"]
        verify_records(output, store.rows, check_record=check, expected_files=store.manifest["files"])
        result = dict(passed=True, parents=len(store.rows), checked_frames=counts["frames"], selected=counts["selected"],
            cohorts={k: dict(v) for k, v in cohorts.items()}, compressed_bytes=store.manifest["compressed_bytes"],
            raw_bytes=store.manifest["raw_bytes"], manifest_sha256=digest(Path(output)/"manifest.json"),
            infrastructure_ready=False, pending=["canonical graph alignment and loader pilot", "Isambard effective quota", "Isambard GPU preflight"])
    atomic_json(Path(output)/"coordinate-verification.json", result)
    return result


def build_graphs(coordinates, output, runtime, atlas_tar):
    import tarfile
    from .augmentation_graphs import INPUT_FIELDS, free_graph, pair_inputs
    from .augmentation_sources import canonical_source
    from .production_inputs import _sequence_index
    from .production_graphs import arrays_digest
    with AugmentationStore(coordinates) as store, tarfile.open(atlas_tar, mode="r:") as archive:
        provenance = dict(kind="augmentation-graph-pilot-v1", coordinate_manifest_sha256=digest(Path(coordinates)/"manifest.json"),
            atlas_tar_sha256=digest(atlas_tar), native_training_enabled=False,
            features="canonical branch-specific backbone and sidechains; geometry-derived supervision omitted")
        if (Path(output)/"manifest.json").exists():
            with AugmentationStore(output) as old:
                if old.manifest["provenance"] != provenance: raise ValueError("graph release provenance differs")
            return verify_graphs(output)
        expected = []; buckets = {}; alignment = []
        with ReleaseWriter(output, provenance) as writer:
            for number, row in enumerate(store.rows, 1):
                record = store.raw(row["id"])
                if record["metadata"]["status"] != "usable": continue
                meta = record["metadata"]; arrays = record["arrays"]
                canonical, bb, chain, partners, atoms, mapping = canonical_source(runtime, record, atlas_archive=archive)
                bound = {k: canonical[k] for k in INPUT_FIELDS}
                native_free = dict(bound)
                if "branch_edge_mask" in canonical:
                    native_free["edge_mask"] = canonical["branch_edge_mask"][1]
                    native_free["site_edge_mask"] = canonical["branch_site_edge_mask"][1]
                sequence = _sequence_index(chain)
                control = pair_inputs(bound, native_free, bb, bb, chain, sequence, atoms, atoms)
                control_hash = fingerprint({k: hashlib_array(v[0]) for k, v in control.items()})
                candidates = [("native", None)] + [(f"c{c:02d}", i) for i, c in enumerate(meta["selected"])]
                for label, index in candidates:
                    parent = row["id"]+"/"+label; expected.append(parent)
                    if parent in writer.ids: continue
                    if index is None: payload = control
                    else:
                        moved = bb.copy(); moved[mapping] = arrays["coordinates"][index][arrays["backbone_atoms"]]
                        free_atoms = [dict(a) for a in atoms]
                        for j, (r, name) in enumerate(zip(arrays["atom_residue"], meta["atom_names"])):
                            free_atoms[mapping[r]][name] = arrays["coordinates"][index, j]
                        free = free_graph(bound, moved, chain, partners)
                        payload = pair_inputs(bound, free, bb, moved, chain, sequence, atoms, free_atoms)
                    if fingerprint({k: hashlib_array(v[0]) for k, v in payload.items()}) != control_hash:
                        # Different free edge capacities can add bound padding; compare native slices instead.
                        for key, value in control.items():
                            slice_ = (0, *(slice(0, n) for n in value.shape[1:]))
                            if not np.array_equal(payload[key][slice_], value[0]):
                                raise ValueError(f"bound branch changed: {key}")
                    dims = [len(bb), payload["neighbors"].shape[-1], payload["query_residue"].shape[-1],
                            payload["site_residue"].shape[-1], payload["site_neighbors"].shape[-1]]
                    graph_meta = dict(source_parent=row["id"], dataset=meta["source"]["dataset"], split=meta["source"]["split"],
                        kind="native" if index is None else "augmented", candidate=0 if index is None else meta["selected"][index],
                        canonical_node_mapping=mapping.tolist(), native_bound_digest=arrays_digest(bound, INPUT_FIELDS),
                        dims=dims, geometry_labels_reused=False)
                    writer.add(parent, graph_meta, payload, summary=graph_meta)
                alignment.append(dict(parent=row["id"], passed=True, residues=len(mapping)))
                if number % 25 == 0: print(json.dumps(dict(graph_parents=number, records=len(writer.rows))), flush=True)
            # Compute capacities over both native and augmented records, including resumed records.
            for row in writer.rows:
                n, k, q, s, sk = row["summary"]["dims"]
                bound = next((b for b in (128, 256, 384, 512, 640, 768, 1024, 1280, 1536) if n <= b), None)
                if bound is None: raise ValueError("canonical complex exceeds supported production buckets")
                dims = [bound, k, q, s, sk]
                buckets[str(bound)] = [max(a, b) for a, b in zip(buckets.get(str(bound), [0]*5), dims)]
            writer.finish(expected_ids=expected)
        atomic_json(Path(output)/"graph-alignment.json", dict(passed=True, parents=alignment, capacities=buckets))
    return verify_graphs(output)


def hashlib_array(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verify_graphs(output):
    with AugmentationStore(output) as store:
        buckets = {}; parents = {}; control = dict(parent=None, arrays=None)
        def check(record, row):
            graph = record["arrays"]
            n, k, q, s, sk = record["metadata"]["dims"]
            bucket = next((b for b in (128, 256, 384, 512, 640, 768, 1024, 1280, 1536) if n <= b), None)
            if bucket is None: raise ValueError("unsupported canonical production bucket")
            dims = [bucket, k, q, s, sk]
            buckets[str(bucket)] = [max(a, b) for a, b in zip(buckets.get(str(bucket), [0]*5), dims)]
            parents[record["metadata"]["source_parent"]] = record["metadata"]["canonical_node_mapping"]
            for branch in range(2):
                for indices, count in (("neighbors", n), ("site_neighbors", s), ("site_residue", n), ("query_residue", n)):
                    if np.any(graph[indices][branch] < 0) or np.any(graph[indices][branch] >= count):
                        raise ValueError(f"graph index out of bounds: {indices}")
                if np.any(graph["query_site"][branch] < -1) or np.any(graph["query_site"][branch] >= s):
                    raise ValueError("query token index out of bounds")
            for key in ("query_residue", "query_group", "site_residue", "site_type", "query_site"):
                if not np.array_equal(graph[key][0], graph[key][1]): raise ValueError("paired graph identity differs")
            parent = record["metadata"]["source_parent"]
            if record["metadata"]["kind"] == "native":
                control.update(parent=parent, arrays=graph)
            if record["metadata"]["kind"] == "augmented":
                if control["parent"] != parent:
                    control.update(parent=parent, arrays=store.raw(parent+"/native")["arrays"])
                native = control["arrays"]
                for key, value in native.items():
                    slice_ = (0, *(slice(0, n) for n in value.shape[1:]))
                    if not np.array_equal(graph[key][slice_], value[0]): raise ValueError(f"native bound array changed: {key}")
                # The current paired releases have one canonical chain per partner.
                if record["metadata"]["dataset"] in ("pinder", "atlas"):
                    chain = graph["nodes"][1, :, -2]
                    neighbors = graph["neighbors"][1]; mask = graph["edge_mask"][1]
                    if np.any((chain[neighbors] != chain[:, None])[mask]): raise ValueError("saved free graph crosses partners")
                    site_chain = chain[graph["site_residue"][1]]
                    site_neighbors = graph["site_neighbors"][1]; smask = graph["site_edge_mask"][1]
                    if np.any((site_chain[site_neighbors] != site_chain[:, None])[smask]):
                        raise ValueError("saved free site graph crosses partners")
        verified = verify_records(output, store.rows, check_record=check, expected_files=store.manifest["files"])
        report = dict(passed=True, checked_records=verified["records_checked"], manifest_sha256=digest(Path(output)/"manifest.json"),
                      infrastructure_ready=False, pending=["Isambard quota and synchronised GPU throughput"])
    atomic_json(Path(output)/"graph-verification.json", report)
    atomic_json(Path(output)/"graph-alignment.json", dict(passed=True, capacities=buckets, parents=parents,
                manifest_sha256=report["manifest_sha256"]))
    return report


def verify_native(output, runtime):
    """Compare the pilot's native controls with the live frozen production stores."""
    from .production_graphs import ProductionStore, output as native_output
    from .augmentation_graphs import INPUT_FIELDS
    from .production_inputs import BackboneTable
    from .production_sidechains import SidechainTable
    paths = {d: native_output(runtime, d)/"store-v2" for d in ("pkpdb", "pinder")}
    paths["atlas"] = Path(runtime)/"pretraining/atlas-prepared-v1/graph-v1/release-v1/store-v2"
    stores = {}; tables = {}; checked = 0; max_error = 0.0
    try:
        with AugmentationStore(output) as pilot:
            for row in pilot.rows:
                if row["summary"]["kind"] != "native": continue
                dataset, cid, _ = row["summary"]["source_parent"].split("/")
                if dataset not in stores: stores[dataset] = ProductionStore(paths[dataset])
                native = stores[dataset].raw(cid); arrays = pilot.raw(row["id"])["arrays"]
                for name in INPUT_FIELDS:
                    expected = native[name]
                    got = arrays[name][(0, *(slice(0, n) for n in expected.shape))]
                    if name in ("edge", "site_edge"):
                        error = float(np.max(np.abs(got-expected), initial=0))
                        if error > 2e-5: raise ValueError(f"production native geometry mismatch: {cid}/{name}")
                        max_error = max(max_error, error)
                    elif not np.array_equal(got, expected):
                        raise ValueError(f"production native identity/mask mismatch: {cid}/{name}")
                if dataset in ("pinder", "atlas"):
                    for name, field in (("edge_mask", "branch_edge_mask"), ("site_edge_mask", "branch_site_edge_mask")):
                        expected = native[field]
                        if not np.array_equal(arrays[name][tuple(slice(0, n) for n in expected.shape)], expected):
                            raise ValueError(f"production native branch mask mismatch: {cid}/{name}")
                if dataset in ("pkpdb", "pinder"):
                    if dataset not in tables:
                        tables[dataset] = (BackboneTable(runtime, dataset), SidechainTable(runtime, dataset))
                    backbone, sidechains = tables[dataset]
                    if not np.allclose(arrays["nodes"][0, :, -14:], backbone(cid), rtol=0, atol=2e-5):
                        raise ValueError(f"production native backbone table mismatch: {cid}")
                    if not np.allclose(arrays["sidechains"][0], sidechains(cid), rtol=0, atol=0.001):
                        raise ValueError(f"production native sidechain table mismatch: {cid}")
                checked += 1
            report = dict(passed=True, native_parents=checked, max_abs_geometry_difference=max_error,
                          manifest_sha256=digest(Path(output)/"manifest.json"),
                          production_verification_hashes={d: digest(paths[d]/"verification.json") for d in stores})
    finally:
        for store in stores.values(): store.close()
    atomic_json(Path(output)/"native-compatibility.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("pack"); p.add_argument("pilot", type=Path); p.add_argument("output", type=Path)
    p = commands.add_parser("inventory"); p.add_argument("pilot", type=Path); p.add_argument("output", type=Path)
    p = commands.add_parser("verify"); p.add_argument("output", type=Path)
    p = commands.add_parser("graphs"); p.add_argument("coordinates", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--runtime", required=True, type=Path); p.add_argument("--atlas-tar", required=True, type=Path)
    p = commands.add_parser("verify-graphs"); p.add_argument("output", type=Path)
    p = commands.add_parser("verify-native"); p.add_argument("output", type=Path)
    p.add_argument("--runtime", required=True, type=Path)
    args = parser.parse_args(argv); require_compute(threads=1)
    if args.command == "pack": result = pack(args.pilot, args.output)
    elif args.command == "verify": result = verify(args.output)
    elif args.command == "graphs": result = build_graphs(args.coordinates, args.output, args.runtime, args.atlas_tar)
    elif args.command == "verify-graphs": result = verify_graphs(args.output)
    elif args.command == "verify-native": result = verify_native(args.output, args.runtime)
    else:
        rows, provenance = inventory(args.pilot)
        result = dict(parents=len(rows), provenance=provenance); atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
