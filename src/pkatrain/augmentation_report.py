"""Conservative infrastructure readiness report for the accepted pilot release."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pkabench.runtime import atomic_json, digest, require_compute
from .augmentation_store import AugmentationStore


def worker_budget(free_cpus, available_bytes, peak_rss_GiB):
    """Apply per-node, then sum; leave 10% RAM plus 25% per-worker margin."""
    if free_cpus < 1 or available_bytes <= 0 or peak_rss_GiB <= 0: return 0
    return min(free_cpus, int(available_bytes*0.90/(peak_rss_GiB*1024**3*1.25)))


def report(root, pilot):
    root = Path(root); pilot = Path(pilot)
    checks = {}; pending = []; recommended = None
    for name, path in (("coordinates", root/"coordinates/coordinate-verification.json"),
                       ("source_evidence", root/"evidence/evidence-verification.json"),
                       ("graphs", root/"graphs/graph-verification.json"),
                       ("production_compatibility", root/"graphs/native-compatibility.json"),
                       ("gpu", root/"gpu-preflight.json"), ("quota", root/"quota-budget.json")):
        value = json.loads(path.read_text()) if path.exists() else None
        checks[name] = dict(passed=bool(value and value.get("passed")),
                            sha256=digest(path) if value else None)
        if not checks[name]["passed"]: pending.append(name)
    with AugmentationStore(root/"coordinates") as store:
        if checks["coordinates"]["passed"]:
            verified = json.loads((root/"coordinates/coordinate-verification.json").read_text())
            if verified["manifest_sha256"] != digest(root/"coordinates/manifest.json"):
                raise ValueError("coordinate verification belongs to a different release")
        population = json.loads((pilot/"population.json").read_text()); projection = {}
        for row in store.rows:
            record = store.raw(row["id"]); source = record["metadata"]["source"]
            key = f"{source['dataset']}|{source['split']}|{source['bin']}"
            weight = population["strata"][key]/population["sample_counts"][key]
            item = projection.setdefault(source["dataset"], dict(parents=0, usable=0, raw_bytes=0, compressed_bytes=0))
            item["parents"] += weight; item["usable"] += weight*(record["metadata"]["status"] == "usable")
            item["raw_bytes"] += weight*row["raw_bytes"]; item["compressed_bytes"] += weight*row["bytes"]
        for item in projection.values():
            item["coverage"] = item["usable"]/item["parents"]
        storage = dict(pilot_records=len(store.rows), pilot_compressed_bytes=store.manifest["compressed_bytes"],
            pilot_raw_bytes=store.manifest["raw_bytes"], population=projection,
            projected_compressed_bytes=sum(v["compressed_bytes"] for v in projection.values()),
            projected_raw_bytes=sum(v["raw_bytes"] for v in projection.values()),
            excludes="Final indexes/manifests, bounded local repair staging, transfer partials, logs and future full graph stores")
    if (root/"evidence/manifest.json").exists():
        evidence_bytes = 0; evidence_raw = 0
        sources = {x["parent"]: x["source"] for x in store.manifest["provenance"]["inventory"]}
        with AugmentationStore(root/"evidence") as evidence_store:
            for row in evidence_store.rows:
                dataset = row["id"].split("/")[0]
                # The coordinate provenance is the immutable source of strata.
                unit = sources[row["id"]]
                key = f"{dataset}|{unit['split']}|{unit['bin']}"
                weight = population["strata"][key]/population["sample_counts"][key]
                evidence_bytes += weight*row["bytes"]; evidence_raw += weight*row["raw_bytes"]
        storage["projected_evidence_compressed_bytes"] = evidence_bytes
        storage["projected_evidence_raw_bytes"] = evidence_raw
        storage["projected_total_payload_compressed_bytes"] = storage["projected_compressed_bytes"]+evidence_bytes
    else: storage["projected_total_payload_compressed_bytes"] = storage["projected_compressed_bytes"]
    own_files = [p for p in root.rglob("*") if p.is_file()]
    storage["current_release_files"] = len(own_files)
    storage["current_release_bytes"] = sum(p.stat().st_size for p in own_files)
    storage["illustrative_380_worker_file_allowance"] = 380*9+math.ceil(storage["projected_total_payload_compressed_bytes"]/(2*1024**3))+10
    if checks["quota"]["passed"]:
        quota = json.loads((root/"quota-budget.json").read_text())
        if quota["snapshot"].get("coordinate_manifest_sha256") != digest(root/"coordinates/manifest.json") or \
                quota["additional_bytes"] < storage["projected_total_payload_compressed_bytes"] or \
                quota["additional_files"] < storage["illustrative_380_worker_file_allowance"]:
            checks["quota"]["passed"] = False; pending.append("quota")
    if checks["graphs"]["passed"]:
        graph_verified = json.loads((root/"graphs/graph-verification.json").read_text())
        if graph_verified["manifest_sha256"] != digest(root/"graphs/manifest.json"):
            raise ValueError("graph verification belongs to a different release")
    if checks["source_evidence"]["passed"]:
        evidence = json.loads((root/"evidence/evidence-verification.json").read_text())
        if evidence["coordinate_manifest_sha256"] != digest(root/"coordinates/manifest.json") or \
                evidence["evidence_manifest_sha256"] != digest(root/"evidence/manifest.json"):
            raise ValueError("source evidence verification belongs to a different release")
    if checks["production_compatibility"]["passed"]:
        compatibility = json.loads((root/"graphs/native-compatibility.json").read_text())
        if compatibility["manifest_sha256"] != digest(root/"graphs/manifest.json"):
            raise ValueError("production compatibility belongs to a different release")
    if checks["gpu"]["passed"]:
        gpu = json.loads((root/"gpu-preflight.json").read_text())
        if gpu["manifest_sha256"] != digest(root/"graphs/manifest.json"):
            raise ValueError("GPU preflight belongs to a different graph release")
        workers, prefetch = min(gpu["viable_settings"])
        recommended = dict(workers=workers, prefetch=prefetch)
    repair_summary = json.loads((pilot/"summary.json").read_text())
    repair_units = json.loads((pilot/"unit-summary.json").read_text())
    max_rss = max(u["peak_rss_GiB"] for u in repair_units)
    result = dict(infrastructure_ready=not pending, checks=checks, pending=pending, storage=storage,
        recommended_loader=recommended,
        performance_scope="Verified pilot; base-A compute-only proxy with a warm working set. Full graph stores and their random I/O remain deferred.",
        repairs=dict(ideal_380_worker_hours=repair_summary["datasets"]["all"]["estimates_380_cpus_h"],
            max_worker_rss_GiB=max_rss, memory_margin=1.25, available_node_memory_reserve=0.10,
            warning="Ideal repair projection excludes queueing, contention, transfer, validation and graph construction; 380 CPUs do not guarantee 380 RAM-safe workers."),
        graph_policy="Pilot graphs only; interface-detection inputs and full graph-store size remain deferred",
        training_changed=False, validation_role="native selection unchanged; augmentation robustness separate")
    atomic_json(root/"readiness.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("root", type=Path); p.add_argument("pilot", type=Path)
    args = p.parse_args(argv); require_compute(threads=1)
    print(json.dumps(report(args.root, args.pilot), indent=2))


if __name__ == "__main__": main()
