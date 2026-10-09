"""Current-PypKa audit on the fixed 400-complex PINDER validation cohort."""
from __future__ import annotations

import json
import os
import gzip
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .adapters.base import Adapter, TEACHER
from .runtime import atomic_json, config_hash, digest, require_compute
from .schema import read_table, write_table


def read(path): return json.loads(Path(path).read_text())
def source(root): return Path(root) / "pretraining/pinder-pkai-v1"
def cohort(root): return Path(root) / "training/ogqt-pinder-factorial-v1/cohort.json"
def output(root): return Path(root) / "audits/pinder-pkai-pypka-v1"


def register(root):
    root = Path(root); out = output(root); out.mkdir(parents=True, exist_ok=True)
    records = [row for row in read(cohort(root))["records"] if row["split"] == "val"]
    if len(records) != 400 or len({row["cluster_id"] for row in records}) != 400:
        raise AssertionError("validation cohort is not the fixed 400-cluster split")
    rows = []
    for row in records:
        folder = source(root) / "entries" / row["id"]
        hashes = {name: digest(folder / name) for name in
                  ("AB.cif.gz", "A.cif.gz", "B.cif.gz", "sites.json", "labels.json")}
        rows.append({"id": row["id"], "cluster_id": row["cluster_id"], "ctype": row["ctype"],
                     "n_res": row["n_res"],
                     "source_hashes": hashes})
    manifest = {"version": "pinder-pkai-pypka-v1", "records": rows,
        "teacher": TEACHER, "teacher_config_sha256": config_hash(TEACHER),
        "code_sha256": digest(Path(__file__)),
        "states": ["AB", "A", "B"], "timeout_seconds_per_complex": 5400,
        "site_matching": ["complex_id", "chain", "resnum", "icode", "group", "state"],
        "test_data_included": False}
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "registration.json", {"passed": True, "complexes": len(rows),
        "manifest_sha256": digest(out / "manifest.json"), "code_sha256": digest(Path(__file__)),
        "test_data_included": False})


def work(root, index):
    root = Path(root); out = output(root); manifest = read(out / "manifest.json")
    if manifest.get("code_sha256") != digest(Path(__file__)):
        raise AssertionError("teacher-audit code changed after registration")
    row = manifest["records"][int(index)]; cid = row["id"]; src = source(root) / "entries" / cid
    destination = out / "results" / cid; receipt = destination / "receipt.json"
    if receipt.exists():
        value = read(receipt)
        prediction_path = destination / "predictions.parquet"
        if (value.get("version") == 2 and prediction_path.exists()
                and value.get("teacher_config_sha256") == manifest["teacher_config_sha256"]
                and digest(prediction_path) == value.get("predictions_sha256")):
            return
    for name, expected in row["source_hashes"].items():
        if digest(src / name) != expected: raise AssertionError((cid, name, "source changed"))
    sites = [{**site, "complex_id": cid} for site in read(src / "sites.json")]
    destination.mkdir(parents=True, exist_ok=True)
    state_files = {}
    for state in ("AB", "A", "B"):
        plain = destination / f"{state}.cif"
        with gzip.open(src / f"{state}.cif.gz", "rb") as source_stream, plain.open("wb") as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream)
        state_files[state] = plain
    adapter = Adapter("pypka", sites, root, timeout=manifest["timeout_seconds_per_complex"])
    predictions = adapter.run(state_files, destination / "work")
    # Midpoints are sufficient for this audit; discard curves to keep 400 outputs compact.
    for item in predictions:
        item["curve"] = None; item["curve_source"] = None
    path = destination / "predictions.parquet"; write_table(path, "predictions", predictions)
    atomic_json(receipt, {"version": 2, "passed": not bool(adapter.errors), "id": cid, "index": int(index),
        "predictions_sha256": digest(path), "rows": len(predictions),
        "timings": adapter.timings, "errors": adapter.errors,
        "teacher_config_sha256": manifest["teacher_config_sha256"]})


def summarize(root):
    root = Path(root); out = output(root); manifest = read(out / "manifest.json")
    matched = []; process = Counter(); failures = []; eligible_state = 0; eligible_paired = 0
    for record in manifest["records"]:
        cid = record["id"]; folder = source(root) / "entries" / cid
        receipt_path = out / "results" / cid / "receipt.json"
        if not receipt_path.exists():
            process["missing"] += 1; failures.append({"id": cid, "reason": "missing receipt"}); continue
        receipt = read(receipt_path); path = out / "results" / cid / "predictions.parquet"
        if (receipt.get("version") != 2 or not path.exists()
                or digest(path) != receipt.get("predictions_sha256")):
            process["invalid"] += 1; failures.append({"id": cid, "reason": "invalid receipt"}); continue
        if receipt["passed"]:
            process["complete"] += 1
        else:
            process["partial"] += 1
            failures.append({"id": cid, "reason": "teacher state failure", "errors": receipt.get("errors", {})})
        pypka = {(row["state"], row["chain"], row["resnum"], row["icode"], row["group"]): row
                 for row in read_table(path)}
        labels = read(folder / "labels.json")["pkai"]
        masks = {(row["chain"], row["resnum"], row["icode"], row["group"]): row
                 for row in read(folder / "sites.json")}
        label_states = defaultdict(dict)
        for state in ("AB", "A", "B"):
            for chain, number, insertion, group, value in labels.get(state, []):
                key = (str(chain), int(number), str(insertion), str(group))
                mask = masks.get(key)
                if value is not None and mask is not None and mask["eval_mask"]:
                    eligible_state += 1; label_states[key][state] = float(value)
        eligible_paired += sum("AB" in states and ("A" in states or "B" in states)
                               for states in label_states.values())
        for state in ("AB", "A", "B"):
            for chain, number, insertion, group, value in labels.get(state, []):
                key = (str(chain), int(number), str(insertion), str(group))
                mask = masks.get(key)
                if value is None or mask is None or not mask["eval_mask"]: continue
                teacher = pypka.get((state, *key))
                if teacher is None or teacher["status"] != "ok" or teacher["pka"] is None: continue
                matched.append({"complex_id": cid, "cluster_id": record["cluster_id"], "ctype": record["ctype"],
                    "state": state, "chain": key[0], "resnum": key[1], "icode": key[2], "group": key[3],
                    "pkai": float(value), "pypka": float(teacher["pka"]), "difference": float(value-teacher["pka"]),
                    "interface": bool(mask["interface"]), "partner_distance_A": mask["partner_distance_A"]})
    by_pair = defaultdict(dict)
    for row in matched: by_pair[(row["complex_id"], row["chain"], row["resnum"], row["icode"], row["group"])][row["state"]] = row
    paired = []
    for key, states in by_pair.items():
        if "AB" not in states: continue
        partner = next((name for name in ("A", "B") if name in states), None)
        if partner is None: continue
        ab, free = states["AB"], states[partner]
        paired.append({**{name: ab[name] for name in ("complex_id", "cluster_id", "ctype", "chain", "resnum", "icode", "group", "interface", "partner_distance_A")},
            "free_state": partner, "pkai_shift": ab["pkai"]-free["pkai"],
            "pypka_shift": ab["pypka"]-free["pypka"],
            "shift_difference": (ab["pkai"]-free["pkai"])-(ab["pypka"]-free["pypka"])})
    out.mkdir(parents=True, exist_ok=True)
    fields = list(matched[0]) if matched else []
    if fields:
        with (out / "matched-state.csv").open("w", newline="") as stream:
            import csv; writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(matched)
    if paired:
        with (out / "matched-paired.csv").open("w", newline="") as stream:
            import csv; writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    state_abs = np.abs([row["difference"] for row in matched]); pair_abs = np.abs([row["shift_difference"] for row in paired])
    def strata(rows, field, names):
        result = {}
        for name in names:
            for value in sorted({row[name] for row in rows}, key=str):
                subset = [abs(row[field]) for row in rows if row[name] == value]
                result[f"{name}={value}"] = {"sites": len(subset), "mae": float(np.mean(subset))}
        return result
    state_fraction = len(matched) / eligible_state if eligible_state else None
    paired_fraction = len(paired) / eligible_paired if eligible_paired else None
    summary = {"passed": process["complete"] + process["partial"] == len(manifest["records"]),
        "all_teacher_states_succeeded": process["complete"] == len(manifest["records"]), "process": dict(process),
        "failures": failures, "matched_state_sites": len(matched), "matched_paired_sites": len(paired),
        "eligible_state_sites": eligible_state, "eligible_paired_sites": eligible_paired,
        "state_match_fraction": state_fraction, "paired_match_fraction": paired_fraction,
        "coverage_qualified": bool(state_fraction is not None and paired_fraction is not None
                                   and state_fraction >= 0.8 and paired_fraction >= 0.8),
        "state_teacher_mae": float(np.mean(state_abs)) if len(state_abs) else None,
        "paired_teacher_mae": float(np.mean(pair_abs)) if len(pair_abs) else None,
        "interface_paired_teacher_mae": float(np.mean([abs(row["shift_difference"]) for row in paired if row["interface"]])) if any(row["interface"] for row in paired) else None,
        "state_strata": strata(matched, "difference", ("group", "ctype", "state")),
        "paired_strata": strata(paired, "shift_difference", ("group", "ctype", "interface")),
        "test_data_included": False}
    atomic_json(out / "summary.json", summary)


def preflight(root):
    manifest = read(output(root) / "manifest.json"); selected = []
    for ctype in sorted({row["ctype"] for row in manifest["records"]}):
        selected.append(next(index for index, row in enumerate(manifest["records"]) if row["ctype"] == ctype))
    selected.append(max(range(len(manifest["records"])), key=lambda index: manifest["records"][index]["n_res"]))
    selected = list(dict.fromkeys(selected))
    receipts = []
    for index in selected:
        work(root, index)
        receipt = read(output(root) / "results" / manifest["records"][index]["id"] / "receipt.json")
        if receipt.get("version") != 2: raise RuntimeError((index, "invalid receipt version"))
        receipts.append({"index": index, "id": manifest["records"][index]["id"],
            "ctype": manifest["records"][index]["ctype"], "n_res": manifest["records"][index]["n_res"],
            "passed": receipt["passed"], "errors": receipt.get("errors", {})})
    required_types = {manifest["records"][index]["ctype"] for index in selected}
    successful_types = {row["ctype"] for row in receipts if row["passed"]}
    passed = required_types <= successful_types
    atomic_json(output(root) / "preflight.json", {"version": 2, "passed": passed, "indices": selected,
        "types": [manifest["records"][index]["ctype"] for index in selected],
        "largest_n_res": max(manifest["records"][index]["n_res"] for index in selected),
        "successful": sum(row["passed"] for row in receipts),
        "failed": sum(not row["passed"] for row in receipts), "receipts": receipts,
        "policy": "Every observed complex type must have a successful end-to-end representative; "
                  "structure-specific teacher failures are retained and measured in production."})
    if not passed: raise RuntimeError(("missing successful complex type", required_types-successful_types))


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    require_compute(threads=min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    if action == "register": register(root)
    elif action == "preflight": preflight(root)
    elif action == "work": work(root, int(sys.argv[2]))
    elif action == "summarize": summarize(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
