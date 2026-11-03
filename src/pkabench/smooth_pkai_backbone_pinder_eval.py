"""Frozen PINDER evaluation of smooth_propka's backbone-only pKAI variants."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .runtime import atomic_json, digest, require_compute
from .schema import read_table


VERSION = "pkai-backbone-pinder-v1"
SMOOTH_REVISION = "616900a4aa89d8ed99883b05faac4c30d6b5420f"
FORK_REVISION = "49ab002684450c55f1e2364dd1b1e9a6e09d6071"
MODELS = ("pKAI-backbone-15A", "pKAI-backbone-20A")
MODEL_NAMES = {
    "pKAI-backbone-15A": "pKAI-joint-aa20-backbone",
    "pKAI-backbone-20A": "pKAI-joint-aa20-backbone-r20s540",
}
PK_MOD = {"ASP": 3.79, "CYS": 8.67, "GLU": 4.20, "HIS": 6.74,
          "LYS": 10.46, "TYR": 9.59}


def read(path):
    return json.loads(Path(path).read_text())


def source(root):
    return Path(root) / "pretraining/pinder-pkai-v1"


def cohort(root):
    return Path(root) / "training/ogqt-pinder-factorial-v1/cohort.json"


def teacher(root):
    return Path(root) / "audits/pinder-pkai-pypka-v1"


def output(root):
    return Path(root) / f"audits/{VERSION}"


def fork_source(root):
    return Path(root) / "sources/pKAI-fork-backbone"


def smooth_source(root):
    return Path(root) / "sources/smooth_propka-pkai-backbone"


def model_paths(root):
    base = fork_source(root) / "pKAI/models"
    return {label: base / f"{name}_model.pt" for label, name in MODEL_NAMES.items()}


def _chains(folder, state):
    return tuple(dict.fromkeys(str(row[2]) for row in read(folder / f"map_{state}.json")))


def register(root):
    require_compute(threads=1, allow_comp1400=True)
    root = Path(root); out = output(root); out.mkdir(parents=True, exist_ok=True)
    if (fork_source(root) / ".source-revision").read_text().strip() != FORK_REVISION:
        raise AssertionError("unexpected pKAI fork revision")
    if (smooth_source(root) / ".source-revision").read_text().strip() != SMOOTH_REVISION:
        raise AssertionError("unexpected smooth_propka feature revision")
    paths = model_paths(root)
    expected = {
        "pKAI-backbone-15A": "01c7e75989ec8befb63090613c8972fb98ffec9c3693d6c42b8c3a6f81a29a65",
        "pKAI-backbone-20A": "078b77b6c6b8851bf2fdd6fa70fe29e616c111fb994b3c3c1ae03e891e8223a9",
    }
    for name, path in paths.items():
        if digest(path) != expected[name]:
            raise AssertionError((name, "weight hash mismatch"))
    records = [row for row in read(cohort(root))["records"] if row["split"] == "val"]
    if len(records) != 400 or len({row["cluster_id"] for row in records}) != 400:
        raise AssertionError("expected fixed 400-cluster validation cohort")
    frozen = []
    for row in records:
        folder = source(root) / "entries" / row["id"]
        names = ["sites.json", *[f"{state}.cif.gz" for state in ("AB", "A", "B")],
                 *[f"map_{state}.json" for state in ("AB", "A", "B")]]
        frozen.append({"id": row["id"], "cluster_id": row["cluster_id"],
            "ctype": row["ctype"], "source_hashes": {name: digest(folder / name) for name in names}})
    manifest = {
        "version": VERSION, "records": frozen, "models": list(MODELS),
        "smooth_propka_revision": SMOOTH_REVISION, "pkai_fork_revision": FORK_REVISION,
        "weights": {name: {"path": str(path), "sha256": expected[name]}
                    for name, path in paths.items()},
        "protocol": {
            "smooth": "PKAIEstimator smooth mode, cutoff_window=0.5 A, temperature=0.01 A, CPU float64; backbone N/O context from site CA; protein_only, reject incomplete residues, infer disulfides",
            "variants": {"pKAI-backbone-15A": "15 A/250 slots", "pKAI-backbone-20A": "20 A/540 slots"},
            "groups": sorted(PK_MOD), "states": ["AB", "A", "B"],
        },
        "source_cohort_sha256": digest(cohort(root)),
        "code_sha256": digest(Path(__file__)), "test_data_included": False,
    }
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "registration.json", {"passed": True, "complexes": len(frozen),
        "manifest_sha256": digest(out / "manifest.json"), "test_data_included": False})
    print(json.dumps({"complexes": len(frozen), "models": list(MODELS)}))


def work(root, index):
    root = Path(root); out = output(root); manifest = read(out / "manifest.json")
    require_compute(threads=min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
                    allow_comp1400=True)
    if manifest.get("code_sha256") != digest(Path(__file__)):
        raise AssertionError("evaluation code changed after registration")
    record = manifest["records"][int(index)]; cid = record["id"]
    folder = source(root) / "entries" / cid; destination = out / "results" / cid
    receipt_path = destination / "receipt.json"; prediction_path = destination / "predictions.json"
    if receipt_path.exists() and prediction_path.exists():
        receipt = read(receipt_path)
        if (receipt.get("version") == 1 and receipt.get("code_sha256") == manifest["code_sha256"]
                and receipt.get("predictions_sha256") == digest(prediction_path)):
            return
    for name, expected in record["source_hashes"].items():
        if digest(folder / name) != expected:
            raise AssertionError((cid, name, "source changed"))
    destination.mkdir(parents=True, exist_ok=True)

    import torch
    from ph_loss import Config, PKAIEstimator, load_structure
    torch.set_num_threads(min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    smooth_models = {
        label: PKAIEstimator(Config(), model_name=name, mode="smooth",
                            cutoff_window=0.5, temperature=0.01).eval()
        for label, name in MODEL_NAMES.items()
    }
    predictions, errors, timings, preparations = [], {}, {}, {}
    for state in ("AB", "A", "B"):
        started = time.monotonic(); plain = destination / f"{state}.cif"
        with gzip.open(folder / f"{state}.cif.gz", "rb") as src, plain.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        try:
            prepared = load_structure(plain, target_chains=_chains(folder, state),
                protein_only=True, incomplete_residues="reject", infer_disulfides=True,
                dtype=torch.float64)
            preparations[state] = prepared.preparation
            with torch.no_grad():
                for model, estimator in smooth_models.items():
                    result = estimator(prepared.coords, prepared.topology)
                    for site, value in zip(result.site_ids, result.pka.tolist(), strict=True):
                        group = str(site.kind)
                        if group in PK_MOD:
                            predictions.append({"model": model, "state": state,
                                "chain": str(site.chain), "resnum": int(site.number),
                                "icode": str(site.insertion), "group": group, "pka": float(value)})
        except Exception as exc:
            for model in MODELS:
                errors[f"{model}:{state}"] = {"type": type(exc).__name__, "message": str(exc)}
        timings[state] = time.monotonic() - started
    atomic_json(prediction_path, predictions)
    atomic_json(receipt_path, {"version": 1, "passed": not errors, "id": cid,
        "index": int(index), "rows": len(predictions), "errors": errors,
        "timings": timings, "preparations": preparations,
        "predictions_sha256": digest(prediction_path), "code_sha256": manifest["code_sha256"],
        "torch_version": torch.__version__})


def _metrics(values):
    x = np.asarray(values, np.float64)
    return {"sites": len(x), "mae": float(np.mean(np.abs(x))),
            "rmse": float(np.sqrt(np.mean(x*x))), "bias": float(np.mean(x))}


def summarize(root, *, smoke=False):
    require_compute(threads=1, allow_comp1400=True)
    root = Path(root); out = output(root); manifest = read(out / "manifest.json")
    selected = manifest["records"][:1] if smoke else manifest["records"]
    teacher_manifest = {row["id"]: row for row in read(teacher(root) / "manifest.json")["records"]}
    matched = []; failures = []; process = Counter()
    for record in selected:
        cid = record["id"]; folder = source(root) / "entries" / cid
        receipt_path = out / "results" / cid / "receipt.json"
        prediction_path = out / "results" / cid / "predictions.json"
        if not receipt_path.exists() or not prediction_path.exists():
            process["missing"] += 1; failures.append({"id": cid, "reason": "missing"}); continue
        receipt = read(receipt_path)
        if receipt.get("predictions_sha256") != digest(prediction_path):
            process["invalid"] += 1; failures.append({"id": cid, "reason": "invalid receipt"}); continue
        process["complete" if receipt["passed"] else "partial"] += 1
        if receipt["errors"]:
            failures.append({"id": cid, "errors": receipt["errors"]})
        pred = {(row["model"], row["state"], row["chain"], int(row["resnum"]),
                 row["icode"], row["group"]): row for row in read(prediction_path)}
        masks = {(str(row["chain"]), int(row["resnum"]), str(row["icode"]), str(row["group"])): row
                 for row in read(folder / "sites.json")}
        teacher_path = teacher(root) / "results" / cid / "predictions.parquet"
        if not teacher_path.exists():
            failures.append({"id": cid, "reason": "missing teacher"}); continue
        for ref in read_table(teacher_path):
            site = (str(ref["chain"]), int(ref["resnum"]), str(ref["icode"]), str(ref["group"]))
            mask = masks.get(site)
            if (ref["status"] != "ok" or ref["pka"] is None or mask is None
                    or not mask["eval_mask"] or site[3] not in PK_MOD):
                continue
            for model in MODELS:
                candidate = pred.get((model, str(ref["state"]), *site))
                if candidate is None:
                    continue
                matched.append({"model": model, "complex_id": cid,
                    "cluster_id": teacher_manifest[cid]["cluster_id"],
                    "ctype": teacher_manifest[cid]["ctype"], "state": str(ref["state"]),
                    "chain": site[0], "resnum": site[1], "icode": site[2], "group": site[3],
                    "pypka": float(ref["pka"]), "prediction": float(candidate["pka"]),
                    "error": float(candidate["pka"] - ref["pka"]),
                    "interface": bool(mask["interface"]),
                    "partner_distance_A": mask.get("partner_distance_A")})
    paired = []
    grouped = defaultdict(dict)
    for row in matched:
        grouped[(row["model"], row["complex_id"], row["chain"], row["resnum"],
                 row["icode"], row["group"])][row["state"]] = row
    for key, states in grouped.items():
        if "AB" not in states:
            continue
        free_name = next((name for name in ("A", "B") if name in states), None)
        if free_name is None:
            continue
        ab, free = states["AB"], states[free_name]
        paired.append({"model": key[0], **{name: ab[name] for name in
            ("complex_id", "cluster_id", "ctype", "chain", "resnum", "icode", "group",
             "interface", "partner_distance_A")}, "free_state": free_name,
            "pypka_ab": ab["pypka"], "pypka_free": free["pypka"],
            "prediction_ab": ab["prediction"], "prediction_free": free["prediction"],
            "pypka_shift": ab["pypka"] - free["pypka"],
            "prediction_shift": ab["prediction"] - free["prediction"],
            "shift_error": (ab["prediction"] - free["prediction"])
                           - (ab["pypka"] - free["pypka"])})

    def write_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    prefix = "smoke-" if smoke else ""
    write_csv(out / f"{prefix}matched-state.csv", matched)
    write_csv(out / f"{prefix}matched-paired.csv", paired)
    results = {}
    for model in MODELS:
        state_rows = [row for row in matched if row["model"] == model]
        pair_rows = [row for row in paired if row["model"] == model]
        interface = [row for row in pair_rows if row["interface"]]
        results[model] = {
            "state": _metrics([row["error"] for row in state_rows]) if state_rows else None,
            "paired": _metrics([row["shift_error"] for row in pair_rows]) if pair_rows else None,
            "interface_paired": _metrics([row["shift_error"] for row in interface]) if interface else None,
            "complexes": len({row["complex_id"] for row in state_rows}),
        }
    summary = {"passed": process["missing"] == 0 and process["invalid"] == 0,
        "smoke": smoke, "process": dict(process), "failures": failures,
        "results": results, "matched_state_rows": len(matched), "matched_pair_rows": len(paired)}
    atomic_json(out / f"{prefix}summary.json", summary)
    lines = ["# smooth_propka backbone-only pKAI evaluation", "",
        "All rows use the frozen 400-complex PINDER/PypKa validation cohort. No parameters were updated.", "",
        "| Estimator | State MAE | Paired-shift MAE | Interface paired MAE | Complexes |",
        "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        row = results[model]
        value = lambda name: "—" if row[name] is None else f"{row[name]['mae']:.3f}"
        lines.append(f"| {model} | {value('state')} | {value('paired')} | "
                     f"{value('interface_paired')} | {row['complexes']} |")
    lines += ["", "Both rows use smooth_propka's differentiable backbone-only feature path at "
        "its documented defaults. No side-chain coordinates enter either estimator.", ""]
    (out / f"{prefix}report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "work", "summarize", "smoke-summary"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "register": register(args.runtime)
    elif args.action == "work":
        if args.index is None: raise ValueError("--index is required")
        work(args.runtime, args.index)
    elif args.action == "smoke-summary": summarize(args.runtime, smoke=True)
    else: summarize(args.runtime)


if __name__ == "__main__":
    main()
