"""Frozen PINDER evaluation of smooth_propka's pKAI adapter and fork fine-tunes."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .prep import export_pdb, read_cif
from .runtime import atomic_json, digest, require_compute
from .schema import read_table


VERSION = "pkai-feature-pinder-v1"
SMOOTH_REVISION = "dd5790879d901e5b1a04addb05200e3a3c7ed3fb"
FORK_REVISION = "1ced84288cf0dfd75b52a038af7f9dc14bdd55e4"
MODELS = ("smooth-pKAI", "smooth-pKAI+", "pKAI-joint", "pKAI-joint-aa20")
DIRECT_MODELS = ("pKAI-joint", "pKAI-joint-aa20")
SMOOTH_MODELS = ("smooth-pKAI", "smooth-pKAI+")
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
    return Path(root) / "sources/pKAI-fork"


def smooth_source(root):
    return Path(root) / "sources/smooth_propka-pkai"


def model_paths(root):
    base = fork_source(root) / "pKAI/models"
    return {name: base / f"{name}_model.pt" for name in ("pKAI", "pKAI+", *DIRECT_MODELS)}


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
        "pKAI": "0f808c2485181620450d59f38f056ef33b6d9e4464344aaa1eadb515d201dabd",
        "pKAI+": "f65ddb1fcb3648d85654edb77bbe11db5719bb10535f524ae93da66ec5c71b34",
        "pKAI-joint": "e379e2c548526182557f6cf4f5ae5893267e2e442d7f89a14e3b4df8c2ce0a93",
        "pKAI-joint-aa20": "633ee8d669cf8a8b784e25978cbb470ef6c4ff132314d5cbc06a9e6daa41fb28",
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
            "smooth": "PKAIEstimator smooth mode, cutoff_window=0.5 A, temperature=0.01 A, CPU float64; protein_only, reject incomplete residues, infer disulfides",
            "fork": "published pKAI Python feature path, 15 A/250 slots, CPU float32 TorchScript, public API 0.01-pKa rounding; canonical benchmark PDB export",
            "groups": sorted(PK_MOD), "states": ["AB", "A", "B"],
        },
        "source_cohort_sha256": digest(cohort(root)),
        "code_sha256": digest(Path(__file__)), "test_data_included": False,
    }
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "registration.json", {"passed": True, "complexes": len(frozen),
        "manifest_sha256": digest(out / "manifest.json"), "test_data_included": False})
    print(json.dumps({"complexes": len(frozen), "models": list(MODELS)}))


def _load_direct_models(root, torch):
    result = {}
    for name in DIRECT_MODELS:
        model = torch.jit.load(str(model_paths(root)[name]), map_location="cpu").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        result[name] = model
    return result


def _direct_predictions(pdb, mapping, models, torch):
    package = fork_source(_RUNTIME) / "pKAI"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from protein import Protein

    predictions = {}
    for name, encoding in (("pKAI-joint", "atom16"),
                           ("pKAI-joint-aa20", "atom16aa20")):
        protein = Protein(str(pdb)); protein.apply_cutoff(encoding=encoding)
        residues = list(protein.iter_residues(titrable_only=True))
        if not residues:
            predictions[name] = []
            continue
        features = torch.stack([residue.input_layer for residue in residues])
        with torch.no_grad():
            shifts = models[name](features).reshape(-1).tolist()
        rows = []
        for residue, shift in zip(residues, shifts, strict=True):
            chain, number, insertion = mapping[(str(residue.chain), int(residue.resnumb))]
            group = str(residue.resname)
            if group in PK_MOD:
                # Match the fork's released Python/CLI API.
                rows.append({"chain": chain, "resnum": number, "icode": insertion,
                             "group": group, "pka": round(float(shift) + PK_MOD[group], 2)})
        predictions[name] = rows
    return predictions


_RUNTIME = None


def work(root, index):
    global _RUNTIME
    root = Path(root); _RUNTIME = root; out = output(root); manifest = read(out / "manifest.json")
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
    direct_models = _load_direct_models(root, torch)
    smooth_models = {
        "smooth-pKAI": PKAIEstimator(Config(), model_name="pKAI", mode="smooth",
                                      cutoff_window=0.5, temperature=0.01).eval(),
        "smooth-pKAI+": PKAIEstimator(Config(), model_name="pKAI+", mode="smooth",
                                       cutoff_window=0.5, temperature=0.01).eval(),
    }
    predictions, errors, timings, preparations = [], {}, {}, {}
    for state in ("AB", "A", "B"):
        started = time.monotonic(); plain = destination / f"{state}.cif"
        with gzip.open(folder / f"{state}.cif.gz", "rb") as src, plain.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        try:
            pdb = destination / f"{state}.pdb"
            mapping = export_pdb(read_cif(plain), pdb)
            for model, rows in _direct_predictions(pdb, mapping, direct_models, torch).items():
                predictions.extend({"model": model, "state": state, **row} for row in rows)
        except Exception as exc:
            for model in DIRECT_MODELS:
                errors[f"{model}:{state}"] = {"type": type(exc).__name__, "message": str(exc)}
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
            for model in SMOOTH_MODELS:
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
    lines = ["# pKAI feature-branch and fork evaluation", "",
        "All rows use the frozen 400-complex PINDER/PypKa validation cohort. No parameters were updated.", "",
        "| Estimator | State MAE | Paired-shift MAE | Interface paired MAE | Complexes |",
        "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        row = results[model]
        value = lambda name: "—" if row[name] is None else f"{row[name]['mae']:.3f}"
        lines.append(f"| {model} | {value('state')} | {value('paired')} | "
                     f"{value('interface_paired')} | {row['complexes']} |")
    lines += ["", "`smooth-pKAI` and `smooth-pKAI+` use the feature branch's differentiable "
        "feature approximation at its documented defaults. The two `joint` rows use the fork's "
        "released inference API and 0.01-pKa output rounding.", ""]
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
