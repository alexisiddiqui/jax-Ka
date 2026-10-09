"""Frozen-cohort evaluation of smooth_propka against current PypKa.

The external estimator is evaluated without calibration or parameter updates.
AB, A and B are independent state predictions; paired shifts are derived only
after matching site identifiers through the existing PINDER maps and masks.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .runtime import atomic_json, digest, require_compute
from .schema import read_table


VERSION = "smooth-propka-pinder-v1"
EXPECTED_REVISION = "1f3a96bd348c54a8b27406a37c43040bb8ddbd37"
STATE_NAMES = ("AB", "A", "B")
GROUP_ALIAS = {"N+": "NTERM", "C-": "CTERM"}


def read(path): return json.loads(Path(path).read_text())
def source(root): return Path(root) / "pretraining/pinder-pkai-v1"
def cohort(root): return Path(root) / "training/ogqt-pinder-factorial-v1/cohort.json"
def teacher(root): return Path(root) / "audits/pinder-pkai-pypka-v1"
def output(root): return Path(root) / "audits/smooth-propka-pinder-v1"
def external(root): return Path(root) / "sources/smooth_propka"


def _revision(root):
    git = external(root) / ".git"; head = (git / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        path = git / head.removeprefix("ref: ")
        if path.exists(): return path.read_text().strip()
        ref = head.removeprefix("ref: ")
        for line in (git / "packed-refs").read_text().splitlines():
            if line and not line.startswith("#") and line.split()[-1] == ref:
                return line.split()[0]
        raise RuntimeError(f"Unresolved git ref {ref}")
    return head


def register(root):
    root = Path(root); out = output(root); out.mkdir(parents=True, exist_ok=True)
    revision = _revision(root)
    if revision != EXPECTED_REVISION: raise AssertionError((revision, EXPECTED_REVISION))
    records = [row for row in read(cohort(root))["records"] if row["split"] == "val"]
    if len(records) != 400 or len({row["cluster_id"] for row in records}) != 400:
        raise AssertionError("expected fixed 400-cluster validation cohort")
    frozen = []
    for row in records:
        folder = source(root) / "entries" / row["id"]
        names = ["sites.json", *[f"{state}.cif.gz" for state in STATE_NAMES],
                 *[f"map_{state}.json" for state in STATE_NAMES]]
        frozen.append({"id": row["id"], "cluster_id": row["cluster_id"],
            "ctype": row["ctype"], "n_res": row["n_res"],
            "source_hashes": {name: digest(folder / name) for name in names}})
    manifest = {"version": VERSION, "records": frozen, "revision": revision,
        "repository": "https://github.com/ovavourakis/smooth_propka",
        "configuration": "package generic defaults: CPU float64, protein_only=True, "
            "infer_disulfides=True, incomplete_residues=reject",
        "state_protocol": "independent SmoothPropka calls for AB, A and B",
        "site_aliases": GROUP_ALIAS, "source_cohort_sha256": digest(cohort(root)),
        "code_sha256": digest(Path(__file__)), "test_data_included": False}
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "registration.json", {"passed": True, "complexes": len(frozen),
        "manifest_sha256": digest(out / "manifest.json"), "revision": revision,
        "test_data_included": False})


def _chains(folder, state):
    # Maps contain every protein residue, including chains with no titratable site.
    return tuple(dict.fromkeys(str(row[2]) for row in read(folder / f"map_{state}.json")))


def work(root, index):
    from importlib.metadata import version
    import torch
    from ph_loss import Config, SmoothPropka, load_structure

    root = Path(root); out = output(root); manifest = read(out / "manifest.json")
    if manifest["revision"] != _revision(root) or manifest["code_sha256"] != digest(Path(__file__)):
        raise AssertionError("registered source/code changed")
    row = manifest["records"][int(index)]; cid = row["id"]
    folder = source(root) / "entries" / cid; destination = out / "results" / cid
    receipt_path = destination / "receipt.json"; prediction_path = destination / "predictions.json"
    if receipt_path.exists() and prediction_path.exists():
        receipt = read(receipt_path)
        if (receipt.get("version") == 1 and receipt.get("predictions_sha256") == digest(prediction_path)
                and receipt.get("code_sha256") == manifest["code_sha256"]): return
    for name, expected in row["source_hashes"].items():
        if digest(folder / name) != expected: raise AssertionError((cid, name, "source changed"))
    destination.mkdir(parents=True, exist_ok=True)
    model = SmoothPropka(Config()).eval(); predictions = []; errors = {}; timings = {}
    preparations = {}
    torch.set_num_threads(min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    for state in STATE_NAMES:
        plain = destination / f"{state}.cif"
        with gzip.open(folder / f"{state}.cif.gz", "rb") as src, plain.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        started = time.monotonic()
        try:
            prepared = load_structure(plain, target_chains=_chains(folder, state),
                protein_only=True, incomplete_residues="reject", infer_disulfides=True,
                dtype=torch.float64)
            with torch.no_grad(): result = model(prepared.coords, prepared.topology)
            for site, value in zip(result.site_ids, result.pka.detach().cpu().tolist(), strict=True):
                predictions.append({"state": state, "chain": str(site.chain),
                    "resnum": int(site.number), "icode": str(site.insertion),
                    "group": GROUP_ALIAS.get(str(site.kind), str(site.kind)),
                    "pka": float(value), "status": "ok"})
            preparations[state] = prepared.preparation
        except Exception as exc:
            errors[state] = {"type": type(exc).__name__, "message": str(exc)}
        timings[state] = time.monotonic() - started
    atomic_json(prediction_path, predictions)
    atomic_json(receipt_path, {"version": 1, "passed": not errors, "id": cid,
        "index": int(index), "rows": len(predictions), "errors": errors,
        "timings": timings, "preparations": preparations,
        "predictions_sha256": digest(prediction_path), "code_sha256": manifest["code_sha256"],
        "packages": {name: version(name) for name in ("ph-loss", "propka", "torch", "numpy", "gemmi")},
        "config_digest": Config().digest, "revision": manifest["revision"]})


def _metrics(rows, field):
    values = np.asarray([float(row[field]) for row in rows], np.float64)
    absolute = np.abs(values)
    return {"sites": len(values), "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(values ** 2))), "bias": float(np.mean(values)),
        "p50": float(np.quantile(absolute, .5)), "p90": float(np.quantile(absolute, .9)),
        "p95": float(np.quantile(absolute, .95)), "max": float(np.max(absolute))}


def summarize(root):
    import csv
    root = Path(root); out = output(root); manifest = read(out / "manifest.json")
    process = Counter(); failures = []; matched = []; eligible_state = 0; eligible_pair = 0
    timings = []
    teacher_manifest = {row["id"]: row for row in read(teacher(root) / "manifest.json")["records"]}
    for record in manifest["records"]:
        cid = record["id"]; src = source(root) / "entries" / cid
        receipt_path = out / "results" / cid / "receipt.json"
        prediction_path = out / "results" / cid / "predictions.json"
        if not receipt_path.exists() or not prediction_path.exists():
            process["missing"] += 1; failures.append({"id": cid, "reason": "missing receipt"}); continue
        receipt = read(receipt_path)
        if receipt.get("predictions_sha256") != digest(prediction_path):
            process["invalid"] += 1; failures.append({"id": cid, "reason": "invalid receipt"}); continue
        process["complete" if receipt["passed"] else "partial"] += 1
        if receipt["errors"]: failures.append({"id": cid, "errors": receipt["errors"]})
        timings.extend(receipt["timings"].values())
        smooth = {(row["state"], row["chain"], int(row["resnum"]), row["icode"], row["group"]): row
                  for row in read(prediction_path) if row["status"] == "ok"}
        pypka_path = teacher(root) / "results" / cid / "predictions.parquet"
        if not pypka_path.exists(): continue
        pypka = {(row["state"], row["chain"], int(row["resnum"]), row["icode"], row["group"]): row
                 for row in read_table(pypka_path) if row["status"] == "ok" and row["pka"] is not None}
        masks = {(row["chain"], int(row["resnum"]), row["icode"], row["group"]): row
                 for row in read(src / "sites.json")}
        eligible_by_site = defaultdict(set)
        for (state, *key), teacher_row in pypka.items():
            mask = masks.get(tuple(key))
            if mask is not None and mask["eval_mask"]:
                eligible_state += 1; eligible_by_site[tuple(key)].add(state)
        eligible_pair += sum("AB" in states and bool(states & {"A", "B"})
                             for states in eligible_by_site.values())
        for key, estimate in smooth.items():
            state, chain, number, insertion, group = key
            site_key = (chain, number, insertion, group); mask = masks.get(site_key)
            reference = pypka.get(key)
            if mask is None or not mask["eval_mask"] or reference is None: continue
            matched.append({"complex_id": cid, "cluster_id": teacher_manifest[cid]["cluster_id"],
                "ctype": teacher_manifest[cid]["ctype"], "state": state, "chain": chain,
                "resnum": number, "icode": insertion, "group": group,
                "smooth_propka": float(estimate["pka"]), "pypka": float(reference["pka"]),
                "error": float(estimate["pka"] - reference["pka"]),
                "interface": bool(mask["interface"]), "partner_distance_A": mask["partner_distance_A"]})
    by_site = defaultdict(dict)
    for row in matched:
        by_site[(row["complex_id"], row["chain"], row["resnum"], row["icode"], row["group"])][row["state"]] = row
    paired = []
    for states in by_site.values():
        if "AB" not in states: continue
        free_name = next((name for name in ("A", "B") if name in states), None)
        if free_name is None: continue
        ab, free = states["AB"], states[free_name]
        paired.append({**{name: ab[name] for name in ("complex_id", "cluster_id", "ctype",
            "chain", "resnum", "icode", "group", "interface", "partner_distance_A")},
            "free_state": free_name, "smooth_propka_shift": ab["smooth_propka"] - free["smooth_propka"],
            "pypka_shift": ab["pypka"] - free["pypka"],
            "shift_error": (ab["smooth_propka"] - free["smooth_propka"]) - (ab["pypka"] - free["pypka"])})
    def write_csv(name, rows):
        if not rows: return
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_csv("matched-state.csv", matched); write_csv("matched-paired.csv", paired)
    interface = [row for row in paired if row["interface"]]
    strata = {}
    for name in ("group", "ctype", "state"):
        strata[name] = {str(value): _metrics([row for row in matched if row[name] == value], "error")
                        for value in sorted({row[name] for row in matched}, key=str)}
    state_fraction = len(matched) / eligible_state if eligible_state else 0.0
    pair_fraction = len(paired) / eligible_pair if eligible_pair else 0.0
    summary = {"passed": process["complete"] + process["partial"] == len(manifest["records"]),
        "process": dict(process), "failures": failures, "revision": manifest["revision"],
        "matched_state_sites": len(matched), "matched_paired_sites": len(paired),
        "eligible_state_sites": eligible_state, "eligible_paired_sites": eligible_pair,
        "state_match_fraction": state_fraction, "paired_match_fraction": pair_fraction,
        "state": _metrics(matched, "error") if matched else None,
        "paired": _metrics(paired, "shift_error") if paired else None,
        "interface_paired": _metrics(interface, "shift_error") if interface else None,
        "strata": strata, "timing": {"state_calls": len(timings),
            "median_seconds": float(np.median(timings)) if timings else None,
            "p95_seconds": float(np.quantile(timings, .95)) if timings else None,
            "total_cpu_seconds": float(np.sum(timings))}, "test_data_included": False}
    atomic_json(out / "summary.json", summary)
    comparison = []
    for label, path, method in (
        ("Regular pKAI", root / "audits/pinder-pkai-backbone-eval-v1/summary.json", "regular_pkai"),
        ("pKAI+", root / "audits/pinder-pkai-plus-eval-v1/summary.json", None),
        ("Backbone-pretrained pKAI", root / "audits/pinder-pkai-backbone-eval-v1/summary.json", "backbone_ensemble")):
        if not path.exists(): continue
        data = read(path)
        metrics = data["methods"][method] if method else data
        comparison.append((label, metrics["state"]["mae"], metrics["paired"]["mae"],
                           metrics["interface_paired"]["mae"]))
    if summary["state"]:
        comparison.insert(0, ("smooth_propka", summary["state"]["mae"],
            summary["paired"]["mae"], summary["interface_paired"]["mae"]))
    lines = ["# smooth_propka on the frozen PINDER validation cohort", "",
        f"External revision: `{manifest['revision']}`. Generic package defaults were used without calibration or training. "
        "AB, A and B were evaluated independently in CPU float64.", "",
        "| Estimator | State MAE | Paired-shift MAE | Interface paired-shift MAE |",
        "|---|---:|---:|---:|"]
    lines += [f"| {name} | {state:.3f} | {pair:.3f} | {interface_mae:.3f} |"
              for name, state, pair, interface_mae in comparison]
    lines += ["", f"Coverage: **{len(matched):,}/{eligible_state:,} state sites ({state_fraction:.1%})** and "
        f"**{len(paired):,}/{eligible_pair:,} paired sites ({pair_fraction:.1%})**.",
        f"Structures: {process.get('complete', 0)} complete, {process.get('partial', 0)} partial, "
        f"{process.get('missing', 0)} missing. Median state runtime: {summary['timing']['median_seconds']:.2f} s.", "",
        "The estimator is a smooth approximation to PROPKA 3.5.1. These numbers measure agreement with current "
        "PypKa on our cleaned cohort; they are not experimental accuracy.", "",
        "Per-site records are in `matched-state.csv` and `matched-paired.csv`; failures and strata are in `summary.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


def smoke(root):
    register(root)
    records = read(output(root) / "manifest.json")["records"]
    selected = []
    for ctype in sorted({row["ctype"] for row in records}):
        selected.append(next(i for i, row in enumerate(records) if row["ctype"] == ctype))
    selected.append(max(range(len(records)), key=lambda i: records[i]["n_res"]))
    selected = list(dict.fromkeys(selected))
    for index in selected: work(root, index)
    receipts = [read(output(root) / "results" / records[i]["id"] / "receipt.json") for i in selected]
    result = {"passed": any(row["passed"] for row in receipts), "indices": selected,
        "complexes": [{"index": i, "id": records[i]["id"], "ctype": records[i]["ctype"],
            "n_res": records[i]["n_res"], "passed": receipt["passed"],
            "rows": receipt["rows"], "errors": receipt["errors"]}
            for i, receipt in zip(selected, receipts, strict=True)]}
    atomic_json(output(root) / "smoke.json", result)
    if not result["passed"]: raise RuntimeError("no representative smooth_propka case passed")


def main():
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    require_compute(threads=min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    if action == "register": register(root)
    elif action == "smoke": smoke(root)
    elif action == "work": work(root, int(sys.argv[2]))
    elif action == "summarize": summarize(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
