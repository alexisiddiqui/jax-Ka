"""Evaluation-only comparison of released pKAI+ with current PypKa on PINDER."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .runtime import atomic_json, digest, require_compute
from .schema import read_table


def read(path): return json.loads(Path(path).read_text())
def source(root): return Path(root) / "pretraining/pinder-pkai-v1"
def cohort(root): return Path(root) / "training/ogqt-pinder-factorial-v1/cohort.json"
def audit(root): return Path(root) / "audits/pinder-pkai-pypka-v1"
def output(root): return Path(root) / "audits/pinder-pkai-plus-eval-v1"


def measures(rows, field):
    values = np.asarray([row[field] for row in rows], np.float64)
    return {"sites": len(values), "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "bias": float(np.mean(values))}


def report(root):
    root = Path(root); out = output(root); out.mkdir(parents=True, exist_ok=True)
    records = [row for row in read(cohort(root))["records"] if row["split"] == "val"]
    if len(records) != 400 or len({row["cluster_id"] for row in records}) != 400:
        raise AssertionError("expected fixed 400-cluster validation cohort")
    audit_manifest = read(audit(root) / "manifest.json")
    audit_records = {row["id"]: row for row in audit_manifest["records"]}
    state_rows = []
    processed = 0
    for record in records:
        cid = record["id"]; folder = source(root) / "entries" / cid
        prediction_path = audit(root) / "results" / cid / "predictions.parquet"
        if not prediction_path.exists(): continue
        processed += 1
        pypka = {(row["state"], str(row["chain"]), int(row["resnum"]),
                  str(row["icode"]), str(row["group"])): row
                 for row in read_table(prediction_path)}
        masks = {(str(row["chain"]), int(row["resnum"]), str(row["icode"]), str(row["group"])): row
                 for row in read(folder / "sites.json")}
        labels = read(folder / "labels.json")["pkai_plus"]
        for state in ("AB", "A", "B"):
            for chain, number, insertion, group, value in labels.get(state, []):
                key = (str(chain), int(number), str(insertion), str(group))
                mask = masks.get(key); teacher = pypka.get((state, *key))
                if (value is None or mask is None or not mask["eval_mask"] or teacher is None
                        or teacher["status"] != "ok" or teacher["pka"] is None):
                    continue
                state_rows.append({"complex_id": cid, "cluster_id": audit_records[cid]["cluster_id"],
                    "ctype": audit_records[cid]["ctype"], "state": state, "chain": key[0],
                    "resnum": key[1], "icode": key[2], "group": key[3],
                    "interface": bool(mask["interface"]),
                    "partner_distance_A": mask["partner_distance_A"],
                    "pkai_plus": float(value), "pypka": float(teacher["pka"]),
                    "error": float(value) - float(teacher["pka"])})
    grouped = defaultdict(dict)
    for row in state_rows:
        key = (row["complex_id"], row["chain"], row["resnum"], row["icode"], row["group"])
        grouped[key][row["state"]] = row
    paired_rows = []
    for states in grouped.values():
        if "AB" not in states: continue
        free_name = next((name for name in ("A", "B") if name in states), None)
        if free_name is None: continue
        ab, free = states["AB"], states[free_name]
        paired_rows.append({**{name: ab[name] for name in ("complex_id", "cluster_id", "ctype",
            "chain", "resnum", "icode", "group", "interface", "partner_distance_A")},
            "free_state": free_name, "pkai_plus_shift": ab["pkai_plus"] - free["pkai_plus"],
            "pypka_shift": ab["pypka"] - free["pypka"],
            "shift_error": (ab["pkai_plus"] - free["pkai_plus"]) - (ab["pypka"] - free["pypka"])})

    def write_rows(name, rows):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if state_rows: write_rows("matched-state.csv", state_rows)
    if paired_rows: write_rows("matched-paired.csv", paired_rows)
    interface = [row for row in paired_rows if row["interface"]]
    nonterminal_state = [row for row in state_rows if row["group"] not in ("NTERM", "CTERM")]
    nonterminal_paired = [row for row in paired_rows if row["group"] not in ("NTERM", "CTERM")]
    nonterminal_interface = [row for row in nonterminal_paired if row["interface"]]
    groups = {group: measures([row for row in state_rows if row["group"] == group], "error")
              for group in sorted({row["group"] for row in state_rows})}
    summary = {"passed": bool(state_rows and paired_rows), "evaluation_only": True,
        "model": "released pKAI+; stored predictions on the exact AB/A/B structures",
        "backbone_checkpoint_available": False, "complexes_with_pypka_results": processed,
        "matched_state_sites": len(state_rows), "matched_paired_sites": len(paired_rows),
        "state": measures(state_rows, "error"), "paired": measures(paired_rows, "shift_error"),
        "interface_paired": measures(interface, "shift_error"), "state_by_group": groups,
        "shared_nonterminal_sites": {"state": measures(nonterminal_state, "error"),
            "paired": measures(nonterminal_paired, "shift_error"),
            "interface_paired": measures(nonterminal_interface, "shift_error")},
        "source_cohort_sha256": digest(cohort(root)), "pypka_manifest_sha256": digest(audit(root) / "manifest.json"),
        "test_data_included": False}
    atomic_json(out / "summary.json", summary)
    lines = ["# Released pKAI+ evaluation", "",
        "This is evaluation only: no parameters were updated. The stored released-pKAI+ predictions for the exact "
        "AB/A/B structures were compared with current PypKa on the fixed PINDER validation cohort.", "",
        f"PypKa structures currently available: **{processed}/400**. Matched coverage: **{len(state_rows):,} state "
        f"sites** and **{len(paired_rows):,} paired sites**.", "",
        "| Model | State MAE | Paired-shift MAE | Interface paired-shift MAE |",
        "|---|---:|---:|---:|",
        f"| Released pKAI+ | {summary['state']['mae']:.3f} | {summary['paired']['mae']:.3f} | "
        f"{summary['interface_paired']['mae']:.3f} |",
        f"| Released pKAI+ (shared nonterminal sites) | {summary['shared_nonterminal_sites']['state']['mae']:.3f} | "
        f"{summary['shared_nonterminal_sites']['paired']['mae']:.3f} | "
        f"{summary['shared_nonterminal_sites']['interface_paired']['mae']:.3f} |", "",
        "No backbone-pretrained pKAI+ checkpoint exists in the registered experiments, so none was evaluated. "
        "Producing one would require a new training run.", "",
        "Per-site values are in `matched-state.csv` and `matched-paired.csv`; complete metrics and residue strata "
        "are in `summary.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    require_compute(threads=min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    report(Path(os.environ["PKABENCH_RUNTIME"]))


if __name__ == "__main__": main()
