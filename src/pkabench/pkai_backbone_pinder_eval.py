"""Evaluation-only comparison of released and backbone-pretrained pKAI on PINDER.

No parameters are updated.  Released pKAI predictions are the frozen labels made
from the exact AB/A/B structures.  Backbone-pretrained checkpoints are evaluated
on a strict N/O-backbone representation of those same structures and sites.
Both are compared with the current-PypKa teacher audit on the fixed validation
cohort.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx

from pkatrain.pkai_scratch import model_class, native
from .runtime import atomic_json, digest, require_compute
from .schema import read_table


SEEDS = (17, 29, 43)
PK_MOD = {"ASP": 3.79, "CYS": 8.67, "TYR": 9.59, "GLU": 4.20,
          "HIS": 6.74, "LYS": 10.46}
RES_OHE = ("CTR", "CYS", "TYR", "GLU", "HIS", "ASP", "LYS", "NTR")
KEY_FIELDS = ("chain", "resnum", "icode", "group")


def read(path):
    return json.loads(Path(path).read_text())


def source(root):
    return Path(root) / "pretraining/pinder-pkai-v1"


def cohort(root):
    return Path(root) / "training/ogqt-pinder-factorial-v1/cohort.json"


def audit(root):
    return Path(root) / "audits/pinder-pkai-pypka-v1"


def output(root):
    return Path(root) / "audits/pinder-pkai-backbone-eval-v1"


def checkpoints(root):
    base = Path(root) / "pretraining/pkai-backbone-ablation-v1/backbone-pretrained"
    return {seed: base / f"seed-{seed}/best.pt" for seed in SEEDS}


def _parameter_digest(path):
    return digest(path)


def _read_cif(path):
    with gzip.open(path, "rt") as stream:
        file = pdbx.CIFFile.read(stream)
    return pdbx.get_structure(file, model=1, altloc="occupancy",
                              use_author_fields=True, include_bonds=False)


def _features(atoms, keys, encoding="atom16"):
    """Reproduce the registered strict-backbone pKAI representation.

    The checkpoint's backbone protocol uses the query residue C-alpha only as
    its origin.  Context consists of N/O atoms from other residues within 15 A;
    the native 4008-wide pKAI schema is otherwise unchanged.
    """
    if encoding == "atom16aa20ori":
        from pkatrain.pkai_orientation import orientation_features
        return orientation_features(atoms, keys)
    protein = ~np.asarray(atoms.hetero)
    atom_name = np.asarray(atoms.atom_name).astype(str)
    context = protein & np.isin(atom_name, ("N", "O"))
    context_coord = np.asarray(atoms.coord[context], np.float64)
    context_chain = np.asarray(atoms.chain_id[context]).astype(str)
    context_resnum = np.asarray(atoms.res_id[context], np.int64)
    context_icode = np.asarray(atoms.ins_code[context]).astype(str)
    context_name = atom_name[context]
    context_residue = np.asarray(atoms.res_name[context]).astype(str)

    ca = protein & (atom_name == "CA")
    ca_lookup = {}
    for chain, number, insertion, coord in zip(
            np.asarray(atoms.chain_id[ca]).astype(str),
            np.asarray(atoms.res_id[ca], np.int64),
            np.asarray(atoms.ins_code[ca]).astype(str),
            np.asarray(atoms.coord[ca], np.float64)):
        key = (chain, int(number), insertion)
        if key in ca_lookup:
            raise AssertionError((key, "duplicate CA"))
        ca_lookup[key] = coord

    from pkatrain.pkai_scratch import CUTOFF, SLOTS, SLOT_WIDTH, aa20_index, feature_width
    slot = SLOT_WIDTH[encoding]  # "aa20": the slot holds the residue type of the backbone N/O atom (see pkai_scratch)
    matrix = np.zeros((len(keys), feature_width(encoding)), np.float32)
    retained = np.zeros(len(keys), bool)
    for row, key in enumerate(keys):
        chain, number, insertion, group = key
        residue = (str(chain), int(number), str(insertion))
        if group not in PK_MOD or residue not in ca_lookup:
            continue
        distance = np.sqrt(np.sum((context_coord - ca_lookup[residue]) ** 2, axis=1))
        different = ((context_chain != residue[0]) | (context_resnum != residue[1]) |
                     (context_icode != residue[2]))
        ids = np.flatnonzero(different & (distance < CUTOFF))
        if np.any(distance[ids] == 0):
            raise ValueError((key, "coincident backbone atom"))
        # Native pKAI sorts by (distance, encoded atom class).  In the strict
        # representation the only possible classes are N (0) and O (9).
        if encoding == "atom16":
            ordered = sorted(((float(distance[j]), 0 if context_name[j] == "N" else 9)
                              for j in ids))[:SLOTS]
        elif encoding in ("atom16aa20", "atom16aa20sc"):
            ordered = sorted(((float(distance[j]), 0 if context_name[j] == "N" else 9, aa20_index(context_residue[j]))
                              for j in ids))[:SLOTS]
        else:
            ordered = sorted(((float(distance[j]), aa20_index(context_residue[j])) for j in ids))[:SLOTS]
        for position, (value, atom_class, *residue) in enumerate(ordered):
            matrix[row, position * slot + atom_class] = 1.0 / value ** 2
            if residue: matrix[row, position * slot + 16 + residue[0]] = 1.0 / value ** 2
            if encoding == "atom16aa20sc": matrix[row, position * slot + 36] = 1.0 / value ** 2
        matrix[row, SLOTS * slot + RES_OHE.index(group)] = 1.0
        retained[row] = True
    return matrix, retained


def predict(root, *, limit=None):
    root = Path(root); out = output(root); out.mkdir(parents=True, exist_ok=True)
    records = [row for row in read(cohort(root))["records"] if row["split"] == "val"]
    if len(records) != 400 or len({row["cluster_id"] for row in records}) != 400:
        raise AssertionError("expected fixed 400-cluster validation cohort")
    if limit is not None:
        records = records[:int(limit)]
    torch, _ = native()
    require_compute(threads=min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
                    gpu_benchmark=True, allow_comp1400=True)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    models = {}
    model_hashes = {}
    for seed, path in checkpoints(root).items():
        if not path.exists(): raise FileNotFoundError(path)
        model = model_class(torch)()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model = model.cuda().eval(); models[seed] = model
        model_hashes[str(seed)] = {"file_sha256": _parameter_digest(path)}

    predictions = []
    with torch.no_grad():
        for number, record in enumerate(records, 1):
            cid = record["id"]; folder = source(root) / "entries" / cid
            labels = read(folder / "labels.json")["pkai"]
            masks = {(str(row["chain"]), int(row["resnum"]), str(row["icode"]), str(row["group"])): row
                     for row in read(folder / "sites.json")}
            for state in ("AB", "A", "B"):
                state_labels = [(str(c), int(n), str(i), str(g), float(v))
                                for c, n, i, g, v in labels.get(state, [])
                                if v is not None and str(g) in PK_MOD]
                keys = [row[:4] for row in state_labels]
                values, retained = _features(_read_cif(folder / f"{state}.cif.gz"), keys)
                ids = np.flatnonzero(retained)
                outputs = {seed: np.empty(len(keys), np.float32) for seed in SEEDS}
                for start in range(0, len(ids), 4096):
                    chosen = ids[start:start + 4096]
                    # PyTorch 1.13 in the released pKAI environment predates
                    # NumPy 2; use the Python buffer boundary deliberately.
                    tensor = torch.tensor(values[chosen].tolist(), device="cuda")
                    for seed, model in models.items():
                        outputs[seed][chosen] = np.asarray(
                            model(tensor).detach().cpu().tolist(), dtype=np.float32)
                for index in ids:
                    key = keys[index]; mask = masks.get(key)
                    row = {"complex_id": cid, "cluster_id": record["cluster_id"],
                           "ctype": record["ctype"], "state": state,
                           **dict(zip(KEY_FIELDS, key)), "regular_pkai": state_labels[index][4],
                           "eval_mask": bool(mask and mask["eval_mask"]),
                           "interface": bool(mask and mask["interface"]),
                           "partner_distance_A": mask.get("partner_distance_A") if mask else None}
                    for seed in SEEDS:
                        row[f"backbone_seed_{seed}"] = float(PK_MOD[key[3]] + outputs[seed][index])
                    row["backbone_ensemble"] = float(np.mean([row[f"backbone_seed_{seed}"] for seed in SEEDS]))
                    predictions.append(row)
            if number % 25 == 0:
                print(json.dumps({"complexes": number, "total": len(records),
                                  "predictions": len(predictions)}), flush=True)
    path = out / ("smoke-predictions.csv" if limit is not None else "predictions.csv")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    atomic_json(out / ("smoke.json" if limit is not None else "prediction.json"), {
        "passed": True, "evaluation_only": True, "complexes": len(records), "rows": len(predictions),
        "models": model_hashes, "predictions_sha256": digest(path),
        "source_cohort_sha256": digest(cohort(root)), "test_data_included": False})


def _measures(rows, field):
    values = np.asarray([row[field] for row in rows], np.float64)
    return {"sites": len(values), "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "bias": float(np.mean(values))}


def report(root):
    root = Path(root); out = output(root)
    prediction_meta = read(out / "prediction.json")
    prediction_path = out / "predictions.csv"
    if digest(prediction_path) != prediction_meta["predictions_sha256"]:
        raise AssertionError("prediction file changed")
    with prediction_path.open() as stream:
        predictions = list(csv.DictReader(stream))
    pred = {(row["complex_id"], row["state"], row["chain"], int(row["resnum"]),
             row["icode"], row["group"]): row for row in predictions}
    methods = ("regular_pkai", "backbone_seed_17", "backbone_seed_29",
               "backbone_seed_43", "backbone_ensemble")
    state_rows = []
    manifest = read(audit(root) / "manifest.json")
    for record in manifest["records"]:
        path = audit(root) / "results" / record["id"] / "predictions.parquet"
        if not path.exists(): continue
        for teacher in read_table(path):
            if teacher["status"] != "ok" or teacher["pka"] is None: continue
            key = (record["id"], teacher["state"], teacher["chain"], int(teacher["resnum"]),
                   teacher["icode"], teacher["group"])
            candidate = pred.get(key)
            if candidate is None or candidate["eval_mask"].lower() != "true": continue
            row = {"complex_id": key[0], "cluster_id": candidate["cluster_id"],
                   "ctype": candidate["ctype"], "state": key[1], "chain": key[2],
                   "resnum": key[3], "icode": key[4], "group": key[5],
                   "interface": candidate["interface"].lower() == "true",
                   "partner_distance_A": candidate["partner_distance_A"],
                   "pypka": float(teacher["pka"])}
            for method in methods:
                row[method] = float(candidate[method])
                row[f"{method}_error"] = row[method] - row["pypka"]
            state_rows.append(row)
    paired_rows = []
    grouped = defaultdict(dict)
    for row in state_rows:
        grouped[(row["complex_id"], row["chain"], row["resnum"], row["icode"], row["group"])][row["state"]] = row
    for key, states in grouped.items():
        if "AB" not in states: continue
        free_name = next((name for name in ("A", "B") if name in states), None)
        if free_name is None: continue
        ab, free = states["AB"], states[free_name]
        row = {name: ab[name] for name in ("complex_id", "cluster_id", "ctype", "chain",
                                            "resnum", "icode", "group", "interface", "partner_distance_A")}
        row["free_state"] = free_name; row["pypka_shift"] = ab["pypka"] - free["pypka"]
        for method in methods:
            row[f"{method}_shift"] = ab[method] - free[method]
            row[f"{method}_shift_error"] = row[f"{method}_shift"] - row["pypka_shift"]
        paired_rows.append(row)

    def write_rows(name, rows):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if state_rows: write_rows("matched-state.csv", state_rows)
    if paired_rows: write_rows("matched-paired.csv", paired_rows)
    summary = {"passed": bool(state_rows and paired_rows), "evaluation_only": True,
               "matched_state_sites": len(state_rows), "matched_paired_sites": len(paired_rows),
               "methods": {}, "test_data_included": False}
    for method in methods:
        state = _measures(state_rows, f"{method}_error")
        paired = _measures(paired_rows, f"{method}_shift_error")
        interface = _measures([row for row in paired_rows if row["interface"]], f"{method}_shift_error")
        strata = {}
        for group in sorted({row["group"] for row in state_rows}):
            strata[group] = _measures([row for row in state_rows if row["group"] == group],
                                      f"{method}_error")
        summary["methods"][method] = {"state": state, "paired": paired,
                                        "interface_paired": interface,
                                        "state_by_group": strata}
    atomic_json(out / "summary.json", summary)
    labels = {
        "regular_pkai": "Regular pKAI",
        "backbone_seed_17": "Backbone-pretrained, seed 17",
        "backbone_seed_29": "Backbone-pretrained, seed 29",
        "backbone_seed_43": "Backbone-pretrained, seed 43",
        "backbone_ensemble": "Backbone-pretrained, 3-seed ensemble",
    }
    lines = ["# Regular and backbone-pretrained pKAI evaluation", "",
             "This is evaluation only: no model parameters were updated. Released pKAI and the three existing "
             "backbone-pretrained checkpoints were scored on the same fixed PINDER validation structures and "
             "compared with current PypKa wherever that audit returned a matched midpoint.", "",
             f"Matched coverage: **{len(state_rows):,} state sites** and **{len(paired_rows):,} paired sites**.", "",
             "| Model | State MAE | Paired-shift MAE | Interface paired-shift MAE |",
             "|---|---:|---:|---:|"]
    for method in methods:
        item = summary["methods"][method]
        lines.append(f"| {labels[method]} | {item['state']['mae']:.3f} | "
                     f"{item['paired']['mae']:.3f} | {item['interface_paired']['mae']:.3f} |")
    lines += ["", "Regular pKAI uses the stored predictions generated on these exact AB/A/B structures. "
              "The backbone model receives only N/O context atoms around each query C-alpha, matching its "
              "registered training protocol. The ensemble is the arithmetic mean of the three checkpoint predictions.",
              "", "Per-site matched values are in `matched-state.csv` and `matched-paired.csv`; complete numerical "
              "metrics and residue strata are in `summary.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    if action == "smoke": predict(root, limit=2)
    elif action == "predict": predict(root)
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__":
    main()
