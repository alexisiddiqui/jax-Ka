#!/usr/bin/env python3
"""Score fixed GQT validation predictions against deposited pKPDB labels.

The comparison holds structures, sites, predictions, and aggregation fixed.  It
only swaps the current PypKa 2.10.0 target for the historical pKPDB target on
sites for which a direct author-chain/residue/type match exists.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.frozen_score import aggregate, measures
from pkabench.runtime import atomic_json, digest, require_compute


ROOT = Path(os.environ["PKABENCH_RUNTIME"])
OUT = ROOT / "audits" / "gqt-historical-pkpdb-validation-v1"
GRAPH_MANIFEST = ROOT / "pretraining" / "graph-pilot-v1" / "manifest.json"
PDB_LOOKUP = ROOT / "audits" / "afdb-overlap-v1" / "rcsb_lookup.csv"
LABELS = ROOT / "pretraining" / "pkpdb-5k-v2" / "labels.sqlite"
PREDICTIONS = {
    "GQT 50k": ROOT / "pretraining" / "gqt-backbone-5k-pkmod-v1" / "unweighted" / "seed-17" / "validation_predictions.csv",
    "GQT 200k": ROOT / "pretraining" / "gqt-pkai-parameter-sweep-v1" / "gqt" / "200k" / "unweighted" / "seed-17" / "validation_predictions.csv",
    "GQT 800k": ROOT / "pretraining" / "gqt-pkai-parameter-sweep-v1" / "gqt" / "800k" / "unweighted" / "seed-17" / "validation_predictions.csv",
}
KIND = {"NTR": "NTERM", "CTR": "CTERM"}


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return (
        row["complex_id"], row["chain"], int(row["resnum"]),
        row.get("icode", "").strip(), row["group"],
    )


def historical_index(validation_ids):
    pdb_for = {
        row["complex_id"]: row["pdb"].lower()
        for row in read_csv(PDB_LOOKUP)
        if row["complex_id"] in validation_ids and row["pdb"]
    }
    missing_pdb = sorted(validation_ids - set(pdb_for))

    connection = sqlite3.connect(f"file:{LABELS}?mode=ro", uri=True)
    by_pdb = defaultdict(list)
    for pdb in sorted(set(pdb_for.values())):
        rows = connection.execute(
            "SELECT chain, kind, number, pka FROM labels WHERE pdb=?", (pdb,)
        ).fetchall()
        by_pdb[pdb].extend(rows)
    connection.close()

    result = {}
    ambiguous = set()
    for cid, pdb in pdb_for.items():
        for chain, kind, number, pka in by_pdb[pdb]:
            text = str(number).strip()
            pos = 0
            if text.startswith("-"):
                pos = 1
            while pos < len(text) and text[pos].isdigit():
                pos += 1
            if pos == 0 or (pos == 1 and text[0] == "-"):
                continue
            residue = int(text[:pos])
            insertion = text[pos:].strip()
            item = (cid, str(chain), residue, insertion, KIND.get(kind, kind))
            value = float(pka)
            if item in result and not np.isclose(result[item], value, rtol=0, atol=1e-8):
                ambiguous.add(item)
            else:
                result[item] = value
    for item in ambiguous:
        result.pop(item, None)
    return result, pdb_for, ambiguous, by_pdb, missing_pdb


def summarize(rows, reference):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["complex_id"]].append(row)
    complexes = []
    for cid, items in grouped.items():
        observed = np.asarray([float(row[reference]) for row in items])
        predicted = np.asarray([float(row["predicted_pka"]) for row in items])
        complexes.append({
            "complex_id": cid,
            "component_id": items[0]["component_id"],
            "n": len(items),
            **measures(observed, predicted),
        })
    return aggregate(complexes, replicates=2000, seed=20261008)[0]


def paired_target_delta(rows, seed=20261008):
    """Historical-target MAE minus current-target MAE, paired by component."""
    by_complex = defaultdict(list)
    for row in rows:
        by_complex[row["complex_id"]].append(row)
    by_group = defaultdict(list)
    for items in by_complex.values():
        prediction = np.asarray([float(row["predicted_pka"]) for row in items])
        current = np.asarray([float(row["current_pypka"]) for row in items])
        historical = np.asarray([float(row["historical_pkpdb"]) for row in items])
        by_group[items[0]["component_id"]].append(
            float(np.mean(np.abs(prediction - historical)) - np.mean(np.abs(prediction - current)))
        )
    values = np.asarray([np.mean(group) for group in by_group.values()])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (2000, len(values)))]
    return {
        "historical_minus_current_mae": float(values.mean()),
        "ci95": np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist(),
        "groups": len(values),
    }


def main():
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_json(GRAPH_MANIFEST)
    validation = [row for row in manifest["records"] if row["split"] == "val"]
    validation_ids = {row["complex_id"] for row in validation}
    components = {row["complex_id"]: row["component_id"] for row in validation}
    roles = {row["complex_id"]: row["role"] for row in validation}
    expected_keys = {
        (str(x[0]), str(x[1]), int(x[2]), str(x[3]), str(x[4]))
        for row in validation for x in row["keys"]
    }

    historical, pdb_for, ambiguous, by_pdb, missing_pdb = historical_index(validation_ids)
    common_keys = expected_keys & set(historical)
    coverage_by_complex = {
        cid: sum(item[0] == cid for item in common_keys) for cid in validation_ids
    }

    predictions = {}
    for name, path in PREDICTIONS.items():
        rows = read_csv(path)
        indexed = {key(row): row for row in rows}
        if len(indexed) != len(rows):
            raise AssertionError(f"Duplicate prediction key in {path}")
        predictions[name] = indexed
    shared = set(common_keys)
    for indexed in predictions.values():
        shared &= set(indexed)

    matched = []
    for item in sorted(shared):
        base = predictions["GQT 50k"][item]
        row = {
            "complex_id": item[0], "pdb": pdb_for[item[0]], "chain": item[1],
            "resnum": item[2], "icode": item[3], "group": item[4],
            "component_id": components[item[0]], "role": roles[item[0]],
            "current_pypka": float(base["teacher_pka"]),
            "historical_pkpdb": historical[item],
        }
        for name, indexed in predictions.items():
            value = float(indexed[item]["predicted_pka"])
            if not np.isclose(float(indexed[item]["teacher_pka"]), row["current_pypka"], rtol=0, atol=1e-6):
                raise AssertionError(f"Current target mismatch at {item}")
            row[name] = value
        matched.append(row)

    long_rows = []
    results = []
    comparisons = []
    # The label-to-label row quantifies the target-domain shift directly.
    label_rows = [dict(row, predicted_pka=row["current_pypka"]) for row in matched]
    label_summary = summarize(label_rows, "historical_pkpdb")
    results.append({"model": "Current PypKa target", "target": "historical pKPDB", **label_summary})
    for name in PREDICTIONS:
        model_rows = [dict(row, predicted_pka=row[name]) for row in matched]
        current = summarize(model_rows, "current_pypka")
        historical_score = summarize(model_rows, "historical_pkpdb")
        results.extend([
            {"model": name, "target": "current PypKa", **current},
            {"model": name, "target": "historical pKPDB", **historical_score},
        ])
        comparisons.append({"model": name, **paired_target_delta(model_rows)})
        for row in model_rows:
            long_rows.append({
                "model": name, **{k: row[k] for k in (
                    "complex_id", "pdb", "chain", "resnum", "icode", "group",
                    "component_id", "role", "current_pypka", "historical_pkpdb",
                    "predicted_pka",
                )},
            })

    structures_with_labels = sum(bool(by_pdb[pdb_for[cid]]) for cid in pdb_for)
    matched_structures = len({row["complex_id"] for row in matched})
    coverage = {
        "validation_structures": len(validation_ids),
        "validation_sites": len(expected_keys),
        "structures_with_any_historical_pkpdb_label": structures_with_labels,
        "structures_with_direct_matched_sites": matched_structures,
        "direct_matched_sites": len(matched),
        "direct_site_coverage": len(matched) / len(expected_keys),
        "ambiguous_historical_keys_dropped": len(ambiguous),
        "complexes_without_pdb_mapping": missing_pdb,
        "complexes_without_matched_sites": sorted(cid for cid, n in coverage_by_complex.items() if not n),
    }
    atomic_json(OUT / "coverage.json", coverage)
    atomic_json(OUT / "results.json", results)
    atomic_json(OUT / "paired_target_deltas.json", comparisons)
    write_csv(OUT / "matched_predictions.csv", long_rows)

    lines = [
        "# GQT scoring against historical pKPDB validation labels", "",
        "The same epoch-20 predictions are scored on direct author-chain/residue/type matches. "
        "Only the target column changes; current PypKa and historical pKPDB scores use identical matched sites.", "",
        f"Coverage: **{len(matched):,}/{len(expected_keys):,} sites** across **{matched_structures}/{len(validation_ids)} complexes**.", "",
        "| Model | Target | Group-macro MAE | 95% CI | RMSE | Sites | Groups |", 
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in results:
        ci = row["mae_ci95"]
        ci_text = f"{ci[0]:.4f}–{ci[1]:.4f}" if ci else "—"
        lines.append(
            f"| {row['model']} | {row['target']} | {row['mae']:.4f} | {ci_text} | "
            f"{row['rmse']:.4f} | {row['sites']:,} | {row['groups']} |"
        )
    lines += ["", "| Model | Historical minus current MAE | Paired 95% CI |", "|---|---:|---|" ]
    for row in comparisons:
        lines.append(
            f"| {row['model']} | {row['historical_minus_current_mae']:+.4f} | "
            f"{row['ci95'][0]:+.4f}–{row['ci95'][1]:+.4f} |"
        )
    lines += ["", "Historical coverage is reported explicitly; unmatched sites are never imputed."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(OUT / "verification.json", {
        "passed": True,
        "inputs": {
            "graph_manifest": digest(GRAPH_MANIFEST),
            "pdb_lookup": digest(PDB_LOOKUP),
            "labels_sqlite": digest(LABELS),
            "predictions": {name: digest(path) for name, path in PREDICTIONS.items()},
        },
        "coverage_sha256": digest(OUT / "coverage.json"),
        "results_sha256": digest(OUT / "results.json"),
        "paired_target_deltas_sha256": digest(OUT / "paired_target_deltas.json"),
        "matched_predictions_sha256": digest(OUT / "matched_predictions.csv"),
        "report_sha256": digest(OUT / "report.md"),
    })
    print(json.dumps({"coverage": coverage, "results": results}, indent=2))


if __name__ == "__main__":
    main()
