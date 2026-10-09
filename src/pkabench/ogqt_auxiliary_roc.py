"""ROC analysis for the registered oGQT auxiliary-loss pilot."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("baseline", "low", "standard")
COLORS = {"baseline": "#4c78a8", "low": "#f58518", "standard": "#54a24b"}
GRID = np.linspace(0.0, 1.0, 201)


def _roc(labels, scores):
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    positives = int(labels.sum()); negatives = int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]; scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(labels)[ends]; fp = 1 + ends - tp
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def _deduplicated(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["complex_id"], row["chain"], int(row["resnum"]), row["icode"])].append(row)
    result = []
    for key, sites in grouped.items():
        result.append({"complex_id": key[0], "interface": sites[0]["interface"].lower() == "true",
            "predicted": float(np.mean([float(x["interface_prediction"]) for x in sites])),
            "structural": float(np.mean([float(x["interface_target"]) for x in sites]))})
    return result


def _curves(rows, field):
    pooled = _roc([x["interface"] for x in rows], [x[field] for x in rows])
    by_complex = defaultdict(list)
    for row in rows: by_complex[row["complex_id"]].append(row)
    curves = []
    for cid in sorted(by_complex):
        items = by_complex[cid]
        result = _roc([x["interface"] for x in items], [x[field] for x in items])
        if result is None: continue
        fpr, tpr, auc = result
        curves.append((cid, np.interp(GRID, fpr, tpr), auc))
    matrix = np.stack([x[1] for x in curves]); aucs = np.asarray([x[2] for x in curves])
    return pooled, curves, matrix.mean(0), float(aucs.mean())


def _bootstrap_macro(curves, seed=17, replicates=2000):
    matrix = np.stack([x[1] for x in curves]); aucs = np.asarray([x[2] for x in curves])
    rng = np.random.default_rng(seed); curve_means = []; auc_means = []
    for _ in range(replicates):
        selected = rng.integers(0, len(curves), len(curves))
        curve_means.append(matrix[selected].mean(0)); auc_means.append(aucs[selected].mean())
    curve_means = np.asarray(curve_means); auc_means = np.asarray(auc_means)
    return (np.quantile(curve_means, .025, axis=0), np.quantile(curve_means, .975, axis=0),
            [float(np.quantile(auc_means, .025)), float(np.quantile(auc_means, .975))])


def main():
    root = Path(os.environ["PKABENCH_RUNTIME"]) / "training/ogqt-auxiliary-pilot-v1"
    output = root / "plots"; output.mkdir(exist_ok=True)
    data = {arm: _deduplicated(root / arm / "seed-17/validation_predictions.csv") for arm in ARMS}
    # The validation keys and labels must be identical across matched arms.
    signatures = [{(x["complex_id"], i, x["interface"]) for i, x in enumerate(rows)} for rows in data.values()]
    if any(value != signatures[0] for value in signatures[1:]): raise AssertionError("unmatched validation rows")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    summary = {"positive_definition": "residue delta-SASA > 10 A^2", "residue_deduplicated": True,
               "bootstrap_replicates": 2000, "bootstrap_unit": "complex", "arms": {}}
    for arm, rows in data.items():
        pooled, curves, macro, macro_auc = _curves(rows, "predicted")
        low, high, interval = _bootstrap_macro(curves)
        axes[0].plot(pooled[0], pooled[1], color=COLORS[arm], lw=2,
                     label=f"{arm} (AUC {pooled[2]:.3f})")
        axes[1].plot(GRID, macro, color=COLORS[arm], lw=2,
                     label=f"{arm} (AUC {macro_auc:.3f})")
        axes[1].fill_between(GRID, low, high, color=COLORS[arm], alpha=.10, linewidth=0)
        summary["arms"][arm] = {"residues": len(rows), "complexes": len({x['complex_id'] for x in rows}),
            "eligible_macro_complexes": len(curves), "pooled_auc": pooled[2], "macro_auc": macro_auc,
            "macro_auc_ci95": interval}
    reference, ref_curves, ref_macro, ref_auc = _curves(data["baseline"], "structural")
    axes[0].plot(reference[0], reference[1], color="#777777", lw=1.7, ls="--",
                 label=f"structural weight (AUC {reference[2]:.3f})")
    axes[1].plot(GRID, ref_macro, color="#777777", lw=1.7, ls="--",
                 label=f"structural weight (AUC {ref_auc:.3f})")
    summary["structural_reference"] = {"pooled_auc": reference[2], "macro_auc": ref_auc,
                                        "eligible_macro_complexes": len(ref_curves)}
    for axis, title in zip(axes, ("Pooled validation residues", "Equal-complex macro average")):
        axis.plot([0, 1], [0, 1], color="#aaaaaa", ls=":", lw=1)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate", title=title)
        axis.grid(alpha=.2); axis.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Can oGQT rank ΔSASA-defined interface residues?", fontsize=15)
    fig.savefig(output / "interface_roc.png", dpi=220)
    fig.savefig(output / "interface_roc.svg")
    plt.close(fig)
    (output / "interface_roc.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    counterfactual = root / "interface-counterfactual"
    if (counterfactual / "summary.json").exists():
        audit = json.loads((counterfactual / "summary.json").read_text())
        with (counterfactual / "predictions.csv").open() as stream:
            augmented = list(csv.DictReader(stream))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        x = np.arange(len(ARMS)); width = .24
        native_fpr = [audit["arms"][arm]["youden"]["native_equal_complex_fpr"]["mean"] for arm in ARMS]
        native_fnr = [audit["arms"][arm]["youden"]["native_equal_complex_fnr"]["mean"] for arm in ARMS]
        separated_fpr = [audit["arms"][arm]["youden"]["separated_equal_complex_fpr"]["mean"] for arm in ARMS]
        axes[0].bar(x - width, native_fpr, width, label="Native false-positive rate", color="#e45756")
        axes[0].bar(x, native_fnr, width, label="Native false-negative rate", color="#4c78a8")
        axes[0].bar(x + width, separated_fpr, width, label="Separated false-positive rate", color="#72b7b2")
        axes[0].set(xticks=x, xticklabels=ARMS, ylim=(0, 1), ylabel="Equal-complex error rate",
                    title="Errors at each arm's descriptive Youden threshold")
        axes[0].legend(frameon=False, fontsize=9); axes[0].grid(axis="y", alpha=.2)
        standard = [row for row in augmented if row["arm"] == "standard"]
        positive = np.asarray([float(x["bound_score"]) for x in standard if x["interface"].lower() == "true"])
        negative = np.asarray([float(x["bound_score"]) for x in standard if x["interface"].lower() == "false"])
        separated = np.asarray([float(x["separated_score"]) for x in standard])
        threshold = audit["arms"]["standard"]["youden"]["native"]["threshold"]
        bins = np.linspace(min(positive.min(), negative.min(), separated.min()),
                           max(positive.max(), negative.max(), separated.max()), 45)
        axes[1].hist(negative, bins=bins, density=True, histtype="step", lw=2, label="Native ΔSASA-negative")
        axes[1].hist(positive, bins=bins, density=True, histtype="step", lw=2, label="Native ΔSASA-positive")
        axes[1].hist(separated, bins=bins, density=True, histtype="step", lw=2, label="Partner edges removed")
        axes[1].axvline(threshold, color="black", ls="--", lw=1.3, label=f"Youden threshold {threshold:.3f}")
        axes[1].set(xlabel="Predicted normalized interface weight", ylabel="Density",
                    title="Standard auxiliary: score distributions")
        axes[1].legend(frameon=False, fontsize=9); axes[1].grid(alpha=.2)
        fig.suptitle("Native and separated-structure interface errors", fontsize=15)
        fig.savefig(output / "interface_fpfn_augmentation.png", dpi=220)
        fig.savefig(output / "interface_fpfn_augmentation.svg")
        plt.close(fig)
    # Basic implementation oracles.
    perfect = _roc([0, 0, 1, 1], [0, .1, .9, 1])
    reversed_ = _roc([0, 0, 1, 1], [1, .9, .1, 0])
    if perfect is None or abs(perfect[2] - 1) > 1e-12 or reversed_ is None or abs(reversed_[2]) > 1e-12:
        raise AssertionError("ROC oracle")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__": main()
