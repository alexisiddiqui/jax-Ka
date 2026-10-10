"""Summarise scripts/auxiliary_target_audit.py output (2026-10-10): RSA intervals for an ordinal burial target and
additive soft-contact scores for an ordinal interface target, each against the teacher shifts they should explain.

  python scripts/auxiliary_target_report.py SAMPLE.json OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
RSA_EDGES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 10.0)
CONTACT = 4.0
SCORES = {
    "plateau_l1.5": lambda d: np.minimum(1, np.exp(-(d - CONTACT) / 1.5)),
    "plateau_l3": lambda d: np.minimum(1, np.exp(-(d - CONTACT) / 3.0)),
    "plateau_l5": lambda d: np.minimum(1, np.exp(-(d - CONTACT) / 5.0)),
    "exp_l3": lambda d: np.exp(-d / 3.0),
    "exp_l5": lambda d: np.exp(-d / 5.0),
    "count_4.5A": lambda d: (d <= 4.5).astype(float),
    "count_10A": lambda d: np.ones_like(d),
}


def auroc(score, label):
    label = np.asarray(label, bool); n1 = label.sum(); n0 = len(label) - n1
    return float((rankdata(score)[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def quantiles(x): return {f"p{q}": float(np.quantile(x, q / 100)) for q in (25, 50, 75, 90, 95, 99)}


def main():
    from pkanet.model import PKPDB_PK_MOD
    from jaxpropka.parameters import GROUPS
    from pkatrain.gqt_paired_pinder import GROUP_ALIAS
    data = json.loads(Path(sys.argv[1]).read_text()); ref = np.asarray(PKPDB_PK_MOD)
    reference = lambda g: ref[GROUPS.index(GROUP_ALIAS.get(g, g))]
    pinder, pkpdb = data["pinder"], data["pkpdb"]; report = {"sites": {"pinder": len(pinder), "pkpdb": len(pkpdb)}}

    # Burial: RSA bins vs |state shift|.
    burial = {}
    for name, rows, rsa_key, value in (("pinder_free", pinder, "rsa_free", lambda r: r["target_free"]), ("pkpdb", pkpdb, "rsa", lambda r: r["pka"])):
        rows = [r for r in rows if r[rsa_key] is not None]
        rsa = np.asarray([r[rsa_key] for r in rows], float); shift = np.abs([value(r) - reference(r["group"]) for r in rows])
        bins = []
        for lo, hi in zip(RSA_EDGES[:-1], RSA_EDGES[1:]):
            m = (rsa >= lo) & (rsa < hi)
            bins.append({"rsa": f"[{lo}, {hi})", "fraction": float(m.mean()), "mean_abs_shift": float(shift[m].mean()) if m.any() else None})
        burial[name] = {"sites": len(rows), "rsa_quantiles": quantiles(rsa), "rsa_above_1": float((rsa > 1).mean()),
                        "spearman_burial_vs_abs_shift": float(spearmanr(1 - np.clip(rsa, 0, 1), shift)[0]), "bins": bins}
    report["burial"] = burial

    # Interface: additive soft-contact scores vs |AB - free|.
    delta = np.abs([r["target_ab"] - r["target_free"] for r in pinder]); nearest = np.asarray([min(r["contacts"], default=np.inf) for r in pinder])
    stored = np.asarray([r["partner_distance_A"] if r["partner_distance_A"] is not None else np.inf for r in pinder], float)
    agree = np.isfinite(stored) & (stored <= 10)
    report["interface_check"] = {"sites_within_10A": int(np.sum(nearest <= 10)), "fraction_within_10A": float(np.mean(nearest <= 10)),
        "max_abs_diff_vs_stored_partner_distance": float(np.max(np.abs(nearest[agree] - stored[agree]))) if agree.any() else None,
        "abs_delta_quantiles_all": quantiles(delta), "abs_delta_quantiles_within_10A": quantiles(delta[nearest <= 10]),
        "spearman_minus_nearest_distance": float(spearmanr(-np.minimum(nearest, 99), delta)[0]),
        "auroc_minus_nearest_distance_delta_gt_0.5": auroc(-np.minimum(nearest, 99), delta > 0.5)}
    interface = {}
    for name, f in SCORES.items():
        score = np.asarray([float(np.sum(f(np.asarray(r["contacts"], float)))) if r["contacts"] else 0.0 for r in pinder])
        near = nearest <= 10; bins = []
        edges = (-np.inf,) + THRESHOLDS + (np.inf,)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (score > lo) & (score <= hi) if np.isfinite(lo) else (score <= hi)
            bins.append({"score": f"({lo}, {hi}]", "fraction": float(m.mean()), "mean_abs_delta": float(delta[m].mean()) if m.any() else None})
        interface[name] = {"quantiles_within_10A": quantiles(score[near]), "fraction_above": {str(t): float(np.mean(score > t)) for t in THRESHOLDS},
            "bins": bins, "spearman_vs_abs_delta": float(spearmanr(score, delta)[0]),
            "spearman_vs_abs_delta_within_10A": float(spearmanr(score[near], delta[near])[0]),
            "auroc_delta_gt_0.5": auroc(score, delta > 0.5), "auroc_delta_gt_1": auroc(score, delta > 1.0)}
    report["interface"] = interface
    Path(sys.argv[2]).write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
