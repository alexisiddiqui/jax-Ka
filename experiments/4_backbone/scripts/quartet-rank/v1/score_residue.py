"""negatives-v1 (2026-10-29): copied unchanged from decoy-rank-jobs (report title only; run with GQT_PARAMETER_SCALING=1
for the width-128 runs).

Score the residue-heads-v1 runs against the default recipe (2026-10-19, user: "train burial on all residues and burial +
residue edges on the default recipe, seed 17").

(1) pKa from each run's stored selected-epoch predictions: selection on all PINDER validation sites and without bonded-Cys
    sites, state, interface, pKPDB val, benchmark.
(2) Per residue on PINDER validation (residue-rsa-v1 labels), every residue with an RSA:
    burial   free branch: AUROC of the burial logit for free RSA < 0.1 / 0.25 / 0.5, Spearman(sigmoid, 1 - RSA);
             binding: Spearman of the bound - free burial-probability difference with RSA_free - RSA_bound.
    interface (residue within 5 A heavy-atom distance of the partner in the bound complex), scored from the FREE branch:
             predicted exposure (-burial logit), edge score (max over the partner's surface residues; edge runs), and the
             measured free RSA; pooled AUROC and per-chain median AUROC (chains with both classes), on all residues and on
             surface residues (free RSA >= 0.1).
    edges    (edge runs) every cross-chain surface pair, contact <= 10 A: pooled AUROC / AP, macro AP over complexes;
             free and bound branches.
  python score_residue.py OUT_DIR NAME=RUN [NAME=RUN ...]
v4 (2026-10-20, user: "can you change the burial eval to just interface residues"): burial metrics (free, bound,
change) on interface residues only (within 5 A of the partner in the bound complex); interface and edge metrics unchanged.
v3 (2026-10-20, residue-siamese-v1): adds bound-branch burial (vs RSA_bound) and the MAE of the predicted change.
v2 (2026-10-19): the edge score of residues without a scored surface pair (free RSA < 0.1) is set below the complex's
lowest edge score, so "all residues" ranks buried residues last (v1 dropped them, making that row surface-only).
"""
import csv
import json
import os
import sys
from pathlib import Path

import cpu_attention  # noqa: F401  (plain-JAX attention under JAX_PLATFORMS=cpu)
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from pkatrain.gqt_paired_pinder import _metrics
from pkatrain.loading import Prefetcher
from pkatrain.production_gate import DisulfideTable, bonded_queries
from pkatrain.production_graphs import ProductionStore, output, read
from pkatrain.production_inputs import BackboneTable
from pkatrain.production_loading import LoaderConfig, PinderSource, select
from pkatrain.production_residue import SURFACE_RSA, ResidueTable, surface_pairs
from pkatrain.production_train import _auroc, _with_policy, run_dir, selected_model

ROOT = Path(os.environ["PKABENCH_RUNTIME"]); OUT = Path(sys.argv[1]); RUNS = dict(a.split("=", 1) for a in sys.argv[2:]); OUT.mkdir(parents=True, exist_ok=True)
manifest = read(output(ROOT, "pinder") / "manifest-v1.json"); table = ResidueTable(ROOT); store = ProductionStore(Path(manifest["store"]))
VAL = [r["id"] for r in select(manifest, "val")]; ss = DisulfideTable(ROOT, "pinder")
bonded = {}
for cid in VAL:
    for i, f in enumerate(bonded_queries(store.raw(cid), ss.flags(cid)[0])): bonded[(cid, i)] = bool(f)


def pka(run):
    sel = read(run_dir(ROOT, run) / "selection.json"); e = sel["selected_epoch"]; v = sel["validation"]; rows = []
    for r in csv.DictReader((run_dir(ROOT, run) / f"predictions-pinder-epoch-{e:03d}.csv").open()):
        rows.append({"state_error": float(r["state_error"]), "paired_error": float(r["paired_error"]), "interface": r["interface"] == "True",
                     "distance": float(r["distance"]), "bonded": bonded[(r["complex_id"], int(r["site"]))]})
    a = _metrics(rows); c = _metrics([r for r in rows if not r["bonded"]])
    return {"epoch": e, "selection": v["selection"], "selection_no_bonded": c["state_mae"] + c["interface_paired_mae"], "state": a["state_mae"],
            "interface": a["interface_paired_mae"], "pkpdb": v["pkpdb"]["site_mae"], "benchmark": v["benchmark"]["overall"]["mae"],
            "state_no_bonded": c["state_mae"], "interface_no_bonded": c["interface_paired_mae"],  # drsa-within-v1
            "paired_all_no_bonded": c["paired_mae"], "paired_bins_no_bonded": {k: b["paired_mae"] for k, b in c["distance_bins"].items()}}


def binary_metrics(y, p):
    """production_edges.binary_metrics (residue-edges-v1 scoring): AP (step), AUROC (trapezoid)."""
    y = np.asarray(y, float); p = np.asarray(p, float); order = np.argsort(-p, kind="stable"); yy = y[order]; pp = p[order]
    ends = np.r_[np.flatnonzero(np.diff(pp) != 0), len(pp) - 1]
    tp = np.cumsum(yy)[ends]; fp = ends + 1 - tp; positives = float(y.sum()); negatives = len(y) - positives
    ap = float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / positives) if positives else None
    auc = None
    if positives and negatives:
        tpr = np.r_[0, tp / positives]; fpr = np.r_[0, fp / negatives]; auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return {"average_precision": ap, "auroc": auc}


def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def edge_scores(e, z, a, b):
    out = []
    for s in range(0, len(a), 65536):
        za, zb = z[a[s:s + 65536]], z[b[s:s + 65536]]; f = np.concatenate((za * zb, np.abs(za - zb)), axis=1)
        out.append((gelu(f @ e["hidden"]["w"] + e["hidden"]["b"]) @ e["out"]["w"] + e["out"]["b"])[:, 0])
    return np.concatenate(out) if out else np.zeros(0)


def per_chain_median(score, label, chain):
    v = [_auroc(score[chain == c], label[chain == c]) for c in np.unique(chain)]; v = [x for x in v if x is not None]
    return (float(np.median(v)), len(v)) if v else (None, 0)


def residue_scores(run):
    from pkanet.model import norm
    protocol = read(run_dir(ROOT, run) / "protocol.json"); params, (embeddings, _) = selected_model(ROOT, run)
    if "sidechains" in protocol: raise NotImplementedError("score backbone-only runs")
    edges = "aux_redge" in params; host = jax.tree.map(np.asarray, params)

    def one(p, g):
        tok = embeddings(p, g)[0]; n = g["nodes"].shape[0]; r = tok[:n]
        return (r @ p["aux_burial"]["w"] + p["aux_burial"]["b"])[:, 0], (norm(p["aux_redge"]["norm"], r) if edges else r)
    fn = jax.jit(jax.vmap(one, in_axes=(None, 0)))
    src = PinderSource(_with_policy(manifest), config=LoaderConfig(), sidechains=BackboneTable(ROOT, "pinder")); src.residues = table
    acc = {k: [] for k in ("logit_free", "logit_bound", "rsa_free", "rsa_bound", "iface", "surface", "chain_key", "edge_res")}; pairs = {"free": [[], [], []], "bound": [[], [], []]}
    for ids, batch in Prefetcher(src, src.policy.plans(select(manifest, "val"), np.random.default_rng(0)), LoaderConfig()):
        graphs = batch[0]; flat = jax.tree.map(lambda v: v.reshape((-1,) + v.shape[2:]), graphs); logit, z = map(np.asarray, fn(params, flat))
        b2 = logit.shape[0] // 2; logit = logit.reshape(b2, 2, -1); z = z.reshape(b2, 2, z.shape[1], -1)
        for slot, cid in enumerate(ids):
            r = table.row(cid); m = len(r["rsa"]); fr, bd = r["rsa"][:, 1], r["rsa"][:, 0]; keep = np.isfinite(fr) & np.isfinite(bd)
            close = np.zeros(m, bool); c5 = r["d"] <= 5.0; close[r["i"][c5]] = True; close[r["j"][c5]] = True
            res_edge = np.full(m, -np.inf)
            if edges:
                a, b = surface_pairs(r)
                if len(a):
                    key = r["i"] * m + r["j"]; y = np.isin(a * m + b, key).astype(float)
                    for name, br in (("free", 1), ("bound", 0)):
                        s = edge_scores(host["aux_redge"], z[slot, br, :m], a, b); pairs[name][0].append(y); pairs[name][1].append(s)
                        if y.any() and (1 - y).any(): pairs[name][2].append(binary_metrics(y, s)["average_precision"])
                        if br == 1: np.maximum.at(res_edge, a, s); np.maximum.at(res_edge, b, s)
                # residues without a scored pair (buried, free RSA < 0.1) rank below every scored residue of the complex
                if np.isfinite(res_edge).any(): res_edge[~np.isfinite(res_edge)] = res_edge[np.isfinite(res_edge)].min() - 1.0
            for k, v in (("logit_free", logit[slot, 1, :m]), ("logit_bound", logit[slot, 0, :m]), ("rsa_free", fr), ("rsa_bound", bd), ("iface", close),
                         ("surface", np.nan_to_num(fr) >= SURFACE_RSA), ("chain_key", np.array([f"{cid}:{c}" for c in r["chain"]])), ("edge_res", res_edge)):
                acc[k].append(v[keep])
    src.close(); v = {k: np.concatenate(x) for k, x in acc.items()}; out = {}; full = dict(v)
    # v4: burial on interface residues only (within 5 A heavy-atom distance of the partner in the bound complex)
    sub = v["iface"].astype(bool); v = {k: x[sub] for k, x in v.items()}; out["burial_subset"] = {"residues": int(sub.sum())}
    out["burial_free"] = {f"auroc_rsa<{t}": _auroc(v["logit_free"], v["rsa_free"] < t) for t in (0.1, 0.25, 0.5)}
    out["burial_free"]["spearman"] = float(spearmanr(1 / (1 + np.exp(-v["logit_free"])), 1 - np.clip(v["rsa_free"], 0, 1))[0])
    out["burial_bound"] = {"auroc_rsa<0.25": _auroc(v["logit_bound"], v["rsa_bound"] < 0.25),
                           "spearman": float(spearmanr(1 / (1 + np.exp(-v["logit_bound"])), 1 - np.clip(v["rsa_bound"], 0, 1))[0])}
    drsa_ = v["rsa_free"] - v["rsa_bound"]; pchange = 1 / (1 + np.exp(-v["logit_bound"])) - 1 / (1 + np.exp(-v["logit_free"]))
    out["burial_binding"] = {"mae_dprob_vs_drsa": float(np.mean(np.abs(pchange - np.clip(v["rsa_free"], 0, 1) + np.clip(v["rsa_bound"], 0, 1))))}
    pb, pf = 1 / (1 + np.exp(-v["logit_bound"])), 1 / (1 + np.exp(-v["logit_free"])); drsa = v["rsa_free"] - v["rsa_bound"]
    out["burial_binding"] = {**out["burial_binding"], "spearman_dprob_vs_drsa": float(spearmanr(pb - pf, drsa)[0]), "auroc_drsa>0.05": _auroc(pb - pf, drsa > 0.05)}
    v = full  # interface prediction over every residue, as before
    scores = {"predicted exposure (free)": -v["logit_free"], "measured free RSA": v["rsa_free"]}
    if edges: scores["edge max (free)"] = v["edge_res"]
    for subset, m in (("all residues", np.ones(len(v["iface"]), bool)), ("surface residues", v["surface"])):
        for name, s in scores.items():
            mm = m & np.isfinite(s); med, chains = per_chain_median(s[mm], v["iface"][mm], v["chain_key"][mm])
            out.setdefault("interface_5A", {}).setdefault(subset, {})[name] = {"pooled_auroc": _auroc(s[mm], v["iface"][mm]), "per_chain_median_auroc": med, "chains": chains}
    out["interface_5A"]["rate_all"] = float(v["iface"].mean())
    if edges:
        for name, (y, s, macro) in pairs.items():
            y, s = np.concatenate(y), np.concatenate(s); bm = binary_metrics(y, s)
            out.setdefault("edges_10A", {})[name] = {"pairs": int(len(y)), "positive_rate": float(y.mean()), "auroc": bm.get("auroc"), "ap": bm["average_precision"], "macro_ap": float(np.mean(macro))}
    return out


report = {}
for name, run in RUNS.items():
    report[name] = {"run": run, "pka": pka(run)}; print(name, json.dumps(report[name]["pka"]), flush=True)
    report[name]["residue"] = residue_scores(run); print(name, json.dumps(report[name]["residue"]), flush=True)
L = ["# negatives-v1: same-class negatives (within-complex listwise, same-stratum swaps) vs run 4 and decoy both (w64), allres 15 A, 50%, seed 17", "",
     "Selection = PINDER val state MAE + interface paired MAE (all sites / without bonded-Cys sites). Residue metrics on every PINDER validation residue with an RSA (residue-rsa-v1).", "",
     "| arm | epoch | selection | no bonded Cys | state | interface | pKPDB | benchmark |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for name, r in report.items():
    p = r["pka"]; L.append(f"| {name} | {p['epoch']} | {p['selection']:.4f} | {p['selection_no_bonded']:.4f} | {p['state']:.4f} | {p['interface']:.4f} | {p['pkpdb']:.4f} | {p['benchmark']:.4f} |")
L += ["", "## Paired (binding-shift) MAE without bonded-Cys sites, by the store's partner distance (A)", "", "| arm | all sites | <=4 | 4-6 | 6-10 | >10 |", "|---|---:|---:|---:|---:|---:|"]
for name, r in report.items():
    p = r["pka"]; b = p["paired_bins_no_bonded"]; L.append(f"| {name} | {p['paired_all_no_bonded']:.4f} | {b['<=4']:.4f} | {b['4-6']:.4f} | {b['6-10']:.4f} | {b['>10']:.4f} |")
L += ["", f"## Burial on interface residues ({next(iter(report.values()))['residue']['burial_subset']['residues']} within 5 A of the partner): free branch vs RSA_free, bound branch vs RSA_bound, change vs dRSA", "",
      "| arm | free AUROC RSA<0.25 | free Spearman | bound AUROC RSA<0.25 | bound Spearman | change: Spearman(dP, dRSA) | change: AUROC dRSA>0.05 | change: MAE(dP, dRSA) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for name, r in report.items():
    b, bo, bb = r["residue"]["burial_free"], r["residue"]["burial_bound"], r["residue"]["burial_binding"]
    L.append(f"| {name} | {b['auroc_rsa<0.25']:.4f} | {b['spearman']:.4f} | {bo['auroc_rsa<0.25']:.4f} | {bo['spearman']:.4f} | {bb['spearman_dprob_vs_drsa']:.4f} | {bb['auroc_drsa>0.05']:.4f} | {bb['mae_dprob_vs_drsa']:.4f} |")
L += ["", "## Interface from the free state (residue within 5 A of the partner)", "", "| arm | residues | score | pooled AUROC | per-chain median AUROC (chains) |", "|---|---|---|---:|---:|"]
for name, r in report.items():
    for subset in ("all residues", "surface residues"):
        for score, m in r["residue"]["interface_5A"][subset].items():
            L.append(f"| {name} | {subset} | {score} | {m['pooled_auroc']:.4f} | {m['per_chain_median_auroc']:.4f} ({m['chains']}) |")
for name, r in report.items():
    if "edges_10A" in r["residue"]:
        L += ["", f"## Edges: {name} (surface cross pairs, contact <= 10 A)", "", "| branch | pairs | positive rate | AUROC | AP | macro AP |", "|---|---:|---:|---:|---:|---:|"]
        for br, m in r["residue"]["edges_10A"].items(): L.append(f"| {br} | {m['pairs']} | {m['positive_rate']:.4f} | {m['auroc']:.4f} | {m['ap']:.4f} | {m['macro_ap']:.4f} |")
(OUT / "report.md").write_text("\n".join(L) + "\n"); (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n"); print("\n".join(L))
