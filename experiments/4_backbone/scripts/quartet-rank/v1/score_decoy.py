"""Decoy-partner ranking of the residue edge head on the 800 PINDER validation complexes (decoy-rank-v1, 2026-10-29).

Free branch only (each chain encoded without its partner), selected checkpoint per selection.json, CPU jax.
Receptor = chain code 0 of the complex, partner = every other chain. Surface = free RSA >= 0.1 (residue-rsa-v1).
Contact pairs C_i = receptor-surface x partner-surface pairs with soft target min(1, exp(-(d - 4) / 3)) > 0.5 (the
training margin's definition). Pair score = sigmoid(edge logit); complex score = mean over its pairs.
Pairwise accuracy = fraction of (true, decoy) comparisons where the true partner scores higher, ties count 1/2.
  (1) re-typed: one chain's interface residues re-typed by production_residue.retype (the training rule) with a fixed
      per-complex seed; true = mean over C_i of the native free-branch pair scores, decoy = the same pairs after a free-
      branch pass on the re-typed nodes. Complexes the rule skips (< 3 re-typable interface residues) are left out.
  (2) cross-complex: 5 other validation complexes j per i (fixed seed); decoy = mean over C_i of score(a, b'), b' drawn
      (fixed seed, with replacement) from j's partner-side contact residues; true as in (1).
  (3) surface-only (mirrors the blind POISK arm): true = mean over every receptor-surface x partner-surface pair of i;
      decoy = receptor surface of i x partner surface of j, j = the first of i's partners in (2).
  edges: pooled AUROC / AP and macro AP over complexes of the free-branch score on the true receptor x partner surface
      pairs, label = heavy-atom contact <= 10 A (score_residue.py's edges_10A free row; identical for two-chain complexes).
  python score_decoy.py OUT_DIR NAME=RUN [NAME=RUN ...]     (PKABENCH_RUNTIME, JAX_PLATFORMS=cpu, cpu_attention.py beside it)
"""
import os

os.environ.setdefault("GQT_PARAMETER_SCALING", "1")  # width-128 runs rebuild through the scaling guard (production_position)
import json
import sys
import time
import zlib
from pathlib import Path

import cpu_attention  # noqa: F401  (plain-JAX attention under JAX_PLATFORMS=cpu)
import jax
import numpy as np

from pkatrain.loading import Prefetcher
from pkatrain.production_graphs import output, read
from pkatrain.production_inputs import BackboneTable
from pkatrain.production_loading import LoaderConfig, PinderSource, select
from pkatrain.production_residue import ALPHABET, SURFACE_RSA, ResidueTable, pair_targets, residue_types, retype
from pkatrain.production_train import _with_policy, run_dir, selected_model

SEED = 20261029; RETYPE_SALT = 52711; PARTNERS = 5; CHUNK = 65536
ROOT = Path(os.environ["PKABENCH_RUNTIME"]); OUT = Path(sys.argv[1]); RUNS = dict(a.split("=", 1) for a in sys.argv[2:]); OUT.mkdir(parents=True, exist_ok=True)
manifest = read(output(ROOT, "pinder") / "manifest-v1.json"); table = ResidueTable(ROOT)
VAL = [r["id"] for r in select(manifest, "val")]; INDEX = {cid: i for i, cid in enumerate(VAL)}
_rng = np.random.default_rng(SEED)
CROSS = [_rng.choice(np.delete(np.arange(len(VAL)), i), PARTNERS, replace=False) for i in range(len(VAL))]


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))  # jax.nn.gelu (approximate)


def logits(e, za, zb):
    """aux_redge pair logits for embedding rows za, zb (n, W), chunked."""
    out = []
    for s in range(0, len(za), CHUNK):
        a, b = za[s:s + CHUNK], zb[s:s + CHUNK]; f = np.concatenate((a * b, np.abs(a - b)), axis=1)
        out.append((gelu(f @ e["hidden"]["w"] + e["hidden"]["b"]) @ e["out"]["w"] + e["out"]["b"])[:, 0])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def mean_cross(e, za_rows, zb_rows):
    """Mean sigmoid score over every za_rows x zb_rows pair, chunked by rows of za."""
    if not len(za_rows) or not len(zb_rows): return None, 0
    total = 0.0; step = max(1, CHUNK // len(zb_rows))
    for s in range(0, len(za_rows), step):
        a = za_rows[s:s + step]; za = np.repeat(a, len(zb_rows), axis=0); zb = np.tile(zb_rows, (len(a), 1))
        total += float(sigmoid(logits(e, za, zb)).sum())
    return total / (len(za_rows) * len(zb_rows)), len(za_rows) * len(zb_rows)


def binary_metrics(y, p):
    """production_edges.binary_metrics (score_residue.py): AP (step), AUROC (trapezoid)."""
    y = np.asarray(y, float); p = np.asarray(p, float); order = np.argsort(-p, kind="stable"); yy = y[order]; pp = p[order]
    ends = np.r_[np.flatnonzero(np.diff(pp) != 0), len(pp) - 1]
    tp = np.cumsum(yy)[ends]; fp = ends + 1 - tp; positives = float(y.sum()); negatives = len(y) - positives
    ap = float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / positives) if positives else None
    auc = None
    if positives and negatives:
        tpr = np.r_[0, tp / positives]; fpr = np.r_[0, fp / negatives]; auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return {"average_precision": ap, "auroc": auc}


def win(true, decoy): return 1.0 if true > decoy else 0.5 if true == decoy else 0.0


def accuracy(values):
    v = np.asarray(values, float); n = len(v)
    return {"accuracy": float(v.mean()) if n else None, "comparisons": n, "binomial_se": float(np.sqrt(v.mean() * (1 - v.mean()) / n)) if n else None}


def layout(cid):
    """Residue-table row, receptor / partner surface indices, contact pairs (a receptor, b partner) with target > 0.5,
    and every receptor x partner surface pair's 10 A contact label."""
    r = table.row(cid); m = len(r["rsa"]); surface = np.nan_to_num(r["rsa"][:, 1], nan=0.0) >= SURFACE_RSA; c0 = r["chain"].min()
    rec = np.flatnonzero(surface & (r["chain"] == c0)); par = np.flatnonzero(surface & (r["chain"] != c0))
    a, b = np.repeat(rec, len(par)), np.tile(par, len(rec)); t = pair_targets(r, a, b)
    key = r["i"] * m + r["j"]; label = np.isin(np.minimum(a, b) * m + np.maximum(a, b), key)
    return {"r": r, "rec": rec, "par": par, "contact": (a[t > 0.5], b[t > 0.5]), "all": (a, b, label), "chains": int(len(np.unique(r["chain"])))}


LAYOUT = {cid: layout(cid) for cid in VAL}
SHAPES = {"complexes": len(VAL), "more_than_two_chains": sum(L["chains"] > 2 for L in LAYOUT.values()),
          "with_contacts": sum(len(L["contact"][0]) > 0 for L in LAYOUT.values()),
          "surface_pairs_true": int(sum(len(L["all"][0]) for L in LAYOUT.values()))}


def score(run):
    from pkanet.model import norm
    began = time.time(); protocol = read(run_dir(ROOT, run) / "protocol.json"); params, (embeddings, _) = selected_model(ROOT, run)
    if "aux_redge" not in params: raise ValueError(f"{run} has no residue edge head")
    host = jax.tree.map(np.asarray, params); e = host["aux_redge"]

    def one(p, g):
        tok = embeddings(p, g)[0]; n = g["nodes"].shape[0]; return norm(p["aux_redge"]["norm"], tok[:n])
    fn = jax.jit(jax.vmap(one, in_axes=(None, 0)))
    src = PinderSource(_with_policy(manifest), config=LoaderConfig(), sidechains=BackboneTable(ROOT, "pinder"))
    Z = {}; arm1 = []; true_contact = {}; composition = {"native": np.zeros(20), "retyped": np.zeros(20)}; skipped = 0
    for ids, batch in Prefetcher(src, src.policy.plans(select(manifest, "val"), np.random.default_rng(0)), LoaderConfig()):
        free = jax.tree.map(lambda v: np.asarray(v[:, 1]), batch[0]); decoy = dict(free); nodes = free["nodes"].copy(); valid = {}
        for slot, cid in enumerate(ids):
            L = LAYOUT[cid]; m = len(L["r"]["rsa"]); aa = residue_types(nodes[slot, :m])
            idx, new, mask, ok = retype(L["r"], aa, np.random.default_rng([RETYPE_SALT, zlib.crc32(cid.encode())]))
            valid[cid] = bool(ok)
            if ok:
                keep = idx[mask > 0]; composition["native"] += np.bincount(aa[keep], minlength=20); composition["retyped"] += np.bincount(new[mask > 0], minlength=20)
                nodes[slot, keep, :20] = np.eye(20, dtype=nodes.dtype)[new[mask > 0]]
        decoy["nodes"] = nodes
        z_true, z_decoy = np.asarray(fn(params, free)), np.asarray(fn(params, decoy))
        for slot, cid in enumerate(ids):
            L = LAYOUT[cid]; m = len(L["r"]["rsa"]); Z[cid] = z_true[slot, :m].astype(np.float32); a, b = L["contact"]
            if not len(a): continue
            true_contact[cid] = float(sigmoid(logits(e, Z[cid][a], Z[cid][b])).mean())
            if not valid[cid]: skipped += 1; continue
            zd = z_decoy[slot, :m]; arm1.append(win(true_contact[cid], float(sigmoid(logits(e, zd[a], zd[b])).mean())))
    src.close()
    arm2 = []; arm2_by_i = []; arm3 = []; ys, ss, macro = [], [], []; pairs3 = 0
    for i, cid in enumerate(VAL):
        L = LAYOUT[cid]; z = Z[cid]; a, b, label = L["all"]; t3 = None
        if len(a):  # edge AUROC / AP on the true receptor x partner surface pairs; their mean score is arm (3)'s true score
            s = logits(e, z[a], z[b]); ys.append(label); ss.append(s); t3 = float(sigmoid(s).mean())
            if label.any() and (~label).any(): macro.append(binary_metrics(label, s)["average_precision"])
        if cid in true_contact:
            rng = np.random.default_rng([SEED, i]); row = []
            for j in CROSS[i]:
                other = VAL[j]; patch = np.unique(LAYOUT[other]["contact"][1])
                if not len(patch): continue
                bp = rng.choice(patch, len(L["contact"][0])); d = float(sigmoid(logits(e, z[L["contact"][0]], Z[other][bp])).mean())
                row.append(win(true_contact[cid], d))
            arm2 += row; arm2_by_i += [float(np.mean(row))] if row else []
        other = VAL[CROSS[i][0]]
        d3, m3 = mean_cross(e, z[L["rec"]], Z[other][LAYOUT[other]["par"]])
        if t3 is not None and d3 is not None: arm3.append(win(t3, d3)); pairs3 += len(a) + m3
    y, s = np.concatenate(ys), np.concatenate(ss); bm = binary_metrics(y, s)
    comp = {k: {ALPHABET[x]: round(float(v[x] / max(v.sum(), 1)), 4) for x in range(20)} for k, v in composition.items()}
    return {"run": run, "width": protocol["architecture"]["width"], "selected_epoch": read(run_dir(ROOT, run) / "selection.json")["selected_epoch"],
            "decoy_protocol": {k: protocol[k] for k in ("decoy_retype", "decoy_batch") if k in protocol},
            "retyped": {**accuracy(arm1), "skipped_by_rule": skipped, "composition_of_retyped_residues": comp},
            "cross_complex": {**accuracy(arm2), "per_complex_mean": float(np.mean(arm2_by_i)) if arm2_by_i else None, "complexes": len(arm2_by_i)},
            "surface_only": {**accuracy(arm3), "pairs_scored": pairs3},
            "edges_10A_free": {"pairs": int(len(y)), "positive_rate": float(y.mean()), "auroc": bm["auroc"], "ap": bm["average_precision"], "macro_ap": float(np.mean(macro))},
            "seconds": round(time.time() - began, 1)}


def fmt(x, d=4): return "-" if x is None else f"{x:.{d}f}"


def run_md(name, r):
    L = [f"# Decoy ranking: {name}", "", f"run `{r['run']}`, width {r['width']}, selected epoch {r['selected_epoch']}; {SHAPES}", "",
         "| arm | pairwise accuracy | comparisons | binomial SE |", "|---|---:|---:|---:|"]
    for arm in ("retyped", "cross_complex", "surface_only"): L.append(f"| {arm} | {fmt(r[arm]['accuracy'])} | {r[arm]['comparisons']} | {fmt(r[arm]['binomial_se'])} |")
    m = r["edges_10A_free"]
    L += ["", f"cross-complex per-complex mean accuracy {fmt(r['cross_complex']['per_complex_mean'])} over {r['cross_complex']['complexes']} complexes; "
          f"re-typed arm skipped {r['retyped']['skipped_by_rule']} complexes (< 3 re-typable interface residues).", "",
          f"Free-branch edges (true receptor x partner surface pairs, contact <= 10 A): pairs {m['pairs']}, positive rate {fmt(m['positive_rate'])}, "
          f"AUROC {fmt(m['auroc'])}, AP {fmt(m['ap'])}, macro AP {fmt(m['macro_ap'])}.", "",
          "Composition of the re-typed residues (native -> re-typed): " + json.dumps(r["retyped"]["composition_of_retyped_residues"])]
    return "\n".join(L) + "\n"


report = {"definition": __doc__, "seed": SEED, "validation": SHAPES, "runs": {}}
for name, run in RUNS.items():
    r = score(run); report["runs"][name] = r; d = OUT / name.replace(" ", "_").replace("/", "_"); d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(r, indent=1) + "\n"); (d / "report.md").write_text(run_md(name, r))
    print(name, json.dumps({k: r[k] for k in ("retyped", "cross_complex", "surface_only", "edges_10A_free", "seconds")}, default=str), flush=True)
L = ["# Decoy-partner ranking of the residue edge head (decoy-rank-v1, PINDER validation, free branch)", "",
     f"{SHAPES}. Pairwise accuracy, ties 1/2; chance 0.5. Definitions in report.json.", "",
     "| arm | width | epoch | re-typed | cross-complex | surface-only | edge AUROC | edge AP | macro AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for name, r in report["runs"].items():
    m = r["edges_10A_free"]
    L.append(f"| {name} | {r['width']} | {r['selected_epoch']} | {fmt(r['retyped']['accuracy'])} | {fmt(r['cross_complex']['accuracy'])} | "
             f"{fmt(r['surface_only']['accuracy'])} | {fmt(m['auroc'])} | {fmt(m['ap'])} | {fmt(m['macro_ap'])} |")
(OUT / "report.md").write_text("\n".join(L) + "\n"); (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n"); print("\n".join(L))
