"""Asymmetric hotspot-masked scoring of the residue edge head on the 800 PINDER validation complexes (hotspot-mask-v2, 2026-10-30).

score_masked.py (hotspot-mask-v1) with per-comparison chain masks. Free branch, selected checkpoint per selection.json, CPU jax.
Oracle: PeSTo (store-hotspot-v1) at each run's keep fraction (protocol hotspot_mask.keep; the control is scored at every keep
of the other runs, --control-keeps overrides). Per complex the v1 eval mask (production_residue.hotspot_keep, soft 0, per-complex
seed [EVAL_SEED, 0 (pesto), crc32(id)], identical to score_masked.py's pesto mode) gives the unmasked set of BOTH chains; the
free branch is chain-isolated, so a chain's masked embedding is read from the both-chains-masked pass (checked: a direct
chain-1-only pass is computed and its rows compared, "isolation" in the report). Modes:
  unmasked  every residue typed (score_masked.py unmasked; same draws, equal up to float summation order).
  varying   mask ONLY the chain that varies between true and decoy in each comparison: the receptor / fixed side (chain code 0
            of the true complex; the antigen in the Ab/Ag wrong-antibody control) reads full embeddings, the partner side
            (true partner and every decoy partner chain) reads masked embeddings. Same-stratum swapped partner (all / hetero /
            homo / Ab/Ag role-matched; Ab/Ag split by the varying chain's role: antibody / antigen), surface-only, wrong-partner
            controls. Within-complex AUROC / top-1 and edge AUROC / AP / macro AP: chain code 1 masked, chain 0 full; the
            chain-0-masked version (chain 0 masked, chain 1 full) is reported beside it (within_c0 / edges_c0).
  both      v1-style symmetric mask (= score_masked.py pesto mode at that keep): reference; hotspot-only arm here only.
Metric definitions, seeds and decoy draws as score_masked.py / score_negatives.py. Re-typed arm: unmasked and both modes.
pKa: clean selection from the stored predictions (score_residue.py pka()).
GATES per run with the control beside, in unmasked and varying@keep: Ab/Ag-vs-Ab/Ag swapped, swapped partner all, wrong-partner
different protein, within AUROC, edge AUROC / AP, pKa clean.
  python score_masked2.py OUT_DIR STORE NAME=RUN [NAME=RUN ...] [--modes unmasked,varying,both] [--control-keeps 0.25,0.5]
  (the first NAME is the control)

min-aug-v1 (2026-10-31) adds mode altrow (--alt-store PATH, production_alt side store): the free branch with the ALT
(minimised) geometry for the validation complexes that have a usable record (production_alt.AltConfig prob 1; use_alt), and
the unmasked-mode metrics (swapped partner, surface-only, wrong partner, within AUROC / top-1, edge AUROC / AP / macro AP) on
that subset twice: every embedding crystal ("crystal" row) and every embedding alt ("alt" row), same-stratum partners redrawn
inside the subset (seed SEED) so true and decoy share the geometry source in both rows. Wrong-partner pools (hetero, Ab/Ag)
are restricted to the subset. Alt-vs-crystal gap per complex (subset): true-pair score (mean sigmoid over contact pairs,
target > 0.5) and surface-only score (mean over receptor surface x partner surface), alt minus crystal; reported as the
mean differences and the fraction of complexes where the alt score is below the crystal score (a model that treats the two
geometries as equivalent sits near 0.5; the rank arm should sit at or below it). Surfaces / contacts are the crystal
residue-table definitions in both rows. Other modes unchanged; with only altrow the masked passes are skipped.

anm-aug-v1 (2026-10-31): --alt-store takes ONE production_alt entry "store[:keysuffix]" (records <cid> or <cid>::<suffix>),
so the same mode scores each geometry (minimised store-min-v1, ANM 1 A views :a1, 2 A views :a2) in its own OUT_DIR. The gap
block adds the logit-level true-pair gap (mean over contact pairs of the edge logit, alt minus crystal, per complex; mean
and fraction of complexes with alt < crystal) beside the sigmoid-score gap.

drsa-within-v1 (2026-11-01): within_complex adds a graded ranking of the same anchors / candidate sets as within AUROC / top-1
(anchor = interface residue with a contact and a non-contact; candidates = every interface residue of the other chain; the
distance metrics are unchanged): per anchor, the free-branch pair logits vs the dRSA-weighted target g(a, c) = soft(d_ac) *
min(dRSA_a, dRSA_c) for contacts (residue-rsa-v1 pairs <= 10 A), 0 otherwise, soft(d) = min(1, exp(-(d - 4) / 3)), dRSA =
clip(RSA_free - RSA_bound, 0, 1) (NaN -> 0). ndcg_drsa = NDCG over the full candidate list (gain g, discount 1 / log2(rank + 1),
anchors with sum g = 0 excluded, n_ndcg); spearman_drsa = Spearman (average ranks) of logits vs g (anchors with constant g
excluded); ndcg_distance = the same NDCG with gain soft(d_ac) for contacts (reference). Means over anchors
(production_residue.ranking_metrics).
"""
import os

os.environ.setdefault("GQT_PARAMETER_SCALING", "1")
import argparse
import csv
import json
import re
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np


def anchor_metrics(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    auc = float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
    return auc, float(labels[int(np.argmax(scores))])


def compose(z_full, z_masked, chain, c):
    """Embeddings with chain code c read from the masked pass and every other chain from the full pass (chain-isolated free branch)."""
    return np.where((np.asarray(chain) == c)[:, None], z_masked, z_full)


if __name__ != "__main__": raise ImportError("score_masked2.py is a script")
_ap = argparse.ArgumentParser(prog="score_masked2.py"); _ap.add_argument("out"); _ap.add_argument("store"); _ap.add_argument("runs", nargs="+")
_ap.add_argument("--modes", default="unmasked,varying,both"); _ap.add_argument("--control-keeps", default=None)
_ap.add_argument("--alt-store", default=None, help="min-aug-v1: alt side store for mode altrow; anm-aug-v1: one store[:keysuffix] entry")
ARGS = _ap.parse_args()
OUT = Path(ARGS.out); OUT.mkdir(parents=True, exist_ok=True); MODES = [m for m in ARGS.modes.split(",") if m]
if set(MODES) - {"unmasked", "varying", "both", "altrow"}: raise ValueError(MODES)
if ("altrow" in MODES) != (ARGS.alt_store is not None): raise ValueError("mode altrow and --alt-store go together")
RUNS = dict(a.split("=", 1) for a in ARGS.runs)
sys.argv = [sys.argv[0], tempfile.mkdtemp(prefix="score_decoy_import_")]  # score_decoy reads OUT / RUNS at import: zero runs
import score_decoy as sd  # noqa: E402  (builds VAL / LAYOUT / CROSS; imports cpu_attention)
import jax  # noqa: E402

from pkatrain.gqt_paired_pinder import _metrics  # noqa: E402
from pkatrain.production_gate import DisulfideTable, bonded_queries  # noqa: E402
from pkatrain.production_graphs import ProductionStore  # noqa: E402
from pkatrain.production_residue import MASK_ORACLES, HotspotStore, hotspot_keep, interface_mask, mask_nodes_np  # noqa: E402
from pkatrain.production_residue import drsa, ranking_metrics, soft  # noqa: E402  (drsa-within-v1)

SEED = 20261031; PARTNERS = 5; CONTROLS = 5; STRATA = ("hetero", "homo", "Ab/Ag"); EVAL_SEED = 20261030; ORACLE = "pesto"
STORE = HotspotStore(ARGS.store)
META = {r["id"]: r for r in csv.DictReader(open(sd.ROOT / "training/pool-v4/pinder-val.tsv"), delimiter="\t")}
assert all(cid in META for cid in sd.VAL), "validation ids missing from pinder-val.tsv"
STRATUM = {cid: META[cid]["stratum"] for cid in sd.VAL}
CHAIN = {cid: np.asarray(sd.LAYOUT[cid]["r"]["chain"]) for cid in sd.VAL}
assert all(set(np.unique(CHAIN[c]).tolist()) <= {0, 1} for c in sd.VAL), "two-chain complexes expected"


# ---------------------------------------------------------------- helpers (score_masked.py)
def chain_name(label):
    base, n = re.match(r"^(.*?)(\d+)$", label).groups(); return base if int(n) == 1 else f"{base}-{n}"


def chain_uniprot(cid):
    sides = [(chain_name(s.split("__")[1].rsplit("_", 1)[0]), s.rsplit("_", 1)[1]) for s in cid.split("--")]
    return [u for _, u in sorted(sides)]


UNIPROT = {cid: chain_uniprot(cid) for cid in sd.VAL}
ROLE = {cid: (UNIPROT[cid].index("UNDEFINED") if STRATUM[cid] == "Ab/Ag" and UNIPROT[cid].count("UNDEFINED") == 1 else None) for cid in sd.VAL}


def same_stratum_partners(i, strata, uniprot, rng, k):
    u = uniprot[i][1]
    cands = [j for j in range(len(strata)) if j != i and strata[j] == strata[i] and (u == "UNDEFINED" or u not in uniprot[j])]
    return [int(x) for x in rng.choice(cands, min(k, len(cands)), replace=False)] if cands else []


def partner_code(i_cid, j_cid):
    ri, rj = ROLE[i_cid], ROLE[j_cid]
    if ri is None or rj is None: return 1, False
    return (rj if ri == 1 else 1 - rj), True


def side_patch(cid, code):
    a, b = sd.LAYOUT[cid]["contact"]; return np.unique(b if code == 1 else a)


def side_surface(cid, code):
    L = sd.LAYOUT[cid]; return L["par"] if code == 1 else L["rec"]


def hot_side(kept, cid, code):
    s = side_surface(cid, code); return s[kept[cid][s]]


def summary(win, margin=None):
    w = np.asarray(win, float); n = len(w)
    out = {"n": n, "accuracy": float(w.mean()) if n else None, "se": float(np.sqrt(w.mean() * (1 - w.mean()) / n)) if n else None}
    if margin is not None: out["margin"] = float(np.mean(margin)) if n else None
    return out


def by_stratum(records, key="win"):
    out = {"all": summary([r[key] for r in records], [r["margin"] for r in records]),
           **{s: summary([r[key] for r in records if r["si"] == s], [r["margin"] for r in records if r["si"] == s]) for s in STRATA}}
    for role in ("antibody", "antigen"):  # Ab/Ag split by the role of the chain that varies (cid's chain 1)
        sel = [r for r in records if r.get("varying_role") == role]; out[f"Ab/Ag varying {role}"] = summary([r[key] for r in sel], [r["margin"] for r in sel])
    return out


def eval_masks(keep):
    """{cid: unmasked (nodes,) bool} for BOTH chains: score_masked.py's pesto eval mask at keep."""
    code = MASK_ORACLES.index(ORACLE); kept = {}; tot = np.zeros(3)
    for cid in sd.VAL:
        k, st = hotspot_keep(sd.LAYOUT[cid]["r"], STORE.record(cid), ORACLE, keep, 0.0, np.random.default_rng([EVAL_SEED, code, zlib.crc32(cid.encode())]))
        kept[cid] = k; tot += st
    return kept, {"kept": int(tot[0]), "surface": int(tot[1]), "kept_fraction": float(tot[0] / max(tot[1], 1)), "fallbacks_random": int(tot[2]), "keep": keep}


_rng = np.random.default_rng(SEED); _strata = [STRATUM[c] for c in sd.VAL]; _uni = [UNIPROT[c] for c in sd.VAL]
SAME = [same_stratum_partners(i, _strata, _uni, _rng, PARTNERS) for i in range(len(sd.VAL))]

store = ProductionStore(Path(sd.manifest["store"])); ss = DisulfideTable(sd.ROOT, "pinder"); BONDED = {}
for cid in sd.VAL:
    for i, f in enumerate(bonded_queries(store.raw(cid), ss.flags(cid)[0])): BONDED[(cid, i)] = bool(f)


def pka(run):
    sel = sd.read(sd.run_dir(sd.ROOT, run) / "selection.json"); e = sel["selected_epoch"]; v = sel["validation"]; rows = []
    for r in csv.DictReader((sd.run_dir(sd.ROOT, run) / f"predictions-pinder-epoch-{e:03d}.csv").open()):
        rows.append({"state_error": float(r["state_error"]), "paired_error": float(r["paired_error"]), "interface": r["interface"] == "True",
                     "distance": float(r["distance"]), "bonded": BONDED[(r["complex_id"], int(r["site"]))]})
    a = _metrics(rows); c = _metrics([r for r in rows if not r["bonded"]])
    return {"epoch": e, "selection": v["selection"], "selection_no_bonded": c["state_mae"] + c["interface_paired_mae"], "state": a["state_mae"],
            "interface": a["interface_paired_mae"], "pkpdb": v["pkpdb"]["site_mae"], "benchmark": v["benchmark"]["overall"]["mae"]}


# ---------------------------------------------------------------- embeddings
def embed(model, kept, retyped_arm=True):
    """Free-branch normalised residue tokens Z under the mask `kept` (None = unmasked); with retyped_arm the re-typed arm
    (score_decoy rule on the native nodes, then the same mask) and the true contact scores."""
    from pkanet.model import norm
    params, embeddings = model; e = jax.tree.map(np.asarray, params)["aux_redge"]

    def one(p, g):
        tok = embeddings(p, g)[0]; n = g["nodes"].shape[0]; return norm(p["aux_redge"]["norm"], tok[:n])
    fn = jax.jit(jax.vmap(one, in_axes=(None, 0)))
    src = sd.PinderSource(sd._with_policy(sd.manifest), config=sd.LoaderConfig(), sidechains=sd.BackboneTable(sd.ROOT, "pinder"))
    Z = {}; arm1 = []; skipped = 0
    for ids, batch in sd.Prefetcher(src, src.policy.plans(sd.select(sd.manifest, "val"), np.random.default_rng(0)), sd.LoaderConfig()):
        free = jax.tree.map(lambda v: np.asarray(v[:, 1]), batch[0]); native = free["nodes"]; nodes = native.copy(); retyped = native.copy(); valid = {}
        for slot, cid in enumerate(ids):
            L = sd.LAYOUT[cid]; m = len(L["r"]["rsa"])
            if retyped_arm:
                aa = sd.residue_types(native[slot, :m])
                idx, new, mask, ok = sd.retype(L["r"], aa, np.random.default_rng([sd.RETYPE_SALT, zlib.crc32(cid.encode())])); valid[cid] = bool(ok)
                if ok: retyped[slot, idx[mask > 0], :20] = np.eye(20, dtype=retyped.dtype)[new[mask > 0]]
            if kept is not None:
                nodes[slot] = mask_nodes_np(nodes[slot], kept[cid])
                if retyped_arm: retyped[slot] = mask_nodes_np(retyped[slot], kept[cid])
        z_true = np.asarray(fn(params, {**free, "nodes": nodes})); z_decoy = np.asarray(fn(params, {**free, "nodes": retyped})) if retyped_arm else None
        for slot, cid in enumerate(ids):
            L = sd.LAYOUT[cid]; m = len(L["r"]["rsa"]); Z[cid] = z_true[slot, :m].astype(np.float32); a, b = L["contact"]
            if not retyped_arm or not len(a): continue
            t = float(sd.sigmoid(sd.logits(e, Z[cid][a], Z[cid][b])).mean())
            if not valid[cid]: skipped += 1; continue
            zd = z_decoy[slot, :m]; arm1.append(sd.win(t, float(sd.sigmoid(sd.logits(e, zd[a], zd[b])).mean())))
    src.close()
    return e, Z, {**sd.accuracy(arm1), "skipped_by_rule": skipped} if retyped_arm else None


def within_edges(e, ZE, val=None):
    """(e) within-complex anchor AUROC / top-1 and (f) edge AUROC / AP / macro AP on the embeddings ZE (score_masked.py);
    val: complex subset (default every validation complex)."""
    anchors = []; ys, ss_, macro = [], [], []
    for cid in (sd.VAL if val is None else val):
        L = sd.LAYOUT[cid]; z = ZE[cid]; a, b, label = L["all"]
        if len(a):
            s = sd.logits(e, z[a], z[b]); ys.append(label); ss_.append(s)
            if label.any() and (~label).any(): macro.append(sd.binary_metrics(label, s)["average_precision"])
        r = L["r"]; iface = interface_mask(r); m = len(r["rsa"]); con = np.zeros((m, m), bool); con[r["i"], r["j"]] = True; con |= con.T
        dm = np.full((m, m), np.inf, np.float32); dm[r["i"], r["j"]] = r["d"]; dm[r["j"], r["i"]] = r["d"]; dr = drsa(r)
        for c in np.unique(r["chain"][iface]):
            rows_ = np.flatnonzero(iface & (r["chain"] == c)); cols = np.flatnonzero(iface & (r["chain"] != c))
            if not len(rows_) or not len(cols): continue
            lab = con[np.ix_(rows_, cols)]; use = lab.any(1) & (~lab).any(1)
            if not use.any(): continue
            rr = rows_[use]; sc = sd.logits(e, np.repeat(z[rr], len(cols), axis=0), np.tile(z[cols], (len(rr), 1))).reshape(len(rr), len(cols))
            gd = np.where(lab[use], soft(dm[np.ix_(rr, cols)]), 0.0); gr = gd * np.minimum(dr[rr][:, None], dr[cols][None, :])
            for k in range(len(rr)):
                auc, top = anchor_metrics(sc[k], lab[use][k]); nd, rho = ranking_metrics(sc[k], gr[k]); ndd, _ = ranking_metrics(sc[k], gd[k])
                anchors.append({"si": STRATUM[cid], "auroc": auc, "top1": top, "cid": cid, "chain": int(c), "ndcg": nd, "rho": rho, "ndcg_d": ndd})
    y, s = np.concatenate(ys), np.concatenate(ss_); bm = sd.binary_metrics(y, s)

    def agg(sel):
        x = [a_ for a_ in anchors if sel(a_)]

        def mean(k):
            v = [a_[k] for a_ in x if a_[k] is not None]; return (float(np.mean(v)) if v else None), len(v)
        (nd, nn), (rho, nr), (ndd, _) = mean("ndcg"), mean("rho"), mean("ndcg_d")
        return {"auroc": float(np.mean([a_["auroc"] for a_ in x])) if x else None, "top1": float(np.mean([a_["top1"] for a_ in x])) if x else None, "n": len(x),
                "ndcg_drsa": nd, "spearman_drsa": rho, "ndcg_distance": ndd, "n_ndcg": nn, "n_spearman": nr}
    within = {"anchors": len(anchors), "complexes": len({x["cid"] for x in anchors}), "all": agg(lambda a_: True),
              **{k: agg(lambda a_, k=k: a_["si"] == k) for k in STRATA}, "anchor_chain0": agg(lambda a_: a_["chain"] == 0), "anchor_chain1": agg(lambda a_: a_["chain"] == 1)}
    return within, {"pairs": int(len(y)), "positive_rate": float(y.mean()), "auroc": bm["auroc"], "ap": bm["average_precision"], "macro_ap": float(np.mean(macro))}


def comparisons(e, ZR, ZP, kept_hot=None, ZH=None, val=None, same=None):
    """(b) swapped partner, (c) surface-only, (d) wrong-partner controls with the fixed side read from ZR and every partner
    side (true and decoy) from ZP; (h) hotspot-only on ZH with kept_hot (both mode). val / same (min-aug-v1 altrow): complex
    subset and its same-stratum partner indices into val (default every validation complex and SAME)."""
    V = sd.VAL if val is None else val; SAME_ = SAME if same is None else same
    swap, surf, ctrl = [], [], []; matched = {"swap": 0, "surface": 0}; hswap, hctrl, th = [], [], {}; hot = kept_hot is not None
    for i, cid in enumerate(V):
        L = sd.LAYOUT[cid]; zr, zp = ZR[cid], ZP[cid]; ca, cb = L["contact"]
        vrole = None if ROLE[cid] is None else ("antibody" if ROLE[cid] == 1 else "antigen")
        if len(ca) and SAME_[i]:  # (b)
            tc = float(sd.sigmoid(sd.logits(e, zr[ca], zp[cb])).mean()); rng = np.random.default_rng([SEED, i])
            for j in SAME_[i]:
                other = V[j]; code, rm = partner_code(cid, other); patch = side_patch(other, code)
                if not len(patch): continue
                bp = rng.choice(patch, len(ca)); d = float(sd.sigmoid(sd.logits(e, zr[ca], ZP[other][bp])).mean())
                swap.append({"si": STRATUM[cid], "win": sd.win(tc, d), "margin": tc - d, "varying_role": vrole}); matched["swap"] += rm
        if hot:
            d0, _ = sd.mean_cross(e, ZH[cid][hot_side(kept_hot, cid, 0)], ZH[cid][hot_side(kept_hot, cid, 1)])
            if d0 is not None:
                th[cid] = d0
                for j in SAME_[i]:
                    other = V[j]; code, _ = partner_code(cid, other)
                    d, _ = sd.mean_cross(e, ZH[cid][hot_side(kept_hot, cid, 0)], ZH[other][hot_side(kept_hot, other, code)])
                    if d is not None: hswap.append({"si": STRATUM[cid], "win": sd.win(th[cid], d), "margin": th[cid] - d})
        if len(L["all"][0]) and SAME_[i]:  # (c)
            t3 = sd.mean_cross(e, zr[L["rec"]], zp[L["par"]])[0]; other = V[SAME_[i][0]]; code, rm = partner_code(cid, other)
            d, _ = sd.mean_cross(e, zr[L["rec"]], ZP[other][side_surface(other, code)])
            if t3 is not None and d is not None: surf.append({"si": STRATUM[cid], "win": sd.win(t3, d), "margin": t3 - d, "varying_role": vrole}); matched["surface"] += rm
    hetero = [c for c in V if STRATUM[c] == "hetero"]; rng = np.random.default_rng(SEED + 1)

    def hot_control(kind, cid, A_code, k, B_code):
        if not hot: return
        A = ZH[cid][hot_side(kept_hot, cid, A_code)]; t = sd.mean_cross(e, A, ZH[cid][hot_side(kept_hot, cid, 1 - A_code)])[0]
        d = sd.mean_cross(e, A, ZH[k][hot_side(kept_hot, k, B_code)])[0]
        if t is not None and d is not None: hctrl.append({"kind": kind, "win": sd.win(t, d), "margin": t - d})
    for cid in hetero:  # (d) score_masked.py order and draws
        if not len(sd.LAYOUT[cid]["all"][0]): continue
        A = ZR[cid][sd.LAYOUT[cid]["rec"]]; t = sd.mean_cross(e, A, ZP[cid][sd.LAYOUT[cid]["par"]])[0]; ub = UNIPROT[cid][1]
        unrelated = [k for k in hetero if k != cid and UNIPROT[k][1] != ub]
        for k in rng.choice(unrelated, min(CONTROLS, len(unrelated)), replace=False):
            d = sd.mean_cross(e, A, ZP[k][side_surface(k, 1)])[0]
            if d is not None: ctrl.append({"kind": "different_protein", "win": sd.win(t, d), "margin": t - d})
            hot_control("different_protein", cid, 0, k, 1)
        if ub == "UNDEFINED": continue
        same = [(k, c) for k in hetero if k != cid for c in (0, 1) if UNIPROT[k][c] == ub]
        for n in (rng.choice(len(same), min(CONTROLS, len(same)), replace=False) if same else []):
            k, c = same[n]; B = ZP[k][side_surface(k, c)]
            if not len(B): continue
            d = sd.mean_cross(e, A, B)[0]; ctrl.append({"kind": "same_uniprot", "win": sd.win(t, d), "margin": t - d})
            hot_control("same_uniprot", cid, 0, k, c)
    abag = [c for c in V if ROLE[c] is not None]
    for cid in abag:  # antigen fixed (ZR), antibody swapped (ZP)
        ab = ROLE[cid]; ag = 1 - ab; A = ZR[cid][side_surface(cid, ag)]; B = ZP[cid][side_surface(cid, ab)]
        if not len(A) or not len(B): continue
        t = sd.mean_cross(e, A, B)[0]; others = [k for k in abag if k != cid and UNIPROT[k][1 - ROLE[k]] != UNIPROT[cid][ag]]
        for k in rng.choice(others, min(CONTROLS, len(others)), replace=False):
            Bk = ZP[k][side_surface(k, ROLE[k])]
            if not len(Bk): continue
            d = sd.mean_cross(e, A, Bk)[0]; ctrl.append({"kind": "abag_wrong_antibody", "win": sd.win(t, d), "margin": t - d})
            hot_control("abag_wrong_antibody", cid, ag, k, ROLE[k])
    kinds = ("different_protein", "same_uniprot", "abag_wrong_antibody")
    return {"swap_same_stratum": {**by_stratum(swap), "role_matched_comparisons": matched["swap"]},
            "surface_same_stratum": {**by_stratum(surf), "role_matched_comparisons": matched["surface"]},
            "wrong_partner": {k: summary([r["win"] for r in ctrl if r["kind"] == k], [r["margin"] for r in ctrl if r["kind"] == k]) for k in kinds},
            "hotspot_only": None if not hot else {
                "complexes_scored": len(th), "mean_true_score": float(np.mean(list(th.values()))) if th else None, "swap_same_stratum": by_stratum(hswap),
                "wrong_partner": {k: summary([r["win"] for r in hctrl if r["kind"] == k], [r["margin"] for r in hctrl if r["kind"] == k]) for k in kinds}}}


def embed_alt(model, store):
    """(Z with the alt free geometry where a record is usable, set of those complexes, loader counts): unmasked free-branch
    normalised residue tokens, production_alt.AltConfig(store, 1.0) on the validation source."""
    from pkanet.model import norm
    from pkatrain.production_alt import AltConfig
    params, embeddings = model

    def one(p, g):
        tok = embeddings(p, g)[0]; n = g["nodes"].shape[0]; return norm(p["aux_redge"]["norm"], tok[:n])
    fn = jax.jit(jax.vmap(one, in_axes=(None, 0)))
    src = sd.PinderSource(sd._with_policy(sd.manifest), config=sd.LoaderConfig(), sidechains=sd.BackboneTable(sd.ROOT, "pinder")); src.alt = AltConfig(store, 1.0)
    Z = {}; used = set()
    for ids, batch in sd.Prefetcher(src, src.policy.plans(sd.select(sd.manifest, "val"), np.random.default_rng(0)), sd.LoaderConfig()):
        free = jax.tree.map(lambda v: np.asarray(v[:, 1]), {k: v for k, v in batch[0].items() if k not in ("use_alt", "alt_source")}); z = np.asarray(fn(params, free))
        for slot, cid in enumerate(ids):
            m = len(sd.LAYOUT[cid]["r"]["rsa"]); Z[cid] = z[slot, :m].astype(np.float32)
            if batch[0]["use_alt"][slot, 1] > 0.5: used.add(cid)
    counts = dict(src.alt.counts); src.close(); return Z, used, counts


def alt_gap(e, ZX, ZA, val):
    """Per complex (val): true-pair score (mean sigmoid over contact pairs) and surface-only score, alt minus crystal."""
    true_d, surf_d, logit_d = [], [], []
    for cid in val:
        L = sd.LAYOUT[cid]; ca, cb = L["contact"]
        if len(ca):
            la, lx = np.asarray(sd.logits(e, ZA[cid][ca], ZA[cid][cb])), np.asarray(sd.logits(e, ZX[cid][ca], ZX[cid][cb]))
            true_d.append(float(sd.sigmoid(la).mean()) - float(sd.sigmoid(lx).mean())); logit_d.append(float(la.mean() - lx.mean()))
        sa = sd.mean_cross(e, ZA[cid][L["rec"]], ZA[cid][L["par"]])[0]; sx = sd.mean_cross(e, ZX[cid][L["rec"]], ZX[cid][L["par"]])[0]
        if sa is not None and sx is not None: surf_d.append(sa - sx)
    t, u, lg = np.asarray(true_d), np.asarray(surf_d), np.asarray(logit_d)
    return {"true_logit_alt_minus_xtal_mean": float(lg.mean()) if len(lg) else None, "frac_true_logit_alt_below_xtal": float(np.mean(lg < 0)) if len(lg) else None,"true_alt_minus_xtal_mean": float(t.mean()) if len(t) else None, "frac_true_alt_below_xtal": float(np.mean(t < 0)) if len(t) else None,
            "true_complexes": int(len(t)), "surface_alt_minus_xtal_mean": float(u.mean()) if len(u) else None,
            "frac_surface_alt_below_xtal": float(np.mean(u < 0)) if len(u) else None, "surface_complexes": int(len(u))}


def altrow(e, ZF, model, store):
    began = time.time(); ZA, used, counts = embed_alt(model, store); V = [c for c in sd.VAL if c in used]
    rng = np.random.default_rng(SEED); st = [STRATUM[c] for c in V]; un = [UNIPROT[c] for c in V]
    SS = [same_stratum_partners(i, st, un, rng, PARTNERS) for i in range(len(V))]
    rows = {}
    for row, Z in (("crystal", ZF), ("alt", ZA)):
        w, ed = within_edges(e, Z, V); rows[row] = {**comparisons(e, Z, Z, val=V, same=SS), "within_complex": w, "edges_10A_free": ed}
    return {"complexes": len(V), "strata": {s: st.count(s) for s in STRATA}, "loader_counts": counts, **rows, "gap": alt_gap(e, ZF, ZA, V),
            "seconds": round(time.time() - began, 1)}


def keep_of(run):
    hm = sd.read(sd.run_dir(sd.ROOT, run) / "protocol.json").get("hotspot_mask"); return float(hm["keep"]) if hm else None


names = list(RUNS); CTRL = names[0]
ALT_STORE = None
if ARGS.alt_store is not None:
    from pkatrain.production_alt import open_sources
    ALT_STORE = open_sources(ARGS.alt_store)
    if len(ALT_STORE) != 1: raise ValueError("--alt-store takes one store[:keysuffix] entry (score each geometry separately)")
other_keeps = sorted({keep_of(RUNS[n]) for n in names[1:]} - {None}) or [0.25]
CONTROL_KEEPS = [float(x) for x in ARGS.control_keeps.split(",")] if ARGS.control_keeps else other_keeps
MASKS = {}
report = {"definition": __doc__, "seed": SEED, "eval_seed": EVAL_SEED, "oracle": ORACLE, "validation": sd.SHAPES, "modes": MODES,
          "store": {"path": str(STORE.path), "verification_sha256": STORE.verification_sha256, "records": len(STORE),
                    "validation_records": sum(STORE.record(c) is not None for c in sd.VAL)},
          "strata": {s: _strata.count(s) for s in STRATA}, "ab_ag_with_role": sum(ROLE[c] is not None for c in sd.VAL),
          "ab_ag_antibody_is_chain1": sum(ROLE[c] == 1 for c in sd.VAL), "masks": {}, "runs": {}}


def save(name, key, r):
    d = OUT / name.replace(" ", "_").replace("/", "_"); d.mkdir(parents=True, exist_ok=True); (d / f"report-{key}.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(name, key, json.dumps({k: r.get(k) for k in ("swap_same_stratum", "wrong_partner", "within_complex", "edges_10A_free", "within_c0", "edges_c0", "isolation", "seconds")},
                                default=str), flush=True)


for name in names:
    run = RUNS[name]; began = time.time(); model = sd.selected_model(sd.ROOT, run); model = (model[0], model[1][0])
    if "aux_redge" not in model[0]: raise ValueError(f"{run} has no residue edge head")
    protocol = sd.read(sd.run_dir(sd.ROOT, run) / "protocol.json"); hm = protocol.get("hotspot_mask")
    if hm is not None and hm["oracle"] != ORACLE: raise AssertionError((name, "scored with the PeSTo oracle only", hm["oracle"]))
    keeps = CONTROL_KEEPS if name == CTRL or hm is None else [float(hm["keep"])]
    base = {"run": run, "width": protocol["architecture"]["width"], "selected_epoch": sd.read(sd.run_dir(sd.ROOT, run) / "selection.json")["selected_epoch"],
            "protocol": {k: protocol[k] for k in ("within_listwise", "hotspot_mask") if k in protocol}, "pka": pka(run)}
    report["runs"][name] = {**base, "keeps": keeps, "modes": {}}
    e, ZF, retF = embed(model, None, retyped_arm="unmasked" in MODES)
    if "unmasked" in MODES:
        t0 = time.time(); w, ed = within_edges(e, ZF)
        r = {"mode": "unmasked", "keep": None, "retyped": retF, **comparisons(e, ZF, ZF), "within_complex": w, "edges_10A_free": ed, "seconds": round(time.time() - t0, 1)}
        report["runs"][name]["modes"]["unmasked"] = r; save(name, "unmasked", r)
    if "altrow" in MODES:
        r = altrow(e, ZF, model, ALT_STORE); report["runs"][name]["altrow"] = r; save(name, "altrow", r)
    for keep in (keeps if {"both", "varying"} & set(MODES) else []):
        if keep not in MASKS: MASKS[keep] = eval_masks(keep); report["masks"][f"{ORACLE}@{keep}"] = MASKS[keep][1]
        kept, stats = MASKS[keep]
        _, ZB, retB = embed(model, kept, retyped_arm="both" in MODES)
        if "both" in MODES:
            t0 = time.time(); w, ed = within_edges(e, ZB)
            r = {"mode": "both", "keep": keep, "mask": stats, "retyped": retB, **comparisons(e, ZB, ZB, kept, ZB), "within_complex": w, "edges_10A_free": ed,
                 "seconds": round(time.time() - t0, 1)}
            report["runs"][name]["modes"][f"both@{keep}"] = r; save(name, f"both@{keep}", r)
        if "varying" in MODES:
            t0 = time.time(); kept1 = {c: kept[c] | (CHAIN[c] != 1) for c in sd.VAL}; _, ZC1, _ = embed(model, kept1, retyped_arm=False)
            iso = {"chain1_rows_vs_both_pass": float(max(np.abs(ZC1[c][CHAIN[c] == 1] - ZB[c][CHAIN[c] == 1]).max() for c in sd.VAL)),
                   "chain0_rows_vs_full_pass": float(max(np.abs(ZC1[c][CHAIN[c] == 0] - ZF[c][CHAIN[c] == 0]).max() for c in sd.VAL))}
            ZC0 = {c: compose(ZF[c], ZB[c], CHAIN[c], 0) for c in sd.VAL}
            w1, ed1 = within_edges(e, ZC1); w0, ed0 = within_edges(e, ZC0)
            r = {"mode": "varying", "keep": keep, "mask": stats, "retyped": None, **comparisons(e, ZF, ZB), "within_complex": w1, "edges_10A_free": ed1,
                 "within_c0": w0, "edges_c0": ed0, "isolation": iso, "seconds": round(time.time() - t0, 1)}
            report["runs"][name]["modes"][f"varying@{keep}"] = r; save(name, f"varying@{keep}", r)
    report["runs"][name]["seconds"] = round(time.time() - began, 1)


# ---------------------------------------------------------------- markdown
def f(x, d=4): return "-" if x is None else f"{x:.{d}f}"


def g(r, *path):
    for p in path:
        if r is None: return None
        r = r.get(p)
    return r


R = report["runs"]
L = ["# Asymmetric hotspot-masked scoring of the residue edge head (hotspot-mask-v2, PINDER validation, free branch)", "",
     f"{report['validation']}; strata {report['strata']}; Ab/Ag with role {report['ab_ag_with_role']} (antibody = chain 1 in {report['ab_ag_antibody_is_chain1']}); "
     f"oracle {ORACLE}; masks {json.dumps(report['masks'])}. Pairwise accuracy, ties 1/2, chance 0.5. varying = only the chain that differs between true and decoy "
     "is masked (within / edge: chain 1 masked); both = v1 symmetric mask. Definitions in report.json.", "",
     "| arm | mode | epoch | re-typed | swap all | swap hetero | swap homo | swap Ab/Ag | Ab/Ag varying antibody | Ab/Ag varying antigen | surface all | surface Ab/Ag | edge AUROC | edge AP | pKa clean |",
     "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for n in R:
    for mode, r in R[n]["modes"].items():
        sw, su = r["swap_same_stratum"], r["surface_same_stratum"]
        L.append(f"| {n} | {mode} | {R[n]['selected_epoch']} | {f(g(r, 'retyped', 'accuracy'))} | " + " | ".join(f(sw[k]['accuracy']) for k in ("all",) + STRATA)
                 + f" | {f(sw['Ab/Ag varying antibody']['accuracy'])} (n {sw['Ab/Ag varying antibody']['n']}) | {f(sw['Ab/Ag varying antigen']['accuracy'])} (n {sw['Ab/Ag varying antigen']['n']})"
                 + f" | {f(su['all']['accuracy'])} | {f(su['Ab/Ag']['accuracy'])} | {f(r['edges_10A_free']['auroc'])} | {f(r['edges_10A_free']['ap'])} | {f(R[n]['pka']['selection_no_bonded'])} |")
L += ["", "## Wrong-partner controls and within-complex ranking", "",
      "| arm | mode | different protein | same UniProt other complex | Ab/Ag wrong antibody | within AUROC | within top-1 | hetero | homo | Ab/Ag |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for n in R:
    for mode, r in R[n]["modes"].items():
        w, wc = r["wrong_partner"], r["within_complex"]
        L.append(f"| {n} | {mode} | {f(w['different_protein']['accuracy'])} (n {w['different_protein']['n']}) | {f(w['same_uniprot']['accuracy'])} (n {w['same_uniprot']['n']}) | "
                 f"{f(w['abag_wrong_antibody']['accuracy'])} (n {w['abag_wrong_antibody']['n']}) | {f(wc['all']['auroc'])} | {f(wc['all']['top1'])} | "
                 + " | ".join(f(wc[s]['auroc']) for s in STRATA) + " |")
L += ["", "## Within ranking vs the dRSA-weighted target (drsa-within-v1; same anchors / candidates as within AUROC)", "",
      "| arm | mode | anchors | NDCG dRSA (n) | Spearman dRSA (n) | NDCG distance | NDCG dRSA hetero | homo | Ab/Ag |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for n in R:
    for mode, r in R[n]["modes"].items():
        wc = r["within_complex"]
        L.append(f"| {n} | {mode} | {wc['anchors']} | {f(wc['all']['ndcg_drsa'])} ({wc['all']['n_ndcg']}) | {f(wc['all']['spearman_drsa'])} ({wc['all']['n_spearman']}) | "
                 f"{f(wc['all']['ndcg_distance'])} | " + " | ".join(f(wc[s_]['ndcg_drsa']) for s_ in STRATA) + " |")
L += ["", "## Chain-1-masked vs chain-0-masked (within AUROC, edge AUROC / AP; the other chain full)", "",
      "| arm | keep | within AUROC c1-masked | within AUROC c0-masked | edge AUROC c1-masked | edge AUROC c0-masked | edge AP c1-masked | edge AP c0-masked | isolation max abs diff (c1 rows, c0 rows) |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
for n in R:
    for mode, r in R[n]["modes"].items():
        if not mode.startswith("varying"): continue
        L.append(f"| {n} | {r['keep']} | {f(r['within_complex']['all']['auroc'])} | {f(r['within_c0']['all']['auroc'])} | {f(r['edges_10A_free']['auroc'])} | "
                 f"{f(r['edges_c0']['auroc'])} | {f(r['edges_10A_free']['ap'])} | {f(r['edges_c0']['ap'])} | {r['isolation']['chain1_rows_vs_both_pass']:.2e}, {r['isolation']['chain0_rows_vs_full_pass']:.2e} |")
L += ["", "## Hotspot-only score (both mode)", "", "| arm | mode | complexes | swap all | swap Ab/Ag | different protein | Ab/Ag wrong antibody |", "|---|---|---:|---:|---:|---:|---:|"]
for n in R:
    for mode, r in R[n]["modes"].items():
        h = r.get("hotspot_only")
        if not h: continue
        L.append(f"| {n} | {mode} | {h['complexes_scored']} | {f(h['swap_same_stratum']['all']['accuracy'])} | {f(h['swap_same_stratum']['Ab/Ag']['accuracy'])} | "
                 f"{f(h['wrong_partner']['different_protein']['accuracy'])} | {f(h['wrong_partner']['abag_wrong_antibody']['accuracy'])} |")


if "altrow" in MODES:
    L += ["", "## Alt (minimised) free geometry vs crystal on the same complexes (unmasked; partners redrawn inside the subset)", "",
          f"Alt store {ARGS.alt_store}. Gap = alt minus crystal per complex; frac below = share of complexes whose alt score is below the crystal score.", "",
          "| arm | row | complexes | swap all | swap Ab/Ag | surface all | different protein | Ab/Ag wrong antibody | within AUROC | edge AUROC | edge AP |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in R:
        a = R[n].get("altrow")
        if not a: continue
        for row in ("crystal", "alt"):
            r = a[row]
            L.append(f"| {n} | {row} | {a['complexes']} | {f(g(r, 'swap_same_stratum', 'all', 'accuracy'))} | {f(g(r, 'swap_same_stratum', 'Ab/Ag', 'accuracy'))} | "
                     f"{f(g(r, 'surface_same_stratum', 'all', 'accuracy'))} | {f(g(r, 'wrong_partner', 'different_protein', 'accuracy'))} | "
                     f"{f(g(r, 'wrong_partner', 'abag_wrong_antibody', 'accuracy'))} | {f(g(r, 'within_complex', 'all', 'auroc'))} | "
                     f"{f(g(r, 'edges_10A_free', 'auroc'))} | {f(g(r, 'edges_10A_free', 'ap'))} |")
    L += ["", "| arm | true-pair logit alt - crystal | frac true logit alt < crystal | true-pair alt - crystal | frac true alt < crystal | surface alt - crystal | frac surface alt < crystal | complexes (true, surface) |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for n in R:
        a = R[n].get("altrow")
        if not a: continue
        gp = a["gap"]
        L.append(f"| {n} | {f(gp.get('true_logit_alt_minus_xtal_mean'))} | {f(gp.get('frac_true_logit_alt_below_xtal'), 3)} | "
                 f"{f(gp['true_alt_minus_xtal_mean'])} | {f(gp['frac_true_alt_below_xtal'], 3)} | {f(gp['surface_alt_minus_xtal_mean'])} | "
                 f"{f(gp['frac_surface_alt_below_xtal'], 3)} | {gp['true_complexes']}, {gp['surface_complexes']} |")


def gate_row(n, mode):
    r = R[n]["modes"].get(mode)
    if r is None: return None
    return (f"| {n} | {mode} | {f(g(r, 'swap_same_stratum', 'Ab/Ag', 'accuracy'))} | {f(g(r, 'swap_same_stratum', 'all', 'accuracy'))} | "
            f"{f(g(r, 'wrong_partner', 'different_protein', 'accuracy'))} | {f(g(r, 'within_complex', 'all', 'auroc'))} | {f(g(r, 'edges_10A_free', 'auroc'))} | "
            f"{f(g(r, 'edges_10A_free', 'ap'))} | {f(R[n]['pka']['selection_no_bonded'])} |")


G = ["| arm | mode | Ab/Ag-vs-Ab/Ag swapped | swapped partner all | wrong-partner different protein | within AUROC | edge AUROC | edge AP | pKa clean |",
     "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for n in names[1:]:
    for keep in R[n]["keeps"]:
        for mode in ("unmasked", f"varying@{keep}"):
            for who in (n, CTRL):
                row = gate_row(who, mode)
                if row and row not in G: G.append(row)
L += ["", f"## GATES (control = {CTRL}; unmasked and varying at each run's keep, control beside)", "", *G]
(OUT / "report.md").write_text("\n".join(L) + "\n"); (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
print("\n".join(L), flush=True)
