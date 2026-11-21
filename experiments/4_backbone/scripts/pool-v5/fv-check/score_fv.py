"""abag-fv-check-v1 (2026-11-20): does scoring a Fab as a whole Fv (POISK's unit) instead of one antibody chain (the
PINDER training unit) change the Ab/Ag gates?

Inputs (cases.json, pdb/ from assemble.py; PINDER val Ab/Ag, prepared pinder-pkai-v1 entries):
  fv_<id>.pdb  chain A = the val record's antibody chain, chain B = the cognate partner chain from the partner-antigen
               PINDER entry of the same PDB (antigen frames identical, RMSD 0.000); ag_<id>.pdb chain C = the antigen alone.
Views of the antibody (pkanet.allres.AllResiduePredictor, the POISK _embed path):
  single = Fv free state (A-B edges masked: chain A is embedded exactly as alone, the training free branch)
  fv     = Fv bound state (A-B edges kept: POISK's whole-Fv input). The antigen is always embedded alone.
Contacts: min heavy-atom distance < 4 + 3 ln 2 A (score_decoy target > 0.5). own = chain A contacts; full = A + B contacts.
Gates (win = true > decoy, half credit ties; pooled over comparisons, plus per-case means for paired differences):
  vary_antigen  antibody fixed: true contact pairs vs the same antibody contact residues x random residues (same count, seed)
                of another antigen's epitope (every other val Ab/Ag antigen with a different UniProt)    [masked2 (b)]
  vary_antibody antigen fixed: true contact pairs vs the same antigen contact residues x random residues of another Fv
                case's paratope (own: its chain A paratope; full: both chains)                        [masked2 (b)]
  blind_*       POISK-like all-pairs mean over every antibody residue x every antigen residue, same decoys
  edge_auroc    per complex, all antibody x antigen pairs vs contact labels, mean over cases
Arms: single/own (training-like), fv/own (context only), fv/full (POISK-like unit).
"""
import json, sys, time, csv
from pathlib import Path
import numpy as np

SEED = 20261120
DATA, MODELS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
CASES = [c for c in json.loads((DATA / "cases.json").read_text())]
FV = [c for c in CASES if "fv" in c and 3 < c["vh_vl_min_ca"] < 8 and c["antigen_rmsd"] < 2]
UNIPROT = {c["id"]: [p.rsplit("_", 1)[1] for p in c["id"].split("--") if not p.endswith("UNDEFINED")][0] for c in CASES}


def auroc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float); pos, neg = p[y], p[~y]
    if not len(pos) or not len(neg): return None
    order = np.argsort(np.concatenate((pos, neg)), kind="mergesort"); ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allp = np.concatenate((pos, neg))
    for v in np.unique(allp):  # average ties
        m = allp == v; ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def win(t, d): return 1.0 if t > d else 0.5 if t == d else 0.0


def main():
    import jax, jax.numpy as jnp
    from pkanet.allres import AllResiduePredictor, pair_logits
    from pkanet.features import prepare_file
    feat = Path("/tmp") / f"fvfeat-{int(time.time())}"; feat.mkdir(parents=True, exist_ok=True)
    names = sorted({c["antigen"] for c in CASES} | {c["fv"] for c in FV})
    prepared = {}
    for n in names:
        try: prepared[n] = prepare_file(DATA / "pdb" / f"{n}.pdb", feat, gap_policy="cap")
        except Exception as e: print("prepare failed", n, repr(e)[:200], flush=True)
    report = {"cases_fv": len(FV), "antigens": sum(1 for c in CASES if c["antigen"] in prepared), "seed": SEED, "models": {}}
    rows = []
    for model in sorted(p for p in MODELS.iterdir() if (p / "model.json").exists()):
        t0 = time.time(); pred = AllResiduePredictor(str(model), device="gpu", backend="jax", batch_size=8)
        recs = [(n, prepared[n]) for n in names if n in prepared]
        Z = {}
        for index, o in pred.predict([r for _, r in recs], states=("bound", "free")):
            n, (_, meta) = recs[index]
            keys = [(k["chain"], int(k["number"])) for k in meta["residues"]]
            Z[n] = {"keys": {k: i for i, k in enumerate(keys)}, "bound": np.asarray(o["bound"]["edge_embedding"], np.float32),
                    "free": np.asarray(o["free"]["edge_embedding"], np.float32)}
        params = pred.params
        score = jax.jit(lambda za, zb: jax.nn.sigmoid(pair_logits(params, jnp.concatenate((za, zb), 0), jnp.arange(za.shape[0]), jnp.arange(zb.shape[0]) + za.shape[0])))

        def sig(za, zb):  # elementwise pairs (len(za) == len(zb)), padded to a power of two to bound recompiles
            m = len(za); cap = 1 << max(4, (m - 1).bit_length())
            pa = np.zeros((cap, za.shape[1]), np.float32); pb = np.zeros_like(pa); pa[:m] = za; pb[:m] = zb
            return np.asarray(score(pa, pb))[:m]

        def allpairs(za, zb):
            a, b = np.repeat(np.arange(len(za)), len(zb)), np.tile(np.arange(len(zb)), len(za))
            out = []
            for s in range(0, len(a), 1 << 16): out.append(sig(za[a[s:s + (1 << 16)]], zb[b[s:s + (1 << 16)]]))
            return np.concatenate(out).reshape(len(za), len(zb))

        # per-case views: antibody rows (own chain A / full A+B), antigen rows, contact index pairs in those rows
        V = {}
        for c in FV:
            if c["fv"] not in Z or c["antigen"] not in Z: continue
            f, g = Z[c["fv"]], Z[c["antigen"]]
            rowA = [f["keys"].get(("A", i + 1)) for i in range(c["n_own"])]; rowB = [f["keys"].get(("B", i + 1)) for i in range(c["n_partner"])]
            rowC = [g["keys"].get(("C", j + 1)) for j in range(c["n_ag"])]
            own = [(rowA[i], rowC[j]) for i, j in c["own_contacts"] if rowA[i] is not None and rowC[j] is not None]
            par = [(rowB[i], rowC[j]) for i, j in c["partner_contacts"] if rowB[i] is not None and rowC[j] is not None]
            a_rows = np.array([r for r in rowA if r is not None]); ab_rows = np.array([r for r in rowA + rowB if r is not None])
            V[c["id"]] = dict(c=c, own=own, full=own + par, a_rows=a_rows, ab_rows=ab_rows, zag=g["bound"],
                              z={"single": f["free"], "fv": f["bound"]})
        AG = {c["id"]: c for c in CASES if c["antigen"] in Z}
        # antigen epitopes (own-chain contacts of every val Ab/Ag record, rows in the antigen-alone embedding)
        EPI = {}
        for c in CASES:
            if c["antigen"] not in Z: continue
            g = Z[c["antigen"]]; rows_ = sorted({g["keys"].get(("C", j + 1)) for _, j in c["own_contacts"]} - {None})
            if rows_: EPI[c["id"]] = np.array(rows_)
        res = {}
        for view, cset in (("single", "own"), ("fv", "own"), ("fv", "full")):
            arm = f"{view}/{cset}"; per = {k: [] for k in ("vary_antigen", "vary_antibody", "blind_vary_antigen", "blind_vary_antibody", "edge_auroc")}
            for cid, v in V.items():
                rng = np.random.default_rng([SEED, len(cid), sum(map(ord, cid))])
                za = v["z"][view]; pairs = v[cset]; rows_ab = v["a_rows"] if cset == "own" else v["ab_rows"]
                if not pairs: continue
                ia = np.array([p[0] for p in pairs]); ic = np.array([p[1] for p in pairs])
                t = float(sig(za[ia], v["zag"][ic]).mean())
                full_true = allpairs(za[rows_ab], v["zag"])
                tb = float(full_true.mean())
                lab = np.zeros(full_true.shape, bool); rpos = {r: k for k, r in enumerate(rows_ab)}
                for a, cc in pairs: lab[rpos[a], cc] = True
                per["edge_auroc"].append(auroc(lab.ravel(), full_true.ravel()))
                wa, wb = [], []
                for oid, epi in EPI.items():  # vary antigen
                    if oid == cid or UNIPROT[oid] == UNIPROT[cid]: continue
                    zo = Z[AG[oid]["antigen"]]["bound"]; d = float(sig(za[ia], zo[rng.choice(epi, len(ia))]).mean())
                    wa.append(win(t, d)); wb.append(win(tb, float(allpairs(za[rows_ab], zo).mean())))
                per["vary_antigen"].append(np.mean(wa)); per["blind_vary_antigen"].append(np.mean(wb))
                wa, wb = [], []
                for oid, o in V.items():  # vary antibody (other Fv cases, same view / contact set)
                    if oid == cid or UNIPROT[oid] == UNIPROT[cid]: continue
                    zo = o["z"][view]; opar = np.array(sorted({p[0] for p in o[cset]})); orows = o["a_rows"] if cset == "own" else o["ab_rows"]
                    if not len(opar): continue
                    d = float(sig(zo[rng.choice(opar, len(ic))], v["zag"][ic]).mean())
                    wa.append(win(t, d)); wb.append(win(tb, float(allpairs(zo[orows], v["zag"]).mean())))
                per["vary_antibody"].append(np.mean(wa)); per["blind_vary_antibody"].append(np.mean(wb))
                rows.append({"model": model.name, "arm": arm, "id": cid, "kind": v["c"]["kind"], "vary_antigen": per["vary_antigen"][-1],
                             "vary_antibody": per["vary_antibody"][-1], "blind_vary_antigen": per["blind_vary_antigen"][-1],
                             "blind_vary_antibody": per["blind_vary_antibody"][-1], "edge_auroc": per["edge_auroc"][-1], "true_mean": t})
            res[arm] = {k: {"mean": float(np.nanmean([x for x in vals if x is not None])), "n": len(vals),
                            "se": float(np.nanstd([x for x in vals if x is not None]) / np.sqrt(max(len(vals), 1)))} for k, vals in per.items()}
        report["models"][model.name] = {"model_json": json.loads((model / "model.json").read_text()).get("run"), "arms": res,
                                         "seconds": round(time.time() - t0, 1), "fv_cases_scored": len(V)}
        print(model.name, json.dumps(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fv_check.json").write_text(json.dumps(report, indent=1))
    with open(OUT / "per_case.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
