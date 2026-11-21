"""quartet-rank-v2 (2026-11-19): chain-level quartet accuracy of the free edge head on the PINDER validation complexes.

Free branch, unmasked, selected checkpoint per selection.json, CPU jax (cpu_attention.py). Per validation complex the
chain sides of the training loss (production_residue.chain_*): surface = free RSA >= 0.1 (residue-rsa-v1, NaN = not
surface) on active free nodes; A / B = chain codes 0 / 1 for hetero / homo, A = antibody (the one UNDEFINED-UniProt chain,
score_masked2 ROLE) / B = antigen for Ab/Ag; Ab/Ag without exactly one UNDEFINED chain are not scored. Pooled score
S(A, B) = tau * logsumexp(s_ab / tau) - tau * log(k) (tau 1, k 32 = the training defaults) over all surface pairs, s = the
aux_redge logits. Quartets: within each class (pinder-val.tsv stratum), in validation order, every complex is paired with
the next PARTNERS complexes of its class (cyclic) that pass the training false-negative guard (different `group`, no shared
non-UNDEFINED UniProt); unordered pairs deduplicated. delta = S(A1,B1) + S(A2,B2) - S(A1,B2) - S(A2,B1); accuracy = mean
[delta > 0] (ties 1/2), softplus = mean softplus(1 - delta), margin = mean delta, per class and over all quartets. Length
shortcut check: Spearman of the native S(A, B) with log(n_A * n_B) (surface pair count), per class and over all complexes.

  python score_chain.py OUT_DIR NAME=RUN [...] [--limit N] [--partners 5]      -> OUT_DIR/report.json, OUT_DIR/report.md
"""
import os

os.environ.setdefault("GQT_PARAMETER_SCALING", "1")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CHUNK = 65536
PARTNERS = 5
TAU = 1.0
K = 32
CLASSES = ("hetero", "homo", "Ab/Ag")  # production_residue.STRATA codes 0, 1, 2


def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))  # jax.nn.gelu (approximate)


def logits(e, za, zb):
    """aux_redge pair logits for embedding rows za, zb (m, W)."""
    out = []
    for s in range(0, len(za), CHUNK):
        a, b = za[s:s + CHUNK], zb[s:s + CHUNK]; f = np.concatenate((a * b, np.abs(a - b)), axis=1)
        out.append((gelu(f @ e["hidden"]["w"] + e["hidden"]["b"]) @ e["out"]["w"] + e["out"]["b"])[:, 0])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def pooled(e, za, zb, tau=TAU, k=K):
    """S = tau * logsumexp(s / tau) - tau * log(k) over every row pair of za (nA, W) x zb (nB, W); None if a side is empty."""
    if not len(za) or not len(zb): return None
    a = np.repeat(np.arange(len(za)), len(zb)); b = np.tile(np.arange(len(zb)), len(za))
    s = logits(e, za[a], zb[b]).astype(np.float64) / tau; m = s.max()
    return float(tau * (m + np.log(np.exp(s - m).sum()) - np.log(k)))


def sides(r, active, role):
    """(A rows, B rows) of one complex: surface (free RSA >= 0.1, NaN not surface) active rows on chain code role / 1 - role."""
    from pkatrain.production_residue import SURFACE_RSA
    surf = (np.nan_to_num(r["rsa"][:, 1], nan=-1.0) >= SURFACE_RSA) & active; ch = np.asarray(r["chain"])
    return np.flatnonzero(surf & (ch == role)), np.flatnonzero(surf & (ch == 1 - role))


def val_pairs(ids, meta, partners=PARTNERS):
    """[(cid1, cid2, class code)]: per class in `ids` order, each complex with the next `partners` guard-passing complexes of its
    class (cyclic); unordered pairs once. Also the guard skip counts per class."""
    from pkatrain.production_residue import chain_guard
    out = []; seen = set(); skipped = {c: {"group": 0, "uniprot": 0} for c in range(3)}
    for code in range(3):
        members = [c for c in ids if c in meta and meta[c][0] == code]
        for i, cid in enumerate(members):
            got = 0
            for step in range(1, len(members)):
                other = members[(i + step) % len(members)]; why = chain_guard(meta[cid], meta[other])
                if why: skipped[code][why] += 1; continue
                key = tuple(sorted((cid, other)))
                if key not in seen: seen.add(key); out.append((key[0], key[1], code))
                got += 1
                if got >= partners: break
    return out, skipped


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3: return None
    rank = lambda v: np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort").astype(float)
    def avg(v):  # average ranks for ties
        r = rank(v); out = r.copy()
        for u in np.unique(v):
            sel = v == u
            if sel.sum() > 1: out[sel] = r[sel].mean()
        return out
    a, b = avg(x), avg(y)
    if a.std() == 0 or b.std() == 0: return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(e, Z, R, meta, pairs, tau=TAU, k=K):
    """Report metrics from embeddings Z {cid: (m, W)}, side rows R {cid: (A rows, B rows)} and quartet pairs."""
    native = {c: pooled(e, Z[c][R[c][0]], Z[c][R[c][1]], tau, k) for c in Z}
    rows = []
    for c1, c2, code in pairs:
        if c1 not in Z or c2 not in Z or native[c1] is None or native[c2] is None: continue
        x12 = pooled(e, Z[c1][R[c1][0]], Z[c2][R[c2][1]], tau, k); x21 = pooled(e, Z[c2][R[c2][0]], Z[c1][R[c1][1]], tau, k)
        rows.append((code, native[c1] + native[c2] - x12 - x21))
    def summary(sel):
        d = np.asarray([r[1] for r in sel], float)
        return {"n": int(len(d)), **({"accuracy": float(np.mean((d > 0) + 0.5 * (d == 0))), "softplus": float(np.mean(np.logaddexp(0.0, 1.0 - d))),
                                      "margin": float(np.mean(d))} if len(d) else {"accuracy": None, "softplus": None, "margin": None})}
    out = {"quartets": {"all": summary(rows), **{CLASSES[c]: summary([r for r in rows if r[0] == c]) for c in range(3)}}, "length_spearman": {}, "native_mean": {}}
    for label, codes in (("all", (0, 1, 2)), *((CLASSES[c], (c,)) for c in range(3))):
        cs = [c for c in Z if native[c] is not None and meta[c][0] in codes]
        out["length_spearman"][label] = spearman([native[c] for c in cs], [np.log(len(R[c][0]) * len(R[c][1])) for c in cs])
        out["native_mean"][label] = float(np.mean([native[c] for c in cs])) if cs else None
    out["complexes_scored"] = int(sum(native[c] is not None for c in Z))
    return out


def markdown(report):
    f = lambda x: "-" if x is None else f"{x:.4f}"
    L = ["# Chain-quartet accuracy of the free edge head (PINDER validation, free branch, unmasked; soft top-k pooled chain scores)", "",
         f"tau {report['tau']}, k {report['k']}, partners {report['partners']}; quartets per class {report['pairs_per_class']}", "",
         "| arm | run | all acc | hetero acc | homo acc | Ab/Ag acc | all softplus | all margin | length Spearman all | hetero | homo | Ab/Ag |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in report["runs"].items():
        q = r["metrics"]["quartets"]; s = r["metrics"]["length_spearman"]
        L.append(f"| {name} | {r['run']} | {f(q['all']['accuracy'])} | {f(q['hetero']['accuracy'])} | {f(q['homo']['accuracy'])} | {f(q['Ab/Ag']['accuracy'])} | "
                 f"{f(q['all']['softplus'])} | {f(q['all']['margin'])} | {f(s['all'])} | {f(s['hetero'])} | {f(s['homo'])} | {f(s['Ab/Ag'])} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(prog="score_chain.py"); ap.add_argument("out"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--partners", type=int, default=PARTNERS)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True); runs = dict(a.split("=", 1) for a in args.runs)
    import cpu_attention  # noqa: F401  (plain-JAX attention under JAX_PLATFORMS=cpu)
    import jax
    from pkanet.model import norm
    from pkatrain.loading import Prefetcher
    from pkatrain.production_graphs import output, read
    from pkatrain.production_inputs import BackboneTable
    from pkatrain.production_loading import LoaderConfig, PinderSource, select
    from pkatrain.production_residue import ResidueTable, chain_metadata
    from pkatrain.production_train import _with_policy, selected_model
    root = Path(os.environ["PKABENCH_RUNTIME"]); manifest = read(output(root, "pinder") / "manifest-v1.json"); table = ResidueTable(root)
    val = select(manifest, "val")[:args.limit] if args.limit else select(manifest, "val")
    val_tsv = root / "training/pool-v4/pinder-val.tsv"; meta = chain_metadata(val_tsv)
    ids = [r["id"] for r in val]; pairs, skipped = val_pairs(ids, meta, args.partners)
    report = {"definition": __doc__, "limit": args.limit, "validation_complexes": len(val), "tau": TAU, "k": K, "partners": args.partners,
              "val_tsv": str(val_tsv), "pairs_per_class": {CLASSES[c]: sum(p[2] == c for p in pairs) for c in range(3)},
              "guard_skips": {CLASSES[c]: v for c, v in skipped.items()}, "unscored_role": sum(1 for c in ids if c in meta and meta[c][0] < 0), "runs": {}}
    for name, run in runs.items():
        began = time.time(); params, fns = selected_model(root, run); embeddings = fns[0]; e = jax.tree.map(np.asarray, params)["aux_redge"]

        def one(p, g):
            tok = embeddings(p, g)[0]; n = g["nodes"].shape[0]; return norm(p["aux_redge"]["norm"], tok[:n])
        fn = jax.jit(jax.vmap(one, in_axes=(None, 0)))
        src = PinderSource(_with_policy(manifest), config=LoaderConfig(), sidechains=BackboneTable(root, "pinder"))
        Z, R = {}, {}
        for batch_ids, batch in Prefetcher(src, src.policy.plans(val, np.random.default_rng(0)), LoaderConfig()):
            free = jax.tree.map(lambda v: np.asarray(v[:, 1]), batch[0]); z = np.asarray(fn(params, free))
            for slot, cid in enumerate(batch_ids):
                if cid not in meta or meta[cid][0] < 0: continue
                r = table.row(cid); m = len(r["rsa"]); active = free["node_mask"][slot, :m] > 0
                Z[cid] = z[slot, :m].astype(np.float32); R[cid] = sides(r, active, meta[cid][1])
        src.close()
        report["runs"][name] = {"run": run, "metrics": evaluate(e, Z, R, meta, pairs), "seconds": round(time.time() - began, 1)}
        q = report["runs"][name]["metrics"]["quartets"]
        print(name, json.dumps({k: q[k]["accuracy"] for k in q}), flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n"); (out / "report.md").write_text(markdown(report))
    print(markdown(report), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
