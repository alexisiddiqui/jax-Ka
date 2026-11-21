"""quartet-rank-v1 (2026-11-15): native-pairing quartet accuracy of the free edge head on the PINDER validation complexes.

Free branch, unmasked, selected checkpoint per selection.json, CPU jax (cpu_attention.py). Per validation complex the
deterministic quartet pools of the training sampler (production_residue.quartet_support / quartet_pool, stream
[QUARTET_POOL_SALT, crc32(id)], <= 2048 general + <= 2048 chemically matched; node types and node_mask of the free branch,
residue-rsa-v1 contacts) are scored EXHAUSTIVELY (no per-visit draw): delta = s_ij + s_kl - s_il - s_kj with s the aux_redge
logits on the normalised free-branch residue tokens. Per complex: accuracy = mean [delta > 0] (ties count 1/2), softplus =
mean softplus(1 - delta), margin = mean delta, over general, chemical, chemical opposite-charge (class_i * class_j = -1)
and all rows (general + chemical); then the macro mean over complexes with >= 1 row, overall and per stratum (pool-v4
pinder-val.tsv: hetero / homo / Ab/Ag). A run NAME=RUN@HEAD.npz grafts an external free edge head (score_graft.py); runs
sharing a base run share one embedding pass.

  python score_quartet.py OUT_DIR NAME=RUN[@HEAD.npz] [...] [--limit N]      -> OUT_DIR/report.json, OUT_DIR/report.md
"""
import os

os.environ.setdefault("GQT_PARAMETER_SCALING", "1")
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

CHUNK = 65536
STRATA = ("hetero", "homo", "Ab/Ag")
SUBSETS = ("all", "general", "chemical", "opposite")


def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))  # jax.nn.gelu (approximate)


def logits(e, za, zb):
    """aux_redge pair logits for embedding rows za, zb (n, W) (score_decoy.logits)."""
    out = []
    for s in range(0, len(za), CHUNK):
        a, b = za[s:s + CHUNK], zb[s:s + CHUNK]; f = np.concatenate((a * b, np.abs(a - b)), axis=1)
        out.append((gelu(f @ e["hidden"]["w"] + e["hidden"]["b"]) @ e["out"]["w"] + e["out"]["b"])[:, 0])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def deltas(e, z, q):
    """delta (R,) = s_ij + s_kl - s_il - s_kj for quartet rows q (R, 4) on embeddings z (m, W)."""
    if not len(q): return np.zeros(0)
    i, j, k, l = q.T; a = np.concatenate((i, k, i, k)); b = np.concatenate((j, l, l, j))
    s = logits(e, z[a], z[b]).astype(np.float64).reshape(4, len(q)); return s[0] + s[1] - s[2] - s[3]


def complex_metrics(e, z, general, chemical, qclass):
    """{subset: {n, accuracy, softplus, margin} or None} for one complex."""
    opp = chemical[(qclass[chemical[:, 0]] * qclass[chemical[:, 1]]) == -1] if len(chemical) else chemical
    rows = {"general": general, "chemical": chemical, "opposite": opp, "all": np.concatenate((general, chemical)) if len(general) or len(chemical) else general}
    out = {}
    for name, q in rows.items():
        d = deltas(e, z, q)
        out[name] = None if not len(d) else {"n": int(len(d)), "accuracy": float(np.mean((d > 0) + 0.5 * (d == 0))),
                                             "softplus": float(np.mean(np.logaddexp(0.0, 1.0 - d))), "margin": float(np.mean(d))}
    return out


def aggregate(per, strata):
    """Macro means over complexes per subset, overall and per stratum."""
    def agg(cids):
        res = {}
        for sub in SUBSETS:
            vals = [per[c][sub] for c in cids if per[c][sub] is not None]
            res[sub] = {"complexes": len(vals), "quartets": int(sum(v["n"] for v in vals)),
                        **{k: (float(np.mean([v[k] for v in vals])) if vals else None) for k in ("accuracy", "softplus", "margin")}}
        return res
    cids = sorted(per)
    return {"overall": agg(cids), **{s: agg([c for c in cids if strata.get(c) == s]) for s in STRATA}, "complexes_scored": len(cids)}


def markdown(report):
    f = lambda x: "-" if x is None else f"{x:.4f}"
    L = ["# Quartet accuracy of the free edge head (PINDER validation, free branch, unmasked; exhaustive deterministic pools)", "",
         "| arm | run | complexes | all acc | general acc | chemical acc | opposite-charge acc | all softplus | all margin | hetero acc | homo acc | Ab/Ag acc |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in report["runs"].items():
        o = r["metrics"]["overall"]
        L.append(f"| {name} | {r['run']} | {o['all']['complexes']} | {f(o['all']['accuracy'])} | {f(o['general']['accuracy'])} | {f(o['chemical']['accuracy'])} | "
                 f"{f(o['opposite']['accuracy'])} | {f(o['all']['softplus'])} | {f(o['all']['margin'])} | "
                 + " | ".join(f(r["metrics"][s]["all"]["accuracy"]) for s in STRATA) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(prog="score_quartet.py"); ap.add_argument("out"); ap.add_argument("runs", nargs="+"); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True); runs = dict(a.split("=", 1) for a in args.runs)
    import cpu_attention  # noqa: F401  (plain-JAX attention under JAX_PLATFORMS=cpu)
    import jax
    from pkanet.model import norm
    from pkatrain.loading import Prefetcher
    from pkatrain.production_graphs import output, read
    from pkatrain.production_inputs import BackboneTable
    from pkatrain.production_loading import LoaderConfig, PinderSource, select
    from pkatrain.production_residue import ResidueTable, quartet_pools, quartet_classes
    from pkatrain.production_train import _with_policy, selected_model
    from score_graft import graft_params, graft_record, load_head, split_run
    root = Path(os.environ["PKABENCH_RUNTIME"]); manifest = read(output(root, "pinder") / "manifest-v1.json"); table = ResidueTable(root)
    val = select(manifest, "val")[:args.limit] if args.limit else select(manifest, "val")
    meta = {r["id"]: r["stratum"] for r in csv.DictReader(open(root / "training/pool-v4/pinder-val.tsv"), delimiter="\t")}
    report = {"definition": __doc__, "limit": args.limit, "validation_complexes": len(val), "runs": {}}
    bases = {}
    for name, run in runs.items(): bases.setdefault(split_run(run)[0], []).append(name)
    for base, names in bases.items():
        began = time.time(); params, fns = selected_model(root, base); embeddings = fns[0]
        heads = {}
        for name in names:
            _, head_path = split_run(runs[name])
            if head_path is None: heads[name] = (params, None)
            else:
                p, diff = graft_params(params, load_head(head_path)); heads[name] = (p, graft_record(runs[name], head_path, diff))
        es = {name: jax.tree.map(np.asarray, p)["aux_redge"] for name, (p, _) in heads.items()}

        def one(p, g):
            tok = embeddings(p, g)[0]; n = g["nodes"].shape[0]; return norm(p["aux_redge"]["norm"], tok[:n])
        fn = jax.jit(jax.vmap(one, in_axes=(None, 0)))
        src = PinderSource(_with_policy(manifest), config=LoaderConfig(), sidechains=BackboneTable(root, "pinder"))
        per = {name: {} for name in names}; empty = 0; iterated = 0
        for ids, batch in Prefetcher(src, src.policy.plans(val, np.random.default_rng(0)), LoaderConfig()):
            free = jax.tree.map(lambda v: np.asarray(v[:, 1]), batch[0]); z = np.asarray(fn(params, free))
            for slot, cid in enumerate(ids):
                r = table.row(cid); m = len(r["rsa"]); iterated += 1
                nodes = free["nodes"][slot, :m]; active = free["node_mask"][slot, :m] > 0
                general, chemical = quartet_pools(r, cid, nodes, active)
                if not len(general) and not len(chemical): empty += 1; continue
                qclass = quartet_classes(nodes).astype(np.int64); zc = z[slot, :m].astype(np.float32)
                for name in names: per[name][cid] = complex_metrics(es[name], zc, general, chemical, qclass)
        src.close()
        for name in names:
            report["runs"][name] = {"run": runs[name], "base_run": base, "graft": heads[name][1], "complexes_iterated": iterated, "complexes_without_quartets": empty,
                                    "metrics": aggregate(per[name], meta), "seconds_base": round(time.time() - began, 1)}
            o = report["runs"][name]["metrics"]["overall"]
            print(name, json.dumps({s: o[s]["accuracy"] for s in SUBSETS}), f"complexes {o['all']['complexes']}", flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n"); (out / "report.md").write_text(markdown(report))
    print(markdown(report), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
