"""Auxiliary-target audit (2026-10-10): data for replacing the oGQT burial/interface regression heads with ordinal BCE
targets. On a random sample of pool-v4 training structures:
- burial: free-state RSA of every labelled PINDER site (sites.json rsa_free) and pKPDB train_mask site (environment.json
  rsa), with |state shift| = |pKa - PKPDB_PK_MOD[group]| per RSA bin, to choose four 1 - clip(RSA) intervals;
- interface: for every labelled PINDER site, the residue-level minimum heavy-atom distance from the site residue to each
  partner-chain residue within 10 A in AB.cif.gz (the min over partners reproduces sites.json partner_distance_A), so
  additive soft-contact scores sum_j f(d_j) can be compared against the teacher |AB - free| shift.

  python scripts/auxiliary_target_audit.py OUT.json [--pinder 3000] [--pkpdb 3000] [--workers 32]
"""
import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CUTOFF = 10.0


def pinder_one(args):
    root, cid = args
    from scipy.spatial import cKDTree
    from pkatrain.gqt_paired_pinder import _paired_rows, _read_cif_gz
    from pkatrain.production_graphs import PINDER, source
    folder = source(root, f"{PINDER}/entries/{cid}")
    rows = _paired_rows(folder, "train")
    if not rows: return []
    atoms = _read_cif_gz(folder / "AB.cif.gz")
    atoms = atoms[(atoms.element != "H") & (atoms.element != "D") & ~atoms.hetero]
    chain = atoms.chain_id.astype(str); resid = atoms.res_id; icode = np.char.strip(atoms.ins_code.astype(str))
    out = []
    for row in rows:
        c, n, i, group = row["key"]
        own = (chain == c) & (resid == n) & (icode == i); partner = chain != c
        if not own.any() or not partner.any(): continue
        tree = cKDTree(atoms.coord[partner]); hits = tree.query_ball_point(atoms.coord[own], CUTOFF + 1e-3)
        index = np.where(partner)[0]; best = {}
        for site_atom, near in zip(atoms.coord[own], hits):
            for h in near:
                a = index[h]; key = (chain[a], int(resid[a]), icode[a]); d = float(np.linalg.norm(site_atom - atoms.coord[a]))
                if d < best.get(key, np.inf): best[key] = d
        out.append({"id": cid, "group": group, "rsa_free": row.get("rsa_free"), "partner_distance_A": row.get("partner_distance_A"),
                    "target_ab": row["target_ab"], "target_free": row["target_free"],
                    "contacts": sorted(round(d, 3) for d in best.values() if d <= CUTOFF)})
    return out


def pkpdb_one(args):
    root, cid = args
    from pkatrain.production_graphs import PKPDB, source
    entry = source(root, f"{PKPDB}/entries/{cid}")
    sites = json.loads((entry / "sites.json").read_text())
    env = {(r["chain"], r["resnum"], r["icode"], r["group"]): r for r in json.loads((entry / "environment.json").read_text())["sites"]}
    return [{"id": cid, "group": s["group"], "pka": s["pka"], "rsa": env[s["chain"], s["resnum"], s["icode"], s["group"]]["rsa"],
             "chain_distance_A": env[s["chain"], s["resnum"], s["icode"], s["group"]]["chain_distance_A"]}
            for s in sites if s["train_mask"] and s.get("pka") is not None]


def ids(path, n, seed):
    with open(path) as handle: rows = [r["id"] for r in csv.DictReader(handle, delimiter="\t")]
    return sorted(np.random.default_rng(seed).choice(rows, min(n, len(rows)), replace=False).tolist())


def main():
    p = argparse.ArgumentParser(); p.add_argument("out"); p.add_argument("--pinder", type=int, default=3000)
    p.add_argument("--pkpdb", type=int, default=3000); p.add_argument("--workers", type=int, default=32)
    a = p.parse_args(); root = Path(os.environ["PKABENCH_RUNTIME"]); pool = root / "training/pool-v4"
    with ProcessPoolExecutor(a.workers) as ex:
        pinder = [r for rows in ex.map(pinder_one, [(root, c) for c in ids(pool / "pinder.tsv", a.pinder, 0)], chunksize=4) for r in rows]
        pkpdb = [r for rows in ex.map(pkpdb_one, [(root, c) for c in ids(pool / "pkpdb.tsv", a.pkpdb, 1)], chunksize=8) for r in rows]
    Path(a.out).write_text(json.dumps({"pinder": pinder, "pkpdb": pkpdb}))
    print(json.dumps({"pinder_sites": len(pinder), "pkpdb_sites": len(pkpdb)}))


if __name__ == "__main__":
    main()
