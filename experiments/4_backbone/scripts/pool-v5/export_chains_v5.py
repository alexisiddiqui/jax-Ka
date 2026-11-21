"""abdata-v5 (2026-11-16): export_chains.py of hotspot-oracle-v1 (sha256 a13f5029...) with ONLY the complex selection changed:
every store-v2 record of the runtime PKABENCH_RUNTIME (the abdata-v5 mini runtime: the Ab/Ag newcomers, or the parity
sample), instead of pool-v4 min_fraction <= 0.5 + val.   python export_chains_v5.py OUT_ZIP
"""
import csv, io, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(os.environ["PKABENCH_RUNTIME"])
W = {}


def complexes(pilot=False):
    from pkatrain.production_graphs import output
    recs = json.loads((output(ROOT, "pinder") / "store-v2/records.json").read_text())
    return sorted((r["id"], r["split"], r["n"]) for r in recs)


def one(arg):
    cid, split, n_store = arg
    try:
        import biotite.structure.io.pdb as pdb
        from jaxpropka.topology import load_topology
        from pkatrain.production_graphs import source, PINDER
        from pkatrain.gqt_paired_pinder import _read_cif_gz
        from pkatrain.production_residue import ResidueTable
        if "rsa" not in W: W["rsa"] = ResidueTable(ROOT)
        folder = source(ROOT, f"{PINDER}/entries/{cid}"); atoms = _read_cif_gz(folder / "AB.cif.gz")
        keys = load_topology(atoms, gap_policy="cap", freeze_disulfides=True).keys; n = len(keys)
        r = W["rsa"].row(cid); names = sorted({k.chain for k in keys}); code = np.array([names.index(k.chain) for k in keys])
        if n != n_store or len(r["rsa"]) != n: raise AssertionError(("n differs", n, n_store, len(r["rsa"])))
        if not np.array_equal(code, r["chain"]): raise AssertionError("chain codes differ from residue-rsa-v1")
        lookup = {(k.chain, k.number, k.insertion): i for i, k in enumerate(keys)}
        heavy = atoms[(atoms.element != "H") & (atoms.element != "D") & ~atoms.hetero]
        node = np.array([lookup.get((str(c), int(x), str(i).strip()), -1) for c, x, i in zip(heavy.chain_id, heavy.res_id, heavy.ins_code)])
        heavy = heavy[node >= 0]; node = node[node >= 0]
        missing = n - len(np.unique(node))  # graph residues without heavy non-hetero atoms: PeSTo leaves them NaN
        chains = {}
        for c in names:
            sub = heavy[heavy.chain_id == c].copy(); sub.chain_id[:] = "A"; f = pdb.PDBFile(); f.set_structure(sub)
            s = io.StringIO(); f.write(s); chains[c] = s.getvalue()
        info = {"id": cid, "split": split, "n": n, "chains": names, "code": code.tolist(),
                "resnum": [int(k.number) for k in keys], "icode": [str(k.insertion) for k in keys], "missing": int(missing)}
        return cid, chains, info, None
    except Exception as e:  # noqa: BLE001
        return cid, None, {"id": cid, "split": split, "n": n_store}, f"{type(e).__name__}: {str(e)[:200]}"


def main():
    from concurrent.futures import ProcessPoolExecutor
    dest = Path(sys.argv[1]); pilot = len(sys.argv) > 2 and sys.argv[2] == "pilot"; items = complexes(pilot); t = time.time()
    print(json.dumps({"complexes": len(items), "val": sum(s == "val" for _, s, _ in items), "pilot": pilot}), flush=True)
    count = {"complexes": 0, "chains": 0, "errors": 0}; workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 2
    part = dest.with_suffix(".partial")
    with ProcessPoolExecutor(workers) as pool, zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for k, (cid, chains, info, err) in enumerate(pool.map(one, items, chunksize=4)):
            if err: info["error"] = err; count["errors"] += 1; print(json.dumps({"error": cid, "why": err}), flush=True)
            else:
                for c, text in chains.items(): z.writestr(f"{cid}/{c}.pdb", text)
                count["chains"] += len(chains)
            z.writestr(f"{cid}/residues.json", json.dumps(info)); count["complexes"] += 1
            if (k + 1) % 2000 == 0: print(json.dumps({**count, "s": round(time.time() - t)}), flush=True)
    part.replace(dest); print(json.dumps({**count, "s": round(time.time() - t), "zip": str(dest)}), flush=True)


if __name__ == "__main__":
    main()
