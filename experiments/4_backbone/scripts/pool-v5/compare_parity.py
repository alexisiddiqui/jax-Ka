"""abdata-v5 parity: records and side tables rebuilt on Coulson (abdata-v5 build of rt-parity, 35 pool-v4 structures)
against the Isambard originals (extract_ref.py -> parity-ref.npz).

  python compare_parity.py RT_PARITY PARITY_REF_NPZ OUT_JSON
Per field: exact equality, else max |diff| (edge / site_edge may differ in the last float32 bit across numpy builds,
production_graphs.compare's EDGE_TOLERANCE 1e-6)."""
import json, sys
from pathlib import Path
import numpy as np
from pkatrain.production_graphs import ProductionStore, decode, output, FIELDS, DIMS

rt, ref_path, dest = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ref = np.load(ref_path); meta = json.loads(bytes(ref["_meta"]).decode()); P = output(rt, "pinder")
store = ProductionStore(P / "store-v2"); ids = [r["id"] for r in json.loads((P / "store-v2/records.json").read_text())]
out = {"structures": len(ids), "store": {}, "tables": {}, "failures": []}

def cmp(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape: return {"shape": [list(a.shape), list(b.shape)]}
    if a.dtype.kind == "f":
        same = np.array_equal(a, b, equal_nan=True)
        return {"exact": bool(same), "max_abs": float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0)),
                "nan_mismatch": int(np.sum(np.isnan(a) != np.isnan(b)))}
    return {"exact": bool(np.array_equal(a, b))}

def agg(bucket, name, r):
    s = bucket.setdefault(name, {"exact": 0, "inexact": 0, "max_abs": 0.0, "shape_mismatch": 0, "nan_mismatch": 0})
    if "shape" in r: s["shape_mismatch"] += 1; return
    s["exact" if r["exact"] else "inexact"] += 1; s["max_abs"] = max(s["max_abs"], r.get("max_abs", 0.0)); s["nan_mismatch"] += r.get("nan_mismatch", 0)

m = meta["store-v2"]
for cid in ids:
    mine = store.raw(cid); dims = [int(x) for x in ref[f"store-dims|{cid}"]]
    import zstandard
    theirs = decode(bytes(ref[f"store|{cid}"]), m["fields"], m["dtypes"], dims)
    if [int(mine["nodes"].shape[0])] != dims[:1]: out["failures"].append((cid, "n differs", dims))
    for f in FIELDS["pinder"]: agg(out["store"], f, cmp(mine[f], theirs[f]))
for table, arrays in (("backbone-v1", ["features"]), ("sidechains-v1", ["features"]), ("contacts-v1", ["scores"]),
                      ("residue-rsa-v1", ["rsa", "chain", "pair_i", "pair_j", "pair_d"])):
    idx = {c: i for i, c in enumerate(json.loads((P / table / "ids.json").read_text()))}
    for a in arrays:
        if a.startswith("pair_"): offs = np.load(P / table / "pair_offsets.npy")
        else: offs = np.load(P / table / "offsets.npy")
        mm = np.load(P / table / f"{a}.npy", mmap_mode="r")
        for cid in ids:
            k = idx[cid]; agg(out["tables"], f"{table}:{a}", cmp(np.asarray(mm[offs[k]:offs[k + 1]]), ref[f"{table}:{a}|{cid}"]))
store.close()
out["passed_exact_except_edges"] = (all(v["inexact"] == 0 and v["shape_mismatch"] == 0 for k, v in out["store"].items() if k not in ("edge", "site_edge"))
                                    and all(v["shape_mismatch"] == 0 and v["inexact"] == 0 for v in out["tables"].values())
                                    and all(out["store"][k]["max_abs"] <= 1e-6 and out["store"][k]["shape_mismatch"] == 0 for k in ("edge", "site_edge")) and not out["failures"])
Path(dest).write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
