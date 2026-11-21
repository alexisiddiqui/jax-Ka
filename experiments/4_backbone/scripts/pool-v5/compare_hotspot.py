"""abdata-v5: PeSTo parity on the 35 pool-v4 parity structures: this job's PeSTo (chains exported on Coulson from the
rebuilt parity records) vs store-hotspot-v2 `pesto` (same structures, original Isambard pipeline).
  python compare_hotspot.py PESTO_PREFIX STORE_HOTSPOT_V2 OUT_JSON"""
import json, sys
import numpy as np
from pkatrain.production_residue import HotspotStore
prefix, store_path, dest = sys.argv[1:4]
f = np.load(prefix + ".npz"); ids = f["ids"].astype(str); off = f["offsets"]; store = HotspotStore(store_path); rows = []
for i, cid in enumerate(ids):
    mine = f["pesto"][off[i]:off[i + 1]].astype(np.float64); rec = store.record(cid)
    if rec is None: rows.append({"id": cid, "missing": True}); continue
    ref = rec["pesto"].astype(np.float64)
    if ref.shape != mine.shape: rows.append({"id": cid, "shape": [len(mine), len(ref)]}); continue
    d = np.abs(mine - ref); both = np.isfinite(d)
    rows.append({"id": cid, "n": len(ref), "exact": bool(np.array_equal(mine, ref, equal_nan=True)), "max_abs": float(d[both].max(initial=0.0)),
                 "gt_0.01": int((d[both] > 0.01).sum()), "nan_mismatch": int((np.isnan(mine) != np.isnan(ref)).sum())})
ok = [r for r in rows if "max_abs" in r]; allres = sum(r["n"] for r in ok)
out = {"structures": len(rows), "compared": len(ok), "exact": sum(r["exact"] for r in ok), "residues": allres,
       "max_abs": max((r["max_abs"] for r in ok), default=None), "residues_gt_0.01": sum(r["gt_0.01"] for r in ok),
       "nan_mismatch": sum(r["nan_mismatch"] for r in ok), "problems": [r for r in rows if "max_abs" not in r], "rows": rows}
open(dest, "w").write(json.dumps(out, indent=1) + "\n"); print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
