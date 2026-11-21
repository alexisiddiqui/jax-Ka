"""abdata-v5 (2026-11-16): the pool-v5 PINDER manifest for quartet-rank-v1b (--pinder-manifest; production_extra).

  python build_manifest_v5.py BASE_MANIFEST POOL_V5_TSV EXTRA_MANIFEST EXTRA_STORE_VERIFICATION OUT_JSON POOL_V5_PATH EXTRA_STORE_PATH
BASE_MANIFEST = gqt-production-v2/pinder/manifest-v1.json (store-v2, pool-v4), EXTRA_MANIFEST = the newcomers'
manifest-v1.json (production_loading build_manifest on the abdata-v5 mini runtime). Pure file manipulation:
- every base validation record unchanged; every base training record in pool-v5 with pool-v5's min_fraction; base
  training records not in pool-v5 dropped (listed);
- pool-v5 training ids not in store-v2 taken from the extra manifest (split train, pool-v5 min_fraction; dims and
  w_burial_mean / w_interface_mean as built); an id missing from both is an error;
- store, store_verification_sha256, bucket_policy, capacities, version, dataset kept from the base; each extra record
  must fit its bucket's capacities (else it is listed and the build stops: capacities are compiled shapes);
- pool / pool_sha256 = pool-v5; extra_store / extra_store_verification_sha256 for production_extra.check_manifest.
POOL_V5_PATH / EXTRA_STORE_PATH are the paths recorded in the manifest (the Isambard paths), files are read locally.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(base_path, pool_path, extra_path, extra_verification, out, pool_recorded, extra_store_recorded):
    base = json.loads(Path(base_path).read_text()); extra = json.loads(Path(extra_path).read_text())
    pool = {r["id"]: r for r in csv.DictReader(open(pool_path), delimiter="\t")}
    bounds = base["bucket_policy"]["bounds"]; caps = base["capacities"]
    def bucket(n): return next(str(b) for b in bounds if int(n) <= b) if int(n) <= bounds[-1] else None
    records, dropped, kept = [], [], set()
    for r in base["records"]:
        if r["split"] == "val": records.append(r); continue
        if r["id"] in pool: records.append({**r, "min_fraction": float(pool[r["id"]]["min_fraction"])}); kept.add(r["id"])
        else: dropped.append(r["id"])
    base_ids = {r["id"] for r in base["records"]}; xidx = {r["id"]: r for r in extra["records"]}
    added, missing, too_big = [], [], []
    for cid, row in pool.items():
        if cid in base_ids: continue
        if cid not in xidx: missing.append(cid); continue
        r = {**xidx[cid], "split": "train", "min_fraction": float(row["min_fraction"])}; b = bucket(r["n"])
        if b is None or any(int(r[k]) > caps[b][i] for i, k in enumerate(("n", "k", "q", "s", "sk"))): too_big.append((cid, b, [r[k] for k in ("n", "k", "q", "s", "sk")])); continue
        records.append(r); added.append(cid)
    if missing or too_big: raise SystemExit(json.dumps({"missing": missing, "too_big": too_big}))
    pool_train = set(pool); have = kept | set(added)
    if have != pool_train: raise AssertionError("manifest training ids != pool-v5")
    manifest = {**base, "records": records, "pool": pool_recorded, "pool_sha256": sha(pool_path),
                "extra_store": extra_store_recorded, "extra_store_verification_sha256": sha(extra_verification),
                "pool_v5": {"base_manifest_sha256": sha(base_path), "extra_manifest_sha256": sha(extra_path), "train_records": len(have),
                            "from_store_v2": len(kept), "from_extra": len(added), "dropped_from_pool_v4": dropped,
                            "built_by": "abdata-v5 build_manifest_v5.py"}}
    Path(out).write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest["pool_v5"].items() if k != "dropped_from_pool_v4"} | {"dropped": len(dropped), "records": len(records)}))


if __name__ == "__main__":
    main(*sys.argv[1:8])
