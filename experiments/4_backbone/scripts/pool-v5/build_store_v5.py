"""abdata-v5 (2026-11-16): build_store.py of hotspot-oracle-v1 (sha256 c94e3004...) with two changes for the Ab/Ag newcomers:
the own-model oracle is not computed (own_edgemax / own_dburial NaN, as store-hotspot-atlas-v1; the asym50 recipe reads
only `pesto`), so the finite-own_dburial check is dropped; the final summary tolerates a store without validation records.

hotspot-oracle-v1 (2026-10-30): pack the PeSTo and own-model oracles into one record store (store-v2 /
store-apo-v1 format, zstd-records-v1; FORMAT.md) and compute the validation sanity AUROCs.

  python build_store.py PESTO_PREFIX OWN_PREFIX STORE_DIR
Per complex: pesto, own_edgemax, own_dburial, rsa_free (residue-rsa-v1 rsa[:, 1]), chain (residue-rsa-v1 codes), all (n,),
n checked against store-v2 and residue-rsa-v1. Writes records.bin, index.npz, records.json, metadata.json, manifest.tsv,
verification.json (every record decoded and checked), sanity.json. Atomic: built in a pending dir, then renamed.
"""
import csv, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from pkatrain.production_graphs import arrays_digest, encode, output
from pkatrain.production_residue import ResidueTable

ROOT = Path(os.environ["PKABENCH_RUNTIME"]); FIELDS = ("pesto", "own_edgemax", "own_dburial", "rsa_free", "chain")
DTYPES = {"pesto": "<f4", "own_edgemax": "<f4", "own_dburial": "<f4", "rsa_free": "<f4", "chain": "|i1"}
SHAPES = {f: ["n"] for f in FIELDS}


class Store:
    def __init__(self, path):
        import zstandard
        self.p = Path(path); self.meta = json.loads((self.p / "metadata.json").read_text()); ix = np.load(self.p / "index.npz")
        self.ids = ix["ids"].astype(str); self.dims = ix["dims"]; self.off = ix["offsets"]; self.by = {c: i for i, c in enumerate(self.ids)}
        self.fd = os.open(self.p / "records.bin", os.O_RDONLY); self.z = zstandard

    def raw(self, cid):
        i = self.by[cid]; n = int(self.dims[i, 0]); a, b = int(self.off[i]), int(self.off[i + 1])
        buf = self.z.ZstdDecompressor().decompress(os.pread(self.fd, b - a, a)); out, p = {}, 0
        for f in self.meta["fields"]:
            dt = np.dtype(self.meta["dtypes"][f]); out[f] = np.frombuffer(buf, dt, n, p); p += n * dt.itemsize
        if p != len(buf): raise ValueError("record length")
        return out


def auroc(s, y):
    y = np.asarray(y, bool); s = np.asarray(s, float); npos = int(y.sum()); nneg = len(y) - npos
    if not npos or not nneg: return None
    r = rankdata(s); return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load(prefix, keys):
    with np.load(prefix + ".npz") as f:
        ids = f["ids"].astype(str); off = f["offsets"]; arr = {k: f[k] for k in keys}
    return {c: {k: arr[k][off[i]:off[i + 1]] for k in keys} for i, c in enumerate(ids)}


def sanity(rows):
    """rows: per val complex (record, residue-table row). Pooled AUROC over surface residues (free RSA >= 0.1) with a finite
    score, interface = min heavy-atom distance to the partner <= 5 / 10 A (residue-rsa-v1 pairs); plus per-chain medians and
    the all-residue pooled AUROC (NaN scores excluded)."""
    acc = {}
    for rec, r in rows:
        m = len(r["rsa"]); lab = {}
        for c in (5.0, 10.0):
            y = np.zeros(m, bool); k = r["d"] <= c; y[r["i"][k]] = True; y[r["j"][k]] = True; lab[c] = y
        surf = np.nan_to_num(rec["rsa_free"].astype(float), nan=0.0) >= 0.1
        for f in ("pesto", "own_edgemax", "own_dburial"):
            for c in (5.0, 10.0):
                for dom, mask in (("surface", surf), ("all", np.ones(m, bool))):
                    a = acc.setdefault((f, c, dom), {"s": [], "y": [], "chain": [], "nan": 0})
                    s = rec[f].astype(float); keep = mask & np.isfinite(s); a["nan"] += int((mask & ~np.isfinite(s)).sum())
                    a["s"].append(s[keep]); a["y"].append(lab[c][keep]); a["chain"].append(np.char.add(rec["_id"] + ":", rec["chain"][keep].astype(str)))
    out = {}
    for (f, c, dom), a in acc.items():
        s, y, ch = np.concatenate(a["s"]), np.concatenate(a["y"]), np.concatenate(a["chain"]); per = []
        order = np.argsort(ch, kind="stable"); s, y, ch = s[order], y[order], ch[order]; cuts = np.flatnonzero(ch[1:] != ch[:-1]) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(ch)]):
            v = auroc(s[lo:hi], y[lo:hi])
            if v is not None: per.append(v)
        out.setdefault(f, {}).setdefault(f"{c:g}A", {})[dom] = {"pooled_auroc": auroc(s, y), "per_chain_median_auroc": float(np.median(per)) if per else None,
                                                             "chains": len(per), "residues": int(len(y)), "positive_rate": float(y.mean()), "excluded_nan": a["nan"]}
    return out


def main():
    pesto_prefix, own_prefix, store_dir = sys.argv[1:4]; out = Path(store_dir); t = time.time()
    if out.exists(): raise FileExistsError(out)
    table = ResidueTable(ROOT); recs2 = {r["id"]: r for r in json.loads((output(ROOT, "pinder") / "store-v2/records.json").read_text())}
    pesto = load(pesto_prefix, ("pesto",)); own = load(own_prefix, ("own_edgemax", "own_dburial"))
    pstat = {r["complex_id"]: r for r in csv.DictReader(open(pesto_prefix + ".tsv"), delimiter="\t")}
    ids = sorted(set(pstat) | set(own)); pending = out.parent / f".{out.name}.pending-{os.getpid()}"; pending.mkdir(parents=True)
    records, manifest, offsets, val_rows = [], [], [0], []
    with open(pending / "records.bin", "wb") as h:
        for cid in ids:
            row = {"complex_id": cid, "split": recs2[cid]["split"], "stored": 0, "n": recs2[cid]["n"], "pesto_status": pstat.get(cid, {}).get("pesto_status", "missing"),
                   "pesto_scored": pstat.get(cid, {}).get("pesto_scored", 0), "own_status": "ok" if cid in own else "missing", "note": ""}
            try:
                r = table.row(cid); n = recs2[cid]["n"]
                if len(r["rsa"]) != n: raise AssertionError("residue-rsa-v1 length differs from store-v2 n")
                p = pesto[cid]["pesto"] if cid in pesto else np.full(n, np.nan, np.float32)
                if cid not in own: raise AssertionError("own scores missing")
                arrays = {"pesto": p, "own_edgemax": own[cid]["own_edgemax"], "own_dburial": own[cid]["own_dburial"], "rsa_free": r["rsa"][:, 1], "chain": r["chain"]}
                for f in FIELDS:
                    arrays[f] = np.ascontiguousarray(arrays[f]).astype(DTYPES[f])
                    if arrays[f].shape != (n,): raise AssertionError((f, arrays[f].shape, n))
                blob = encode(arrays, FIELDS); h.write(blob); offsets.append(offsets[-1] + len(blob))
                records.append({"id": cid, "split": row["split"], "n": n, "arrays_sha256": arrays_digest(arrays, FIELDS)}); row["stored"] = 1
                if row["split"] == "val": val_rows.append(({**arrays, "_id": cid}, r))
            except Exception as e:  # noqa: BLE001
                row["note"] = f"{type(e).__name__}: {str(e)[:160]}"
            manifest.append(row)
    np.savez(pending / "index.npz", ids=np.asarray([r["id"] for r in records]), dims=np.asarray([[r["n"]] for r in records], np.int32).reshape(-1, 1),
             offsets=np.asarray(offsets, np.int64))
    (pending / "records.json").write_text(json.dumps(records) + "\n")
    with np.load(own_prefix + ".npz") as f: run = str(f["run"])
    meta = {"version": "hotspot-oracle-v1", "format": "zstd-records-v1", "zstd_level": 3, "dataset": "pinder-hotspot", "fields": list(FIELDS),
            "dtypes": DTYPES, "shapes": SHAPES, "dims": ["n"], "structures": len(records),
            "train_records": sum(r["split"] == "train" for r in records), "val_records": sum(r["split"] == "val" for r in records),
            "compressed_bytes": int(offsets[-1]), "own_run": run, "pesto_model": "LBM-EPFL/PeSTo@ba651aa i_v4_1_2021-09-07_11-21 (CC BY-NC-SA 4.0)",
            "residue_table": "residue-rsa-v1", "surface_rsa": 0.1, "definition": __doc__,
            "sources": {k: hashlib.sha256(open(p, "rb").read()).hexdigest() for k, p in (("pesto_npz", pesto_prefix + ".npz"), ("own_npz", own_prefix + ".npz"))}}
    (pending / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    st = Store(pending); bad = [r["id"] for r in records if arrays_digest(st.raw(r["id"]), FIELDS) != r["arrays_sha256"]]
    if bad: raise AssertionError(f"{len(bad)} records differ after packing, e.g. {bad[:5]}")
    cols = ["complex_id", "split", "stored", "n", "pesto_status", "pesto_scored", "own_status", "note"]
    with open(pending / "manifest.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(manifest)
    s = sanity(val_rows); s["val_complexes"] = len(val_rows); (pending / "sanity.json").write_text(json.dumps(s, indent=1) + "\n")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pending.iterdir()) if p.name != "records.bin"}
    (pending / "verification.json").write_text(json.dumps({"passed": True, "records_checked": len(records), "files": files,
                                                           "bytes": sum(p.stat().st_size for p in pending.iterdir())}, indent=1) + "\n")
    os.replace(pending, out)
    summary = {f: {c: {d: (None if v[d]["pooled_auroc"] is None else round(v[d]["pooled_auroc"], 4)) for d in ("surface", "all")} for c, v in s[f].items()} for f in ("pesto", "own_edgemax", "own_dburial") if f in s}
    print(json.dumps({"store": str(out), "records": len(records), "requested": len(ids), "not_stored": len(ids) - len(records),
                      "pesto_not_ok": sum(r["pesto_status"] != "ok" for r in manifest), "bytes": sum(p.stat().st_size for p in out.iterdir()),
                      "val": len(val_rows), "pooled_auroc": summary, "s": round(time.time() - t)}), flush=True)


if __name__ == "__main__":
    main()
