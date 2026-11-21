"""abdata-v5 / quartet-rank-v1b loader check on the real stores (CPU only, no model step).

  python loader_check.py MODE RUN OUT_JSON [extra train args...]
Runs production_train.main(["train", RUN, <asym50 recipe + q03>, extra args]) with make_model replaced by a stop: train()
validates, registers the run directory (protocol.json only) and builds every source exactly as for training, then this
script takes the configured training PINDER source and loads batches from the epoch-1 plan of fraction 0.5:
  MODE v4:  the first 40 PINDER batches; every array of every batch hashed (the v1 and v1b trees must agree bitwise).
  MODE v5:  (with --pinder-manifest/--pinder-extra/--mask-store-extra) the first 40 batches, plus every batch of the epoch
            that contains an extra (newcomer) record; per batch the hash, and for every extra record in it: loaded, residue
            and hotspot fields present, hotspot fallback 0 (mask_stats[:, 2]); counts of records per source.
"""
import hashlib, json, sys
import numpy as np

mode, run, dest = sys.argv[1], sys.argv[2], sys.argv[3]; extra_args = sys.argv[4:]
from pkatrain import production_train as pt  # noqa: E402

captured = {}


class Stop(Exception): pass


def stop(*a, **k):
    f = sys._getframe(1); captured.update(train_pinder=f.f_locals["train_pinder"], manifests=f.f_locals["manifests"]); raise Stop


pt.make_model = stop
try: pt.main(["train", run, *extra_args])
except Stop: pass
src = captured["train_pinder"]; manifests = captured["manifests"]
plans, _ = pt.epoch_plans(manifests, 0.5, 1, 16, 17)
base_ids = (set(src.store._base_ids) if hasattr(src.store, "_base_ids") else set(src.store.by_id)); extra_ids = set(getattr(src.store, "_extra_ids", ()))


def leaves(x, out):
    if isinstance(x, dict):
        for k in sorted(x): leaves(x[k], out)
    elif isinstance(x, (list, tuple)):
        for v in x: leaves(v, out)
    else: out.append(np.ascontiguousarray(np.asarray(x)))
    return out


def sha(batch):
    h = hashlib.sha256()
    for a in leaves(batch, []): h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


report = {"mode": mode, "plans": len(plans), "train_records_0.5": sum(len(p) for p in plans), "batches": {}, "extra": {}}
pick = list(range(min(40, len(plans))))
if mode == "v5": pick += [i for i, p in enumerate(plans) if any(c in extra_ids for c in p) and i not in pick]
for i in pick:
    spec = plans[i]; batch = src.load(spec); report["batches"][i] = {"ids": list(spec), "sha256": sha(batch)}
    if mode == "v5":
        g = batch[0]; stats = np.asarray(g["mask_stats"]) if "mask_stats" in g else None
        for j, cid in enumerate(spec):
            if cid not in extra_ids: continue
            report["extra"][cid] = {"batch": i, "has_rsa_residue": "rsa_residue" in g, "has_mask": stats is not None,
                                    "fallback": None if stats is None else float(stats[j, :, 2].sum()) if stats.ndim == 3 else float(stats[j, 2]),
                                    "masked_any": None if stats is None else bool(np.asarray(stats[j]).sum() > 0)}
report["extra_records_in_epoch"] = sum(1 for p in plans for c in p if c in extra_ids)
report["extra_checked"] = len(report["extra"])
report["extra_fallbacks"] = sum(1 for v in report["extra"].values() if v["fallback"])
report["extra_missing_fields"] = sum(1 for v in report["extra"].values() if not (v["has_rsa_residue"] and v["has_mask"]))
open(dest, "w").write(json.dumps(report, indent=1) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("batches", "extra")}))
