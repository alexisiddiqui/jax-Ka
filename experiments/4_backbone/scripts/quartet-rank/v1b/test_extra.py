"""Local CPU tests of quartet-rank-v1b (2026-11-16): production_extra (pool-v5 extra PINDER records) and the flags-off
identity of the v1b tree with quartet-rank-v1.

  python3 test_extra.py [V1B_SRC] [V1_SRC]
1. tree: v1b = v1 + production_extra.py + production_train.py, whose diff touches only the flag plumbing; every added
   statement of the train() body sits under `pinder_* is not None` (flags off -> v1 code path).
2. union tables on disk (real table classes on two runtime-shaped roots): base ids read byte-identically to the base table
   alone, extra ids read the extra table, unknown ids raise, overlapping ids are refused; hotspot store falls back to None.
3. attach on a stand-in source with every hook the asym50 / quartet recipes set; refusals (sidechains, strata, alt,
   rsa-bound, missing --mask-store-extra).
4. check_manifest: accepts a pool manifest built like build_manifest_v5.py, refuses changed validation / capacities /
   foreign records / a changed extra store.
5. CLI: train() refuses half-set flags before touching a run directory.
"""
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NEW = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parents[4] / "src"  # this repository
OLD = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else HERE.parent / "quartet-model" / "src"
sys.path.insert(0, str(NEW))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
try:
    import zstandard  # noqa: F401
except ImportError:  # the Mac has no zstandard: an identity codec is enough for the union logic (store-v2 encode/decode)
    z = types.ModuleType("zstandard")
    class _C:
        def __init__(self, *a, **k): pass
        def compress(self, b): return bytes(b)
        def decompress(self, b): return bytes(b)
    z.ZstdCompressor = _C; z.ZstdDecompressor = _C; sys.modules["zstandard"] = z

import importlib.abc, importlib.machinery, importlib.util  # noqa: E402


class _Stub(types.ModuleType):  # optional trainer deps missing on the Mac (as test_quartet.py)
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return lambda *a, **k: None


STUBS = {m for m in ("lineax", "optimistix", "equinox", "diffrax", "biotite", "gemmi", "pyarrow", "jax_triton") if importlib.util.find_spec(m) is None}


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        return importlib.machinery.ModuleSpec(name, self, is_package=True) if name.split(".")[0] in STUBS else None
    def create_module(self, spec): return _Stub(spec.name)
    def exec_module(self, module): module.__path__ = []


sys.meta_path.insert(0, _Finder())
from pkabench.runtime import atomic_json, digest  # noqa: E402
from pkatrain import production_extra as px  # noqa: E402
from pkatrain import production_graphs as pg  # noqa: E402
from pkatrain.production_aux import ContactTable  # noqa: E402
from pkatrain.production_inputs import BackboneTable  # noqa: E402
from pkatrain.production_residue import HotspotStore, ResidueTable  # noqa: E402
from pkatrain.production_sidechains import SidechainTable  # noqa: E402


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def test_tree():
    new, old = files(NEW), files(OLD)
    assert set(new) - set(old) == {"pkatrain/production_extra.py"}, set(new) ^ set(old)
    assert not set(old) - set(new)
    changed = sorted(k for k in old if old[k] != new[k]); assert changed == ["pkatrain/production_train.py"], changed
    a = (OLD / "pkatrain/production_train.py").read_text().splitlines(); b = (NEW / "pkatrain/production_train.py").read_text().splitlines()
    removed = [l for l in difflib.ndiff(a, b) if l.startswith("- ")]; added = [l[2:] for l in difflib.ndiff(a, b) if l.startswith("+ ")]
    # removed lines: the signature tail and the CLI call tail only (each re-added with the new arguments)
    assert len(removed) == 2 and "smoke_batches=5, quartet_grad_norms_every_epoch=False):" in removed[0] and "args.quartet_grad_norms)" in removed[1], removed
    body = "\n".join(added)
    for needle in ("pinder_manifest=None, pinder_extra=None, mask_store_extra=None):", "if pinder_extra is not None: production_extra.attach(",
                   "if pinder_manifest is not None:", "if pinder_extra is not None else {}),"):
        assert needle in body, needle
    # every added executable statement in train() is under a pinder_* condition, a CLI argument, or a comment / continuation
    for line in added:
        s = line.strip()
        ok = (not s or s.startswith("#") or "pinder_" in s or "mask_store_extra" in s or s.startswith(("production_extra.refuse", "swap_weight=", "geometry_weight=",
              "from . import production_extra", "pool_manifest =", "\"pool_sha256\"", "\"extra_verification_sha256\"", "**({\"mask_store_extra\"", "if mask_store_extra",
              "\"rule\":", "\"production_extra_sha256\"", "p.add_argument(\"--", "args.pinder_manifest", "args.quartet_quartet")) or s.startswith("if mask_store is not None and mask_store_extra")
              or s.startswith("args.quartet_weight, args.quartet_count, args.smoke_batches, args.quartet_grad_norms,")
              or s.startswith("quartet_weight=0.0, quartet_count=None, smoke_batches=5, quartet_grad_norms_every_epoch=False,") or s == "if mask_store_extra is not None else {}),")
        assert ok, line


# ---------------------------------------------------------------- synthetic tables on disk
def make_root(root, ids, seed):
    """Real-format store-v2 / backbone-v1 / sidechains-v1 / contacts-v1 / residue-rsa-v1 for `ids` under output(root, 'pinder')."""
    rng = np.random.default_rng(seed); out = pg.output(root, "pinder"); out.mkdir(parents=True)
    dims = {cid: (int(rng.integers(5, 12)), 4, int(rng.integers(1, 4)), 2, 2) for cid in ids}
    shards = out / "shards" / "0000-of-0001"; shards.mkdir(parents=True); records = []; pos = 0; fields = pg.FIELDS["pinder"]
    dtypes = {f: ("<i4" if f in ("neighbors", "query_residue", "query_group", "site_residue", "site_type", "site_neighbors", "query_site") else
                  "|b1" if f in ("node_mask", "edge_mask", "site_mask", "site_edge_mask", "branch_edge_mask", "branch_site_edge_mask", "interface") else "<f4") for f in fields}
    with open(shards / "records.bin", "wb") as h:
        for cid in ids:
            arrays = {f: rng.normal(size=pg.shape(f, *dims[cid])).astype(dtypes[f]) if dtypes[f] == "<f4" else
                      rng.integers(0, 2, pg.shape(f, *dims[cid])).astype(dtypes[f]) for f in fields}
            blob = pg.encode(arrays, fields); h.write(blob)
            records.append({"id": cid, "split": "train", **dict(zip(pg.DIMS, dims[cid])), "arrays_sha256": pg.arrays_digest(arrays, fields), "blob": [pos, len(blob)]}); pos += len(blob)
    atomic_json(out / "ids.json", {"ids": [[c, "train"] for c in ids]})
    atomic_json(shards / "receipt.json", {"dataset": "pinder", "task": 0, "tasks": 1, "records": records, "skipped": [], "failures": [], "dtypes": dtypes})
    pg.pack(root, "pinder", workers=1)
    n = {c: dims[c][0] for c in ids}; q = {c: dims[c][2] for c in ids}
    def flat(name, per, width, dtype):
        d = out / name; d.mkdir(exist_ok=True); rows = [per(c) for c in ids]
        np.save(d / ("features.npy" if name != "contacts-v1" else "scores.npy"), np.concatenate(rows).astype(dtype))
        np.save(d / "offsets.npy", np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)); atomic_json(d / "ids.json", list(ids))
        atomic_json(d / "verification.json", {"table": name, "seed": seed})
    flat("backbone-v1", lambda c: rng.normal(size=(n[c], 14)), 14, np.float32)
    flat("sidechains-v1", lambda c: rng.normal(size=(n[c], 128)), 128, np.float16)
    flat("contacts-v1", lambda c: rng.normal(size=(q[c],)), 1, np.float32)
    d = out / "residue-rsa-v1"; d.mkdir(); rsa = [rng.random((n[c], 2)).astype(np.float32) for c in ids]; pairs = [int(rng.integers(0, 4)) for c in ids]
    np.save(d / "rsa.npy", np.concatenate(rsa)); np.save(d / "chain.npy", np.concatenate([(np.arange(n[c]) >= n[c] // 2).astype(np.int16) for c in ids]))
    np.save(d / "offsets.npy", np.cumsum([0] + [n[c] for c in ids]).astype(np.int64)); np.save(d / "pair_offsets.npy", np.cumsum([0] + pairs).astype(np.int64))
    for k in ("i", "j"): np.save(d / f"pair_{k}.npy", np.concatenate([rng.integers(0, 4, p) for p in pairs]).astype(np.int32))
    np.save(d / "pair_d.npy", np.concatenate([rng.random(p) for p in pairs]).astype(np.float32)); atomic_json(d / "ids.json", list(ids))
    atomic_json(d / "verification.json", {"table": "residue-rsa-v1", "seed": seed})
    return n


def make_hotspot(path, n, seed):
    rng = np.random.default_rng(seed); path.mkdir(parents=True); fields = ["pesto", "own_edgemax", "own_dburial", "rsa_free", "chain"]; offs = [0]
    with open(path / "records.bin", "wb") as h:
        for cid, k in n.items():
            arrays = {"pesto": rng.random(k).astype("<f4"), "own_edgemax": np.full(k, np.nan, "<f4"), "own_dburial": np.full(k, np.nan, "<f4"),
                      "rsa_free": rng.random(k).astype("<f4"), "chain": (np.arange(k) >= k // 2).astype("|i1")}
            blob = pg.encode(arrays, fields); h.write(blob); offs.append(offs[-1] + len(blob))
    np.savez(path / "index.npz", ids=np.asarray(list(n)), dims=np.asarray([[k] for k in n.values()], np.int32), offsets=np.asarray(offs, np.int64))
    (path / "metadata.json").write_text(json.dumps({"fields": fields, "dtypes": {"pesto": "<f4", "own_edgemax": "<f4", "own_dburial": "<f4", "rsa_free": "<f4", "chain": "|i1"},
                                                    "shapes": {f: ["n"] for f in fields}, "dims": ["n"]}))
    (path / "verification.json").write_text(json.dumps({"passed": True, "seed": seed}))


def roots():
    t = Path(tempfile.mkdtemp()); base, extra = t / "base", t / "extra"
    nb = make_root(base, ["b1", "b2", "b3"], 1); nx = make_root(extra, ["x1", "x2"], 2)
    make_hotspot(t / "hot-base", {"b1": nb["b1"], "b2": nb["b2"]}, 3); make_hotspot(t / "hot-extra", nx, 4)
    return t, base, extra, nb, nx


def same(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return np.asarray(a).dtype == np.asarray(b).dtype and np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def stand_in(base, hot=None):
    s = types.SimpleNamespace(store=pg.ProductionStore(pg.output(base, "pinder") / "store-v2"), sidechains=BackboneTable(base, "pinder"),
                              contacts=ContactTable(base, "contacts-v1"), residues=ResidueTable(base), hotspot=None, strata=None, alt=None)
    from pkatrain.production_gate import DisulfideTable
    s.disulfides = DisulfideTable(base, "pinder")
    if hot is not None: s.hotspot = (HotspotStore(hot), "pesto", 0.5, 0.0, "one", 0.5)
    return s


def reads(s, cid):
    out = {"raw": s.store.raw(cid), "bb": s.sidechains(cid), "contacts": s.contacts(cid), "res": s.residues.row(cid), "flags": s.disulfides.flags(cid)}
    if s.hotspot is not None: out["hot"] = s.hotspot[0].record(cid)
    return out


def test_union_reads():
    t, base, extra, nb, nx = roots()
    plain = stand_in(base, t / "hot-base"); joined = stand_in(base, t / "hot-base")
    prov = px.attach(joined, base, extra, t / "hot-extra")
    assert set(prov["tables_verification_sha256"]) == {"store-v2", "backbone-v1", "contacts-v1", "sidechains-v1", "residue-rsa-v1", "hotspot"}
    for cid in nb:
        a, b = reads(plain, cid), reads(joined, cid)
        for k in a:
            if k == "hot" and a[k] is None: assert b[k] is None; continue
            assert same(a[k], b[k]) if k not in ("flags",) else all(same(x, y) for x, y in zip(a[k], b[k])), (cid, k)
    alone = stand_in(extra, t / "hot-extra")
    for cid in nx:
        a, b = reads(alone, cid), reads(joined, cid)
        for k in a:
            assert same(a[k], b[k]) if k != "flags" else all(same(x, y) for x, y in zip(a[k], b[k])), (cid, k)
    assert joined.hotspot[0].record("b3") is None and joined.hotspot[0].record("nope") is None and joined.hotspot[1:] == plain.hotspot[1:]
    for f in (lambda: joined.store.raw("nope"), lambda: joined.sidechains("nope"), lambda: joined.residues.row("nope"), lambda: joined.contacts("nope")):
        try: f(); raise AssertionError("unknown id read")
        except KeyError: pass
    try: px.attach(stand_in(base), base, base); raise AssertionError("overlap accepted")
    except AssertionError as e: assert "already in the base" in str(e), e
    shutil.rmtree(t)


def test_attach_refusals():
    t, base, extra, nb, nx = roots()
    s = stand_in(base, t / "hot-base")
    try: px.attach(s, base, extra); raise AssertionError("missing hotspot extra accepted")
    except ValueError as e: assert "--mask-store-extra" in str(e)
    s = stand_in(base); s.strata = {"b1": 0}
    try: px.attach(s, base, extra); raise AssertionError("strata accepted")
    except ValueError as e: assert "strata" in str(e)
    s = stand_in(base); s.sidechains = SidechainTable(base, "pinder")
    try: px.attach(s, base, extra); raise AssertionError("sidechains accepted")
    except ValueError as e: assert "backbone table only" in str(e)
    s = stand_in(base); (pg.output(base, "pinder") / "rsa-bound-v1").mkdir();
    for f in ("scores.npy", "offsets.npy", "ids.json"): shutil.copy(pg.output(base, "pinder") / "contacts-v1" / f, pg.output(base, "pinder") / "rsa-bound-v1" / f)
    atomic_json(pg.output(base, "pinder") / "rsa-bound-v1" / "verification.json", {"table": "rsa-bound"}); s.contacts = ContactTable(base, "rsa-bound-v1")
    try: px.attach(s, base, extra); raise AssertionError("rsa-bound accepted")
    except ValueError as e: assert "contacts-v1" in str(e)
    s = stand_in(base)
    try: px.attach(s, base, extra, t / "hot-extra"); raise AssertionError("hotspot extra without a mask store accepted")
    except ValueError as e: assert "--mask-store-extra needs" in str(e)
    try: px.refuse(sidechains=False, alt_source="x"); raise AssertionError
    except ValueError as e: assert "alt_source" in str(e)
    shutil.rmtree(t)


def manifests(base, extra):
    def recs(root, split="train"):
        return [{"id": r["id"], "split": split, **{d: r[d] for d in pg.DIMS}, "w_burial_mean": 0.5, "w_interface_mean": 0.5}
                for r in json.loads((pg.output(root, "pinder") / "store-v2" / "records.json").read_text())]
    b = recs(base); b[2]["split"] = "val"
    for r in b[:2]: r["min_fraction"] = 0.5
    m0 = {"version": pg.VERSION, "dataset": "pinder", "store": str(pg.output(base, "pinder") / "store-v2"),
          "store_verification_sha256": digest(pg.output(base, "pinder") / "store-v2" / "verification.json"),
          "bucket_policy": {"bounds": [16], "batch_sizes": [4]}, "capacities": {"16": [16, 4, 4, 2, 2]}, "records": b}
    xs = pg.output(extra, "pinder") / "store-v2"
    m5 = {**m0, "records": [b[0], {**b[1], "min_fraction": 1.0}, b[2]] + [{**r, "min_fraction": 0.25} for r in recs(extra)],
          "extra_store": str(xs), "extra_store_verification_sha256": digest(xs / "verification.json")}
    return m0, m5


def test_check_manifest():
    t, base, extra, nb, nx = roots(); m0, m5 = manifests(base, extra)
    px.check_manifest(m5, m0, extra)
    bad = json.loads(json.dumps(m5)); bad["records"][2]["n"] += 1
    for name, m in (("validation", bad), ("capacities", {**m5, "capacities": {"16": [32, 4, 4, 2, 2]}}),
                    ("foreign", {**m5, "records": m5["records"] + [{**m5["records"][-1], "id": "zz"}]}), ("store", {**m5, "store": "/elsewhere"}),
                    ("extra", {**m5, "extra_store_verification_sha256": "0" * 64})):
        try: px.check_manifest(m, m0, extra); raise AssertionError(f"{name} accepted")
        except AssertionError as e:
            if "accepted" in str(e): raise
    shutil.rmtree(t)


def test_cli_flags():
    import jax  # noqa: F401  (production_train imports jax at module level)
    try:
        from pkatrain import production_train as pt
    except Exception:  # noqa: BLE001  the Mac may lack optional deps of the trainer; the AST check below still runs
        pt = None
    src = (NEW / "pkatrain/production_train.py").read_text()
    for flag in ("--pinder-manifest", "--pinder-extra", "--mask-store-extra"): assert f'p.add_argument("{flag}"' in src
    assert "args.pinder_manifest, args.pinder_extra, args.mask_store_extra)" in src
    if pt is None: return
    t = Path(tempfile.mkdtemp())
    for kw, msg in (({"pinder_manifest": "x"}, "go together"), ({"pinder_extra": "x"}, "go together"), ({"mask_store_extra": "x"}, "needs --pinder-extra"),
                    ({"pinder_manifest": "x", "pinder_extra": "y", "sidechains": True}, "does not support"),
                    ({"pinder_manifest": "x", "pinder_extra": "y", "mask_store": "z"}, "needs --mask-store-extra")):
        try: pt.train(t, "never", **kw); raise AssertionError(f"{kw} accepted")
        except ValueError as e: assert msg in str(e), (kw, e)
    assert not (t / "training").exists(), "run directory touched before validation"
    shutil.rmtree(t)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]; passed = 0
    for test in tests:
        try: test(); passed += 1; print("PASS", test.__name__, flush=True)
        except Exception: print("FAIL", test.__name__); traceback.print_exc()
    print(f"passed {passed}/{len(tests)}"); sys.exit(0 if passed == len(tests) else 1)
