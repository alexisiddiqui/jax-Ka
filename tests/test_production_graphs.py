import json

import numpy as np
import pytest

zstandard = pytest.importorskip("zstandard")
from pkatrain import production_graphs as pg


def _fake(dataset, cid, split):
    rng = np.random.default_rng(abs(hash(cid)) % 2**32); dims = {"n": 5 + len(cid), "k": 3, "q": 2, "s": 4, "sk": 2}
    arrays = {}
    for f in pg.FIELDS[dataset]:
        s = pg.shape(f, *(dims[d] for d in pg.DIMS))
        arrays[f] = (rng.random(s) > .5) if f.endswith("mask") or f == "interface" else \
            rng.integers(0, 5, s).astype(np.int32) if f in ("neighbors", "query_residue", "query_group", "site_residue",
                                                             "site_type", "site_neighbors", "query_site") else rng.random(s).astype(np.float32)
    if cid == "skipme": return None, {"id": cid, "split": split, "reason": "no sites"}
    return arrays, {"id": cid, "split": split, **dims, "arrays_sha256": pg.arrays_digest(arrays, pg.FIELDS[dataset])}


@pytest.mark.parametrize("dataset", pg.DATASETS)
def test_build_pack_read_roundtrip(tmp_path, monkeypatch, dataset):
    monkeypatch.setattr(pg, "build_one", lambda root, d, cid, split: _fake(d, cid, split))
    out = pg.output(tmp_path, dataset); out.mkdir(parents=True)
    listing = [[f"id{i}", "train"] for i in range(9)] + [["skipme", "train"], ["v1", "val"]]
    (out / "ids.json").write_text(json.dumps({"ids": listing}))
    for task in range(3): pg.build_shard(tmp_path, dataset, task, 3)
    assert pg.build_shard(tmp_path, dataset, 0, 3)["records"]  # finished shard is reused
    report = pg.pack(tmp_path, dataset, workers=2)
    assert report["passed"] and report["records_checked"] == 10
    store = pg.ProductionStore(pg.output(tmp_path, dataset) / "store-v2")
    assert list(store.ids) == [cid for cid, _ in listing if cid != "skipme"] and store.compressed
    for cid, split in listing:
        if cid == "skipme": continue
        expected, _ = _fake(dataset, cid, split); got = store.raw(cid)
        assert all(np.array_equal(expected[f], got[f]) and expected[f].dtype == got[f].dtype for f in pg.FIELDS[dataset])
    assert store.metadata["skipped"][0]["id"] == "skipme" and store.metadata["compressed_bytes"] < store.metadata["raw_bytes"]
    store.close()
