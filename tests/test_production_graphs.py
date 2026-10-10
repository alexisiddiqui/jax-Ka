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


@pytest.mark.parametrize("dataset", ("pkpdb", "pinder"))  # benchmark-val has no training pool
def test_production_loader_manifest_and_batches(tmp_path, monkeypatch, dataset):
    pytest.importorskip("jax")
    from pkatrain import production_loading as pl
    monkeypatch.setattr(pg, "build_one", lambda root, d, cid, split: _fake(d, cid, split))
    out = pg.output(tmp_path, dataset); out.mkdir(parents=True)
    listing = [[f"id{i}", "train"] for i in range(9)] + [["v1", "val"]]
    (out / "ids.json").write_text(json.dumps({"ids": listing}))
    for task in range(2): pg.build_shard(tmp_path, dataset, task, 2)
    pg.pack(tmp_path, dataset, workers=2)
    pool = tmp_path / (pg.PKPDB if dataset == "pkpdb" else pg.PINDER) / "pool-v3.tsv"; pool.parent.mkdir(parents=True)
    pool.write_text("id\tmin_fraction\n" + "".join(f"id{i}\t{(i + 1) / 10}\n" for i in range(9)))
    manifest = pl.build_manifest(tmp_path, dataset, workers=2)
    assert len(pl.select(manifest, "train", 0.5)) == 5 and len(pl.select(manifest, "train")) == 9 and len(pl.select(manifest, "val")) == 1
    source = pl.PinderSource(manifest, fraction=0.5) if dataset == "pinder" else pl.PkpdbSource(manifest)
    if dataset == "pinder":
        expected = np.mean([np.mean(_fake(dataset, f"id{i}", "train")[0]["w_burial"]) for i in range(5)])
        assert np.isclose(source.norms["burial"], expected)
    ids = ["id0", "id3", "id5"]; batch = source.load(ids); valid = batch[-1]
    assert valid.tolist() == [True] * 3 + [False] * (len(valid) - 3) and len(valid) == source.policy.batch_size("128")
    for slot, cid in enumerate(ids):
        raw, record = _fake(dataset, cid, "train"); n, q = record["n"], record["q"]
        nodes = batch[0]["nodes"][slot][0] if dataset == "pinder" else batch[0]["nodes"][slot]
        assert np.array_equal(nodes[:n], raw["nodes"]) and not nodes[n:].any()
        if dataset == "pkpdb": assert np.array_equal(batch[1][slot][:q], raw["labels"]) and np.array_equal(batch[2][slot][:q], raw["train_mask"])
        else: assert np.array_equal(batch[1][slot][:, :q], raw["targets"]) and batch[2][slot][:q].all() and not batch[2][slot][q:].any()
    assert not batch[2][3:].any() and not batch[1][3:].any()  # padded slots carry no supervision
    source.close()


def test_spread_padding_changes_only_masked_indices():
    from pkatrain.production_loading import spread_padding
    rng = np.random.default_rng(0); n, k, s, sk = 6, 4, 5, 3
    edge_mask = rng.random((2, n, k)) < 0.5; edge_mask[:, 4:] = False          # two padded nodes, both branches
    site_mask = np.array([True, True, True, False, False])
    graph = {"neighbors": np.where(edge_mask, rng.integers(0, 4, (2, n, k)), 0).astype(np.int32), "edge_mask": edge_mask,
             "site_neighbors": np.zeros((2, s, sk), np.int32), "site_edge_mask": np.zeros((2, s, sk), bool),
             "site_residue": np.stack([np.array([1, 2, 3, 0, 0], np.int32)] * 2), "site_mask": np.stack([site_mask] * 2),
             "node_mask": np.stack([np.arange(n) < 4] * 2)}
    graph["site_edge_mask"][:, :3, 0] = True; graph["site_neighbors"][:, :3, 0] = [2, 0, 1]
    before = {key: value.copy() for key, value in graph.items()}
    after = spread_padding(graph)
    assert np.array_equal(after["neighbors"][edge_mask], before["neighbors"][edge_mask])
    own = np.broadcast_to(np.arange(n)[:, None], (2, n, k))
    assert np.array_equal(after["neighbors"][~edge_mask], own[~edge_mask])
    assert np.array_equal(after["site_neighbors"][:, :3, 0], before["site_neighbors"][:, :3, 0])
    assert (after["site_neighbors"][:, 3:] == np.arange(3, 5)[:, None]).all()
    assert np.array_equal(after["site_residue"][:, :3], before["site_residue"][:, :3])
    assert np.array_equal(after["site_residue"][:, 3:], [[3, 0], [3, 0]])     # 3 % 4, 4 % 4
    assert after["neighbors"].dtype == np.int32 and after["site_residue"].dtype == np.int32
