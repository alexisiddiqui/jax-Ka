import json

import numpy as np
import pytest

from pkatrain import pkai_scratch as ps


def _dense(encoding, n, seed):
    """Dense rows built as feature_matrix fills them: sorted 1/d^2 values in the first m slots, a class/type per slot, the
    site one-hot; some rows with no environment at all."""
    rng = np.random.default_rng(seed); slot = ps.SLOT_WIDTH[encoding]; x = np.zeros((n, ps.feature_width(encoding)), np.float32)
    for i in range(n):
        m = 0 if i % 7 == 3 else int(rng.integers(1, 251))
        value = np.sort(1 / rng.uniform(1, 15, m) ** 2)[::-1].astype(np.float32)
        value[1:3] = value[1] if m > 2 else value[1:3]  # ties
        atom = rng.integers(0, 16, m); aa = rng.integers(0, 20, m); j = np.arange(m) * slot
        if encoding != "aa20": x[i, j + atom] = value
        if encoding != "atom16": x[i, j + (16 if encoding == "atom16aa20" else 0) + aa] = value
        x[i, 250 * slot + rng.integers(0, 8)] = 1
    return x


@pytest.mark.parametrize("encoding", sorted(ps.SLOT_WIDTH))
def test_compact_round_trip(encoding):
    x = _dense(encoding, 40, 1); slots = ps.compact(x, encoding)
    assert tuple(slots) == ps.compact_fields(encoding)
    assert slots["value"].dtype == np.float32 and all(slots[k].dtype == np.uint8 for k in slots if k != "value")
    assert np.array_equal(ps.expand(slots, encoding), x)


@pytest.mark.parametrize("encoding", sorted(ps.SLOT_WIDTH))
def test_compact_rejects_non_slot_form(encoding):
    x = _dense(encoding, 4, 2); x[0, 1] = x[0, 2] = 0.5  # two entries in one slot's atom (or residue) block
    with pytest.raises(ValueError):
        ps.compact(x, encoding)


@pytest.mark.parametrize("encoding", sorted(ps.SLOT_WIDTH))
def test_expand_torch_matches_numpy(encoding):
    torch = pytest.importorskip("torch")
    x = _dense(encoding, 16, 3); slots = ps.compact(x, encoding)
    out = ps.expand_torch(torch, {k: torch.from_numpy(v) for k, v in slots.items()}, encoding)
    assert np.array_equal(out.numpy(), x)


def test_store_pack_select_and_exclusions(tmp_path, monkeypatch):
    pytest.importorskip("zstandard")
    from pkatrain import pkai_joint_scale as js
    monkeypatch.setattr(js, "ENCODING", "atom16aa20"); monkeypatch.setattr(js, "WIDTH", ps.feature_width("atom16aa20"))
    base = js.feature_root(tmp_path, "pkpdb"); base.mkdir(parents=True)
    ids = [[f"{c}{n}", "train"] for c in "abc" for n in range(4)]
    (base / "ids.json").write_text(json.dumps({"encoding": "atom16aa20", "ids": ids}))
    dense = {cid: {"full": _dense("atom16aa20", 3 + k, k), "backbone": _dense("atom16aa20", 3 + k, 100 + k),
                   "target": np.arange(3 + k, dtype=np.float32), "weight": np.ones(3 + k, np.float32)}
             for k, (cid, _) in enumerate(ids)}

    def produce(cid, split, _):
        if cid == "b2": raise ValueError((cid, "no mapped pKPDB sites"))
        return js._compact_record("pkpdb", dense[cid])
    # two shards, one overlapping record (identical content is allowed)
    js._write_shard(base, "0000-of-0002", [(c, s, None) for c, s in ids[0::2]], produce, 6)
    js._write_shard(base, "0001-of-0002", [(c, s, None) for c, s in ids[1::2] + [ids[0]]], produce, 7)
    js.pack_store(tmp_path, "pkpdb", workers=2)
    store_path = base / "store"; meta = json.loads((store_path / "metadata.json").read_text())
    assert not (base / "shards").exists() and meta["records"] == 11 and [f["id"] for f in meta["excluded"]] == ["b2"]
    store = js.FeatureStore(store_path)
    names = ["train-c1", "train-a0"]
    got = store.select(names, [f"full.{f}" for f in ps.compact_fields("atom16aa20")] + ["target"])
    store.close()
    full = ps.expand({f: got[f"full.{f}"] for f in ps.compact_fields("atom16aa20")}, "atom16aa20")
    assert np.array_equal(full, np.concatenate([dense["c1"]["full"], dense["a0"]["full"]]))
    assert np.array_equal(got["target"], np.concatenate([dense["c1"]["target"], dense["a0"]["target"]]))


def test_shard_blocking_failure_is_not_installed(tmp_path, monkeypatch):
    pytest.importorskip("zstandard")
    from pkatrain import pkai_joint_scale as js
    base = tmp_path / "store"

    def produce(cid, split, _): raise KeyError("bug")
    with pytest.raises(RuntimeError):
        js._write_shard(base, "0000-of-0001", [("a", "train", None)], produce, 1)
    assert not (base / "shards" / "0000-of-0001").exists() and (base / "shards" / "0000-of-0001.failures.json").exists()
