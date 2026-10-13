import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("zstandard")
from pkatrain.augmentation_store import AugmentationStore, ReleaseWriter, encode, decode, finalize_workers
from pkatrain.augmentation_graphs import align_residues, free_graph, pair_inputs, capacities
from pkatrain.site_graph_data import build_site_graph, candidate_sites
from pkanet.graph import geometry


def test_resume_discards_only_uncommitted_tail_and_rejects_changed_recipe(tmp_path):
    data = np.arange(300, dtype=np.float32).reshape(100, 3)
    with ReleaseWriter(tmp_path, {"source": "frozen"}) as writer:
        writer.add("a", {"selected": [2]}, {"coordinates": data[None]})
        writer.checkpoint()
        writer.add("uncommitted", {}, {"coordinates": data[None]})
    with pytest.raises(ValueError, match="provenance"):
        ReleaseWriter(tmp_path, {"source": "changed"})
    with ReleaseWriter(tmp_path, {"source": "frozen"}) as writer:
        assert writer.ids == {"a"}
        writer.add("b", {"selected": [3]}, {"coordinates": (data+1)[None]})
        writer.finish(expected_ids=["a", "b"])
    with AugmentationStore(tmp_path) as store, ThreadPoolExecutor(4) as threads:
        results = list(threads.map(store.raw, ["a", "b"]*8))
        assert len(results) == 16
        np.testing.assert_array_equal(store.conformer("a", 2), data)
        assert not results[0]["arrays"]["coordinates"].flags.writeable
    with pytest.raises(FileExistsError): ReleaseWriter(tmp_path, {"source": "frozen"})


def test_shard_boundaries_corruption_and_incomplete_inventory(tmp_path):
    rng = np.random.default_rng(42); arrays = {"xyz": rng.normal(size=(200, 3)).astype(np.float32)}
    frame, _ = encode({"parent": "a"}, arrays)
    with ReleaseWriter(tmp_path, {}, shard_bytes=len(frame)+100) as writer:
        writer.add("a", {}, arrays); writer.add("b", {}, arrays)
        with pytest.raises(ValueError, match="duplicate"): writer.add("b", {}, arrays)
        with pytest.raises(ValueError, match="inventory"): writer.finish(expected_ids=["a", "b", "c"])
        assert not (tmp_path/"manifest.json").exists()
        writer.finish(expected_ids=["a", "b"])
    assert len(list(tmp_path.glob("records-*.bin"))) == 2
    with AugmentationStore(tmp_path) as store:
        row = store.by_id["a"]
        with (tmp_path/row["shard"]).open("r+b") as f:
            f.seek(row["offset"]+10); f.write(b"BAD")
        with pytest.raises(ValueError, match="digest"): store.raw("a")


def test_decompressed_size_and_whole_file_digest_are_verified(tmp_path):
    frame, size = encode({"parent": "a"}, {"xyz": np.ones((3, 3), np.float32)})
    with pytest.raises(ValueError, match="size"):
        decode(frame, max_output_size=size+1)
    with ReleaseWriter(tmp_path, {}) as writer:
        writer.add("a", {}, {"xyz": np.ones((3, 3), np.float32)}); writer.finish()
    with (tmp_path/"records-00000.bin").open("ab") as f: f.write(b"unindexed tail")
    with AugmentationStore(tmp_path) as store:
        np.testing.assert_array_equal(store.raw("a")["arrays"]["xyz"], np.ones((3, 3), np.float32))
        with pytest.raises(ValueError, match="file digest"): store.verify_files()


def native_fixture():
    from jaxpropka.parameters import THREE_TO_INDEX
    ca = np.array([[0, 0, 0], [3.8, 0, 0], [0, 4, 0], [3.8, 4, 0]], np.float32)
    bb = ca[:, None]+np.array([[-1, 1, 0], [0, 0, 0], [1.5, 0, 0], [1.8, 1, 0]], np.float32)[None]
    chain = np.array([0, 0, 1, 1]); nodes = np.zeros((4, 24), np.float32)
    nodes[:, THREE_TO_INDEX["ASP"]] = 1; nodes[[0, 2], 20] = 1; nodes[[1, 3], 21] = 1
    qr = np.array([0, 2], np.int32); qg = np.array([0, 0], np.int32)
    graph, valid = geometry(bb, chain); nodes[:, 23] = valid
    graph.update(nodes=nodes, node_mask=np.ones(4, bool), query_residue=qr, query_group=qg)
    graph.update(build_site_graph(bb, chain, nodes, qr, qg))
    return graph, bb, chain


def test_deformed_free_branch_has_own_features_and_no_interpartner_edges():
    native, bb, chain = native_fixture(); before = {k: v.copy() for k, v in native.items()}
    moved = bb.copy(); moved[1] += [35, 0, 0]
    free = free_graph(native, moved, chain, chain)
    assert not free["edge_mask"][0, free["neighbors"][0] == 1].any()
    for i, row in enumerate(free["neighbors"]):
        assert np.all(chain[row[free["edge_mask"][i]]] == chain[i])
    paired = pair_inputs(native, free, bb, moved, chain, [0, 1, 0, 1])
    np.testing.assert_array_equal(paired["nodes"][0, :, -14:-2], bb.reshape(4, 12))
    np.testing.assert_array_equal(paired["nodes"][1, :, -14:-2], moved.reshape(4, 12))
    for k, a in before.items(): np.testing.assert_array_equal(native[k], a)
    for k in ("query_residue", "query_group", "site_residue", "site_type", "query_site"):
        np.testing.assert_array_equal(paired[k][0], paired[k][1])
    assert capacities([paired])["128"][1] == paired["neighbors"].shape[-1]


def test_coordinate_mapping_handles_reordering_but_rejects_alias_ambiguity():
    _, bb, _ = native_fixture(); identity = [1, 1, 1, 1]
    np.testing.assert_array_equal(align_residues(bb[[2, 0]], [1, 1], bb, identity), [2, 0])
    duplicate = np.concatenate((bb, bb[:1]))
    with pytest.raises(ValueError, match="ambiguous"):
        align_residues(bb[:1], [1], duplicate, identity+[1])
    np.testing.assert_array_equal(align_residues(bb[:1], [1], duplicate, identity+[1], allowed=[0]), [0])


def test_worker_finalisation_adopts_shards_without_copying(tmp_path):
    workers = [tmp_path/"workers"/str(i) for i in range(2)]
    for i, path in enumerate(workers):
        with ReleaseWriter(path, {"worker": i}) as writer:
            writer.add(str(i), {}, {"xyz": np.ones((3, 3), np.float32)*i}); writer.finish()
    sizes = [(p/"records-00000.bin").stat().st_size for p in workers]
    finalize_workers(tmp_path, workers, {"release": 1}, expected_ids=["1", "0"])
    assert not list(tmp_path.glob("records-*.bin"))
    assert sizes == [(p/"records-00000.bin").stat().st_size for p in workers]
    with AugmentationStore(tmp_path) as store:
        assert [r["id"] for r in store.rows] == ["1", "0"]
        np.testing.assert_array_equal(store.raw("1")["arrays"]["xyz"], np.ones((3, 3), np.float32))


def test_quota_gate_requires_effective_limits_and_staging_headroom():
    from pkatrain.augmentation_preflight import budget
    snapshot = dict(effective_limits_verified=True, used_bytes=700, limit_bytes=1000,
                    used_files=700, limit_files=1000)
    assert budget(snapshot, additional_bytes=100, additional_files=100)["passed"]
    assert not budget(snapshot, additional_bytes=201, additional_files=10)["passed"]
    assert not budget(dict(snapshot, limit_files=0), additional_bytes=10, additional_files=10)["passed"]
    assert not budget(dict(snapshot, effective_limits_verified=False), additional_bytes=10, additional_files=10)["passed"]


def test_original_input_mask_distinguishes_upstream_completion_and_unknown_atoms(tmp_path):
    import gzip
    from pkatrain.augmentation_evidence import upstream_mask
    evidence = dict(atoms=[dict(key=["X", 10, ""], expected_atoms=["CA", "CB"], input_atom_mask=[True, False])])
    with gzip.open(tmp_path/"input_atom_mask.json.gz", "wt") as f: json.dump(evidence, f)
    (tmp_path/"map_A.json").write_text(json.dumps([["A", 1, "X", 10, ""]]))
    record = dict(metadata=dict(source=dict(dataset="pinder", state="A", path=str(tmp_path/"A.cif.gz")),
                     residue_keys=[["X", 10, ""]], atom_names=["CA", "CB", "OXT", "O"]),
                  arrays=dict(atom_residue=np.zeros(4, np.int32), source_present=np.array([True, True, True, False])))
    arrays, _ = upstream_mask(record, None, None)
    np.testing.assert_array_equal(arrays["original_input_present"], [True, False, False, False])
    np.testing.assert_array_equal(arrays["original_input_known"], [True, True, False, False])


def test_worker_budget_obeys_ram_and_cpu_limits():
    from pkatrain.augmentation_report import worker_budget
    assert worker_budget(72, 144*1024**3, 2.05) == 50
    assert worker_budget(8, 144*1024**3, 2.05) == 8
    assert worker_budget(72, 0, 2.05) == 0


def test_production_geometry_features_are_finite_for_both_branches():
    from pkatrain.production_inputs import orientation_edges, backbone_dihedrals
    native, bb, chain = native_fixture(); moved = bb.copy(); moved[1] += [1, 0.1, 0.2]
    free = free_graph(native, moved, chain, chain)
    inputs = pair_inputs(native, free, bb, moved, chain, [0, 1, 0, 1])
    oriented = []
    for branch in range(2):
        table = inputs["nodes"][branch, :, -14:]
        edges = orientation_edges(table, inputs["neighbors"][branch], xp=np)
        angles = backbone_dihedrals(table, xp=np)
        assert np.isfinite(edges).all() and np.isfinite(angles).all()
        oriented.append(edges)
    assert not np.array_equal(oriented[0], oriented[1])
