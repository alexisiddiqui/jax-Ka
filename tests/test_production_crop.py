import numpy as np
import pytest

pytest.importorskip("scipy")
from pkanet.graph import geometry
from pkatrain import production_crop as pc


def _structure(n=90, sites=24, seed=0, box=45.0):
    """A random backbone with the production residue graph (pkanet.graph.geometry, 20 A) and a site graph over a
    subset of residues; returns (graph, C-alpha coordinates, chain)."""
    rng = np.random.default_rng(seed); ca = rng.random((n, 3)) * box
    backbone = np.stack((ca + rng.normal(0, 1, (n, 3)), ca, ca + rng.normal(0, 1, (n, 3))), axis=1)
    chain = (np.arange(n) >= n // 2).astype(np.int32)
    g, _ = geometry(backbone, chain, radius=20.0); k = g["neighbors"].shape[1]  # pad slots to n (fixed shapes, masked padding)
    g = {name: np.pad(value, ((0, 0), (0, n - k)) + ((0, 0),) * (value.ndim - 2)) for name, value in g.items()}
    residues = np.sort(rng.choice(n, sites, replace=False)); d = np.linalg.norm(ca[residues][:, None] - ca[residues][None], axis=-1)
    graph = {"nodes": np.zeros((n, 24), np.float32), "node_mask": np.ones(n, bool), **g,
             "query_residue": residues.astype(np.int32), "query_group": np.zeros(sites, np.int32),
             "site_residue": residues.astype(np.int32), "site_type": np.zeros(sites, np.int32), "site_mask": np.ones(sites, bool),
             "site_neighbors": np.tile(np.arange(sites, dtype=np.int32), (sites, 1)), "site_edge": np.zeros((sites, sites, 34), np.float32),
             "site_edge_mask": d < 20.0, "site_switch": np.ones((sites, sites), np.float32), "query_site": np.arange(sites, dtype=np.int32)}
    return graph, ca, chain


def _crossing(graph):
    keep = np.asarray(graph["node_mask"], bool); live = np.asarray(graph["edge_mask"], bool)
    sk = np.asarray(graph["site_mask"], bool); sl = np.asarray(graph["site_edge_mask"], bool)
    return (live & ~(keep[:, None] & keep[graph["neighbors"]])).any() or (sl & ~(sk[:, None] & sk[graph["site_neighbors"]])).any()


@pytest.mark.parametrize("radius,inner", [(20.0, 5.0), (15.0, 0.0), (12.0, 6.0)])
def test_crop_matches_true_distances(radius, inner):
    graph, ca, _ = _structure(seed=int(radius))
    original = {k: v.copy() for k, v in graph.items()}; supervised = np.ones(len(graph["query_residue"]), bool); supervised[3] = False
    q = 5; centre = graph["query_residue"][q]; true = np.linalg.norm(ca - ca[centre], axis=-1)
    other = np.arange(len(true)) != centre
    assert not np.any(np.abs(true[other] - radius) < 1e-3) and not np.any(np.abs(true[other] - inner) < 1e-3)
    result, stats = pc.crop(graph, supervised, q, radius, inner)
    keep = np.asarray(graph["node_mask"], bool)
    assert np.array_equal(keep, true <= radius)
    assert np.array_equal(result, (supervised & (true[graph["query_residue"]] <= inner)) | (np.arange(len(result)) == q))
    assert result[q] and not result[3]
    assert not _crossing(graph)
    # every surviving edge is an original edge between kept residues; nothing else changed
    assert np.array_equal(graph["edge_mask"], original["edge_mask"] & keep[:, None] & keep[graph["neighbors"]])
    for name in ("nodes", "neighbors", "edge", "query_residue", "site_residue", "site_neighbors"): assert np.array_equal(graph[name], original[name])
    assert np.array_equal(graph["site_mask"], keep[graph["site_residue"]])
    # each supervised site keeps its full (radius - inner) neighbourhood
    for s in np.flatnonzero(result):
        r = graph["query_residue"][s]; near = np.flatnonzero(np.linalg.norm(ca - ca[r], axis=-1) <= radius - inner)
        assert keep[near].all()
        row = original["edge_mask"][r]; reached = set(original["neighbors"][r][row & (np.linalg.norm(ca[original["neighbors"][r]] - ca[r], axis=-1) <= radius - inner)])
        assert reached <= set(graph["neighbors"][r][graph["edge_mask"][r]])
    assert stats["kept_nodes"] == keep.sum() and stats["nodes"] == len(keep) and stats["kept_supervised"] == result.sum()
    assert stats["supervised"] == supervised.sum()


def test_branches_share_one_crop():
    graph, ca, chain = _structure(seed=3)
    free = {k: v.copy() for k, v in graph.items()}
    free["edge_mask"] = graph["edge_mask"] & (chain[:, None] == chain[graph["neighbors"]])  # partner edges removed
    pair = {k: np.stack((graph[k], free[k])) for k in graph}; before = pair["edge_mask"].copy()
    result, _ = pc.crop(pair, np.ones(len(graph["query_residue"]), bool), 2, 20.0, 5.0, branched=True)
    keep = pair["node_mask"][0].astype(bool)
    assert np.array_equal(pair["node_mask"][0], pair["node_mask"][1]) and np.array_equal(pair["site_mask"][0], pair["site_mask"][1])
    inside = keep[:, None] & keep[graph["neighbors"]]
    assert np.array_equal(pair["edge_mask"][0], before[0] & inside) and np.array_equal(pair["edge_mask"][1], before[1] & inside)
    assert not _crossing({k: v[0] for k, v in pair.items()}) and not _crossing({k: v[1] for k, v in pair.items()})
    assert result[2]


def test_config_and_centre_checks():
    graph, _, _ = _structure(seed=4)
    with pytest.raises(ValueError): pc.check_config(25.0, 5.0, 0.25)
    with pytest.raises(ValueError): pc.check_config(20.0, 21.0, 0.25)
    with pytest.raises(ValueError): pc.crop(graph, np.zeros(len(graph["query_residue"]), bool), 0, 20.0, 5.0)


class _FakeSource:
    """Stacked batches in production_loading's layout: (graphs, targets, mask, valid)."""

    def __init__(self, n_ids=64):
        self.spread = False; self.graphs = {f"s{i}": _structure(seed=100 + i)[0] for i in range(n_ids)}

    def load(self, ids):
        import copy
        graphs = [copy.deepcopy(self.graphs[i]) for i in ids]; q = len(graphs[0]["query_residue"])
        stacked = {k: np.stack([g[k] for g in graphs]) for k in graphs[0]}
        return stacked, np.zeros((len(ids), q), np.float32), np.ones((len(ids), q), bool), np.ones(len(ids), bool)

    def close(self): pass

    def provenance(self): return {"source": "fake"}


def test_cropped_source_is_deterministic_and_spreads_after():
    fake = _FakeSource(); ids = sorted(fake.graphs)
    a = pc.CroppedSource(fake, "pkpdb", 20.0, 5.0, 0.5, seed=17); b = pc.CroppedSource(fake, "pkpdb", 20.0, 5.0, 0.5, seed=17)
    a.set_epoch(1); b.set_epoch(1)
    x, y = a.load(ids), b.load(ids)
    assert np.array_equal(x[2], y[2]) and np.array_equal(x[0]["node_mask"], y[0]["node_mask"])
    cropped = ~x[0]["node_mask"].all(axis=1); stats = a.stats()
    assert stats["cropped"] == cropped.sum() and 0.25 < stats["crop_fraction"] < 0.75 and stats["draws"] == len(ids)
    assert np.array_equal(x[2][~cropped], np.ones_like(x[2][~cropped]))  # uncropped draws keep all their sites
    assert (x[2][cropped].sum(axis=1) >= 1).all() and (x[2][cropped].sum(axis=1) < x[2].shape[1]).any()
    # spread_padding ran after the crop: masked slots point at spread rows, live slots unchanged
    raw = fake.load(ids)[0]; live = x[0]["edge_mask"]
    assert np.array_equal(x[0]["neighbors"][live], raw["neighbors"][live]) and not np.array_equal(x[0]["neighbors"], raw["neighbors"])
    b.set_epoch(2); z = b.load(ids)
    assert not np.array_equal(z[0]["node_mask"], x[0]["node_mask"])
    with pytest.raises(AssertionError): pc.CroppedSource(fake, "pkpdb", 20.0, 5.0, 0.5, seed=17).load(ids)
    fake.spread = True
    with pytest.raises(ValueError): pc.CroppedSource(fake, "pkpdb", 20.0, 5.0, 0.5, seed=17)
