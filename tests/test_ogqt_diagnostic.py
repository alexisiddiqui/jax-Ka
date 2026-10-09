import numpy as np
import pytest

import jax

from pkanet.site_model import initialize_site, predict_site_shift_indexed
from pkatrain.gqt_diagnostic_campaign import _mask_one, nested_subset
from pkatrain.site_graph_data import build_site_graph, candidate_sites
from test_pkanet import graph as residue_graph


def site_graph():
    graph, backbone = residue_graph()
    candidates = candidate_sites(graph["nodes"])
    queries = candidates[candidates[:, 1] != 6][:4]
    graph["query_residue"] = queries[:, 0]
    graph["query_group"] = queries[:, 1]
    graph.update(build_site_graph(backbone, np.repeat([0, 1], 6), graph["nodes"],
                                  graph["query_residue"], graph["query_group"]))
    return graph


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="indexed attention requires CUDA")
def test_site_dropout_zero_and_shared_branch_rng():
    with jax.experimental.enable_x64(False):
        params = initialize_site(jax.random.PRNGKey(17)); graph = site_graph()
        deterministic = predict_site_shift_indexed(params, graph)
        zero = predict_site_shift_indexed(params, graph, key=jax.random.PRNGKey(8), dropout_rate=0.0)
        np.testing.assert_array_equal(deterministic, zero)
        key = jax.random.PRNGKey(9)
        left = predict_site_shift_indexed(params, graph, key=key, dropout_rate=0.1)
        right = predict_site_shift_indexed(params, graph, key=key, dropout_rate=0.1)
        np.testing.assert_array_equal(left, right)
        changed = predict_site_shift_indexed(params, graph, key=jax.random.PRNGKey(10), dropout_rate=0.1)
        assert np.max(np.abs(np.asarray(left - changed))) > 1e-7


def test_context_mask_is_branch_matched_and_protects_queries():
    one = site_graph(); paired = {name: np.stack((value.copy(), value.copy())) for name, value in one.items()}
    eligible = np.ones(len(one["query_group"]), bool)
    count = _mask_one(paired, eligible, "synthetic", 1, branches=True)
    assert count >= 0
    for name in ("nodes", "edge", "edge_mask", "switch", "site_mask", "site_edge_mask", "site_switch"):
        np.testing.assert_array_equal(paired[name][0], paired[name][1])
    assert paired["site_mask"][0, paired["query_site"][0]].all()


def test_nested_subsets_are_deterministic_and_stratified():
    records = [{"id": f"x{i:03d}", "ctype": "antibody" if i % 4 == 0 else "heteromer"}
               for i in range(100)]
    small = nested_subset(records, 25); medium = nested_subset(records, 50)
    assert {row["id"] for row in small} <= {row["id"] for row in medium}
    assert len(small) == 25 and len(medium) == 50
    assert sum(row["ctype"] == "antibody" for row in small) in (6, 7)

