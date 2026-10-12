import numpy as np
import pytest

from pkatrain import production_ph as ph


def row(cid, value, split="train", parent=1.0, sources=("crystallization_point",)):
    eligible = value is not None
    return {"id": cid, "split": split, "ph_target": value, "ph_mask": eligible, "ph_quality_weight": float(eligible),
            "parent_weight": parent, "ph_sources": list(sources), "ph_warnings": []}


def test_fit_uses_parent_weights_and_ignores_masked():
    stats = ph.fit([row("a", 5.0, parent=0.5), row("b", 5.0, parent=0.5), row("c", 8.0), row("d", None)])
    assert stats["structures"] == 3 and stats["mean"] == pytest.approx(6.5)
    assert len(stats["bin_means"]) == len(ph.THRESHOLDS) + 1


def test_expected_ph_recovers_bin_means_at_confident_logits():
    stats = ph.fit([row(str(i), v) for i, v in enumerate((4.0, 5.5, 6.5, 7.5, 8.0, 9.0))])
    for k, value in enumerate(stats["bin_means"]):
        logits = np.where(np.arange(len(ph.THRESHOLDS)) < k, 30.0, -30.0)
        assert ph.expected_ph(logits, stats) == pytest.approx(value, abs=1e-6)
    # non-monotone survival is clipped, so the bin probabilities stay non-negative
    assert np.isfinite(ph.expected_ph(np.array([-5.0, 5.0, -5.0, 5.0, -5.0]), stats))


def test_loss_masks_structures_and_is_zero_without_targets():
    import jax.numpy as jnp
    logits = jnp.zeros((3, len(ph.THRESHOLDS)))
    assert float(ph.loss(logits, jnp.zeros(3), jnp.zeros(3))) == 0.0
    one = float(ph.loss(logits[:1], jnp.asarray([7.0]), jnp.ones(1)))
    padded = float(ph.loss(logits, jnp.asarray([7.0, 3.0, 11.0]), jnp.asarray([1.0, 0.0, 0.0])))
    assert padded == pytest.approx(one) and one == pytest.approx(np.log(2))


def test_pooled_logits_ignore_padded_sites():
    import jax
    import jax.numpy as jnp
    params = ph.initialize({"head": {"w": jnp.zeros((8, 1))}}, 0)["aux_ph"]
    tokens = jax.random.normal(jax.random.PRNGKey(1), (5, 8)); mask = jnp.asarray([1, 1, 1, 0, 0])
    noisy = tokens.at[3:].set(100.0)
    assert np.allclose(ph.pooled_logits(params, tokens, mask), ph.pooled_logits(params, noisy, mask), atol=1e-5)


def test_table_requires_complete_coverage_and_matching_split(monkeypatch):
    train = {"p1": row("p1", 7.0), "p2": row("p2", None)}; val = {"v1": row("v1", 6.0, split="val")}
    monkeypatch.setattr(ph, "load_v4", lambda root, d: train if d == "pinder" else {"k1": row("k1", 5.0)})
    monkeypatch.setattr(ph, "load_validation", lambda root, c: val if c == "pinder-pool-v4" else {})
    manifest = lambda records: {"records": [{"id": i, "split": s, "min_fraction": 0.0} for i, s in records]}
    manifests = {"pinder": manifest([("p1", "train"), ("p2", "train"), ("v1", "val")]), "pkpdb": manifest([("k1", "train")])}
    table = ph.PhTable("root", manifests, 1.0)
    target, weight = table.arrays("pinder", ["p2", "p1"], 4)
    assert target.tolist() == [0, 7, 0, 0] and weight.tolist() == [0, 1, 0, 0]
    assert table.coverage["pinder_train"] == {"structures": 2, "eligible": 1, "by_source": {"crystal": 1, "em_buffer": 0, "nmr": 0}}
    with pytest.raises(KeyError):
        ph.PhTable("root", {**manifests, "pkpdb": manifest([("k1", "train"), ("k2", "train")])}, 1.0)
    with pytest.raises(ValueError):
        ph.PhTable("root", {**manifests, "pinder": manifest([("p1", "train"), ("v1", "train")])}, 1.0)
