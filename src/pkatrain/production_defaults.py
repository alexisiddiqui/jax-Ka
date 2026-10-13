"""Pinned default for frozen GQT representation experiments, separate from the legacy prediction CLI."""
from pathlib import Path
import hashlib
import json
import os

DEFAULT_RUN = "control-v2-50-relations-s17"
DEFAULT_SEED = 17
DEFAULT_FRACTION = 0.5
DEFAULT_EPOCH = 18
DEFAULT_CHECKPOINT_SHA256 = "b79d6e37eaebb7bdbdd73ff1f5af6995a1c6bdf29ad36f93352f0ec455539347"
DEFAULT_EXPORT = "ogqt-relational-50-frozen"


def default_model_path():
    """Repository export, or an explicit directory for installations outside a checkout."""
    override = os.environ.get("GQT_FROZEN_MODEL")
    return Path(override) if override else Path(__file__).resolve().parents[2] / "models" / DEFAULT_EXPORT


def load_frozen_default(path=None, *, input_gradients=False):
    """Return (embeddings(graph), predict_shift(graph), metadata), with outputs detached by default.

    Graphs need the backbone table appended by BackboneTable, as in production
    training. The bound/free graph masks determine permitted information flow.
    With input_gradients=True, use the portable forward and retain derivatives
    with respect to graph inputs (parameters remain captured, not trainable).
    Parameters are deliberately not returned to a head optimizer. Explicit paths
    still must identify the pinned default; choose another loader for other models.
    """
    path = Path(path) if path is not None else default_model_path()
    info = json.loads((path / "model.json").read_text())
    if (info["run"], info["selected_epoch"], info["checkpoint_sha256"]) != (
            DEFAULT_RUN, DEFAULT_EPOCH, DEFAULT_CHECKPOINT_SHA256):
        raise ValueError("Export does not match the pinned frozen relational checkpoint")
    if hashlib.sha256((path / "params.npz").read_bytes()).hexdigest() != info["params_sha256"]:
        raise ValueError("Export parameter checksum mismatch")
    import jax
    from .production_train import load_exported, exported_functions
    params = load_exported(path)
    if input_gradients:
        from pkanet.relational import relational_functions
        embed, predict = relational_functions()
    else:
        embed, predict = exported_functions(path)

    def embeddings(graph):
        value = embed(params, graph)
        return value if input_gradients else jax.tree.map(jax.lax.stop_gradient, value)

    def predict_shift(graph):
        value = predict(params, graph)
        return value if input_gradients else jax.lax.stop_gradient(value)

    return embeddings, predict_shift, info
