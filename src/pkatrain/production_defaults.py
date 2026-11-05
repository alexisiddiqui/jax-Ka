"""Pinned default GQT checkpoints.

Default (2026-10-21): the all-residue model ``ogqt-allres15-siamese-50`` (pkanet.allres): every residue a pKa token
(side chain, N-terminus, C-terminus outputs), titratable gate, burial and residue contact-edge heads; backbone and
sequence only; 50% training fraction, seed 17, selected epoch 17.

Alternative (2026-10-31): ``ogqt-allres15-within-hmask-50`` (HMASK_*), the same recipe plus a within-complex listwise
partner-ranking term on the residue edge head and PeSTo hotspot-mask augmentation during training (the exported model
runs unmasked). It is not the default; select it with ``load_default("ogqt-allres15-within-hmask-50")``,
``GQT_DEFAULT_MODEL`` or ``ogqt predict --model``. ``load_default`` accepts the pinned all-residue exports. It
replaces the decoy-ranking export (2026-10-29), withdrawn because its gain was a family-similarity shortcut.

Alternative (2026-11-02): ``ogqt-allres15-within-drsa-50`` (seed 17), with ``-s29`` and ``-s43`` replicas,
uses dRSA-weighted within-complex ranking with the same hotspot masking. These improve interface identification;
large binding-shift improvements are inconsistent across seeds. Native RSA and PeSTo are training targets only.

Alternative (2026-11-05): ``ogqt-allres15-within-drsa-100-w128`` (DRSA100_W128_*), the dRSA recipe trained on the full
(100%) training pool at width 128 (ff 256; 693,856 parameters), seed 17, selected epoch 20. Same heads and inputs
(backbone and sequence only); selectable, not the default.

The previous default, the frozen relational backbone ``ogqt-relational-50-frozen`` (2026-10-13), stays available
through ``load_relational_frozen`` / ``RELATIONAL_*`` for the frozen-representation experiments built on it.
"""
from pathlib import Path
import hashlib
import json
import os

DEFAULT_RUN = "control-v2-50-relations-4-1-allres-cut15-cysmask-gate01-ressiamese-redge01-s17"
DEFAULT_SEED = 17
DEFAULT_FRACTION = 0.5
DEFAULT_EPOCH = 17
DEFAULT_CHECKPOINT_SHA256 = "395beb5c43382987e5ca69ecb12cc78eb3577a83c2f55145489211f444cedbfd"
DEFAULT_EXPORT = "ogqt-allres15-siamese-50"

HMASK_RUN = "hmask-v2-50-within03-asym50-w064-s17"
HMASK_EPOCH = 18
HMASK_CHECKPOINT_SHA256 = "9fe46802253020c1691fc0e88d1725e5507d6855424753a1df7718009bfde1da"
HMASK_EXPORT = "ogqt-allres15-within-hmask-50"

DRSA_EXPORT = "ogqt-allres15-within-drsa-50"
DRSA_RUN = "drsa-v1-50-within03-asym50-w064-s17"
DRSA_EPOCH = 18
DRSA_CHECKPOINT_SHA256 = "703e17a7e2fcb109d44e510eb58db802f4f0c1cf2a3e994fad73d1d7a2427607"
DRSA_EXPORTS = {
    DRSA_EXPORT: (DRSA_RUN, DRSA_EPOCH, DRSA_CHECKPOINT_SHA256),
    DRSA_EXPORT + "-s29": ("drsa-v1-50-within03-asym50-w064-s29", 20,
                           "1de40443be8dcfc7e9294821a133c6ef794095481dbd481421b24adfa0dc12d2"),
    DRSA_EXPORT + "-s43": ("drsa-v1-50-within03-asym50-w064-s43", 16,
                           "d457cd9a6edaa0b619d0cfa063415d4a6e0c2d46f5da96ca4ecc5d8dce8ffc63"),
}

DRSA100_W128_EXPORT = "ogqt-allres15-within-drsa-100-w128"
DRSA100_W128_RUN = "drsa100-v1-asym50-w128-s17"
DRSA100_W128_EPOCH = 20
DRSA100_W128_CHECKPOINT_SHA256 = "50d53402c43c7e330e9db9e3e0ee86f00450236b04247f9f82e8a51da45ab25b"
DRSA100_W128_PARAMETERS = 693856

# pinned all-residue exports load_default accepts: export name -> (run, selected epoch, checkpoint sha256)
ALLRES_EXPORTS = {DEFAULT_EXPORT: (DEFAULT_RUN, DEFAULT_EPOCH, DEFAULT_CHECKPOINT_SHA256),
                  HMASK_EXPORT: (HMASK_RUN, HMASK_EPOCH, HMASK_CHECKPOINT_SHA256), **DRSA_EXPORTS,
                  DRSA100_W128_EXPORT: (DRSA100_W128_RUN, DRSA100_W128_EPOCH, DRSA100_W128_CHECKPOINT_SHA256)}

RELATIONAL_RUN = "control-v2-50-relations-s17"
RELATIONAL_EPOCH = 18
RELATIONAL_CHECKPOINT_SHA256 = "b79d6e37eaebb7bdbdd73ff1f5af6995a1c6bdf29ad36f93352f0ec455539347"
RELATIONAL_EXPORT = "ogqt-relational-50-frozen"

_MODELS = Path(__file__).resolve().parents[2] / "models"


def resolve_model(name_or_path):
    """An export directory: an existing path as given, else a name under the repository's models/ directory."""
    path = Path(name_or_path)
    if not path.exists() and len(path.parts) == 1 and (_MODELS / path).is_dir(): return _MODELS / path
    return path


def default_model_path():
    """Repository export of the default model, or GQT_DEFAULT_MODEL (a path, or a name under models/)."""
    override = os.environ.get("GQT_DEFAULT_MODEL")
    return resolve_model(override) if override else _MODELS / DEFAULT_EXPORT


def relational_model_path():
    """Repository export of the frozen relational backbone, or GQT_FROZEN_MODEL."""
    override = os.environ.get("GQT_FROZEN_MODEL")
    return Path(override) if override else _MODELS / RELATIONAL_EXPORT


def _check(path, *pinned):
    """model.json of the export at path, which must be one of the pinned (run, epoch, checkpoint sha256) identities."""
    info = json.loads((path / "model.json").read_text())
    if (info["run"], info["selected_epoch"], info["checkpoint_sha256"]) not in pinned:
        raise ValueError(f"Export at {path} is not a pinned checkpoint ({', '.join(p[0] for p in pinned)})")
    if hashlib.sha256((path / "params.npz").read_bytes()).hexdigest() != info["params_sha256"]:
        raise ValueError("Export parameter checksum mismatch")
    return info


class DefaultModel:
    """The default all-residue GQT with every head (pkanet.allres).

    outputs(graph) -> dict: query_shift / query_pka (per query site), and per residue (N, 3) shift / pka / gate_logit
    (side chain, N-terminus, C-terminus), (N,) burial_logit, (N, W) edge_embedding; pair_logits(z, a, b) scores
    residue contacts from edge embeddings. Graphs carry the backbone table (pkanet.allres.AllResidueStructureFeatures,
    or pkanet.allres.graph_from_features for cached files); the free state is pkanet.allres.state_graph(graph, "free").
    Outputs are detached unless input_gradients=True (parameters stay fixed either way).
    """

    def __init__(self, path=None, *, input_gradients=False):
        import jax
        from pkanet import allres
        self.path = resolve_model(path) if path is not None else default_model_path()
        self.info = _check(self.path, *ALLRES_EXPORTS.values())
        self.params, _ = allres.load(self.path); self._embeddings = allres.embedding_function(self.info)
        detach = (lambda v: v) if input_gradients else (lambda v: jax.tree.map(jax.lax.stop_gradient, v))
        self.outputs = lambda graph: detach(allres.outputs(self.params, self._embeddings, graph))
        self.embeddings = lambda graph: detach(self._embeddings(self.params, graph))
        self.predict_shift = lambda graph: self.outputs(graph)["query_shift"]
        self.pair_logits = lambda z, a, b: allres.pair_logits(self.params, z, a, b)


def load_default(path=None, *, input_gradients=False):
    """The default all-residue model with all heads (DefaultModel); path may name another pinned all-residue export
    (ALLRES_EXPORTS, e.g. "ogqt-allres15-within-hmask-50") or give its directory."""
    return DefaultModel(path, input_gradients=input_gradients)


def load_frozen_default(path=None, *, input_gradients=False):
    """(embeddings(graph), predict_shift(graph), metadata) of the default model.

    embeddings returns (tokens (3N, W): side-chain, N-terminus, C-terminus rows; query rows); predict_shift the shift of
    each query site from its group reference. For the previous frozen relational backbone use load_relational_frozen.
    """
    model = load_default(path, input_gradients=input_gradients)
    return model.embeddings, model.predict_shift, model.info


def load_relational_frozen(path=None, *, input_gradients=False):
    """(embeddings(graph), predict_shift(graph), metadata) of the frozen relational backbone (previous default).

    Graphs need the backbone table appended by BackboneTable, as in production training. With input_gradients=True,
    use the portable forward and retain derivatives with respect to graph inputs (parameters remain captured).
    """
    path = Path(path) if path is not None else relational_model_path()
    info = _check(path, (RELATIONAL_RUN, RELATIONAL_EPOCH, RELATIONAL_CHECKPOINT_SHA256))
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
