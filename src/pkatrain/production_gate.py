"""Titratable gate on all-residue tokens and bonded-cysteine masking (2026-10-19, user: "complete the all-site conversion
and cys-bond masking"; gate design agreed 2026-10-17: binary gate, trained jointly with its own BCE, labels from
chemistry, pKa loss masked by the true labels, cysteines up-weighted, termini as the token's extra outputs).

Disulfides (DisulfideTable): a cysteine is bonded when its SG lies within 2.3 A of another cysteine's SG (jaxpropka
load_topology's rule), from the deposited side chains (sidechains-v1 SG slot, decoded with backbone-v1 frames). Computed
on the fly in the loader (no new files). The PINDER free branch counts only partners on the same chain index.

--drop-bonded-cys (training sources only): Cys queries (group 3) on bonded cysteines leave the pKa loss (PINDER mask:
AB and free together, from the bound structure; pKPDB eligible). Validation and selection are unchanged, so runs stay
comparable with earlier ones; score_gate.py reports validation with and without those sites.

--gate-weight W (needs --all-residue-tokens and --aux-weight): head aux_gate (one logit) on each of the 3N all-residue
tokens (side chain, N-terminus, C-terminus outputs), W x weighted BCE against
  side chain: residue type in D/E/H/C/Y/K/R and not a bonded cysteine (that branch's disulfide flag);
  N-terminus: node column 20; C-terminus: node column 21 (load_topology's chain termini);
weights 1, cysteine side-chain rows --gate-cys-weight (default 10); per structure, then over structures (both PINDER
branches averaged). The gate never multiplies the pKa outputs in training.
"""
from __future__ import annotations

import numpy as np

from .production_inputs import BackboneTable
from .production_sidechains import SidechainTable

SS_CUTOFF = 2.3
CYS_GROUP = 3


class DisulfideTable:
    """flags(cid) -> (bonded to any cysteine, bonded within the same chain index), per node."""

    def __init__(self, root, dataset):
        from pkanet.graph import SIDECHAIN_ATOMS
        self.sidechains = SidechainTable(root, dataset); self.backbone = BackboneTable(root, dataset); self.slot = SIDECHAIN_ATOMS.index("SG")
        self.verification_sha256 = "disulfide(" + self.sidechains.verification_sha256 + "+" + self.backbone.verification_sha256 + ")"

    def flags(self, cid):
        table = self.backbone(cid); side = self.sidechains(cid).reshape(len(table), -1, 4)[:, self.slot]
        n = len(table); bonded = np.zeros(n, bool); own = np.zeros(n, bool); rows = np.flatnonzero(side[:, 3] > 0.5)
        if len(rows) < 2: return bonded, own
        nn, ca, c = table[rows, 0:3], table[rows, 3:6], table[rows, 6:9]
        x = c - ca; x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
        y = nn - ca; y = y - (y * x).sum(-1, keepdims=True) * x; y = y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-6)
        sg = ca + np.einsum("nij,nj->ni", np.stack((x, y, np.cross(x, y)), axis=-1), side[rows, :3] * 10)
        d = np.linalg.norm(sg[:, None] - sg[None], axis=-1); np.fill_diagonal(d, np.inf); close = d < SS_CUTOFF
        same = table[rows, 12][:, None] == table[rows, 12][None]
        bonded[rows] = close.any(1); own[rows] = (close & same).any(1)
        return bonded, own


def bonded_queries(raw, bonded):
    """Queries that are the side chain of a bonded cysteine."""
    return (np.asarray(raw["query_group"]) == CYS_GROUP) & bonded[np.asarray(raw["query_residue"], int)]


def initialize(params, seed):
    import jax
    import jax.numpy as jnp
    width = params["head"]["w"].shape[0]; key = jax.random.fold_in(jax.random.PRNGKey(seed), 31337)
    return {**params, "aux_gate": {"w": jax.random.normal(key, (width, 1), jnp.float32) / jnp.sqrt(float(width)), "b": jnp.zeros(1, jnp.float32)}}


def targets(graph, cys_weight):
    """(labels, weights), each (3N,) in all-residue token order (side chains, N-termini, C-termini)."""
    import jax.numpy as jnp
    from jaxpropka.parameters import GROUP_AA
    nodes = graph["nodes"]; aa = jnp.argmax(nodes[:, :20], axis=1); valid = graph["node_mask"].astype(jnp.float32)
    titratable = jnp.zeros(20, bool).at[jnp.asarray(GROUP_AA)].set(True)[aa]; cys = aa == int(GROUP_AA[CYS_GROUP])
    side = titratable & (graph["disulfide"] < 0.5)
    labels = jnp.concatenate((side, nodes[:, 20] > 0.5, nodes[:, 21] > 0.5)).astype(jnp.float32)
    weights = jnp.concatenate((valid * jnp.where(cys, cys_weight, 1.0), valid, valid))
    return labels, weights


def logits(params, tokens):
    from pkanet.model import linear
    return linear(params["aux_gate"], tokens)[:, 0]


def provenance(root, gate_weight, cys_weight, drop):
    out = {}
    if drop: out["drop_bonded_cys"] = "training pKa loss excludes Cys queries whose SG is within 2.3 A of another SG (sidechains-v1, bound structure; PINDER AB and free together); validation unchanged (production_gate)"
    if gate_weight:
        out["titratable_gate"] = {"weight": float(gate_weight), "cys_weight": float(cys_weight),
                                  "definition": "aux_gate logit on every all-residue token (side chain / N-terminus / C-terminus); BCE vs D/E/H/C/Y/K/R minus bonded Cys (per branch; free = same chain), node columns 20 / 21; per-structure weighted mean, PINDER branches averaged (production_gate)"}
    if out: out["disulfide_tables"] = {d: DisulfideTable(root, d).verification_sha256 for d in ("pinder", "pkpdb", "benchmark-val")}
    return out
