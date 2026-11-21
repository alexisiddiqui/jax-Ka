"""Per-residue burial and residue contact edges on all-residue tokens (2026-10-19, user: "Build the full per-residue
bound/free RSA table for all PINDER complexes, then train burial on all residues and burial + residue edges on the default
recipe, seed 17").

Table residue-rsa-v1 (<pinder output>/residue-rsa-v1, every manifest record; `python -m pkatrain.production_residue build`
under scripts/sqfs_run.sh; definitions of residue-edges-v1/residue_pairs.py, which it reproduces on the shared complexes):
  rsa.npy        (nodes, 2) float32 relative SASA: column 0 bound (AB together), column 1 free (each chain alone); biotite
                 Shrake-Rupley, ProtOr radii, 100 points, Tien et al. 2013 maxima; NaN for a node without heavy atoms
  chain.npy      (nodes,) int16 chain code (0, 1, ... in sorted chain-id order)
  offsets.npy, ids.json
  pair_offsets.npy, pair_i.npy, pair_j.npy (int32 node indices, i < j, different chains), pair_d.npy (float32 minimum
                 heavy-atom distance, every cross-chain pair <= 10 A in the bound complex)
  verification.json
Graph nodes = load_topology(AB.cif.gz, gap cap, frozen disulfides) keys, asserted equal to the store's node count.

--residue-burial (all-residue tokens, AuxEngine burial): on PINDER the burial head scores every residue token (side-chain
rows) of the free branch against 1 - clip(rsa_free) (cross-entropy, aux weight), replacing the query-site burial term;
pKPDB keeps its query-site burial (no per-residue RSA there).
--residue-edge-weight W: aux_redge, a symmetric pair scorer on layer-normalised residue tokens (features z_a * z_b,
|z_a - z_b|; GELU hidden layer of width W_model), trained on both branches with the residue-edges-v1 recipe: per training
complex 256 pairs of surface residues (free RSA >= 0.1) on different chains, half from the contacts, half uniform over all
surface cross pairs; soft target min(1, exp(-(d - 4) / 3)) within 10 A, else 0; loss = (free BCE + bound BCE) / 2 +
BCE(free logits, sigmoid(stop_gradient(bound logits))) (Siamese distillation), per complex then over complexes, x W.
Every complex counts (no crystallisation-pH requirement: pH conditioning added nothing, 2026-10-14) and there are no
charge features. Validation sources carry the RSA (scoring) but no sampled pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest
from .production_graphs import ProductionStore, output, read

TABLE = "residue-rsa-v1"
CUTOFF_A = 10.0
SURFACE_RSA = 0.1
PAIRS = 256
SALT = 4099
MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
           "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
W = {}


def soft(distance): return np.minimum(1.0, np.exp(-(np.asarray(distance, float) - 4.0) / 3.0)).astype(np.float32)


# ---------------------------------------------------------------- table
def _init(store_path): W["store"] = ProductionStore(Path(store_path))


def _one(cid):
    import biotite.structure as struc
    from scipy.spatial import cKDTree
    from jaxpropka.topology import load_topology
    from .gqt_paired_pinder import _read_cif_gz
    from .production_graphs import PINDER, source
    root = Path(os.environ["PKABENCH_RUNTIME"]); atoms = _read_cif_gz(source(root, f"{PINDER}/entries/{cid}") / "AB.cif.gz")
    keys = load_topology(atoms, gap_policy="cap", freeze_disulfides=True).keys; n = int(W["store"].raw(cid)["nodes"].shape[0])
    if n != len(keys): raise AssertionError((cid, "topology keys differ from graph nodes", n, len(keys)))
    lookup = {(k.chain, k.number, k.insertion): i for i, k in enumerate(keys)}
    heavy = atoms[(atoms.element != "H") & (atoms.element != "D") & ~atoms.hetero]
    node = np.array([lookup.get((str(c), int(r), str(i).strip()), -1) for c, r, i in zip(heavy.chain_id, heavy.res_id, heavy.ins_code)], np.int64)
    heavy = heavy[node >= 0]; node = node[node >= 0]
    chain = np.unique(np.array([k.chain for k in keys]), return_inverse=True)[1].astype(np.int16)
    names = {}
    for i, r in zip(node, heavy.res_name): names.setdefault(int(i), str(r))

    def rsa(selection):
        total = np.zeros(n); idx = node[selection]; out = np.full(n, np.nan, np.float32)
        np.add.at(total, idx, np.nan_to_num(struc.sasa(heavy[selection], vdw_radii="ProtOr", point_number=100)))
        for i in np.unique(idx): out[i] = total[i] / MAX_ASA.get(names[int(i)], 200)
        return out
    bound = rsa(np.ones(len(heavy), bool)); free = np.full(n, np.nan, np.float32)
    for c in np.unique(heavy.chain_id):
        m = heavy.chain_id == c; part = rsa(m); rows = np.unique(node[m]); free[rows] = part[rows]
    tree = cKDTree(heavy.coord.astype(np.float64)); ap = tree.query_pairs(CUTOFF_A, output_type="ndarray")
    a, b = node[ap[:, 0]], node[ap[:, 1]]; cross = chain[a] != chain[b]
    d = np.linalg.norm(heavy.coord[ap[cross, 0]] - heavy.coord[ap[cross, 1]], axis=1); a, b = a[cross], b[cross]
    lo, hi = np.minimum(a, b), np.maximum(a, b); key = lo * n + hi
    uniq, inv = np.unique(key, return_inverse=True); mind = np.full(len(uniq), np.inf); np.minimum.at(mind, inv, d)
    return cid, np.stack((bound, free), axis=1), chain, (uniq // n).astype(np.int32), (uniq % n).astype(np.int32), mind.astype(np.float32)


def build(root, workers=70, limit=None):
    root = Path(root); manifest = read(output(root, "pinder") / "manifest-v1.json"); ids = [r["id"] for r in manifest["records"]][:limit]
    out = output(root, "pinder") / (TABLE + ("-smoke" if limit else "")); out.mkdir(parents=True, exist_ok=True); began = time.time()
    with Pool(workers, initializer=_init, initargs=(manifest["store"],)) as pool: done = pool.map(_one, ids, chunksize=4)
    offsets = np.cumsum([0] + [len(r[1]) for r in done]).astype(np.int64); pair_offsets = np.cumsum([0] + [len(r[3]) for r in done]).astype(np.int64)
    rsa = np.concatenate([r[1] for r in done]); np.save(out / "rsa.npy", rsa); np.save(out / "chain.npy", np.concatenate([r[2] for r in done]))
    np.save(out / "offsets.npy", offsets); np.save(out / "pair_offsets.npy", pair_offsets)
    for k, name in ((3, "pair_i"), (4, "pair_j"), (5, "pair_d")): np.save(out / f"{name}.npy", np.concatenate([r[k] for r in done]))
    atomic_json(out / "ids.json", ids); files = ["rsa.npy", "chain.npy", "offsets.npy", "ids.json", "pair_offsets.npy", "pair_i.npy", "pair_j.npy", "pair_d.npy"]
    summary = {"version": TABLE, "definition": __doc__.split("\n\n")[1].replace("\n", " "), "structures": len(ids), "nodes": int(offsets[-1]),
               "pairs_10A": int(pair_offsets[-1]), "nan_free": int(np.isnan(rsa[:, 1]).sum()), "nan_bound": int(np.isnan(rsa[:, 0]).sum()),
               "bound_above_free": int(np.sum(rsa[:, 0] > rsa[:, 1] + 1e-6)), "buried_on_binding_gt_0.05": float(np.nanmean((rsa[:, 1] - rsa[:, 0]) > 0.05)),
               "manifest_sha256": digest(output(root, "pinder") / "manifest-v1.json"), "files": {f: digest(out / f) for f in files},
               "seconds": round(time.time() - began, 1), "test_data_included": False}
    atomic_json(out / "verification.json", summary); return summary


class ResidueTable:
    """Per-residue RSA (bound, free), chain codes and cross-chain contacts by complex id."""

    def __init__(self, root, table=TABLE):
        path = output(root, "pinder") / table; self.verification_sha256 = digest(path / "verification.json")
        self.index = {cid: i for i, cid in enumerate(read(path / "ids.json"))}; self.offsets = np.load(path / "offsets.npy")
        self.rsa = np.load(path / "rsa.npy", mmap_mode="r"); self.chain = np.load(path / "chain.npy", mmap_mode="r")
        self.pair_offsets = np.load(path / "pair_offsets.npy")
        self.i, self.j, self.d = (np.load(path / f"pair_{k}.npy", mmap_mode="r") for k in ("i", "j", "d"))

    def row(self, cid):
        k = self.index[cid]; a, b = self.offsets[k], self.offsets[k + 1]; p, q = self.pair_offsets[k], self.pair_offsets[k + 1]
        return {"rsa": np.asarray(self.rsa[a:b], np.float32), "chain": np.asarray(self.chain[a:b], np.int64),
                "i": np.asarray(self.i[p:q], np.int64), "j": np.asarray(self.j[p:q], np.int64), "d": np.asarray(self.d[p:q], np.float32)}


def surface_pairs(r):
    """Every cross-chain pair of surface residues (free RSA >= 0.1), a < b."""
    surf = np.flatnonzero(np.nan_to_num(r["rsa"][:, 1], nan=0.0) >= SURFACE_RSA); c = r["chain"][surf]; pa, pb = [], []
    codes = np.unique(c)
    for x in range(len(codes)):
        for y in range(x + 1, len(codes)):
            sa, sb = surf[c == codes[x]], surf[c == codes[y]]; pa.append(np.repeat(sa, len(sb))); pb.append(np.tile(sb, len(sa)))
    if not pa: return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a, b = np.concatenate(pa), np.concatenate(pb); return np.minimum(a, b), np.maximum(a, b)


def pair_targets(r, a, b):
    """Soft targets for node pairs (0 where the pair is not a <= 10 A contact)."""
    n = len(r["rsa"]); key = r["i"] * n + r["j"]; q = np.minimum(a, b) * n + np.maximum(a, b)
    if not len(key): return np.zeros(len(a), np.float32)
    pos = np.minimum(np.searchsorted(key, q), len(key) - 1); return np.where(key[pos] == q, soft(r["d"][pos]), 0.0).astype(np.float32)


def sample_pairs(r, rng, size=PAIRS):
    """(a, b, target, weight) for one training complex: half contacts between surface residues, half uniform surface pairs."""
    out = (np.zeros(size, np.int32), np.zeros(size, np.int32), np.zeros(size, np.float32), np.zeros(size, np.float32))
    surf = np.nan_to_num(r["rsa"][:, 1], nan=0.0) >= SURFACE_RSA; s = np.flatnonzero(surf); c = r["chain"]
    if len(np.unique(c[s])) < 2: return out
    both = surf[r["i"]] & surf[r["j"]]; ci, cj = r["i"][both], r["j"][both]; npos = size // 2 if len(ci) else 0
    k = rng.choice(len(ci), npos) if npos else np.zeros(0, np.int64); m = size - npos
    a = rng.choice(s, m); b = np.empty(m, np.int64)
    for ca in np.unique(c[a]):
        sel = c[a] == ca; b[sel] = rng.choice(s[c[s] != ca], int(sel.sum()))
    aa = np.concatenate((ci[k], a)); bb = np.concatenate((cj[k], b))
    return aa.astype(np.int32), bb.astype(np.int32), pair_targets(r, aa, bb), np.ones(size, np.float32)


def attach(graph, table, cid, n, sample, visits, decoy=False, within=False, mask=None):
    """Add rsa_residue (2, n: bound, free; NaN -> -1) and, for training sources, edge_a / edge_b / edge_target / edge_weight
    (2, P, both branches identical) to a PINDER paired graph. decoy (decoy-rank-v1): also the re-typed decoy fields
    decoy_idx / decoy_type / decoy_mask (2, DECOY_SLOTS) and decoy_valid (2,) from retype(), drawn from their own RNG
    stream [DECOY_SALT, crc32(id), visit], so the sampled pairs are unchanged. within (negatives-v1): also the
    within-complex listwise fields within_a (2, A), within_c / within_t / within_mask (2, A, C), within_valid (2, A) from
    within_sample(), own RNG stream [WITHIN_SALT, crc32(id), visit]. mask (hotspot-mask-v1): (store, oracle, keep, soft)
    adds mask_keep_idx (2, n) int32 / mask_keep_mask (2, n) float32 (the UNMASKED nodes, padded 0 / 0) and mask_stats (2, 3)
    float32 (kept, surface, fallback) from hotspot_keep(), own RNG stream [MASK_SALT, crc32(id), visit]. hotspot-mask-v2:
    mask = (store, oracle, keep, soft, sides, prob) with (sides, prob) != (both, 1) -> hotspot_fields2() + mask_side (2,) int32."""
    r = table.row(cid)
    if len(r["rsa"]) > n: raise AssertionError((cid, "residue table larger than the padded graph"))
    rsa = np.full((2, n), -1.0, np.float32); rsa[:, :len(r["rsa"])] = np.nan_to_num(r["rsa"].T, nan=-1.0); graph["rsa_residue"] = rsa
    if sample:
        visits[cid] = visits.get(cid, 0) + 1
        a, b, t, w = sample_pairs(r, np.random.default_rng([SALT, zlib.crc32(cid.encode()), visits[cid]]))
        for name, value in (("edge_a", a), ("edge_b", b), ("edge_target", t), ("edge_weight", w)): graph[name] = np.stack((value, value))
        if decoy:
            values = retype(r, residue_types(graph["nodes"][0, :len(r["rsa"])]), np.random.default_rng([DECOY_SALT, zlib.crc32(cid.encode()), visits[cid]]))
            for name, value in zip(DECOY_FIELDS, values): graph[name] = np.stack((value, value))
        if within:  # True / "distance" (negatives-v1) or "drsa" (drsa-within-v1)
            values = within_sample(r, np.random.default_rng([WITHIN_SALT, zlib.crc32(cid.encode()), visits[cid]]),
                                   target="drsa" if within == "drsa" else "distance", stats=True)
            for name, value in zip(WITHIN_FIELDS, values): graph[name] = np.stack((value, value))
        if mask is not None:
            store, oracle, keep, soft, *extra = mask; sides, prob = extra if extra else ("both", 1.0)
            rng = np.random.default_rng([MASK_SALT, zlib.crc32(cid.encode()), visits[cid]])
            if sides == "both" and prob == 1.0:  # hotspot-mask-v1 path, unchanged
                values = hotspot_fields(r, cid, n, store, oracle, keep, soft, rng); names = MASK_FIELDS
            else:  # hotspot-mask-v2: augmentation draw on its own stream [MASK2_SALT, crc32(id), visit]
                values = hotspot_fields2(r, cid, n, store, oracle, keep, soft, rng, sides, prob,
                                         np.random.default_rng([MASK2_SALT, zlib.crc32(cid.encode()), visits[cid]])); names = MASK2_FIELDS
            for name, value in zip(names, values): graph[name] = np.stack((value, value))
    return graph


# ---------------------------------------------------------------- decoy partners (decoy-rank-v1, 2026-10-29)
# Teach the edge head to reject decoy partners, not only to recognise interface-like surfaces (default off).
# A. re-typed decoys (--decoy-retype-weight): per training complex and visit, one chain at random; its interface residues
#    (heavy-atom distance <= 6 A to the other chain in the bound complex, residue-rsa-v1 pairs) except G / P / C; ceil(50%)
#    of them re-typed (at least 3, else the complex is skipped for this loss): D/E -> K/R, K/R -> D/E, any other residue ->
#    a type drawn from INTERFACE_COMPOSITION outside its chemistry class (never to G / P / C, so the G/P/C count is kept).
#    Same backbone, different chemistry, interface-like composition. The model runs a third pass (free-branch edges, nodes
#    re-typed) and scores the same sampled pairs; loss over the contact pairs (soft target > 0.5):
#    softplus(DECOY_MARGIN - (logit_true - logit_decoy)), per complex, then over complexes.
# B. in-batch cross-complex decoys (--decoy-batch-weight): no extra pass; for complex i and partner j = i + s (mod B),
#    s = 1..DECOY_PARTNERS, each contact pair (a_i, b_i) is paired with a residue b'_j from complex j's own contact pairs
#    (the free-branch embeddings: chain-isolated, so z_j[b'] carries only j's partner chain); same margin loss.
#    Batches may hold homologous complexes (cluster ids are not in the batch), a known source of false decoys.
DECOY_SALT = 8191
DECOY_SLOTS = 32
DECOY_INTERFACE_A = 6.0
DECOY_FRACTION = 0.5
DECOY_MIN = 3
DECOY_MARGIN = 1.0
DECOY_PARTNERS = 4
DECOY_FIELDS = ("decoy_idx", "decoy_type", "decoy_mask", "decoy_valid")
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # node one-hot order (jaxpropka.parameters.ALPHABET)
CHEMISTRY = {**dict.fromkeys("AVLIM", "hydrophobic"), **dict.fromkeys("FWY", "aromatic"), **dict.fromkeys("STNQ", "polar"),
             **dict.fromkeys("KR", "positive"), **dict.fromkeys("DE", "negative"), "H": "his", "G": "gly", "P": "pro", "C": "cys"}
NEVER_RETYPED = "GPC"
# Approximate amino-acid composition (%) of protein-protein interface residues, hand-set from published interface
# composition surveys (e.g. Jones & Thornton 1996 PNAS 93:13; Levy 2010 JMB 403:660; interfaces enriched in L, R, Y, W, M,
# depleted in K, D, E relative to the surface); not fitted to PINDER. Only relative values inside the allowed classes matter.
INTERFACE_COMPOSITION = {"A": 5.5, "C": 1.0, "D": 5.0, "E": 6.0, "F": 4.5, "G": 6.0, "H": 2.5, "I": 5.5, "K": 6.0, "L": 8.5,
                         "M": 2.5, "N": 4.5, "P": 4.5, "Q": 4.0, "R": 7.0, "S": 6.0, "T": 5.5, "V": 6.0, "W": 2.0, "Y": 6.0}


def _transition():
    """(21, 20) probabilities of the new type given the original (row 20: no / unknown type, never re-typed)."""
    out = np.zeros((21, 20)); freq = np.array([INTERFACE_COMPOSITION[x] for x in ALPHABET])
    for i, x in enumerate(ALPHABET):
        if x in NEVER_RETYPED: continue
        cls = CHEMISTRY[x]
        allowed = np.array([CHEMISTRY[y] == ("positive" if cls == "negative" else "negative") if cls in ("positive", "negative")
                            else CHEMISTRY[y] != cls and y not in NEVER_RETYPED for y in ALPHABET])
        out[i] = freq * allowed / (freq * allowed).sum()
    return out


TRANSITION = _transition()
RETYPABLE = TRANSITION.sum(axis=1) > 0  # (21,): index -1 (unknown) is row 20, False


def residue_types(onehot):
    """Type index per node from the 20 one-hot node columns, -1 where none is set."""
    onehot = np.asarray(onehot)[:, :20]; return np.where(onehot.max(axis=1) > 0.5, onehot.argmax(axis=1), -1).astype(np.int64)


def retype(r, aa, rng, slots=DECOY_SLOTS):
    """Re-typed decoy for one complex: (idx (slots,) int32 padded -1, new type (slots,) int32, mask (slots,) float32,
    valid () float32). r: ResidueTable row; aa: residue_types of its nodes."""
    idx = np.full(slots, -1, np.int32); new = np.zeros(slots, np.int32); mask = np.zeros(slots, np.float32); skip = (idx, new, mask, np.float32(0))
    codes = np.unique(r["chain"])
    if len(codes) < 2 or not len(r["d"]): return skip
    chosen = codes[rng.integers(len(codes))]; close = r["d"] <= DECOY_INTERFACE_A
    ends = np.concatenate((r["i"][close], r["j"][close])); interface = np.unique(ends[r["chain"][ends] == chosen])
    candidates = interface[RETYPABLE[aa[interface]]]; k = min(int(np.ceil(DECOY_FRACTION * len(candidates))), slots)
    if k < DECOY_MIN: return skip
    pick = np.sort(rng.choice(candidates, k, replace=False))
    idx[:k] = pick; new[:k] = [rng.choice(20, p=TRANSITION[aa[i]]) for i in pick]; mask[:k] = 1.0
    return idx, new, mask, np.float32(1)


# ---------------------------------------------------------------- model
def initialize_edges(params, seed):
    import jax
    import jax.numpy as jnp
    width = params["head"]["w"].shape[0]; kh, ko = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), SALT))
    dense = lambda k, a, b: {"w": jax.random.normal(k, (a, b), jnp.float32) / jnp.sqrt(float(a)), "b": jnp.zeros(b, jnp.float32)}
    return {**params, "aux_redge": {"norm": jnp.stack((jnp.ones(width), jnp.zeros(width))), "hidden": dense(kh, 2 * width, width), "out": dense(ko, width, 1)}}


def residue_embedding(p, tokens, n):
    from pkanet.model import norm
    return norm(p["aux_redge"]["norm"], tokens[:n])


def pair_logits(p, z, a, b):
    import jax
    import jax.numpy as jnp
    from pkanet.model import linear
    za, zb = z[a], z[b]; e = p["aux_redge"]
    return linear(e["out"], jax.nn.gelu(linear(e["hidden"], jnp.concatenate((za * zb, jnp.abs(za - zb)), axis=-1))))[..., 0]


def pair_logits_from(p, za, zb):
    """pair_logits on explicit embedding rows (..., W) (decoy pairs across complexes)."""
    import jax
    import jax.numpy as jnp
    from pkanet.model import linear
    e = p["aux_redge"]
    return linear(e["out"], jax.nn.gelu(linear(e["hidden"], jnp.concatenate((za * zb, jnp.abs(za - zb)), axis=-1))))[..., 0]


def retype_nodes(nodes, idx, new, mask):
    """One graph's nodes (n, F) with the 20 type columns of rows idx[mask > 0] set to one_hot(new); padding is dropped."""
    import jax
    import jax.numpy as jnp
    safe = jnp.where(mask > 0, idx, nodes.shape[0])
    return nodes.at[safe, :20].set(jax.nn.one_hot(new, 20, dtype=nodes.dtype), mode="drop")


def contact_weight(target, weight, valid):
    """(B, P) float mask of the margin pairs: sampled contacts with soft target > 0.5 of valid complexes."""
    import jax.numpy as jnp
    return ((target > 0.5) & (weight > 0) & valid[:, None]).astype(jnp.float32)


def margin_loss(true, decoy, weight, margin=DECOY_MARGIN):
    """softplus(margin - (true - decoy)) averaged per complex over weight > 0 (leading axis = complex; true broadcasts
    against decoy), then over complexes with at least one weighted pair."""
    import jax
    import jax.numpy as jnp
    per = jax.nn.softplus(margin - (true - decoy)); w = jnp.broadcast_to(weight, per.shape)
    b = per.shape[0]; per = per.reshape(b, -1); w = w.reshape(b, -1); n = jnp.sum(w, axis=1); counted = n > 0
    return jnp.sum(jnp.where(counted, jnp.sum(per * w, axis=1) / jnp.maximum(n, 1e-6), 0.0)) / jnp.maximum(jnp.sum(counted), 1)


def batch_decoy_index(b, contact, valid, partners=DECOY_PARTNERS):
    """In-batch decoys: (j (B, K) partner complex, b' (B, K, P) node of j, mask (B, K, P)). Partner j = (i + s) mod B,
    s = 1..K; slot p of complex i takes j's contact slot order_j[p mod count_j] (contacts first), so b' is a residue of j's
    own interface patch. Masked: i's slot not a contact, j padding / without contacts, j == i (batch <= K)."""
    import jax.numpy as jnp
    batch, pairs = b.shape; contact = contact > 0
    count = jnp.sum(contact, axis=1); order = jnp.argsort(1 - contact.astype(jnp.int32), axis=1)  # stable: contacts first
    i = jnp.arange(batch); j = (i[:, None] + jnp.arange(1, partners + 1)[None, :]) % batch
    slot = jnp.arange(pairs)[None, None, :] % jnp.maximum(count[j], 1)[..., None]
    q = jnp.take_along_axis(order[j], slot, axis=-1); bprime = jnp.take_along_axis(b[j], q, axis=-1)
    mask = contact[:, None, :] & ((count[j] > 0) & valid[j] & (j != i[:, None]))[..., None]
    return j, bprime, mask.astype(jnp.float32)


def decoy_provenance(retype_weight, batch_weight):
    out = {}
    if retype_weight: out["decoy_retype"] = {"weight": float(retype_weight), "margin": DECOY_MARGIN, "slots": DECOY_SLOTS, "interface_A": DECOY_INTERFACE_A,
        "fraction": DECOY_FRACTION, "minimum": DECOY_MIN, "never_retyped": NEVER_RETYPED, "composition": INTERFACE_COMPOSITION,
        "definition": "third pass on the free branch with one chain's interface residues re-typed (D/E <-> K/R, others to an interface-composition type "
                      "outside their chemistry class); softplus(margin - (logit_true - logit_decoy)) over sampled contacts with target > 0.5 (production_residue)"}
    if batch_weight: out["decoy_batch"] = {"weight": float(batch_weight), "margin": DECOY_MARGIN, "partners": DECOY_PARTNERS,
        "definition": "free-branch embeddings: contact pair (a_i, b_i) vs (a_i, b'_j), b'_j from complex j = i + 1..K (mod batch)'s own contact pairs; "
                      "softplus(margin - (logit_true - logit_decoy)); homologous complexes may share a batch (production_residue)"}
    return out


# ---------------------------------------------------------------- negatives-v1 (2026-10-29)
# The in-batch decoys of decoy-rank-v1 (--decoy-batch-weight) became a contrastive loss that aligned chain embeddings by
# protein family (homodimer swaps 0.95, Ab/Ag vs Ab/Ag decoys at chance). Negatives here share family / class with the
# positive, so only the pairing differs (default off):
# A. within-complex listwise (--within-weight): per training complex and visit, up to WITHIN_ANCHORS anchors drawn from
#    the interface residues of both chains (interface = free RSA >= 0.1 and at least one cross-chain partner <= 10 A in
#    residue-rsa-v1's pairs); per anchor up to WITHIN_CANDIDATES candidates, all interface residues of the other chain(s):
#    every contact of the anchor (pair <= 10 A whose partner is an interface residue; target soft(d)) when they fit, the
#    rest filled with that chain's other interface residues (target 0, > 10 A from the anchor), random subsample when there
#    are more. When the contacts alone exceed the slots, WITHIN_RESERVE slots stay for non-contacts (contacts subsampled).
#    Anchors without a contact or without a non-contact are not used. Loss on the free branch: softmax over the candidates
#    of pair_logits_from(z[a], z[c]), cross-entropy against target / sum(target), mean over valid anchors per complex,
#    then over complexes.
# B. same-class swaps (--swap-weight, --swap-mode): in-batch partners j of complex i must share i's stratum (pool-v4
#    pinder.tsv / pinder-val.tsv: hetero 0, homo 1, Ab/Ag 2), j != i, never homo (homodimers are neither anchors nor
#    decoys); the first SWAP_PARTNERS such j in cyclic batch order i+1, i+2, ...; fewer (masked) when the batch has fewer.
#    Each contact pair (a_i, b_i) vs (a_i, b'_j), margin loss as decoy-rank-v1. class: b'_j = j's contact slot p mod
#    count_j (as batch_decoy_index); curv: b'_j = the residue among j's contact slots whose curvature proxy is nearest to
#    that of b_i (first slot on ties).
# Curvature proxy (curvature()): per residue, free-branch (intra-chain) neighbours of the 20 A residue graph, self
#    excluded: c_i = || sum_j w_ij d_ij || / sum_j w_ij, d_ij = edge[..., 16:19] (unit CA_i -> CA_j direction in i's
#    backbone frame), w_ij = sum of the first CURV_RBF_BINS RBF columns edge[..., :8] = sum_{b<8} exp(-((d - 4b/3)/1.5)^2)
#    (centres 0 - 9.33 A of linspace(0, 20, 16)): 1.99 for 2-8 A, 1.68 at 9 A, 1.00 at 10 A, 0.31 at 11 A, 0.04 at 12 A
#    (1.50 at 0 A), i.e. a soft 10 A sphere, flat inside, non-increasing beyond 4 A. c ~ 0 when neighbours surround the residue, -> 1 when they all sit on one side (convex tip).
#    Residues with an invalid backbone frame (direction columns zeroed by the graph builder) get c = 0.
WITHIN_SALT = 12289
WITHIN_ANCHORS = 64
WITHIN_CANDIDATES = 32
WITHIN_RESERVE = 8
WITHIN_FIELDS = ("within_a", "within_c", "within_t", "within_mask", "within_valid")
# drsa-within-v1 (2026-11-01): --within-target {distance,drsa}. drsa: contact candidate c of anchor a gets
# soft(d_ac) * min(dRSA_a, dRSA_c), dRSA = clip(RSA_free - RSA_bound, 0, 1) (residue-rsa-v1 rsa[:, 1] free, rsa[:, 0] bound,
# NaN -> 0); non-contacts 0; anchors whose summed target is < WITHIN_DRSA_MIN are dropped (row zeroed like padding, valid 0).
# Anchor eligibility and candidate sampling stay distance-defined (identical RNG draws), so only t and valid change.
WITHIN_TARGETS = ("distance", "drsa")
WITHIN_DRSA_MIN = 1e-6
# cumulative per-process counters of within_sample(stats=True) calls (threaded loader: guarded by the lock); history loader_counts.within
WITHIN_STATS = {"complexes": 0, "anchors": 0, "dropped_drsa": 0}
WITHIN_LOCK = threading.Lock()
SWAP_PARTNERS = 4
STRATA = {"hetero": 0, "homo": 1, "Ab/Ag": 2}
HOMO = STRATA["homo"]
CURV_RBF_BINS = 8


def interface_mask(r):
    """(nodes,) bool: surface (free RSA >= 0.1) residues with at least one cross-chain partner <= 10 A."""
    has = np.zeros(len(r["rsa"]), bool); has[r["i"]] = True; has[r["j"]] = True
    return has & (np.nan_to_num(r["rsa"][:, 1], nan=0.0) >= SURFACE_RSA)


def drsa(r):
    """(nodes,) float32 burial on binding dRSA = clip(RSA_free - RSA_bound, 0, 1), NaN -> 0 (residue-rsa-v1: rsa[:, 0] bound, rsa[:, 1] free)."""
    rsa = np.asarray(r["rsa"], np.float32)
    return np.clip(np.nan_to_num(rsa[:, 1] - rsa[:, 0], nan=0.0), 0.0, 1.0).astype(np.float32)


def within_sample(r, rng, anchors=WITHIN_ANCHORS, candidates=WITHIN_CANDIDATES, reserve=WITHIN_RESERVE, target="distance", stats=False):
    """Within-complex listwise sample for one complex: (a (A,) int32, c (A, C) int32, t (A, C) float32 soft targets,
    mask (A, C) float32, valid (A,) float32), padding 0. target "drsa" (drsa-within-v1): contact targets x min(dRSA_a, dRSA_c),
    anchors with summed target < WITHIN_DRSA_MIN zeroed (valid 0); same draws as "distance". stats: add to WITHIN_STATS."""
    if target not in WITHIN_TARGETS: raise ValueError(target)
    out = _within_distance(r, rng, anchors, candidates, reserve)
    dropped = 0
    if target == "drsa":
        a, c, t, mask, valid = out; dr = drsa(r)
        for s in np.flatnonzero(valid > 0):
            t[s] = t[s] * np.minimum(dr[a[s]], dr[c[s]]) * (t[s] > 0)
            if float(t[s].sum()) < WITHIN_DRSA_MIN: a[s] = 0; c[s] = 0; t[s] = 0.0; mask[s] = 0.0; valid[s] = 0.0; dropped += 1
    if stats:
        with WITHIN_LOCK:
            WITHIN_STATS["complexes"] += 1; WITHIN_STATS["anchors"] += int((out[4] > 0).sum()) + dropped; WITHIN_STATS["dropped_drsa"] += dropped
    return out


def _within_distance(r, rng, anchors, candidates, reserve):
    """negatives-v1 within_sample (distance targets), unchanged."""
    a = np.zeros(anchors, np.int32); c = np.zeros((anchors, candidates), np.int32); t = np.zeros((anchors, candidates), np.float32)
    mask = np.zeros((anchors, candidates), np.float32); valid = np.zeros(anchors, np.float32); out = (a, c, t, mask, valid)
    iface = interface_mask(r); idx = np.flatnonzero(iface); chain = r["chain"]
    if len(np.unique(chain[idx])) < 2: return out
    keep = iface[r["i"]] & iface[r["j"]]; pi, pj, pd = r["i"][keep], r["j"][keep], r["d"][keep]
    sets = {}
    for x in idx:  # contacts and non-contacts of every interface residue
        pos = np.concatenate((pj[pi == x], pi[pj == x])); dist = np.concatenate((pd[pi == x], pd[pj == x]))
        other = idx[chain[idx] != chain[x]]; neg = other[~np.isin(other, pos)]
        if len(pos) and len(neg): sets[int(x)] = (pos, dist, neg)
    usable = np.array(sorted(sets), np.int64)
    if not len(usable): return out
    pick = usable if len(usable) <= anchors else np.sort(rng.choice(usable, anchors, replace=False))
    for s, x in enumerate(pick):
        pos, dist, neg = sets[int(x)]; npos = min(len(pos), candidates - min(len(neg), reserve))
        if npos < len(pos): k = np.sort(rng.choice(len(pos), npos, replace=False)); pos, dist = pos[k], dist[k]
        nneg = min(len(neg), candidates - npos)
        if nneg < len(neg): neg = np.sort(rng.choice(neg, nneg, replace=False))
        m = npos + nneg; a[s] = x; c[s, :m] = np.concatenate((pos, neg)); t[s, :npos] = soft(dist); mask[s, :m] = 1.0; valid[s] = 1.0
    return out


def ranking_metrics(scores, gains):
    """Per-anchor ranking of `scores` against graded relevance `gains` (>= 0) over one candidate list (drsa-within-v1 scorer):
    NDCG over the full list (gain / log2(rank + 1), ties in scores broken by candidate order after a stable sort on -score)
    and Spearman (average ranks). None where undefined (all gains 0 for NDCG; constant gains or scores for Spearman)."""
    s = np.asarray(scores, np.float64); g = np.asarray(gains, np.float64)
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2)); ideal = float(np.sum(np.sort(g)[::-1] * disc))
    ndcg = float(np.sum(g[np.argsort(-s, kind="stable")] * disc) / ideal) if ideal > 0 else None
    rho = None
    if np.ptp(g) > 0 and np.ptp(s) > 0:
        rs, rg = _avg_rank(s), _avg_rank(g); rs -= rs.mean(); rg -= rg.mean(); rho = float(np.sum(rs * rg) / np.sqrt(np.sum(rs * rs) * np.sum(rg * rg)))
    return ndcg, rho


def _avg_rank(x):
    order = np.argsort(x, kind="stable"); ranks = np.empty(len(x)); ranks[order] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True); sums = np.bincount(inv, ranks)
    return sums[inv] / counts[inv]


def load_strata(root):
    """{PINDER id: stratum code} from pool-v4 pinder.tsv (training) and pinder-val.tsv (validation); unknown -> -1 at use."""
    import csv
    from .production_graphs import POOL_V4
    out = {}
    for name in ("pinder.tsv", "pinder-val.tsv"):
        with open(Path(root) / POOL_V4 / name) as handle:
            for row in csv.DictReader(handle, delimiter="\t"): out[row["id"]] = STRATA.get(row["stratum"], -1)
    return out


def strata_provenance(root):
    from .production_graphs import POOL_V4
    return {name: digest(Path(root) / POOL_V4 / name) for name in ("pinder.tsv", "pinder-val.tsv")}


def curvature(edge, edge_mask, neighbors, bins=CURV_RBF_BINS):
    """Curvature proxy (..., n) from residue-graph arrays edge (..., n, k, 20), edge_mask (..., n, k) (pass the free
    branch's intra-chain mask), neighbors (..., n, k); self excluded. See the block comment above."""
    import jax.numpy as jnp
    n = edge.shape[-3]; own = neighbors == jnp.arange(n)[:, None]
    w = jnp.sum(edge[..., :bins], axis=-1) * (edge_mask > 0) * ~own
    vec = jnp.sum(w[..., None] * edge[..., 16:19], axis=-2)
    return jnp.sqrt(jnp.sum(vec * vec, axis=-1)) / jnp.maximum(jnp.sum(w, axis=-1), 1e-6)


def within_loss(logits, target, mask, anchor_valid, valid):
    """Listwise cross-entropy: per anchor softmax over its masked candidates vs target / sum(target); mean over valid
    anchors per complex, then over complexes with at least one (leading axis = complex)."""
    import jax
    import jax.numpy as jnp
    m = mask > 0; logp = jax.nn.log_softmax(jnp.where(m, logits, -1e9), axis=-1)
    q = jnp.where(m, target, 0.0); q = q / jnp.maximum(jnp.sum(q, axis=-1, keepdims=True), 1e-9)
    ce = -jnp.sum(jnp.where(m, q * logp, 0.0), axis=-1)
    w = anchor_valid * valid[:, None]; n = jnp.sum(w, axis=1); counted = n > 0
    return jnp.sum(jnp.where(counted, jnp.sum(ce * w, axis=1) / jnp.maximum(n, 1e-6), 0.0)) / jnp.maximum(jnp.sum(counted), 1)


def swap_decoy_index(b, contact, valid, stratum, curv=None, partners=SWAP_PARTNERS):
    """Same-class in-batch decoys: (j (B, K), b' (B, K, P), mask (B, K, P)). j = the first K complexes in cyclic order
    i+1, i+2, ... with stratum[j] == stratum[i] != homo (and >= 0), j != i, j valid with contacts; b' from j's contact
    slots: slot p mod count_j (curv None, mode class) or the slot whose curv value is nearest to curv[i, b_i[p]]
    (curv (B, n), mode curv). Masked: i's slot not a contact, i invalid or homo, j missing."""
    import jax.numpy as jnp
    batch, pairs = b.shape; contact = contact > 0; count = jnp.sum(contact, axis=1)
    i = jnp.arange(batch); span = max(batch, partners + 1); shift = jnp.arange(1, span)
    jj = (i[:, None] + shift[None, :]) % batch
    ok = (jj != i[:, None]) & (stratum[jj] == stratum[:, None]) & (stratum[:, None] != HOMO) & (stratum[:, None] >= 0) \
        & valid[jj] & valid[:, None] & (count[jj] > 0)
    rank = jnp.argsort(jnp.where(ok, shift[None, :], span + shift[None, :]), axis=1)[:, :partners]
    j = jnp.take_along_axis(jj, rank, axis=1); jok = jnp.take_along_axis(ok, rank, axis=1)
    if curv is None:
        order = jnp.argsort(1 - contact.astype(jnp.int32), axis=1)  # stable: contacts first
        slot = jnp.arange(pairs)[None, None, :] % jnp.maximum(count[j], 1)[..., None]
        q = jnp.take_along_axis(order[j], slot, axis=-1)
    else:
        cb = jnp.take_along_axis(curv, b, axis=1)  # (B, P) curvature of each slot's b
        gap = jnp.abs(cb[:, None, :, None] - cb[j][:, :, None, :])  # (B, K, P_i, P_j)
        q = jnp.argmin(jnp.where(contact[j][:, :, None, :], gap, jnp.inf), axis=-1)
    bprime = jnp.take_along_axis(b[j], q, axis=-1)
    mask = contact[:, None, :] & jok[..., None]
    return j, bprime, mask.astype(jnp.float32)


def negatives_provenance(root, within_weight, swap_weight, swap_mode, within_target="distance"):
    out = {}
    if within_weight: out["within_listwise"] = {"weight": float(within_weight), "anchors": WITHIN_ANCHORS, "candidates": WITHIN_CANDIDATES, "reserve": WITHIN_RESERVE,
        "surface_rsa_min": SURFACE_RSA, "contact_A": CUTOFF_A,
        "definition": "free branch; anchors = interface residues (surface, a cross-chain partner <= 10 A) of both chains; candidates = interface residues of "
                      "the other chain, all contacts (soft target min(1, exp(-(d - 4) / 3))) + non-contacts (target 0); softmax cross-entropy vs normalised "
                      "target, mean over valid anchors per complex, then over complexes (production_residue.within_sample / within_loss)",
        **({"target": "drsa", "drsa_min": WITHIN_DRSA_MIN,
            "target_definition": "drsa-within-v1: contact target soft(d_ac) * min(dRSA_a, dRSA_c), dRSA = clip(RSA_free - RSA_bound, 0, 1) (residue-rsa-v1, "
                                 "NaN -> 0); non-contacts 0; anchors with summed target < 1e-6 dropped (valid 0; counted in history loader_counts.within); "
                                 "anchor eligibility and candidate sampling distance-defined, same draws as distance mode"} if within_target == "drsa" else {})}
    if swap_weight: out["swap_decoys"] = {"weight": float(swap_weight), "mode": swap_mode, "margin": DECOY_MARGIN, "partners": SWAP_PARTNERS, "strata": STRATA,
        "strata_sha256": strata_provenance(root),
        "curvature": {"rbf_bins": CURV_RBF_BINS, "definition": "c_i = ||sum_j w_ij d_ij|| / sum_j w_ij over free-branch neighbours (self excluded), d_ij unit "
                      "direction in i's frame, w_ij = sum of RBF bins 0-7 (centres 0-9.33 A)"} if swap_mode == "curv" else None,
        "definition": "free-branch embeddings: contact pair (a_i, b_i) vs (a_i, b'_j), j = first K same-stratum complexes in cyclic batch order (homo never "
                      "anchor or decoy), b'_j from j's contact slots (class: cyclic; curv: nearest curvature proxy); softplus(margin - (l_true - l_decoy)) "
                      "(production_residue.swap_decoy_index)"}
    return out


# ---------------------------------------------------------------- hotspot-mask-v1 (2026-10-30)
# Hotspot mask (default off; --mask-oracle {pesto,own,random}): the re-typed third pass of decoy-rank-v1 is replaced by a
# free-branch pass in which every residue OUTSIDE a small "hotspot" set has its residue-type one-hot zeroed (an all-zero
# one-hot, the existing unknown-type encoding; geometry, termini flags and frame validity are kept). Per training complex
# and visit (own RNG stream [MASK_SALT, crc32(id), visit]) the UNMASKED set is, per chain, the top round(keep x |S_c|)
# (at least 1) of that chain's surface residues S_c (free RSA >= 0.1, residue-rsa-v1) ranked by the oracle score: pesto =
# PeSTo protein-protein interface probability, own = the frozen within-0.3 model's free-branch edge-max score
# (OWN_FIELD); NaN ranks below every finite score, equal scores are ordered by a uniform draw from the visit's stream.
# random = a uniformly random round(keep x |S_c|) of S_c. soft > 0 additionally unmasks round(soft x |rest|) of the
# remaining SURFACE residues at random. Buried residues (free RSA < 0.1, or no RSA) are always masked: they carry no
# interface chemistry for a partner to read. A complex without a store record falls back to the random oracle for that
# visit (mask_stats[2] = 1, counted in the history). The edge BCE's free part, its free -> bound distillation and the
# within listwise loss read the masked pass's embeddings (production_train.AuxEngine); bound BCE, Siamese burial, gate and
# pKa keep the full sequence.
MASK_SALT = 16381
MASK_ORACLES = ("pesto", "own", "random")
ORACLE_FIELDS = {"pesto": "pesto", "own": "own_edgemax"}
OWN_FIELD = ORACLE_FIELDS["own"]
MASK_FIELDS = ("mask_keep_idx", "mask_keep_mask", "mask_stats")
HOTSPOT_STORE_FIELDS = ("pesto", "own_edgemax", "own_dburial", "rsa_free", "chain")


class HotspotStore:
    """Oracle side store by complex id (hotspot-mask-v1 FORMAT.md). store-v2 layout (records.bin, one zstd frame per record
    = every field's C-order bytes in metadata["fields"] order; index.npz ids / dims / offsets; metadata.json dtypes and
    symbolic shapes, default (n,)); record(cid) -> {field: (n,) array} or None when the complex has no record."""

    def __init__(self, path):
        import os
        self.path = Path(path); self.metadata = json.loads((self.path / "metadata.json").read_text())
        verification = self.path / "verification.json"
        if verification.exists() and not json.loads(verification.read_text()).get("passed", True): raise AssertionError((self.path, "verification failed"))
        self.verification_sha256 = digest(verification) if verification.exists() else None
        with np.load(self.path / "index.npz") as index:
            self.ids = index["ids"].astype(str); self.dims = np.asarray(index["dims"]).reshape(len(self.ids), -1); self.offsets = index["offsets"]
        names = self.metadata.get("dims"); cols = self.dims.shape[1]  # hotspot store: dims (N, 1) = [n]
        if not (isinstance(names, (list, tuple)) and len(names) == cols and all(isinstance(x, str) for x in names)): names = ["n", "k", "q", "s", "sk"][:cols]
        self.by_id = {cid: i for i, cid in enumerate(self.ids)}; self.names = list(names)
        self.fields = list(self.metadata["fields"]); self.dtypes = self.metadata["dtypes"]; self.shapes = self.metadata.get("shapes", {})
        missing = [f for f in ("pesto", OWN_FIELD) if f not in self.fields]
        if missing: raise AssertionError((self.path, "oracle fields missing", missing))
        self.fd = os.open(self.path / "records.bin", os.O_RDONLY)

    def __len__(self): return len(self.ids)

    def record(self, cid):
        import os
        import zstandard
        i = self.by_id.get(cid)
        if i is None: return None
        dims = dict(zip(self.names, (int(x) for x in self.dims[i])))
        blob = os.pread(self.fd, int(self.offsets[i + 1]) - int(self.offsets[i]), int(self.offsets[i]))
        raw = zstandard.ZstdDecompressor().decompress(blob); out = {}; position = 0
        for f in self.fields:
            dtype = np.dtype(self.dtypes[f]); sym = self.shapes.get(f, ["n"]) if isinstance(self.shapes, dict) else ["n"]
            shp = tuple(dims[d] if isinstance(d, str) else int(d) for d in sym)
            size = int(np.prod(shp)); out[f] = np.frombuffer(raw, dtype, count=size, offset=position).reshape(shp); position += size * dtype.itemsize
        if position != len(raw): raise ValueError((cid, "record length does not match its dims"))
        return out


class MemoryHotspotStore:
    """In-memory oracle store (tests, synthetic stand-in): {cid: {field: (n,) array}}."""

    def __init__(self, records): self.records = dict(records); self.verification_sha256 = "memory"; self.path = "memory"

    def __len__(self): return len(self.records)

    def record(self, cid): return self.records.get(cid)


def hotspot_keep(r, record, oracle, keep, soft, rng):
    """(unmasked (nodes,) bool, stats (kept, surface, fallback) float32) for one complex. r: ResidueTable row; record: the
    oracle store record (None -> random oracle, fallback 1, unless the oracle is random already). See the block comment."""
    if oracle not in MASK_ORACLES: raise ValueError(oracle)
    m = len(r["rsa"]); surface = np.nan_to_num(r["rsa"][:, 1], nan=0.0) >= SURFACE_RSA; chain = np.asarray(r["chain"])
    fallback = oracle != "random" and record is None; mode = "random" if oracle == "random" or fallback else oracle
    score = None
    if mode != "random":
        score = np.asarray(record[ORACLE_FIELDS[mode]], np.float64)
        if score.shape != (m,): raise AssertionError(("oracle record does not match the residue table", score.shape, m))
    out = np.zeros(m, bool)
    for c in np.unique(chain):
        s = np.flatnonzero(surface & (chain == c))
        if not len(s) or keep <= 0: continue
        k = min(len(s), max(1, int(np.floor(keep * len(s) + 0.5))))
        if mode == "random": pick = rng.choice(s, k, replace=False)
        else:
            v = score[s]; v = np.where(np.isfinite(v), v, -np.inf); tie = rng.random(len(s))
            pick = s[np.lexsort((tie, -v))[:k]]  # descending score, NaN last, ties by the random draw
        out[pick] = True
    if soft > 0:
        rest = np.flatnonzero(surface & ~out); extra = min(len(rest), int(np.floor(soft * len(rest) + 0.5)))
        if extra: out[rng.choice(rest, extra, replace=False)] = True
    return out, np.array([out.sum(), surface.sum(), float(fallback)], np.float32)


def hotspot_fields(r, cid, n, store, oracle, keep, soft, rng):
    """(mask_keep_idx (n,) int32, mask_keep_mask (n,) float32, mask_stats (3,) float32): the unmasked nodes of one complex,
    padded to the graph size n (fixed per bucket)."""
    record = None if oracle == "random" or store is None else store.record(cid)
    kept, stats = hotspot_keep(r, record, oracle, keep, soft, rng); sel = np.flatnonzero(kept)
    idx = np.zeros(n, np.int32); mask = np.zeros(n, np.float32); idx[:len(sel)] = sel; mask[:len(sel)] = 1.0
    return idx, mask, stats


# ---------------------------------------------------------------- hotspot-mask-v2 (2026-10-30)
# Asymmetric augmentation (default off = v1: --mask-sides both --mask-prob 1). Per training complex and visit, from its own
# RNG stream [MASK2_SALT, crc32(id), visit] (two draws, always both: u ~ U(0, 1), side ~ {0, 1} uniform): with probability
# prob (u < prob) the visit is masked, otherwise mask_keep covers every residue (identity mask: the masked pass equals the
# free pass, so that visit's losses are the unmasked engine's). A masked visit with sides = one applies the v1 rule
# (hotspot_keep, the v1 stream [MASK_SALT, ...] unchanged) to chain code `side` only; every residue of the other chain
# (buried included) keeps its type. sides = both applies the v1 rule to both chains. mask_side (int32): -1 unmasked visit,
# 0 / 1 the masked chain, 2 both chains. mask_stats (kept, surface, fallback) count the MASKED chain(s) only (0, 0, 0 for
# an unmasked visit), so the history kept fraction stays ~keep. A visit that is not masked never reads the store.
MASK2_SALT = 32749
MASK_SIDES = ("both", "one")
MASK2_FIELDS = MASK_FIELDS + ("mask_side",)


def hotspot_choice(prob, sides, choice_rng):
    """mask_side for one visit: -1 (not masked), 0 / 1 (sides one), 2 (sides both). Always two draws from choice_rng."""
    if sides not in MASK_SIDES: raise ValueError(sides)
    u = float(choice_rng.random()); side = int(choice_rng.integers(2))
    if not u < prob: return -1
    return side if sides == "one" else 2


def hotspot_keep2(r, record, oracle, keep, soft, rng, side):
    """(unmasked (nodes,) bool, stats (kept, surface, fallback) of the masked chain(s)) for mask_side `side` (see above)."""
    m = len(r["rsa"])
    if side == -1: return np.ones(m, bool), np.zeros(3, np.float32)
    kept, stats = hotspot_keep(r, record, oracle, keep, soft, rng)
    if side == 2: return kept, stats
    chain = np.asarray(r["chain"]); on = chain == side; surface = np.nan_to_num(r["rsa"][:, 1], nan=0.0) >= SURFACE_RSA
    out = kept | ~on
    return out, np.array([(kept & on & surface).sum(), (on & surface).sum(), stats[2]], np.float32)


def hotspot_fields2(r, cid, n, store, oracle, keep, soft, rng, sides, prob, choice_rng):
    """hotspot_fields + mask_side (int32 scalar): the v2 augmentation (block comment above)."""
    side = hotspot_choice(prob, sides, choice_rng)
    record = None if side == -1 or oracle == "random" or store is None else store.record(cid)
    kept, stats = hotspot_keep2(r, record, oracle, keep, soft, rng, side); sel = np.flatnonzero(kept)
    idx = np.zeros(n, np.int32); mask = np.zeros(n, np.float32); idx[:len(sel)] = sel; mask[:len(sel)] = 1.0
    return idx, mask, stats, np.int32(side)


def mask_nodes(nodes, idx, mask):
    """One graph's nodes (n, F) with the 20 type columns zeroed on every row NOT listed in idx[mask > 0] (jax)."""
    import jax.numpy as jnp
    safe = jnp.where(mask > 0, idx, nodes.shape[0]); keep = jnp.zeros(nodes.shape[0], bool).at[safe].set(True, mode="drop")
    return nodes.at[:, :20].set(jnp.where(keep[:, None], nodes[:, :20], jnp.zeros((), nodes.dtype)))


def mask_nodes_np(nodes, kept):
    """numpy mask_nodes for the scorers: kept (m,) bool over the first m rows; rows >= m (padding) untouched."""
    out = np.array(nodes, copy=True); m = len(kept); out[:m, :20] = np.where(np.asarray(kept)[:, None], out[:m, :20], 0); return out


def hotspot_provenance(oracle, keep, soft, store, sides="both", prob=1.0):
    if oracle is None: return {}
    return {"hotspot_mask": {"oracle": oracle, "keep": float(keep), "soft": float(soft), "buried_masked": True,
        "sides": sides, "prob": float(prob),
        **({"augmentation": "hotspot-mask-v2: per visit (own stream [MASK2_SALT, crc32(id), visit]) masked with probability prob, else identity mask "
                            "(every residue kept); sides one: the v1 rule on one chain drawn uniformly, the other chain fully unmasked; mask_side -1 / 0 / 1 / 2 "
                            "(history pinder_mask_visit_fraction, pinder_mask_side_fraction = chain-1 share of one-sided visits)", "salt2": MASK2_SALT}
           if (sides, float(prob)) != ("both", 1.0) else {}),
        "surface_rsa_min": SURFACE_RSA, "oracle_field": ORACLE_FIELDS.get(oracle), "salt": MASK_SALT,
        "store": None if store is None else str(store.path), "store_verification_sha256": None if store is None else store.verification_sha256,
        "store_records": None if store is None else len(store),
        "fallback": "complex without a store record -> random oracle for that visit (counted: history pinder_mask_fallbacks)",
        "definition": "third pass on the free branch with the residue-type one-hot zeroed on every residue outside the unmasked set: per chain the top "
                      "round(keep x surface) surface residues (free RSA >= 0.1) by the oracle (NaN lowest, ties by a random draw; random: uniform), plus "
                      "round(soft x rest) of the remaining surface residues at random; buried residues always masked (no interface chemistry). Edge BCE free "
                      "part + free -> bound distillation and the within listwise loss on the masked pass; bound BCE, Siamese burial, gate, pKa unchanged "
                      "(production_residue.hotspot_keep / mask_nodes; replaces the re-typed pass)"}}


def provenance(root, residue_burial, edge_weight):
    out = {}
    if residue_burial: out["residue_burial"] = "PINDER burial on every residue token of the free branch vs 1 - clip(rsa_free) (residue-rsa-v1), replacing the query-site term; pKPDB unchanged (production_residue)"
    if edge_weight: out["residue_edges"] = {"weight": float(edge_weight), "pairs_per_complex": PAIRS, "surface_rsa_min": SURFACE_RSA,
                                            "definition": "aux_redge symmetric pair scorer on normalised residue tokens; (free + bound) / 2 + Siamese distillation; soft target min(1, exp(-(d - 4) / 3)) <= 10 A (production_residue)"}
    if out: out["residue_table_verification_sha256"] = ResidueTable(root).verification_sha256
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="pkatrain.production_residue"); sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build"); b.add_argument("--workers", type=int, default=70); b.add_argument("--limit", type=int)
    a = p.parse_args(argv)
    print(json.dumps(build(Path(os.environ["PKABENCH_RUNTIME"]), a.workers, a.limit), indent=1), flush=True)


if __name__ == "__main__":
    main()
