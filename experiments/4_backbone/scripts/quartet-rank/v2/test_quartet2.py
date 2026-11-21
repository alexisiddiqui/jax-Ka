"""Local CPU tests of quartet-rank-v2 (2026-11-19): the chain-quartet tree (work/quartet2-model/src) against its base, the
quartet-rank-v1b tree (work/quartet-model-v1b/src = remote _submission/quartet-rank-v1b/src, pool-v5 loader, read-only).
The residue-quartet tests of quartet-rank-v1 are kept (they run on the v2 tree); v1's graft / compile tests are dropped.

  JAX_PLATFORMS=cpu GQT_PARAMETER_SCALING=1 python test_quartet2.py [MODEL_SRC] [BASE_SRC]
"""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import traceback
import types
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("GQT_PARAMETER_SCALING", "1"); os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROBE = "--probe" in sys.argv[1:]; _ARGS = [a for a in sys.argv[1:] if a != "--probe"]
MODEL = Path(_ARGS[0]).resolve() if _ARGS else HERE.parents[4] / "src"  # this repository
BASE = Path(_ARGS[1]).resolve() if len(_ARGS) > 1 else HERE.parent / "quartet-model-v1b" / "src"
PAIRING = Path(os.environ.get("PAIRING_RANK_CORE", "/Users/alexi/Documents/ChatGPT/GQT/pairing_rank_v1/core.py"))  # read-only reference (pairing-rank-v1)
sys.path[:0] = [str(MODEL), str(HERE)]


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return lambda *a, **k: None


STUBS = {m for m in ("lineax", "optimistix", "equinox", "diffrax", "biotite", "gemmi", "pyarrow", "zstandard", "jax_triton") if importlib.util.find_spec(m) is None}


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        return importlib.machinery.ModuleSpec(name, self, is_package=True) if name.split(".")[0] in STUBS else None
    def create_module(self, spec): return _Stub(spec.name)
    def exec_module(self, module): module.__path__ = []


sys.meta_path.insert(0, _Finder())
import cpu_attention  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402,F401

from pkatrain import production_residue as pr  # noqa: E402
from pkatrain import production_gate, production_loading as pl  # noqa: E402
from pkatrain import production_train as pt  # noqa: E402

HAS_Q = hasattr(pr, "quartet_terms"); HAS_C = hasattr(pr, "chain_quartet_terms")
VARIANT = "relations-4-1-allres-cut15"; N, K, Q = 48, 12, 10; REAL = 44; SPLIT = 24; S_, SK_ = 16, 8
ASYM50 = ("pesto", 0.5, 0.0, "one", 0.5); KEY = jax.random.PRNGKey(5); QC = 32


def synthetic(seed, charged=True):
    """Two chains (24 + 20 real rows, 4 padding); node table = 24 features + backbone-v1 (test_anm_edge.synthetic); residue
    table row with every cross-chain CA pair at CA distance - 3 <= 10 A. charged: node types include D/E/K/R."""
    r = np.random.default_rng(seed); chain = np.r_[np.zeros(SPLIT), np.ones(REAL - SPLIT), np.zeros(N - REAL)].astype(np.float32)
    walk = np.cumsum(r.normal(0, 2.2, (SPLIT, 3)), axis=0); ca = np.zeros((N, 3)); ca[:SPLIT] = walk
    ca[SPLIT:REAL] = walk[:REAL - SPLIT] + np.array([6.0, 0.0, 0.0]) + r.normal(0, 1.0, (REAL - SPLIT, 3)); ca[REAL:] = walk[:N - REAL]
    n_ = ca + r.normal(0, .8, (N, 3)); c_ = ca + r.normal(0, .8, (N, 3)); o_ = c_ + r.normal(0, .8, (N, 3))
    seq = np.r_[np.arange(SPLIT), np.arange(REAL - SPLIT), np.zeros(N - REAL)].astype(np.float32)
    types_ = r.integers(0, 20, N); onehot = np.zeros((N, 24), np.float32); onehot[np.arange(N), types_] = 1; onehot[REAL:] = 0
    onehot[0, 20] = 1; onehot[SPLIT - 1, 21] = 1; onehot[SPLIT, 20] = 1; onehot[REAL - 1, 21] = 1  # chain termini flags
    table = np.concatenate((n_, ca, c_, o_, chain[:, None], seq[:, None]), axis=1).astype(np.float32)
    nodes = np.concatenate((onehot, table), axis=1); node_mask = (np.arange(N) < REAL).astype(np.float32)
    dist = np.linalg.norm(ca[:, None] - ca[None], axis=-1); np.fill_diagonal(dist, 1e9); dist[:, REAL:] = 1e9
    neighbors = np.argsort(dist, axis=1)[:, :K].astype(np.int32); emask = (np.take_along_axis(dist, neighbors, 1) < 1e8) & (node_mask[:, None] > 0)
    same = (chain[:, None] == chain[neighbors]).astype(np.float32)
    edge = np.concatenate((r.normal(0, 1, (N, K, 19)).astype(np.float32), same[..., None]), axis=-1)
    site_residue = r.choice(REAL, S_, replace=False).astype(np.int32); site_type = r.integers(0, 9, S_).astype(np.int32)
    query = np.arange(Q).astype(np.int32)
    g = {"nodes": nodes, "node_mask": node_mask, "neighbors": neighbors, "edge": edge, "edge_mask": emask, "switch": np.ones((N, K), np.float32),
         "query_residue": site_residue[query], "query_group": np.array([0, 1, 2, 3, 4, 5, 7, 8], np.int32)[site_type[query] % 8], "site_residue": site_residue, "site_type": site_type,
         "site_mask": np.ones(S_, np.float32), "site_neighbors": r.integers(0, S_, (S_, SK_)).astype(np.int32), "site_edge": r.normal(0, 1, (S_, SK_, 34)).astype(np.float32),
         "site_edge_mask": np.ones((S_, SK_), bool), "site_switch": np.ones((S_, SK_), np.float32), "query_site": query, "disulfide": np.zeros(N, np.float32)}
    for f in ("nodes", "neighbors", "edge", "edge_mask", "switch"): g[f] = g[f].copy(); g[f][REAL:] = 0
    free = {**g, "edge_mask": g["edge_mask"] & (same > 0)}
    i, j = np.triu_indices(REAL, 1); cross = chain[i] != chain[j]; d = dist[:REAL, :REAL][i, j] - 3.0; keep = cross & (d <= pr.CUTOFF_A)
    row = {"rsa": np.stack((r.uniform(0, .8, REAL), r.uniform(0.0, 1, REAL)), axis=1).astype(np.float32), "chain": chain[:REAL].astype(np.int64),
           "i": i[keep].astype(np.int64), "j": j[keep].astype(np.int64), "d": np.maximum(d[keep], 2.5).astype(np.float32)}
    row["rsa"][r.random(REAL) < 0.05, 1] = np.nan
    return g, free, row


class Table:
    def __init__(self, rows): self.rows = rows
    def row(self, cid): return self.rows[cid]


def oracle_record(seed, row):
    r = np.random.default_rng(10_000 + seed); m = len(row["rsa"]); pesto = r.uniform(0, 1, m).astype(np.float32); pesto[r.random(m) < 0.1] = np.nan
    return {"pesto": pesto, "own_edgemax": r.normal(0, 2, m).astype(np.float32), "own_dburial": r.normal(0, 1, m).astype(np.float32),
            "rsa_free": row["rsa"][:, 1].copy(), "chain": row["chain"].astype(np.int16)}


ROWS = {f"syn{s}": synthetic(s)[2] for s in range(8)}
# syn7: a single cross-chain contact -> no valid quartet (masked complex)
_r7 = ROWS["syn7"]; ROWS["syn7"] = {**_r7, "i": _r7["i"][:1], "j": _r7["j"][:1], "d": _r7["d"][:1]}
HOT = pr.MemoryHotspotStore({cid: oracle_record(s, ROWS[cid]) for s, cid in enumerate(ROWS) if s != 1})
_CACHE = {}


def recipe(seed=17):
    if seed not in _CACHE:
        params, _, _ = pt.make_model(seed, width=64, orientation=True, table=True, dihedrals=True, variant=VARIANT)
        params = pt.initialize_aux(params, seed, "ce", "burial"); params = production_gate.initialize(params, seed); _CACHE[seed] = pr.initialize_edges(params, seed)
    return _CACHE[seed]


def emb(): return pt.model_inputs(1, True, "all", 0.0, True, True, False, VARIANT)[0]


def make_engine(params, rate=0.1, **extra):
    kw = dict(gate_weight=0.1, gate_cys_weight=10.0, residue_burial=True, residue_edge_weight=0.1, residue_burial_siamese=True, within_weight=0.3, hotspot_mask="pesto")
    kw.update(extra); return pt.AuxEngine(params, 1.0, rate, "ce", "bound", "burial", emb(), **kw)


def aux_source(quartets=0, mask=ASYM50, spread=True):
    src = object.__new__(pl.AuxPinderSource); src.residues = Table(ROWS); src.by_id = {cid: {"n": REAL} for cid in ROWS}; src.sample_edges = True
    src._visits = {}; src.decoy_retype = False; src.within_sample = "drsa"; src.strata = None; src.alt = None; src.spread = spread
    src.hotspot = None if mask is None else (HOT, *mask)
    if quartets: src.quartet_count = quartets
    return src


def load_one(src, seed):
    ab, free, _ = synthetic(seed); cid = f"syn{seed}"; g = {k: np.stack((ab[k], free[k])) for k in ab}
    g = src._residue(g, cid, N); return src._spread(g)


def to_batch(graphs, B):
    real = len(graphs); graphs = graphs + [graphs[-1]] * (B - real)
    stacked = {k: jnp.asarray(np.stack([g[k] for g in graphs])) for k in graphs[0]}; r = np.random.default_rng(99)
    return stacked, jnp.asarray(r.normal(5, 2, (B, 2, Q)), jnp.float32), jnp.ones((B, Q), bool), jnp.asarray(r.uniform(0, 1, (B, Q)), jnp.float32), \
        jnp.asarray(r.uniform(0, 3, (B, Q)), jnp.float32), jnp.asarray(np.arange(B) < real)


def batch_of(quartets=0, seeds=(0, 2, 3, 0, 2, 3), B=8):
    src = aux_source(quartets); return to_batch([load_one(src, s) for s in seeds], B)


def digest_tree(tree): return hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree.leaves(tree))).hexdigest()


def hexs(aux): return [float(x).hex() for x in jax.tree.leaves(aux)]


def probe():
    """Under each tree (subprocess): the asym50 dRSA control objective on a fixed batch (loss, aux, gradient sha, attach sha);
    in the quartet tree also --quartet-weight 0 explicitly, and weight 0 with the quartet fields attached."""
    params = recipe(); data = batch_of(); (loss, aux), grad = make_engine(params).paired_value_grad(params, *data, KEY)
    out = {"parameters": pt.parameter_count(params),
           "base": {"loss": float(loss).hex(), "aux": hexs(aux), "grad_sha256": digest_tree(grad), "attach_sha256": digest_tree(data[0]), "keys": sorted(data[0])}}
    if HAS_Q:
        (l0, a0), g0 = make_engine(params, quartet_weight=0.0).paired_value_grad(params, *data, KEY)
        out["q0"] = {"loss": float(l0).hex(), "aux": hexs(a0), "grad_sha256": digest_tree(g0), "attach_sha256": digest_tree(data[0]), "keys": sorted(data[0])}
        dq = batch_of(QC); (l1, a1), g1 = make_engine(params, quartet_weight=0.0).paired_value_grad(params, *dq, KEY)
        rest = {k: v for k, v in dq[0].items() if not k.startswith("quartet_")}
        out["q0_fields"] = {"loss": float(l1).hex(), "aux": hexs(a1), "grad_sha256": digest_tree(g1), "attach_sha256": digest_tree(rest), "keys": sorted(rest),
                            "quartet_keys": sorted(k for k in dq[0] if k.startswith("quartet_")), "mask_sum": float(jnp.sum(dq[0]["quartet_mask"]))}
    if HAS_C:  # quartet-rank-v2: chain weight 0 explicit, alone and with the residue quartet weight 0
        (l2, a2), g2 = make_engine(params, chain_quartet_weight=0.0).paired_value_grad(params, *data, KEY)
        out["c0"] = {"loss": float(l2).hex(), "aux": hexs(a2), "grad_sha256": digest_tree(g2), "attach_sha256": digest_tree(data[0]), "keys": sorted(data[0])}
        (l3, a3), g3 = make_engine(params, quartet_weight=0.0, chain_quartet_weight=0.0).paired_value_grad(params, *data, KEY)
        out["qc0"] = {"loss": float(l3).hex(), "aux": hexs(a3), "grad_sha256": digest_tree(g3), "attach_sha256": digest_tree(data[0]), "keys": sorted(data[0])}
        # residue quartet on, chain 0: identical to the v1 quartet objective (compared against the base tree's q-on run)
    if HAS_Q:
        dq = batch_of(QC); (l4, a4), g4 = make_engine(params, quartet_weight=0.3, **({"chain_quartet_weight": 0.0} if HAS_C else {})).paired_value_grad(params, *dq, KEY)
        out["q03"] = {"loss": float(l4).hex(), "aux": hexs(a4), "grad_sha256": digest_tree(g4)}
    print(json.dumps(out)); sys.exit(0)


def run_probe(tree):
    run = subprocess.run([sys.executable, __file__, "--probe", str(tree)], capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert run.returncode == 0, (str(tree), run.stderr[-3000:]); return json.loads(run.stdout.strip().splitlines()[-1])


# ================================================================ tests
def test_lambda0_bit_identical_to_base():
    """Chain weight 0 (default, explicit 0, with residue quartet 0, and residue quartet 0.3 + chain 0): loss, every aux term,
    gradient and the loader fields bit-identical to the base tree (quartet-model-v1b/src) on a fixed asym50 batch; the
    residue-quartet lambda-0 checks of v1 also hold."""
    q, b = run_probe(MODEL), run_probe(BASE)
    assert q["parameters"] == b["parameters"] == 180000
    assert "c0" in q and "c0" not in b and q["base"] == b["base"], [k for k in b["base"] if q["base"][k] != b["base"][k]]
    for case in ("c0", "qc0"):
        for k in ("loss", "aux", "grad_sha256", "attach_sha256", "keys"): assert q[case][k] == b["base"][k], (case, k)
    assert q["q03"] == b["q03"], "residue quartet 0.3 with chain 0 differs from the base tree"
    for k in ("loss", "aux", "grad_sha256", "attach_sha256", "keys"): assert q["q0"][k] == b["base"][k], ("q0", k)
    for k in ("loss", "aux", "grad_sha256", "attach_sha256", "keys"): assert q["q0_fields"][k] == b["base"][k], ("q0_fields", k)
    assert q["q0_fields"]["quartet_keys"] == ["quartet_idx", "quartet_mask"] and q["q0_fields"]["mask_sum"] > 0


def _pairing_functions():
    """quartet_pool / validate_quartets parsed from pairing-rank-v1 core.py (read-only; its module imports need the HPC)."""
    tree = ast.parse(PAIRING.read_text()); ns = {"np": np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("quartet_pool", "validate_quartets"):
            exec(compile(ast.Module([node], []), str(PAIRING), "exec"), ns)
    return ns["quartet_pool"], ns["validate_quartets"]


def table_sets(r):
    within = {(int(a), int(b)) for a, b, d in zip(r["i"], r["j"], r["d"]) if d <= 10.0}; return within | {(b, a) for a, b in within}


def test_quartet_pool_matches_pairing_rank():
    """The ported pool equals pairing-rank-v1's quartet_pool on the same support and seed (general and chemical)."""
    if not PAIRING.is_file(): raise AssertionError(f"reference {PAIRING} not found")
    their_pool, their_check = _pairing_functions(); checked = 0
    for s in range(7):
        cid = f"syn{s}"; _, free, _ = synthetic(s); r = ROWS[cid]; nodes = free["nodes"][:REAL]; active = free["node_mask"][:REAL] > 0
        g = pr.quartet_support(r, active, pr.quartet_classes(nodes))
        mine = pr.quartet_pool(g, np.random.default_rng([pr.QUARTET_POOL_SALT, 1, s])); theirs = their_pool(g, np.random.default_rng([pr.QUARTET_POOL_SALT, 1, s]))
        for a, b in zip(mine, theirs): assert np.array_equal(np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4))
        their_check(g, mine[0]); their_check(g, mine[1], True); checked += len(mine[0]) + len(mine[1])
    assert checked > 500, checked


def test_quartet_labels():
    """Every attached quartet (mask 1): i-j and k-l are <= 10 A table contacts with surface active endpoints; i-l and k-j are
    absent from the FULL <= 10 A table; same oriented chain pair (chain i = chain k < chain j = chain l); i != k, j != l;
    chemical rows share charge classes with a charged endpoint. Checked against the raw residue-table rows."""
    src = aux_source(QC); n_rows = 0; n_chem = 0
    for s in range(7):
        cid = f"syn{s}"; g = load_one(src, s); r = ROWS[cid]; idx = g["quartet_idx"][1]; m = g["quartet_mask"][1]
        assert idx.dtype == np.int32 and m.dtype == np.float32 and idx.shape == (QC, 4) and m.shape == (QC,)
        assert np.array_equal(g["quartet_idx"][0], idx) and np.array_equal(g["quartet_mask"][0], m)
        T = table_sets(r); surf = np.nan_to_num(r["rsa"][:, 1], nan=0.0) >= pr.SURFACE_RSA; ch = r["chain"]
        cls = pr.quartet_classes(synthetic(s)[1]["nodes"][:REAL]); general, chemical = pr.QUARTET_POOLS[cid]
        first_chem = (QC // 2 if len(chemical) else QC) if len(general) else 0  # draw layout: general rows, then chemical rows
        for pos, ((i, j, k, l), v) in enumerate(zip(idx.tolist(), m.tolist())):
            if v == 0: assert (i, j, k, l) == (0, 0, 0, 0); continue
            assert (i, j) in T and (k, l) in T and surf[[i, j, k, l]].all()
            assert (i, l) not in T and (k, j) not in T
            assert ch[i] == ch[k] and ch[j] == ch[l] and ch[i] < ch[j]
            assert i != k and j != l; n_rows += 1
            if pos >= first_chem:
                assert cls[i] == cls[k] and cls[j] == cls[l] and (cls[i] != 0 or cls[j] != 0); n_chem += 1
        if len(general) and len(chemical): assert m[:QC // 2].sum() == QC // 2 and m[QC // 2:].sum() == QC - QC // 2
    assert n_rows > 100 and n_chem > 10, (n_rows, n_chem)
    # the masked complex: one contact -> no quartet
    g7 = load_one(src, 7); assert float(g7["quartet_mask"].sum()) == 0 and pr.QUARTET_POOLS["syn7"][0].shape == (0, 4)


def test_quartet_charge_classes():
    nodes = np.zeros((7, 24), np.float32)
    for row, aa in enumerate("DEKRHAK"): nodes[row, pr.ALPHABET.index(aa)] = 1
    nodes[5, 20] = 1; nodes[6, 21] = 1  # N-terminal Ala -> +1, C-terminal Lys -> 0
    assert pr.quartet_classes(nodes).tolist() == [-1, -1, 1, 1, 0, 1, 0]


def test_quartet_streams_and_determinism():
    """Pools are deterministic per complex; visit draws differ; every other attached field is unchanged by the quartet streams."""
    a, b = aux_source(QC), aux_source(0)
    g1a, g1b = load_one(a, 0), load_one(b, 0); g2a = load_one(a, 0)
    for k in g1b: assert np.array_equal(np.asarray(g1a[k]), np.asarray(g1b[k])), k
    assert set(g1a) - set(g1b) == {"quartet_idx", "quartet_mask"}
    assert not np.array_equal(g1a["quartet_idx"], g2a["quartet_idx"])  # second visit, new draw
    pools = pr.QUARTET_POOLS.pop("syn0"); again = pr.quartet_pools(ROWS["syn0"], "syn0", synthetic(0)[1]["nodes"][:REAL], synthetic(0)[1]["node_mask"][:REAL] > 0)
    assert all(np.array_equal(x, y) for x, y in zip(pools, again))


def test_quartet_terms_values_and_masking():
    """quartet_terms vs a numpy reference: per-complex means over masked quartets, complexes without quartets (or invalid) excluded;
    an all-empty batch gives loss 0, accuracy 0, covered 0 and a zero, finite gradient."""
    params = recipe(); r = np.random.default_rng(3); B, n, W, Qn = 4, 30, 64, 6
    z = jnp.asarray(r.normal(0, 1, (B, n, W)), jnp.float32); idx = jnp.asarray(r.integers(0, n, (B, Qn, 4)), jnp.int32)
    mask = np.ones((B, Qn), np.float32); mask[1, 3:] = 0; mask[2] = 0; valid = jnp.asarray([True, True, True, False])
    loss, acc, cov = pr.quartet_terms(params, z, idx, jnp.asarray(mask), valid)
    e = jax.tree.map(np.asarray, params)["aux_redge"]
    def s(b, a, c):
        za, zc = np.asarray(z[b, a]), np.asarray(z[b, c]); f = np.concatenate((za * zc, np.abs(za - zc)))
        h = np.asarray(jax.nn.gelu(jnp.asarray(f @ e["hidden"]["w"] + e["hidden"]["b"]))); return float(h @ e["out"]["w"][:, 0] + e["out"]["b"][0])
    per_l, per_a = [], []
    for b in (0, 1):
        ls, as_ = [], []
        for q in range(Qn):
            if mask[b, q] == 0: continue
            i, j, k, l = np.asarray(idx[b, q]); d = s(b, i, j) + s(b, k, l) - s(b, i, l) - s(b, k, j); ls.append(np.logaddexp(0, 1 - d)); as_.append(float(d > 0))
        per_l.append(np.mean(ls)); per_a.append(np.mean(as_))
    assert abs(float(loss) - np.mean(per_l)) < 1e-5 and abs(float(acc) - np.mean(per_a)) < 1e-6 and abs(float(cov) - 2 / 3) < 1e-6, (float(loss), np.mean(per_l))
    zero = jnp.zeros((B, Qn), jnp.float32)
    out = pr.quartet_terms(params, z, idx, zero, valid); assert [float(x) for x in out] == [0.0, 0.0, 0.0]
    g = jax.grad(lambda p, zz: pr.quartet_terms(p, zz, idx, zero, valid)[0], argnums=(0, 1))(params, z)
    leaves = jax.tree.leaves(g); assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves) and all(float(jnp.abs(x).max()) == 0.0 for x in leaves)


def test_additive_score_cancels():
    """With an additive pair score s(a, b) = u(z_a) + v(z_b) (pair_logits_from replaced) delta is exactly 0 for every quartet:
    loss = softplus(1) and no gradient reaches the embeddings."""
    r = np.random.default_rng(4); B, n, W, Qn = 3, 25, 8, 10
    z = jnp.asarray(r.normal(0, 1, (B, n, W)), jnp.float32); idx = jnp.asarray(r.integers(0, n, (B, Qn, 4)), jnp.int32)
    u = jnp.asarray(r.normal(0, 1, W), jnp.float32); v = jnp.asarray(r.normal(0, 1, W), jnp.float32); real = pr.pair_logits_from
    pr.pair_logits_from = lambda p, za, zb: (za @ u) ** 3 + jnp.sin(zb @ v)
    try:
        loss, acc, _ = pr.quartet_terms({}, z, idx, jnp.ones((B, Qn)), jnp.ones(B, bool))
        gz = jax.grad(lambda zz: pr.quartet_terms({}, zz, idx, jnp.ones((B, Qn)), jnp.ones(B, bool))[0])(z)
    finally: pr.pair_logits_from = real
    # delta = 0 up to float rounding (so the accuracy is only rounding noise and is not asserted)
    assert abs(float(loss) - float(np.logaddexp(0, 1))) < 1e-5 and 0.0 <= float(acc) <= 1.0 and float(jnp.abs(gz).max()) < 1e-5, (float(loss), float(jnp.abs(gz).max()))


def test_objective_quartet_on():
    """lambda 1 on the asym50 batch: finite loss and gradient; total(lambda) - total(0) = lambda x slot 10; slot 10 equals
    quartet_terms on the unmasked free embeddings; the quartet term reaches the encoder and aux_redge, not the other heads;
    slots 0-9 equal the lambda-0 run."""
    params = recipe(); data = batch_of(QC)
    (l0, a0), g0 = make_engine(params).paired_value_grad(params, *data, KEY)
    (l1, a1), g1 = make_engine(params, quartet_weight=1.0).paired_value_grad(params, *data, KEY)
    (l3, a3), _ = make_engine(params, quartet_weight=0.3).paired_value_grad(params, *data, KEY)
    t1 = a1[4]; assert len(t1) == 13 and len(a0[4]) == 5
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g1)) and np.isfinite(float(l1))
    q, acc, cov = (float(x) for x in t1[10:13]); assert q > 0 and 0 <= acc <= 1 and abs(cov - 1.0) < 1e-6, (q, acc, cov)
    assert abs((float(l1) - float(l0)) - q) < 1e-5 and abs((float(l3) - float(l0)) - 0.3 * q) < 1e-5
    for s in range(5): assert float(t1[s]) == float(a0[4][s]), s
    assert all(float(t1[s]) == 0.0 for s in range(5, 10))
    eng = make_engine(params, quartet_weight=1.0)
    gq = jax.grad(lambda p: eng._paired_objective(p, *data, KEY)[1][4][10])(params)
    norm = lambda t: float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(t))))
    assert norm(gq["aux_redge"]) > 0 and norm({k: v for k, v in gq.items() if not k.startswith("aux_")}) > 0
    assert norm(gq["aux_burial"]) == 0 and norm(gq["aux_gate"]) == 0
    diff = jax.tree.map(lambda a, b: a - b, g1, g0); assert abs(norm(diff) - norm(gq)) < 1e-3 * max(norm(gq), 1e-6), (norm(diff), norm(gq))
    # reference: the slot equals quartet_terms on the unmasked free-branch embeddings from predict_all
    graphs = data[0]; flat = jax.tree.map(lambda x: x[:, 1], graphs); E = emb()
    keys = jax.random.split(KEY, graphs["nodes"].shape[0])  # paired_all: one key per complex, shared by its AB and free branches
    z = jax.vmap(lambda g, k: pr.residue_embedding(params, E(params, g, key=k, dropout_rate=0.1)[0], g["nodes"].shape[0]))(flat, keys)
    ref = pr.quartet_terms(params, z, graphs["quartet_idx"][:, 1], graphs["quartet_mask"][:, 1], data[-1])
    assert abs(float(ref[0]) - q) < 1e-5 and abs(float(ref[1]) - acc) < 1e-6, (float(ref[0]), q)


def test_masked_complex_excluded():
    """A batch whose only valid complexes have no quartet: slot 10 = 0, covered 0, total = the lambda-0 total, finite grads."""
    params = recipe(); data = batch_of(QC, seeds=(7, 7), B=4)
    assert float(jnp.sum(data[0]["quartet_mask"])) == 0
    (l0, _), _ = make_engine(params).paired_value_grad(params, *data, KEY)
    (l1, a1), g1 = make_engine(params, quartet_weight=1.0).paired_value_grad(params, *data, KEY)
    assert float(a1[4][10]) == 0.0 and float(a1[4][12]) == 0.0 and float(l1) == float(l0)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g1))
    # mixed: covered = 2 of 3 valid complexes
    (_, a2), _ = make_engine(params, quartet_weight=1.0).paired_value_grad(params, *batch_of(QC, seeds=(0, 2, 7), B=4), KEY)
    assert abs(float(a2[4][12]) - 2 / 3) < 1e-6


def test_grad_norms():
    """quartet_grad_norms: finite positive encoder norms that match per-term jax.grad references; ratio fields consistent."""
    params = recipe(); data = batch_of(QC); eng = make_engine(params, quartet_weight=1.0)
    rec = pt.quartet_grad_norms(eng, params, *data, KEY, epoch=1, number=3, updates=2)
    n = rec["norms"]; assert all(np.isfinite(v) and v > 0 for v in n.values()), n
    ref = {}
    for name, slot in (("within", 2), ("quartet", 10)):
        g = jax.grad(lambda p: eng._paired_objective(p, *data, KEY)[1][4][slot])(params)
        ref[name] = float(np.sqrt(sum(float(jnp.sum(jnp.square(x))) for k, v in g.items() if not k.startswith("aux_") for x in jax.tree.leaves(v))))
    assert abs(ref["within"] - n["within_encoder"]) < 1e-4 * ref["within"] and abs(ref["quartet"] - n["quartet_encoder"]) < 1e-4 * ref["quartet"], (ref, n)
    assert abs(rec["ratio_quartet_over_within_encoder"] - n["quartet_encoder"] / n["within_encoder"]) < 1e-9
    assert abs(rec["ratio_weighted_quartet_over_within_encoder"] - n["quartet_encoder"] / (0.3 * n["within_encoder"])) < 1e-9
    assert "aux_redge" not in rec["encoder_subtree"] and "blocks" in rec["encoder_subtree"] and rec["updates_before"] == 2


def test_flag_validation_and_cli():
    ok = dict(aux_weight=1.0, aux_heads="burial", residue_edge_weight=0.1, within_weight=0.3, variant=VARIANT, orientation=True, dihedrals=True)
    cases = [({**ok, "quartet_weight": 1.0, "within_weight": 0.0}, "--quartet-weight needs"), ({**ok, "quartet_weight": 1.0, "residue_edge_weight": 0.0}, "--quartet-weight needs"),
             ({**ok, "quartet_weight": 1.0, "alt_source": "x", "alt_prob": 1.0}, "--quartet-weight needs"), ({**ok, "quartet_weight": -1.0}, "-1.0"),
             ({**ok, "quartet_weight": 1.0, "quartet_count": 0}, "0"), ({**ok, "quartet_grad_norms_every_epoch": True}, "--quartet-grad-norms"),
             ({**ok, "smoke_batches": 50}, "--smoke-batches")]
    with tempfile.TemporaryDirectory() as tmp:
        for kw, text in cases:
            try: pt.train(tmp, "x", **kw)
            except ValueError as exc: assert text in str(exc), (kw, str(exc))
            else: raise AssertionError(("expected ValueError", kw))
        assert not any(Path(tmp).iterdir())
    P = recipe()
    for kw in ({"quartet_weight": 1.0, "within_weight": 0.0}, {"quartet_weight": -0.1}, {"quartet_weight": 1.0, "alt_rank_weight": 0.1}):
        try: make_engine(P, **kw)
        except ValueError: pass
        else: raise AssertionError(("engine accepted", kw))
    seen = {}; real = pt.train
    pt.train = lambda *a, **k: seen.update(a=a, k=k) or {"selected_epoch": 0, "validation": {"selection": 0.0}}
    try:
        os.environ.setdefault("PKABENCH_RUNTIME", tempfile.gettempdir())
        pt.main(["train", "x", "--quartet-weight", "0.3", "--quartet-count", "64", "--smoke", "--smoke-batches", "150", "--quartet-grad-norms"])
        b = inspect.signature(real).bind(*seen["a"], **seen["k"]).arguments
        assert b["quartet_weight"] == 0.3 and b["quartet_count"] == 64 and b["smoke_batches"] == 150 and b["quartet_grad_norms_every_epoch"] is True
        pt.main(["train", "x"]); b = inspect.signature(real).bind(*seen["a"], **seen["k"]).arguments
        assert b["quartet_weight"] == 0.0 and b["quartet_count"] is None and b["smoke_batches"] == 5 and b["quartet_grad_norms_every_epoch"] is False
    finally: pt.train = real


def test_provenance_and_history_names():
    assert pr.quartet_provenance(0.0, 128) == {}
    p = pr.quartet_provenance(0.3, 128)["quartet_rank"]
    assert p["weight"] == 0.3 and p["count"] == 128 and p["general"] == 64 and p["chemical"] == 64 and p["margin"] == 1.0 and "unmasked" in p["embeddings"]
    src = inspect.getsource(pt.train)
    assert '"pinder_quartet_softplus", "pinder_quartet_accuracy", "pinder_quartet_covered"' in src and "quartet_counts" in src
    assert pt.QUARTET_SLOT == 10 and pt.WITHIN_SLOT == 2


# ================================================================ quartet-rank-v2: chain quartets
POOL5 = Path(os.environ.get("QUARTET2_POOL5_DIR", "/private/tmp/claude-501/-Users-alexi-isambard-ssh/ff273e5c-e06f-4d00-b3bb-87e52de7a000/scratchpad/remote/p5"))
CHAIN_SEEDS = (0, 2, 3, 4, 5, 6)


def chain_data(rows=((0, 1, 0), (2, 3, 0), (4, 5, 1), (1, 4, 2)), count=16, role=None):
    """The asym50 batch of six distinct synthetic complexes (B 8) with chain_role / chain_quartet fields."""
    graphs, *rest = batch_of(QC, seeds=CHAIN_SEEDS, B=8)
    r = np.full(8, -1, np.int32); r[:len(CHAIN_SEEDS)] = 0
    if role is not None: r[:] = role
    cq = np.zeros((count, 4), np.int32)
    for i, (a, b, c) in enumerate(rows): cq[i] = (a, b, c, 1)
    return ({**graphs, "chain_role": jnp.asarray(r), "chain_quartet": jnp.asarray(cq)}, *rest)


def np_logit(e, za, zb):
    f = np.concatenate((za * zb, np.abs(za - zb)), axis=-1)
    h = np.asarray(jax.nn.gelu(jnp.asarray(f @ e["hidden"]["w"] + e["hidden"]["b"]))); return h @ e["out"]["w"][:, 0] + e["out"]["b"][0]


def np_pool(e, za, zb, tau=1.0, k=32):
    s = np.array([[np_logit(e, a, b) for b in zb] for a in za], np.float64) / tau; m = s.max()
    return tau * (m + np.log(np.exp(s - m).sum()) - np.log(k))


def test_chain_partner_independence():
    """Free-branch residue tokens of chain A are unchanged (exactly) when the partner chain's residue types, coordinates and own
    edges and A's masked cross-chain edge slots (features and partner indices) are replaced; the bound branch (control) changes."""
    params = recipe(); E = emb(); ab, free, _ = synthetic(0)
    A = np.arange(SPLIT); Bc = np.arange(SPLIT, REAL); r = np.random.default_rng(123)
    g2 = {k: v.copy() for k, v in free.items()}
    oh = np.zeros((len(Bc), 24), np.float32); oh[np.arange(len(Bc)), r.integers(0, 20, len(Bc))] = 1; oh[0, 20] = 1; oh[-1, 21] = 1; g2["nodes"][Bc, :24] = oh
    shift = r.normal(0, 3.0, 3)
    for c in range(4): g2["nodes"][Bc, 24 + 3 * c:27 + 3 * c] += shift + r.normal(0, 0.5, (len(Bc), 3))
    g2["edge"][Bc, :, :19] = r.normal(0, 1, (len(Bc), K, 19))
    cross = ~free["edge_mask"][A] & ab["edge_mask"][A]; assert cross.sum() > 50
    ea = g2["edge"][A]; ea[..., :19] = np.where(cross[..., None], r.normal(0, 1, ea[..., :19].shape), ea[..., :19]); g2["edge"][A] = ea
    nb = g2["neighbors"][A]; nb[cross] = r.integers(SPLIT, REAL, int(cross.sum())); g2["neighbors"][A] = nb
    z = lambda g, rate: np.asarray(pr.residue_embedding(params, E(params, {k: jnp.asarray(v) for k, v in g.items()}, key=KEY, dropout_rate=rate)[0], N))
    for rate in (0.0, 0.1):
        d_a = np.abs(z(free, rate)[A] - z(g2, rate)[A]).max(); d_b = np.abs(z(free, rate)[Bc] - z(g2, rate)[Bc]).max()
        assert d_a == 0.0 and d_b > 0.1, (rate, d_a, d_b)
    bound = np.abs(z(ab, 0.0)[A] - z({**g2, "edge_mask": ab["edge_mask"]}, 0.0)[A]).max(); assert bound > 0.1, bound


def test_chain_pool_matches_numpy():
    """chain_pool = tau * logsumexp(s / tau) - tau * log(k) over the masked pairs (numpy reference), several tau / k / masks."""
    params = recipe(); e = jax.tree.map(np.asarray, params)["aux_redge"]; r = np.random.default_rng(7)
    za = r.normal(0, 1, (9, 64)).astype(np.float32); zb = r.normal(0, 1, (7, 64)).astype(np.float32)
    for tau, k in ((1.0, 32), (0.5, 8), (2.0, 1)):
        ma = r.random(9) < 0.6; mb = r.random(7) < 0.6; ma[0] = mb[0] = True
        got = float(pr.chain_pool(params, jnp.asarray(za), jnp.asarray(ma), jnp.asarray(zb), jnp.asarray(mb), tau, k))
        ref = np_pool(e, za[ma], zb[mb], tau, k); assert abs(got - ref) < 2e-4 * max(1.0, abs(ref)), (tau, k, got, ref)
    # an empty side: finite value at the mask level, finite gradient
    g = jax.grad(lambda zz: pr.chain_pool(params, zz, jnp.zeros(9, bool), jnp.asarray(zb), jnp.ones(7, bool)))(jnp.asarray(za))
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) == 0.0


def test_chain_terms_values_and_masking():
    """chain_quartet_terms vs numpy (native / crossed pooled scores, softplus(1 - delta), accuracy, per-class counts); quartets
    with ok 0, an invalid complex, a role -1 complex or an empty side are excluded; none left -> 0 loss, zero finite gradient."""
    params = recipe(); e = jax.tree.map(np.asarray, params)["aux_redge"]; r = np.random.default_rng(11); B, n, W = 5, 12, 64
    z = r.normal(0, 1, (B, n, W)).astype(np.float32); chain = (np.arange(n) >= 6).astype(np.float32)[None].repeat(B, 0)
    nodes = np.zeros((B, n, 10), np.float32); nodes[..., -2] = chain; node_mask = np.ones((B, n), np.float32); node_mask[:, -1] = 0
    rsa = r.uniform(0, 0.5, (B, n)).astype(np.float32); rsa[3, 6:] = 0.0  # complex 3: B side not surface -> empty
    role = np.array([0, 1, 0, 0, -1], np.int32)  # complex 1: A side = chain code 1; complex 4: unusable
    sa, sb = pr.chain_sides(jnp.asarray(nodes), jnp.asarray(node_mask), jnp.asarray(rsa), jnp.asarray(role))
    sa, sb = np.asarray(sa), np.asarray(sb)
    assert not sb[3].any() and not sa[4].any() and not sb[4].any() and (sa[1] == ((chain[1] == 1) & (rsa[1] >= 0.1) & (node_mask[1] > 0))).all()
    q = np.array([[0, 1, 0, 1], [0, 2, 2, 1], [1, 2, 1, 1], [0, 3, 0, 1], [0, 4, 0, 1], [1, 2, 0, 0], [0, 2, 0, 1]], np.int32)
    valid = np.array([True, True, True, True, True])
    out = pr.chain_quartet_terms(params, jnp.asarray(z), jnp.asarray(sa), jnp.asarray(sb), jnp.asarray(q), jnp.asarray(valid))
    S = lambda a, b: np_pool(e, z[a][sa[a]], z[b][sb[b]])
    used = [0, 1, 2, 6]; d = np.array([S(q[i, 0], q[i, 0]) + S(q[i, 1], q[i, 1]) - S(q[i, 0], q[i, 1]) - S(q[i, 1], q[i, 0]) for i in used])
    loss, acc, count, spsum, corr, cnt = [np.asarray(x) for x in out]
    assert int(count) == 4 and abs(float(loss) - np.mean(np.logaddexp(0, 1 - d))) < 1e-4 and abs(float(acc) - np.mean(d > 0)) < 1e-6, (float(loss), d)
    assert abs(float(spsum) - np.sum(np.logaddexp(0, 1 - d))) < 4e-4 and cnt.tolist() == [2.0, 1.0, 1.0]
    assert corr.tolist() == [float(np.sum(d[[0, 3]] > 0)), float(d[2] > 0), float(d[1] > 0)]
    # complex 2 invalid -> quartets 1, 2, 6 drop
    out2 = pr.chain_quartet_terms(params, jnp.asarray(z), jnp.asarray(sa), jnp.asarray(sb), jnp.asarray(q), jnp.asarray([True, True, False, True, True]))
    assert int(out2[2]) == 1
    none = q.copy(); none[:, 3] = 0
    f = lambda p, zz: pr.chain_quartet_terms(p, zz, jnp.asarray(sa), jnp.asarray(sb), jnp.asarray(none), jnp.asarray(valid))[0]
    assert float(f(params, jnp.asarray(z))) == 0.0
    leaves = jax.tree.leaves(jax.grad(f, argnums=(0, 1))(params, jnp.asarray(z)))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves) and all(float(jnp.abs(x).max()) == 0.0 for x in leaves)


def test_chain_additive_score_cancels():
    """With an additive pair score s(a, b) = u(z_a) + v(z_b) the pooled score is f(A) + g(B), every chain quartet has delta 0:
    loss = softplus(1) and the embedding gradient is ~0."""
    r = np.random.default_rng(4); B, n, W = 4, 10, 8
    z = jnp.asarray(r.normal(0, 1, (B, n, W)), jnp.float32); u = jnp.asarray(r.normal(0, 1, W), jnp.float32); v = jnp.asarray(r.normal(0, 1, W), jnp.float32)
    sa = jnp.asarray(r.random((B, n)) < 0.5).at[:, 0].set(True); sb = jnp.asarray(r.random((B, n)) < 0.5).at[:, 1].set(True)
    q = jnp.asarray([[0, 1, 0, 1], [1, 2, 0, 1], [2, 3, 1, 1], [0, 3, 2, 1]], jnp.int32); valid = jnp.ones(B, bool); real = pr.pair_logits_from
    pr.pair_logits_from = lambda p, za, zb: jnp.tanh(za @ u) * 3 + (zb @ v) ** 2
    try:
        out = pr.chain_quartet_terms({}, z, sa, sb, q, valid)
        gz = jax.grad(lambda zz: pr.chain_quartet_terms({}, zz, sa, sb, q, valid)[0])(z)
    finally: pr.pair_logits_from = real
    assert abs(float(out[0]) - float(np.logaddexp(0, 1))) < 1e-5 and float(jnp.abs(gz).max()) < 1e-5, (float(out[0]), float(jnp.abs(gz).max()))


def meta_table(tmp):
    rows = [("1aaa__A1_P1--1aaa__B1_P2", "hetero", "g1"), ("2bbb__A1_P3--2bbb__B1_P4", "hetero", "g2"), ("3ccc__A1_P1--3ccc__B1_P5", "hetero", "g3"),
            ("4ddd__A1_P6--4ddd__B1_P7", "hetero", "g1"), ("5eee__A1_P8--5eee__A2_P8", "homo", "g5"), ("6fff__A1_P9--6fff__A2_P9", "homo", "g6"),
            ("6u42__LP1_A8HSW0--6u42__HP1_UNDEFINED", "Ab/Ag", "g7"), ("7ggg__A1_UNDEFINED--7ggg__C1_Q1", "Ab/Ag", "g8"),
            ("8hhh__H1_UNDEFINED--8hhh__L1_UNDEFINED", "Ab/Ag", "g9"), ("9iii__B1_UNDEFINED--9iii__C1_A8HSW0", "Ab/Ag", "g10"),
            ("9jjj__B1_UNDEFINED--9jjj__C1_Q2", "Ab/Ag", "g11")]
    path = Path(tmp) / "pinder.tsv"
    path.write_text("id\tpdb_id\tgroup\tstratum\n" + "".join(f"{i}\t{i[:4]}\t{g}\t{s}\n" for i, s, g in rows)); return path, [r[0] for r in rows]


def test_chain_metadata_and_batch_pairs():
    """Roles (Ab/Ag antibody = the UNDEFINED chain in chain-name order; two UNDEFINED -> unclassed), same-class pairs only,
    guard skips (same group; shared non-UNDEFINED UniProt, antibodies never match), count cap, determinism, padding slots."""
    with tempfile.TemporaryDirectory() as tmp:
        path, ids = meta_table(tmp); meta = pr.chain_metadata(path)
    assert meta[ids[0]] == (0, 0, "g1", ("P1", "P2")) and meta[ids[4]][:2] == (1, 0)
    assert meta[ids[6]][:2] == (2, 0) and meta[ids[6]][3] == ("UNDEFINED", "A8HSW0")  # HP < LP: antibody = chain code 0
    assert meta[ids[7]][:2] == (2, 0) and meta[ids[8]][:2] == (-1, -1) and meta[ids[9]][:2] == (2, 0)
    assert pr.chain_guard(meta[ids[0]], meta[ids[3]]) == "group" and pr.chain_guard(meta[ids[0]], meta[ids[2]]) == "uniprot"
    assert pr.chain_guard(meta[ids[6]], meta[ids[9]]) == "uniprot" and pr.chain_guard(meta[ids[7]], meta[ids[10]]) is None
    stats = pr.chain_stats(); role, cq = pr.chain_quartet_batch(ids, 16, meta, 16, np.random.default_rng(1), stats)
    assert role.tolist()[:11] == [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0] and (role[11:] == -1).all()
    got = {tuple(sorted((int(a), int(b)))) for a, b, c, ok in cq if ok}
    assert got == {(0, 1), (1, 2), (1, 3), (2, 3), (4, 5), (6, 7), (6, 10), (7, 9), (7, 10), (9, 10)}, got  # (0,2) uniprot, (0,3) group, (6,9) uniprot
    for a, b, c, ok in cq:
        if ok: assert meta[ids[a]][0] == meta[ids[b]][0] == c
    assert stats["pairs_same_class"] == [6, 1, 6] and stats["skipped_group"] == [1, 0, 0] and stats["skipped_uniprot"] == [1, 0, 1]
    assert stats["drawn"] == [4, 1, 5] and stats["abag_slots"] == 4 and stats["batches_abag_ge2"] == 1 and stats["unclassed_slots"] == 1
    _, cap = pr.chain_quartet_batch(ids, 16, meta, 3, np.random.default_rng(1)); assert int(cap[:, 3].sum()) == 3
    _, again = pr.chain_quartet_batch(ids, 16, meta, 3, np.random.default_rng(1)); assert np.array_equal(cap, again)
    _, other = pr.chain_quartet_batch(ids, 16, meta, 3, np.random.default_rng(2)); assert not np.array_equal(cap, other)


def check_regroup(plans, new, info, bucket_of, meta, minimum):
    abag = lambda c: meta.get(c, (-1,))[0] == 2
    assert len(new) == len(plans) and [len(b) for b in new] == [len(b) for b in plans]
    assert sorted(c for b in new for c in b) == sorted(c for b in plans for c in b)
    for old, b in zip(plans, new): assert {bucket_of[c] for c in b} == {bucket_of[c] for c in old} and len({bucket_of[c] for c in b}) == 1
    per = {}
    for b in new: per.setdefault(bucket_of[b[0]], []).append(sum(abag(c) for c in b))
    for name, counts in per.items():
        total = sum(counts)
        if total >= minimum:
            sizes = info["buckets"][name]["sizes"]; assert sorted(c for c in counts if c) == sorted(sizes), (name, counts, sizes)
            assert min(sizes) >= min(minimum, total // len(sizes)), (name, sizes)
    return per


def test_chain_regroup_synthetic():
    """Regrouping permutes slots inside a bucket only: same sizes, buckets and visited multiset; every Ab/Ag-bearing batch holds
    >= minimum Ab/Ag when the bucket has that many; no Ab/Ag -> identical plans; deterministic."""
    r = np.random.default_rng(0); meta = {}; bucket_of = {}; ids = []
    for i in range(400):
        cid = f"c{i}"; code = int(r.choice([0, 1, 2], p=[0.55, 0.4, 0.05])); meta[cid] = (code, 0, f"g{i}", (f"U{i}", f"V{i}")); bucket_of[cid] = str([128, 256, 384][i % 3]); ids.append(cid)
    by = {}
    for c in ids: by.setdefault(bucket_of[c], []).append(c)
    plans = [b[i:i + 16] for b in by.values() for i in range(0, len(b), 16)]; plans = [plans[i] for i in r.permutation(len(plans))]
    for minimum in (2, 3, 4, 16):
        new, info = pr.chain_regroup(plans, meta, bucket_of, minimum); check_regroup(plans, new, info, bucket_of, meta, minimum)
        assert new == pr.chain_regroup(plans, meta, bucket_of, minimum)[0] and info["abag"] == sum(meta[c][0] == 2 for c in ids)
    none = {c: (0,) + v[1:] for c, v in meta.items()}; same, info = pr.chain_regroup(plans, none, bucket_of, 2); assert same == plans and info["moved"] == 0
    try: pr.chain_regroup([[ids[0], ids[1]]], meta, bucket_of, 2)
    except AssertionError: pass
    else: raise AssertionError("mixed-bucket plan accepted")


def test_chain_regroup_pool5_real():
    """On the real pool-v5 manifest (local copy of training/pool-v5; skipped when absent): epochs 1-3 at seed 17, fraction 0.5,
    batch 16: invariants hold; prints the class mix and the expected Ab/Ag chain quartets per epoch."""
    if not (POOL5 / "pinder-manifest.json").is_file(): print("  (pool-v5 copy not found; real regroup check skipped)"); return
    from pkatrain.loading import BucketPolicy
    manifest = json.loads((POOL5 / "pinder-manifest.json").read_text()); meta = pr.chain_metadata(POOL5 / "pinder.tsv")
    train = pl.select(manifest, "train", 0.5); assert all(r["id"] in meta for r in train)
    policy = pt._policy(manifest, 16); bucket_of = {r["id"]: policy.bucket(r["n"]) for r in train}
    for epoch in (1, 2, 3):
        plans = policy.plans(train, np.random.default_rng((17, epoch, 1)))  # = production_train.epoch_plans PINDER plans
        for minimum in ((2, 4, 8, 16) if epoch == 1 else (2,)):
            new, info = pr.chain_regroup(plans, meta, bucket_of, minimum); check_regroup(plans, new, info, bucket_of, meta, minimum)
            stats = pr.chain_stats()
            for number, ids in enumerate(new, 1): pr.chain_quartet_batch(ids, 16, meta, 16, np.random.default_rng([pr.CHAIN_SALT, 17, epoch, number]), stats)
            base = pr.chain_stats()
            for number, ids in enumerate(plans, 1): pr.chain_quartet_batch(ids, 16, meta, 16, np.random.default_rng([pr.CHAIN_SALT, 17, epoch, number]), base)
            print(f"  epoch {epoch} abag_min {minimum}: batches {len(new)}, Ab/Ag (role) {info['abag']}, Ab/Ag batches {info['abag_batches']}, moved {info['moved']}, "
                  f"unpaired {info['unpaired_abag']}; drawn quartets hetero/homo/abag {stats['drawn']} (without regroup {base['drawn']}); "
                  f"skipped group {stats['skipped_group']} uniprot {stats['skipped_uniprot']}; batches with >= 2 Ab/Ag {stats['batches_abag_ge2']} (without {base['batches_abag_ge2']})")
            assert stats["drawn"][2] > base["drawn"][2] and info["unpaired_abag"] < minimum * len(set(bucket_of.values())) and stats["batches_abag_ge2"] == info["abag_batches"]


def test_objective_chain_on():
    """Residue quartet 0.3 + chain 1 on the asym50 batch: 23 aux slots; total(c) - total(0) = c x slot 13 (c 1, 0.5); slots 0-12
    equal the chain-0 run; slot 13 equals chain_quartet_terms on independently recomputed unmasked free embeddings (same
    dropout keys); the chain gradient reaches the encoder and aux_redge, not aux_burial / aux_gate; chain-only (quartet 0) works."""
    params = recipe(); data = chain_data(); plain = (dict((k, v) for k, v in data[0].items() if k not in pr.CHAIN_FIELDS), *data[1:])
    (l0, a0), g0 = make_engine(params, quartet_weight=0.3).paired_value_grad(params, *plain, KEY)
    (l1, a1), g1 = make_engine(params, quartet_weight=0.3, chain_quartet_weight=1.0).paired_value_grad(params, *data, KEY)
    (l5, a5), _ = make_engine(params, quartet_weight=0.3, chain_quartet_weight=0.5).paired_value_grad(params, *data, KEY)
    t1 = a1[4]; assert len(t1) == 23 and len(a0[4]) == 13
    c = float(t1[13]); assert c > 0 and int(t1[15]) == 4 and [float(x) for x in t1[20:23]] == [2.0, 1.0, 1.0]
    assert abs((float(l1) - float(l0)) - c) < 1e-5 and abs((float(l5) - float(l0)) - 0.5 * c) < 1e-5
    for s in range(13): assert float(t1[s]) == float(a0[4][s]), s
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g1))
    eng = make_engine(params, quartet_weight=0.3, chain_quartet_weight=1.0)
    gc = jax.grad(lambda p: eng._paired_objective(p, *data, KEY)[1][4][13])(params)
    norm = lambda t: float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(t))))
    assert norm(gc["aux_redge"]) > 0 and norm({k: v for k, v in gc.items() if not k.startswith("aux_")}) > 0 and norm(gc["aux_burial"]) == 0 and norm(gc["aux_gate"]) == 0
    graphs = plain[0]; flat = jax.tree.map(lambda x: x[:, 1], graphs); E = emb(); keys = jax.random.split(KEY, graphs["nodes"].shape[0])
    z = jax.vmap(lambda g, k: pr.residue_embedding(params, E(params, g, key=k, dropout_rate=0.1)[0], g["nodes"].shape[0]))(flat, keys)
    sa, sb = pr.chain_sides(graphs["nodes"][:, 1], graphs["node_mask"][:, 1], graphs["rsa_residue"][:, 1], data[0]["chain_role"])
    # the chain column of the raw nodes equals the residue table's chain code
    for slot, seed in enumerate(CHAIN_SEEDS): assert np.array_equal(np.asarray(graphs["nodes"][slot, 1, :REAL, -2]), ROWS[f"syn{seed}"]["chain"].astype(np.float32))
    ref = pr.chain_quartet_terms(params, z, sa, sb, data[0]["chain_quartet"], data[-1]); assert abs(float(ref[0]) - c) < 1e-5, (float(ref[0]), c)
    (lc, ac), gco = make_engine(params, chain_quartet_weight=1.0).paired_value_grad(params, *data, KEY)  # chain only
    assert len(ac[4]) == 23 and float(ac[4][10]) == 0.0 and abs(float(ac[4][13]) - c) < 1e-5 and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(gco))
    (lp, _), _ = make_engine(params).paired_value_grad(params, *plain, KEY); assert abs((float(lc) - float(lp)) - c) < 1e-5
    # no usable quartet: term 0, total = chain-0 total
    empty = chain_data(role=-1); (le, ae), ge = make_engine(params, quartet_weight=0.3, chain_quartet_weight=1.0).paired_value_grad(params, *empty, KEY)
    assert float(ae[4][13]) == 0.0 and float(ae[4][15]) == 0.0 and float(le) == float(l0) and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(ge))


def test_grad_norms_chain():
    """The diagnostic's within / quartet / chain encoder norms match per-term jax.grad references; ratio fields consistent."""
    params = recipe(); data = chain_data(); eng = make_engine(params, quartet_weight=0.3, chain_quartet_weight=1.0)
    rec = pt.quartet_grad_norms(eng, params, *data, KEY, epoch=1, number=3, updates=2); n = rec["norms"]
    assert all(np.isfinite(v) and v > 0 for v in n.values()), n
    for name, slot in (("within", 2), ("quartet", 10), ("chain", 13)):
        g = jax.grad(lambda p: eng._paired_objective(p, *data, KEY)[1][4][slot])(params)
        ref = float(np.sqrt(sum(float(jnp.sum(jnp.square(x))) for k, v in g.items() if not k.startswith("aux_") for x in jax.tree.leaves(v))))
        assert abs(ref - n[f"{name}_encoder"]) < 1e-4 * ref, (name, ref, n)
    assert abs(rec["ratio_chain_over_within_encoder"] - n["chain_encoder"] / n["within_encoder"]) < 1e-9
    assert abs(rec["ratio_weighted_chain_over_within_encoder"] - n["chain_encoder"] / (0.3 * n["within_encoder"])) < 1e-9
    assert abs(rec["ratio_weighted_quartet_over_within_encoder"] - 0.3 * n["quartet_encoder"] / (0.3 * n["within_encoder"])) < 1e-9
    only = make_engine(params, chain_quartet_weight=2.0); r2 = pt.quartet_grad_norms(only, params, *data, KEY)
    assert "quartet_encoder" not in r2["norms"] and "chain_encoder" in r2["norms"] and r2["weights"] == {"within": 0.3, "chain": 2.0}


def test_chain_flags_and_cli():
    ok = dict(aux_weight=1.0, aux_heads="burial", residue_edge_weight=0.1, within_weight=0.3, variant=VARIANT, orientation=True, dihedrals=True)
    cases = [({**ok, "chain_quartet_weight": 1.0, "within_weight": 0.0}, "--chain-quartet-weight needs"),
             ({**ok, "chain_quartet_weight": 1.0, "residue_edge_weight": 0.0}, "--chain-quartet-weight needs"),
             ({**ok, "chain_quartet_weight": 1.0, "alt_source": "x", "alt_prob": 1.0}, "--chain-quartet-weight needs"),
             ({**ok, "chain_quartet_weight": -1.0}, "chain quartet"), ({**ok, "chain_quartet_weight": 1.0, "chain_quartet_count": 0}, "chain quartet"),
             ({**ok, "chain_quartet_weight": 1.0, "chain_quartet_tau": 0.0}, "chain quartet"), ({**ok, "chain_quartet_weight": 1.0, "chain_quartet_abag_min": 1}, "chain quartet"),
             ({**ok, "chain_quartet_count": 8}, "need --chain-quartet-weight"), ({**ok, "quartet_grad_norms_every_epoch": True}, "--quartet-grad-norms")]
    with tempfile.TemporaryDirectory() as tmp:
        for kw, text in cases:
            try: pt.train(tmp, "x", **kw)
            except ValueError as exc: assert text in str(exc), (kw, str(exc))
            else: raise AssertionError(("expected ValueError", kw))
        assert not any(Path(tmp).iterdir())
    P = recipe()
    for kw in ({"chain_quartet_weight": 1.0, "within_weight": 0.0}, {"chain_quartet_weight": -0.1}, {"chain_quartet_weight": 1.0, "alt_rank_weight": 0.1},
               {"chain_quartet_weight": 1.0, "chain_quartet_tau": 0.0}):
        try: make_engine(P, **kw)
        except ValueError: pass
        else: raise AssertionError(("engine accepted", kw))
    seen = {}; real = pt.train
    pt.train = lambda *a, **k: seen.update(a=a, k=k) or {"selected_epoch": 0, "validation": {"selection": 0.0}}
    try:
        os.environ.setdefault("PKABENCH_RUNTIME", tempfile.gettempdir())
        pt.main(["train", "x", "--quartet-weight", "0.3", "--chain-quartet-weight", "1.5", "--chain-quartet-count", "8", "--chain-quartet-tau", "0.5",
                 "--chain-quartet-k", "16", "--chain-quartet-abag-min", "3", "--pinder-manifest", "m", "--pinder-extra", "e", "--mask-store-extra", "s"])
        b = inspect.signature(real).bind(*seen["a"], **seen["k"]).arguments
        assert (b["chain_quartet_weight"], b["chain_quartet_count"], b["chain_quartet_tau"], b["chain_quartet_k"], b["chain_quartet_abag_min"]) == (1.5, 8, 0.5, 16, 3)
        assert b["quartet_weight"] == 0.3 and (b["pinder_manifest"], b["pinder_extra"], b["mask_store_extra"]) == ("m", "e", "s")
        pt.main(["train", "x"]); b = inspect.signature(real).bind(*seen["a"], **seen["k"]).arguments
        assert b["chain_quartet_weight"] == 0.0 and all(b[k] is None for k in ("chain_quartet_count", "chain_quartet_tau", "chain_quartet_k", "chain_quartet_abag_min"))
    finally: pt.train = real


def test_chain_history_provenance_and_sampler_guard():
    """chain_history pooled ratios; provenance block; the train loop touches the sampler / batch only under chain weight > 0."""
    means = np.zeros(23); means[13:23] = [0.9, 0.5, 4.0, 3.2, 1.0, 0.5, 0.25, 2.0, 1.5, 0.5]
    h = pt.chain_history(means)
    assert abs(h["pinder_chain_softplus"] - 0.8) < 1e-12 and abs(h["pinder_chain_accuracy"] - 1.75 / 4) < 1e-12 and h["pinder_chain_quartets_per_batch"] == 4.0
    assert h["pinder_chain_accuracy_hetero"] == 0.5 and abs(h["pinder_chain_accuracy_homo"] - 1 / 3) < 1e-12 and h["pinder_chain_accuracy_abag"] == 0.5
    assert h["pinder_chain_quartets_abag_per_batch"] == 0.5
    zero = np.zeros(23); zero[15] = 0.0; assert pt.chain_history(zero)["pinder_chain_accuracy"] is None
    assert pr.chain_provenance(0.0, 16, 1.0, 32, 2, "x") == {}
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = meta_table(tmp); p = pr.chain_provenance(1.0, 16, 1.0, 32, 2, path)["chain_quartet"]
    assert (p["weight"], p["count"], p["tau"], p["k"], p["abag_min"], p["margin"]) == (1.0, 16, 1.0, 32, 2, 1.0) and len(p["pool_tsv_sha256"]) == 64
    src = inspect.getsource(pt.train)
    assert "if chain_weight: pair, chain_regroup = production_residue.chain_regroup(" in src and "                if chain_weight:  # quartet-rank-v2: batch-level" in src
    assert src.count("chain_regroup(") == 1 and src.count("chain_quartet_batch(") == 1 and pt.CHAIN_SLOT == 13 and len(pt.CHAIN_NAMES) == pr.CHAIN_SLOTS
    assert pr.chain_pool_path("/R", {"pool": "/x/pool-v5/pinder.tsv"}) == Path("/x/pool-v5/pinder.tsv") and pr.chain_pool_path("/R", {}) == Path("/R/training/pool-v4/pinder.tsv")


def test_score_chain_functions():
    """score_chain: pooled() equals chain_pool; val_pairs class / guard / partner rules; evaluate() quartet deltas and Spearman."""
    import score_chain as sc
    params = recipe(); e = jax.tree.map(np.asarray, params)["aux_redge"]; r = np.random.default_rng(5)
    za = r.normal(0, 1, (6, 64)).astype(np.float32); zb = r.normal(0, 1, (5, 64)).astype(np.float32)
    ref = float(pr.chain_pool(params, jnp.asarray(za), jnp.ones(6, bool), jnp.asarray(zb), jnp.ones(5, bool)))
    assert abs(sc.pooled(e, za, zb) - ref) < 2e-4 and sc.pooled(e, za[:0], zb) is None
    with tempfile.TemporaryDirectory() as tmp:
        path, ids = meta_table(tmp); meta = pr.chain_metadata(path)
    pairs, skipped = sc.val_pairs(ids, meta, partners=2)
    assert all(meta[a][0] == meta[b][0] == c for a, b, c in pairs) and len({tuple(sorted(p[:2])) for p in pairs}) == len(pairs)
    assert all(pr.chain_guard(meta[a], meta[b]) is None for a, b, _ in pairs) and ids[8] not in {x for p in pairs for x in p[:2]}
    assert skipped[0]["group"] > 0 and skipped[0]["uniprot"] > 0
    Z = {c: r.normal(0, 1, (8, 64)).astype(np.float32) for c in ids}; R = {c: (np.arange(0, 4), np.arange(4, 8)) for c in ids}
    out = sc.evaluate(e, Z, R, meta, pairs)
    a, b, _ = pairs[0]; S = lambda x, y: sc.pooled(e, Z[x][R[x][0]], Z[y][R[y][1]])
    d0 = S(a, a) + S(b, b) - S(a, b) - S(b, a); assert out["quartets"]["all"]["n"] == len(pairs) and np.isfinite(d0)
    assert sc.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and sc.spearman([1, 2], [1, 2]) is None and abs(sc.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert "| a | r |" in sc.markdown({"tau": 1.0, "k": 32, "partners": 2, "pairs_per_class": {}, "runs": {"a": {"run": "r", "metrics": out}}})

def _write_reports(aud, where, arm, run, g):
    """Minimal masked2 / residue / quartet / chain reports for one arm with gate level g."""
    acc = lambda x: {"accuracy": x, "n": 10}
    m2 = {"runs": {arm: {"run": run, "modes": {"unmasked": {"swap_same_stratum": {"all": acc(0.80 + g), "Ab/Ag": acc(0.68 + g), "Ab/Ag varying antibody": acc(0.69 + g),
                                                                                    "Ab/Ag varying antigen": acc(0.67 + g)},
                                                              "wrong_partner": {"different_protein": acc(0.60 + 2 * g), "abag_wrong_antibody": acc(0.51 + g)},
                                                              "within_complex": {"all": {"auroc": 0.66}}, "edges_10A_free": {"auroc": 0.73, "ap": 0.05}}}}}}
    res = {arm: {"run": run, "pka": {"selection_no_bonded": 0.61 + g / 10, "state_no_bonded": 0.5, "interface_no_bonded": 0.3}}}
    qu = {"runs": {arm: {"run": run, "metrics": {"overall": {"all": acc(0.6)}, "hetero": {"all": acc(0.6)}, "homo": {"all": acc(0.65)}, "Ab/Ag": {"all": acc(0.57)}}}}}
    ch = {"runs": {arm: {"run": run, "metrics": {"quartets": {"all": acc(0.55 + g), "hetero": acc(0.56), "homo": acc(0.6), "Ab/Ag": acc(0.5 + g)},
                                                  "length_spearman": {"all": -0.2, "hetero": -0.1, "homo": 0.0, "Ab/Ag": -0.3}}}}}
    for step, rep in (("masked2", m2), ("residue", res), ("quartet", qu)):
        d = aud / where / "score" / arm / step; d.mkdir(parents=True, exist_ok=True); (d / "report.json").write_text(json.dumps(rep))
    d = aud / "quartet-rank-v2" / "score" / arm / "chain"; d.mkdir(parents=True, exist_ok=True); (d / "report.json").write_text(json.dumps(ch))


def test_compile_v2():
    """compile.py: tables with deltas vs control and vs q03, decision flags, training-term table from history / protocol; exit 1
    with every missing input listed."""
    with tempfile.TemporaryDirectory() as tmp:
        aud = Path(tmp) / "audits"; runs = Path(tmp) / "runs"
        for arm, where, g in (("control", "abdata-v5", 0.0), ("q03", "abdata-v5", 0.02), ("qc", "quartet-rank-v2", 0.03), ("conly", "quartet-rank-v2", -0.01)):
            run = f"pool5-v1-50-asym50-{arm}-w064-s17" if arm in ("control", "q03") else f"pool5-v2-50-asym50-{arm}-w064-s17"; _write_reports(aud, where, arm, run, g)
            if arm in ("qc", "conly"):
                out = runs / run; out.mkdir(parents=True)
                (out / "selection.json").write_text(json.dumps({"selected_epoch": 2, "epochs_run": 3, "parameter_count": 180000}))
                (out / "protocol.json").write_text(json.dumps({"chain_quartet": {"weight": 0.5}, **({"quartet_rank": {"weight": 0.3}} if arm == "qc" else {})}))
                tl = {"pinder_chain_softplus": 1.2, "pinder_chain_accuracy": 0.6, "pinder_within_listwise": 2.0}
                rows = [{"epoch": e, "train_seconds": 180.0, "train_loss": tl, "chain_regroup": {"abag": 264, "moved": 90, "buckets": {}}, "chain_quartet_counts": {"drawn": [1, 2, 3]},
                         **({"debug_quartet_grad_norms": [{"epoch": 1, "batch": 150, "ratio_chain_over_within_encoder": 0.5, "ratio_weighted_chain_over_within_encoder": 1.0}]} if e == 1 else {})}
                        for e in (1, 2, 3)]
                (out / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        cmd = [sys.executable, str(HERE / "compile.py"), "--aud", str(aud), "--runs-dir", str(runs)]
        ok = subprocess.run(cmd, capture_output=True, text=True); assert ok.returncode == 0, ok.stderr; s = ok.stdout
        assert "| qc | pool5-v2-50-asym50-qc-w064-s17 | 0.6130 (+0.0030; q03 +0.0010)" in s, s
        assert "- qc: Ab/Ag swap >= control yes, wrong-partner >= +0.04 yes, swap all >= +0.02 yes, pKa clean <= +0.012 yes -> SUCCESS" in s, s
        assert "- conly: Ab/Ag swap >= control NO" in s and "chain/within (encoder, unweighted) 0.500" in s and "quartet 0.3, chain 0.5" in s
        (aud / "quartet-rank-v2/score/conly/chain/report.json").unlink(); (aud / "abdata-v5/score/q03/residue/report.json").unlink()
        bad = subprocess.run(cmd, capture_output=True, text=True)
        assert bad.returncode == 1 and "conly" in bad.stderr and "chain" in bad.stderr and "q03" in bad.stderr and "residue" in bad.stderr, bad.stderr


def main():
    tests = [(name, f) for name, f in globals().items() if name.startswith("test_") and callable(f)]
    only = os.environ.get("TESTS"); tests = [t for t in tests if not only or any(o in t[0] for o in only.split(","))]
    failed = []
    for name, f in tests:
        try: f(); print(f"PASS {name}", flush=True)
        except Exception:  # noqa: BLE001
            failed.append(name); print(f"FAIL {name}", flush=True); traceback.print_exc()
    print(f"passed {len(tests) - len(failed)}/{len(tests)}" + (f"  failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    probe() if PROBE else main()
