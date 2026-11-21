"""Local CPU tests of quartet-rank-v1 (2026-11-15): the quartet tree (work/quartet-model/src) against its base, the tree that trained
the control drsa-v1-50-within03-asym50-w064-s17 (work/drsa-within-model/src, read-only).

  JAX_PLATFORMS=cpu GQT_PARAMETER_SCALING=1 python test_quartet.py [MODEL_SRC] [BASE_SRC]
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
BASE = Path(_ARGS[1]).resolve() if len(_ARGS) > 1 else HERE.parent / "drsa-within-model" / "src"
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

HAS_Q = hasattr(pr, "quartet_terms")
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
    print(json.dumps(out)); sys.exit(0)


def run_probe(tree):
    run = subprocess.run([sys.executable, __file__, "--probe", str(tree)], capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert run.returncode == 0, (str(tree), run.stderr[-3000:]); return json.loads(run.stdout.strip().splitlines()[-1])


# ================================================================ tests
def test_lambda0_bit_identical_to_base():
    """lambda 0 (default, explicit 0, and 0 with quartet fields attached): loss, every aux term, gradient and the other loader
    fields bit-identical to the base tree (drsa-within-model/src) on a fixed asym50 batch."""
    q, b = run_probe(MODEL), run_probe(BASE)
    assert q["parameters"] == b["parameters"] == 180000
    assert "q0" not in b and q["base"] == b["base"], [k for k in b["base"] if q["base"][k] != b["base"][k]]
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


def test_score_graft_functions():
    import score_graft as sg
    assert sg.split_run("r") == ("r", None) and sg.split_run("r@/a/b.npz") == ("r", "/a/b.npz")
    params = recipe(); cur = {k: np.asarray(v) for k, v in sg.current_head(params).items()}
    with tempfile.TemporaryDirectory() as tmp:
        same = Path(tmp) / "baseline-edge.npz"; np.savez(same, **cur)
        g, diff = sg.graft_params(params, sg.load_head(same)); assert all(v == 0.0 for v in diff.values())
        assert digest_tree(g) == digest_tree(params) and jax.tree.structure(g) == jax.tree.structure(params)
        other = {k: v + 0.5 for k, v in cur.items()}; (Path(tmp) / "h").mkdir(); p2 = Path(tmp) / "h" / "params.npz"; np.savez(p2, **other)
        (Path(tmp) / "h" / "selection.json").write_text(json.dumps({"epoch": 7, "tune_anchor_ap": 0.3}))
        g2, d2 = sg.graft_params(params, sg.load_head(p2)); assert all(abs(v - 0.5) < 1e-6 for v in d2.values())
        assert np.array_equal(np.asarray(g2["aux_redge"]["norm"]), np.asarray(params["aux_redge"]["norm"]))
        assert np.array_equal(np.asarray(g2["aux_redge"]["out"]["b"]), other["out_b"])
        assert digest_tree({k: v for k, v in g2.items() if k != "aux_redge"}) == digest_tree({k: v for k, v in params.items() if k != "aux_redge"})
        rec = sg.graft_record("RUN@" + str(p2), p2, d2); assert rec["head_selection"]["epoch"] == 7 and rec["base_run"] == "RUN"
        bad = Path(tmp) / "bad.npz"; np.savez(bad, **{**cur, "out_w": cur["out_w"][:, :0]})
        try: sg.graft_params(params, sg.load_head(bad))
        except ValueError: pass
        else: raise AssertionError("shape mismatch accepted")
        bad2 = Path(tmp) / "bad2.npz"; np.savez(bad2, hidden_w=cur["hidden_w"])
        try: sg.load_head(bad2)
        except ValueError: pass
        else: raise AssertionError("missing keys accepted")
        # install(): run_dir strips the head; selected_model grafts and writes graft.json
        fake = types.SimpleNamespace(run_dir=lambda root, run: Path(root) / run, selected_model=lambda root, run: (params, ("emb",)))
        recs = sg.install(fake, Path(tmp) / "out")
        assert fake.run_dir("/R", "ctl@" + str(p2)) == Path("/R/ctl")
        gp, fns = fake.selected_model("/R", "ctl@" + str(p2)); assert fns == ("emb",) and np.array_equal(np.asarray(gp["aux_redge"]["out"]["b"]), other["out_b"])
        assert fake.selected_model("/R", "ctl")[0] is params and json.loads((Path(tmp) / "out" / "graft.json").read_text()) == json.loads(json.dumps(recs))


def test_score_quartet_metrics():
    import score_quartet as sq
    params = recipe(); e = jax.tree.map(np.asarray, params)["aux_redge"]; r = np.random.default_rng(8)
    z = r.normal(0, 1, (30, 64)).astype(np.float32); general = r.integers(0, 30, (12, 4)).astype(np.int32); chemical = r.integers(0, 30, (6, 4)).astype(np.int32)
    qclass = r.integers(-1, 2, 30)
    zj = jnp.asarray(z)[None]; ref = pr.quartet_terms(params, zj, jnp.asarray(np.concatenate((general, chemical)))[None], jnp.ones((1, 18)), jnp.ones(1, bool))
    m = sq.complex_metrics(e, z, general, chemical, qclass)
    assert abs(m["all"]["softplus"] - float(ref[0])) < 1e-4 and abs(m["all"]["accuracy"] - float(ref[1])) < 1e-6, (m["all"], [float(x) for x in ref])
    assert m["general"]["n"] == 12 and m["chemical"]["n"] == 6
    opp = chemical[qclass[chemical[:, 0]] * qclass[chemical[:, 1]] == -1]; assert (m["opposite"] is None) == (len(opp) == 0)
    per = {"a": m, "b": sq.complex_metrics(e, z, general[:0], chemical, qclass)}
    agg = sq.aggregate(per, {"a": "hetero", "b": "Ab/Ag"}); assert agg["overall"]["general"]["complexes"] == 1 and agg["overall"]["all"]["complexes"] == 2
    assert agg["hetero"]["all"]["complexes"] == 1 and agg["homo"]["all"]["accuracy"] is None
    md = sq.markdown({"runs": {"x": {"run": "R", "metrics": agg}}}); assert "| x | R | 2 |" in md


def _m2(acc):
    return {"swap_same_stratum": {"Ab/Ag": {"accuracy": acc, "n": 645}, "all": {"accuracy": acc + .1, "n": 4000}}, "wrong_partner": {"different_protein": {"accuracy": acc - .1, "n": 2330}},
            "within_complex": {"all": {"auroc": acc - .05}}, "edges_10A_free": {"auroc": acc + .02, "ap": acc / 10, "pairs": 18956558}}


def test_compile_graft():
    import compile_graft as cg
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp); (t / "g/masked2").mkdir(parents=True); (t / "c/masked2").mkdir(parents=True); (t / "g/quartet").mkdir(parents=True)
        (t / "c/masked2/report.json").write_text(json.dumps({"runs": {"drsa": {"run": cg.CONTROL_RUN, "modes": {"unmasked": _m2(0.7)}}}}))
        runs = {"parity": {"run": cg.CONTROL_RUN + "@/x/baseline-edge.npz", "modes": {"unmasked": _m2(0.7)}}}; grafts = {runs["parity"]["run"]: {"max_abs_diff_vs_base_head": {"out_b": 0.0}}}
        q = {"runs": {}}
        for s in cg.SEEDS:
            for tag, acc in (("w0p0", 0.70), ("w0p3", 0.72)):
                n = cg.name_of(tag, s); run = f"{cg.CONTROL_RUN}@/m/seed-{s}/{tag}/params.npz"; runs[n] = {"run": run, "modes": {"unmasked": _m2(acc)}}
                grafts[run] = {"head_selection": {"epoch": 5, "tune_anchor_ap": 0.25}}; q["runs"][n] = {"metrics": {"overall": {"all": {"accuracy": acc}, "chemical": {"accuracy": acc - .1}}}}
        (t / "g/masked2/report.json").write_text(json.dumps({"runs": runs})); (t / "g/masked2/graft.json").write_text(json.dumps(grafts))
        (t / "g/quartet/report.json").write_text(json.dumps(q))
        cmd = [sys.executable, str(HERE / "compile_graft.py"), "--graft-dir", str(t / "g"), "--control-dir", str(t / "c")]
        out = subprocess.run(cmd, capture_output=True, text=True); assert out.returncode == 0, out.stderr
        assert "**parity PASS**" in out.stdout and "| seed 17: w0p3 - w0p0 | +0.0200 |" in out.stdout and "| mean over seeds: w0p3 - w0p0 | +0.0200 |" in out.stdout, out.stdout
        assert "| graft-w0p3-h29 | seed-29/w0p3 | 5 (0.2500) | 0.7200 |" in out.stdout, out.stdout
        runs["parity"]["modes"]["unmasked"] = _m2(0.7001); (t / "g/masked2/report.json").write_text(json.dumps({"runs": runs}))
        assert "**parity FAIL**" in subprocess.run(cmd, capture_output=True, text=True).stdout
        (t / "g/masked2/graft.json").unlink(); bad = subprocess.run(cmd, capture_output=True, text=True); assert bad.returncode == 1 and "graft.json" in bad.stderr


def test_compile_partb():
    import compile as cp
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp); aud = t / "aud"; runs_dir = t / "runs"
        def write(d, key, run, acc, pka, quart=None):
            (d / "masked2").mkdir(parents=True); (d / "residue").mkdir(parents=True)
            (d / "masked2/report.json").write_text(json.dumps({"runs": {key: {"run": run, "modes": {"unmasked": _m2(acc)}}}}))
            (d / "residue/report.json").write_text(json.dumps({key: {"run": run, "pka": {"selection_no_bonded": pka, "state_no_bonded": .38, "interface_no_bonded": .22}}}))
            if quart is not None:
                (d / "quartet").mkdir(parents=True)
                (d / "quartet/report.json").write_text(json.dumps({"runs": {key: {"run": run, "metrics": {"overall": {"all": {"accuracy": quart}, "chemical": {"accuracy": quart - .05}}}}}}))
        write(aud / "drsa-within-v1", "drsa", cp.control_run(17), 0.70, 0.610)
        (aud / "quartet-rank-v1/control/quartet").mkdir(parents=True)
        (aud / "quartet-rank-v1/control/quartet/report.json").write_text(json.dumps({"runs": {"control": {"run": cp.control_run(17), "metrics": {"overall": {"all": {"accuracy": .6}, "chemical": {"accuracy": .55}}}}}}))
        for arm, acc, pka in (("q03", 0.715, 0.615), ("q10", 0.705, 0.630)):
            write(aud / "quartet-rank-v1" / arm, arm, cp.arm_run(arm, 17), acc, pka, 0.65)
            d = runs_dir / cp.arm_run(arm, 17); d.mkdir(parents=True)
            (d / "selection.json").write_text(json.dumps({"selected_epoch": 2, "epochs_run": 2, "parameter_count": 180000}))
            rows = [{"epoch": e, "train_seconds": 100.0, "train_loss": {"pinder_quartet_softplus": 0.9 - e / 10, "pinder_quartet_accuracy": 0.5 + e / 10, "pinder_quartet_covered": 0.95},
                     "quartet_counts": {"complexes": 10}, **({"debug_quartet_grad_norms": [{"epoch": e, "batch": 3, "ratio_quartet_over_within_encoder": 0.5,
                                                                                            "ratio_weighted_quartet_over_within_encoder": 0.5}]} if e == 1 else {})} for e in (1, 2)]
            (d / "history.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        cmd = [sys.executable, str(HERE / "compile.py"), "--aud", str(aud), "--runs-dir", str(runs_dir)]
        out = subprocess.run(cmd, capture_output=True, text=True); assert out.returncode == 0, out.stderr; s = out.stdout
        assert "| control | drsa-v1-50-within03-asym50-w064-s17 | 0 | 0.6100 |" in s and "0.6000 | 0.5500 |" in s, s
        assert "| q03 | quartet-v1-50-asym50-q03-w064-s17 | 0.3 | 0.6150 (+0.0050) |" in s and "0.7150 (+0.0150)" in s, s
        assert "- q03: gates >= +0.01: ['Ab/Ag swap', 'swap all', 'wrong-partner', 'within AUROC', 'edge AUROC']" in s and "meets the advance rule" in s
        assert "- q10: gates >= +0.01: none; pKa clean +0.0200 (NOT clean)" in s and "| q10 | selected | 0.7000 | 0.7000 | 0.9500 |" in s, s
        assert "ratio quartet/within (encoder, unweighted) 0.500" in s
        (aud / "quartet-rank-v1/q10/residue/report.json").unlink(); bad = subprocess.run(cmd, capture_output=True, text=True)
        assert bad.returncode == 1 and "q10" in bad.stderr and "residue" in bad.stderr


def main():
    tests = [(name, f) for name, f in globals().items() if name.startswith("test_") and callable(f)]
    failed = []
    for name, f in tests:
        try: f(); print(f"PASS {name}", flush=True)
        except Exception:  # noqa: BLE001
            failed.append(name); print(f"FAIL {name}", flush=True); traceback.print_exc()
    print(f"passed {len(tests) - len(failed)}/{len(tests)}" + (f"  failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    probe() if PROBE else main()
