"""Phase-1 edge cases; numerical checks run on a Coulson CPU allocation."""
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpb import maps as MP, pb as PB, records as RC


def test_lazy_memtrack_import():
    code = ('import jax; '
            'jax.devices = lambda *a, **k: (_ for _ in ()).throw(AssertionError("eager devices")); '
            'from jaxpb import api; assert api.MEMTRACK.last == -1')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_geometry_last_class_overflow(monkeypatch):
    tried = []

    def fail(*args, **kwargs):
        tried.append(args[2])
        raise OverflowError('capacity')
    monkeypatch.setattr(PB.GEO, 'geometry_state', fail)
    rec = SimpleNamespace(params={}, size_class=len(RC.NA_C)-1, pos=np.empty((0, 3)), rad=np.empty(0))
    with pytest.raises(OverflowError, match='last size class'):
        PB.solve_state(rec, PB.PBConfig(specrad='estimate'), specfn=lambda t, k: .99)
    assert tried == [len(RC.NA_C)-1]


def test_csr_candidate_order_ties_and_overflow():
    # Equidistant points in adjacent cells: offset order beats original point index.
    P = jnp.array([[1.5, .5, .5], [.5, .5, .5], [.5, .5, .5]], jnp.float32)
    w = jnp.zeros(3, jnp.float32); lo = jnp.zeros(3, jnp.float32)
    cells, maximum = MP.build_cells(P, w, lo, 3, 1., 1, batch_size=1 << 30)
    assert int(maximum) == 2                         # occupancy is uncapped although capacity is 1
    q = jnp.array([[1., .5, .5]], jnp.float32)
    d, a = MP.query_min(q, P, w, cells, lo, 3, 1., chunk=1)
    assert np.asarray(a).tolist() == [1]             # first candidate, not smallest point index
    assert np.asarray(d).tolist() == [.5]
    order, start, stop, slots = cells
    assert order.shape == (3,) and start.shape == stop.shape == (27,) and slots.shape == (1,)
    for B in (1, 4, 16):
        assert MP._q_chunk(1, 768, B) * B * 27 * min(768, MP.CSR_SLAB) * 48 <= MP.QUERY_BYTES


def test_face_setup_bitwise():
    assert len(PB.face_indices(81)) == 38402
    n = 9
    rng = np.random.default_rng(17)
    a = jnp.array([.3, -.7, 1.2], jnp.float32)
    qp = jnp.asarray(rng.normal(0, .3, (24, 3)), jnp.float32)
    q = jnp.asarray(rng.normal(0, .3, 24), jnp.float32)

    @jax.jit
    def compare(qp, q, a):
        x, y, z = MP.grid_coords(a, .25, n)
        G = jnp.stack(jnp.meshgrid(x, y, z, indexing='ij'), -1)
        r = jnp.linalg.norm(G[..., None, :] - qp, axis=-1)
        val = jnp.sum(PB.EPKT * q * jnp.exp(-r / PB.DEBL) / (PB.EPSOUT * r), axis=-1)
        old = jnp.where(PB.face_mask(n), val, 0.)
        new = PB.coulomb_bc(qp, q, a, .25, n, PB.EPSOUT, PB.DEBL)
        phi = jnp.arange(n**3, dtype=jnp.float32).reshape(n, n, n) / 17
        interp_old = jnp.where(PB.face_mask(n), PB.interp(phi, G.reshape(-1, 3), a, 1., n).reshape(n, n, n), 0.)
        interp_new = PB.face_grid(PB.interp(phi, PB.face_points(a, .25, n), a, 1., n), n)
        return old, new, interp_old, interp_new
    with PB.GEO.x32():
        old, new, io, ni = (np.asarray(x) for x in compare(qp, q, a))
    assert old.tobytes() == new.tobytes()
    assert io.tobytes() == ni.tobytes()


@pytest.mark.parametrize("capacity", [48, 64, 768])
def test_streamed_candidates_match_dense_order(capacity):
    # Cell 0's later slab ties with cell 1's first slot, visited earlier by the streaming loop.
    points = np.full((80, 3), .1, np.float32)
    points[:40, 0] = .01
    points[40] = [.5, .5, .5]
    points[41:] = [1.5, .5, .5]
    P = jnp.asarray(points); w = jnp.zeros(80, jnp.float32); lo = jnp.zeros(3, jnp.float32)
    cells, occupancy = MP.build_cells(P, w, lo, 3, 1., capacity, batch_size=1 << 30)
    Q = jnp.asarray([[1., .5, .5], [10., 10., 10.], [.7, .8, .6]], jnp.float32)
    order, start, stop, slots = cells
    kk = start[:, None] + slots
    table = jnp.where(kk < stop[:, None], order[jnp.minimum(kk, len(points)-1)], -1)
    offs = MP._offs(1)
    c = jnp.floor((Q - lo) / 1.).astype(jnp.int32)
    cc = c[:, None] + offs[None]
    ok = jnp.all((cc >= 0) & (cc < 3), axis=2)
    cid = jnp.where(ok, (cc[..., 0] * 3 + cc[..., 1]) * 3 + cc[..., 2], 0)
    idx = jnp.where(ok[..., None], table[cid], -1).reshape(len(Q), -1)
    expected = MP._pick(Q, P, w, idx)
    actual = jax.jit(lambda q: MP._q_one(q, P, w, cells, lo, 3, 1., offs))(Q)
    for a, b in zip(actual, expected):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert int(actual[1][0]) == 40
    assert int(occupancy) == 41


def test_padding_skips_map_work(monkeypatch):
    calls = []
    sz = MP.Sizes(NBc=1, NBf=1, NBs=1)

    def group(pos, rad, probes, acent, spos, srad, sprb, sz, active):
        width = acent.shape[0]
        jax.debug.callback(lambda n: calls.append(int(n)), jnp.sum(active), ordered=True)
        shape = (width, RC.GSIZE, RC.GSIZE, RC.GSIZE)
        grid = dict(ex=jnp.zeros(shape, bool), ey=jnp.zeros(shape, bool), ez=jnp.zeros(shape, bool),
                    ion=jnp.zeros(shape, bool), bidx=jnp.full((width, 1), -1, jnp.int32),
                    surf=jnp.zeros((width, 1, 3), jnp.float32))
        return grid, grid, grid, jnp.zeros((width, 3, 7), jnp.int32)

    monkeypatch.setattr(MP, '_maps_group', group)
    jax.clear_caches()
    p = jnp.zeros((1, 3), jnp.float32)
    sr = jnp.zeros((5, 1), jnp.float32)  # Every real site has zero model radii.
    out = MP.maps_batch(p, jnp.ones(1, jnp.float32), p, jnp.zeros((5, 3), jnp.float32),
                        jnp.zeros((5, 1, 3), jnp.float32), sr, jnp.zeros((5, 1, 3), jnp.float32),
                        sz=sz, active=jnp.ones(5, bool))
    jax.block_until_ready(out)
    jax.effects_barrier()
    assert calls == [4, 1]  # Full first group, one real site in the partial second group; padding never calls maps.
    jax.clear_caches()


def test_cpu_query_padding_skips_candidates():
    calls = []

    def query(q, site):
        jax.debug.callback(lambda s: calls.append(int(s)), site, ordered=True)
        return jnp.sum(q, axis=1), jnp.zeros(q.shape[0], jnp.int32)

    @jax.jit
    def run(q, cnt):
        return MP._dyn_rows(query, q, cnt, 4, (jnp.arange(4, dtype=jnp.int32),), (0,))
    d, a = run(jnp.zeros((4, 8, 3), jnp.float32), jnp.array([3, 0, 0, 0], jnp.int32))
    jax.block_until_ready((d, a)); jax.effects_barrier()
    assert sorted(calls) == [0, 1, 2, 3]  # Preserve B=4 arithmetic, even with empty lanes.
    assert np.isinf(np.asarray(d)[1:]).all() and (np.asarray(a)[1:] == -1).all()


def test_padded_lane_maps_are_clean():
    n = 9
    pos = jnp.asarray([[0., 0., 0.], [RC.FAR, RC.FAR, RC.FAR]], jnp.float32)
    rad = jnp.asarray([1.5, 0.], jnp.float32)
    probes = jnp.full((64, 3), RC.FAR, jnp.float32)
    acent = jnp.zeros((4, 3), jnp.float32)
    active = jnp.asarray([True, False, False, False])

    @jax.jit
    def run():
        return MP.build_maps_b(pos, rad, probes, acent, .25, n, 48, 64,
            MP.ncells(.25, n, 'a'), MP.ncells(.25, n, 'p'), 1024, 1024, 64,
            False, .25, True, active=active)
    with PB.GEO.x32():
        res, ctr = jax.tree_util.tree_map(np.asarray, run())
    for key, value in res.items():
        assert (value[1:] == (-1 if key == 'bidx' else 0)).all(), key
    assert not ctr[1:].any()
    assert res['ex'].dtype == np.bool_
    assert res['ex'][0].any()


def test_dense_budget_and_x64_csr_ties():
    points = np.full((400, 3), [.01, .5, .5], np.float32)
    points[140:300] = [.5, .5, .5]
    points[300:] = [1.5, .5, .5]
    with PB.GEO.x64():
        p = jnp.asarray(points); w = jnp.zeros(400, jnp.float32); lo = jnp.zeros(3, jnp.float32)
        dense, ov = MP.build_cells(p, w, lo, 3, 1., 768)
        csr, ov2 = MP.build_cells(p, w, lo, 3, 1., 768, batch_size=1 << 30)
        assert not isinstance(dense, tuple) and isinstance(csr, tuple)
        assert int(ov) == int(ov2) == 300
        q = jnp.asarray([[1., .5, .5], [100., 100., 100.]], jnp.float32)
        query = jax.jit(lambda table: MP._q_one(q, p, w, table, lo, 3, 1., MP._offs(1)))
        a, b = query(dense), query(csr)
        for x, y in zip(a, b):
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
        assert np.asarray(b[1]).tolist() == [140, -1]
        assert b[1].dtype == jnp.int32
