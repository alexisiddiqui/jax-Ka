"""M5 CPU test: the Pallas SOR solve (interpret=True) against the XLA lane pool on a small grid.
Random spherical solutes (eps_in inside, ion map outside an exclusion shell) on an n^3 grid, random charges and face
values, several operators, mixed omega, lanes of both kinds plus invalid lanes. Criteria: sweep counts 100% equal,
phi max |diff| <= 1e-5 * max|phi|, invalid lanes 0, solve bitwise reproducible.
--gpu: compiled Triton kernel instead of interpret mode, plus wall times of both solvers (second call, compiled).
usage: python tests/test_sor_pallas.py [--n 17] [--W 8] [--gpu]"""
import argparse, json, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=17); ap.add_argument("--W", type=int, default=8)
ap.add_argument("--gpu", action="store_true")
A = ap.parse_args()
import jax
import jax.numpy as jnp
from jaxpb import sor as SOR, pb as PB

n, W, E = A.n, A.W, 4
rng = np.random.default_rng(3)
I = np.arange(n) - (n - 1) / 2
X, Y, Z = np.meshgrid(I, I, I, indexing="ij")
maps = []
for e in range(E):
    c = rng.normal(size=3) * 1.5; r = n / 4 + rng.uniform(-1, 1)
    d = lambda ox, oy, oz: np.sqrt((X + ox - c[0]) ** 2 + (Y + oy - c[1]) ** 2 + (Z + oz - c[2]) ** 2)
    maps.append(dict(ex=jnp.asarray(d(.5, 0, 0) < r), ey=jnp.asarray(d(0, .5, 0) < r), ez=jnp.asarray(d(0, 0, .5) < r),
                     ion=jnp.asarray(d(0, 0, 0) > r + 2)))
h = [1.0, 0.25, 1.0, 0.25]
def coef_n(m, hh):   # pb._coefs on an n-point grid (pb._coefs is fixed to GSIZE)
    ex, ey, ez, diag = PB.operator_coeffs(m, PB.EPSIN, PB.EPSOUT, PB._kap2h2(hh))
    return ex, ey, ez, jnp.where(jnp.asarray(SOR.face_mask_np(n)), 0.0, 1.0 / diag)


coefs = [coef_n(m, hh) for m, hh in zip(maps, h)]
codes = [PB._codes(m, hh) for m, hh in zip(maps, h)]
coefs = tuple(jnp.stack([c[k] for c in coefs]) for k in range(4))
codes = (jnp.stack([c[0] for c in codes]), jnp.stack([c[1] for c in codes]))
fm = SOR.face_mask_np(n)
phi0 = np.where(fm[None], rng.normal(size=(W, n, n, n)), 0).astype(np.float32)
rhs0 = np.where(fm[None], 0, rng.normal(size=(W, n, n, n)) * (rng.uniform(size=(W, n, n, n)) < 0.01) * 50).astype(np.float32)
om0 = rng.uniform(1.6, 1.9, W).astype(np.float32)
op0 = (np.arange(W) % E).astype(np.int32)
valid = np.ones(W, bool); valid[-1] = False
args = (jnp.asarray(phi0), jnp.asarray(rhs0), jnp.asarray(om0), jnp.asarray(op0))
s_x, it_x, mc_x = SOR.run_pool(*args, valid, coefs)
st = []
ip = not A.gpu
s_p, it_p, mc_p = SOR.run_pool_pallas(*args, valid, codes, stats=st, interpret=ip)
s_p2, it_p2, _ = SOR.run_pool_pallas(*args, valid, codes, interpret=ip)
tim = {}
if A.gpu:
    for name, fn in (("xla", lambda: SOR.run_pool(*args, valid, coefs)), ("pallas", lambda: SOR.run_pool_pallas(*args, valid, codes))):
        t = time.perf_counter(); jax.block_until_ready(fn()); tim[name + "_s"] = time.perf_counter() - t
s_x, s_p, s_p2 = (np.asarray(a) for a in (s_x, s_p, s_p2))
scale = float(np.abs(s_x).max())
r = dict(n=n, W=W, sweeps_xla=it_x.tolist(), sweeps_pallas=it_p.tolist(), sweeps_equal=float(np.mean(it_x == it_p)),
         phi_maxabs=float(np.abs(s_x - s_p).max()), phi_scale=scale, invalid_zero=bool((s_p[~valid] == 0).all()),
         bitwise_rerun=bool(np.array_equal(s_p, s_p2) and np.array_equal(it_p, it_p2)), lane_blocks=st[0]["lane_blocks"],
         needed=st[0]["needed"], **tim)
r["pass"] = r["sweeps_equal"] == 1.0 and r["phi_maxabs"] <= 1e-5 * scale and r["invalid_zero"] and r["bitwise_rerun"]
print(json.dumps(r, indent=1))
assert r["pass"], r
