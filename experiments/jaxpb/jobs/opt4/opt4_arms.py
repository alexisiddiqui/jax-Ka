"""Build independent CPU-only arms from the frozen baseline without editing it."""
import importlib.util,sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxpb import maps as M,geometry as G
from opt4_kernels import compact_query,reach_need,span_hash,span_cand,compact_burial

def site_burial(pts,own,cand_all,sp,sR,bound,knn):
 ch=4096; N=len(pts); pad=(-N)%ch
 ps=jnp.pad(pts,((0,pad),(0,0))).reshape(-1,ch,3)
 ow=jnp.pad(own,((0,pad),(0,0))).reshape(-1,ch,2)
 b,ba=jax.lax.map(lambda xs:G._burial(xs[0],xs[1],jnp.broadcast_to(cand_all,(ch,len(cand_all))),sp,sR,bound,knn),(ps,ow))
 return b.reshape(-1)[:N],ba.reshape(-1)[:N]

def module(name,source,extra):
 spec=importlib.util.spec_from_loader(name,loader=None); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; mod.__dict__.update(extra); exec(compile(source,name,'exec'),mod.__dict__); return mod

def maps_arm(step):
 source=Path(M.__file__).read_text()
 if step in (1,2,3,5):
  source=source.replace("if batch_size * ncell ** 3 * K * 4 <= table_budget:", "if batch_size * ncell ** 3 * K * 4 <= table_budget and not (jax.default_backend() == \"cpu\" and K <= 48):")
  needle='    c = jnp.floor((q - lo) / cell).astype(jnp.int32)\n'
  start=source.index('def _q_one('); end=source.index('\ndef _q_one_hashed',start); body=source[start:end]
  body=body.replace(needle,'    if jax.default_backend() == "cpu" and _table_K(table) <= 48:\n        return _compact_query(q, P, w, table, lo, ncell, cell, offs)\n'+needle,1); source=source[:start]+body+source[end:]
 if step in (2,3,4):
  needle='        rm = reach_mask_b(S["preg"], S["pval"], S["npreg"], acent, h, n, ax, hstat, PB, rp)'
  source=source.replace(needle,'        if jax.default_backend() == "cpu":\n            rm = _reach_need(need, S["G"], S["preg"], S["pval"], acent, h, n, ax, NS, ncell_p, rp)\n        else:\n    '+needle)
 if step in (3,5):
  start=source.index('def _q_chunk('); end=source.index('\ndef _brute_chunk',start); body=source[start:end]
  body=body.replace('    if jax.default_backend() == "cpu":', '    if jax.default_backend() == "cpu" and K <= 48:\n        return min(4096, max(1, 8192 // B))\n    if jax.default_backend() == "cpu":',1)
  source=source[:start]+body+source[end:]
 return module('jaxpb._opt4_maps'+str(step),source,dict(_compact_query=compact_query,_reach_need=reach_need))

def geometry_arm(step):
 source=Path(G.__file__).read_text()
 if step>=1:
  source=source.replace('    return _hash_cells(P, valid, lo, cell)','    if jax.default_backend() == "cpu":\n        return _span_hash(P, valid, lo, cell)\n    return _hash_cells(P, valid, lo, cell)')
  needle='    c = _cell_of(pts, lo, cell)[:, None, :] + jnp.asarray(OFFS27)[None]'
  source=source.replace(needle,'    if isinstance(scid, tuple):\n        return _span_cand(pts, lo, cell, scid, order, K)\n'+needle)
 if step>=2:
  for own in ('jnp.stack([a, a], 1)','jnp.stack([Pi[p], Pj[p]], 1)'):
   old=f'    cand, occ = _cand(pts, lo, cell, scid, order, K)\n    b, ball = _burial(pts, {own}, cand, P, Rj, bound, knn)'
   new=f'    if isinstance(scid, tuple):\n        b, ball, occ = _compact_burial(pts, {own}, lo, cell, scid, order, P, Rj, bound, K, knn)\n    else:\n        cand, occ = _cand(pts, lo, cell, scid, order, K)\n        b, ball = _burial(pts, {own}, cand, P, Rj, bound, knn)'
   assert old in source; source=source.replace(old,new)
  old='        cand, oc = _cand(ps, lo, cell, scid, order, K)\n        b, ball = _burial(ps, own[sub], cand, P, Rj, bound, knn)'
  new='        if isinstance(scid, tuple):\n            b, ball, oc = _compact_burial(ps, own[sub], lo, cell, scid, order, P, Rj, bound, K, knn)\n        else:\n            cand, oc = _cand(ps, lo, cell, scid, order, K)\n            b, ball = _burial(ps, own[sub], cand, P, Rj, bound, knn)'
  assert old in source; source=source.replace(old,new)
 if step>=3:
  old='    b, ball = _burial(pts, own, jnp.broadcast_to(cand_all, (pts.shape[0], na)), sp, sR, bound, knn)'
  source=source.replace(old,'    if jax.default_backend() == "cpu":\n        b, ball = _site_burial(pts, own, cand_all, sp, sR, bound, knn)\n    else:\n    '+old)
  old='    return jax.vmap(lambda a, b, c: _site_probes(a, b, c, U, nvert, MXs, OUT, knn, arc_step))(sp, sR, sv)'
  source=source.replace(old,'    if jax.default_backend() == "cpu":\n        return jax.lax.map(lambda xs: _site_probes(*xs, U, nvert, MXs, OUT, knn, arc_step), (sp, sR, sv))\n'+old)
 if step>=4:
  marker='    from scipy.spatial import cKDTree\n    Pf, Rf, n = surface_atoms'
  source=source.replace(marker,'    from scipy.spatial import cKDTree\n    if jax.default_backend() == "cpu": CH = min(CH, 8192)\n    Pf, Rf, n = surface_atoms')
  source=source.replace('CH=CH, nvert=nvert, K=K, NPmax=NPmax)\n    off, occmax', 'CH=CH, nvert=nvert, K=K, NPmax=NPmax, SC=min(CH, RC.GEO_CHUNK // 8))\n    off, occmax')
 return module('jaxpb._opt4_geo'+str(step),source,dict(_span_hash=span_hash,_span_cand=span_cand,_compact_burial=compact_burial,_site_burial=site_burial))

def boundary_laplacian(phi,bidx):
 n=phi.shape[0]; bb=jnp.maximum(bidx,0); p=phi.reshape(-1); i=bb//(n*n); j=(bb//n)%n; k=bb%n
 return (p[(((i+1)%n)*n+j)*n+k]+p[(((i-1)%n)*n+j)*n+k]
         +p[(i*n+(j+1)%n)*n+k]+p[(i*n+(j-1)%n)*n+k]
         +p[(i*n+j)*n+(k+1)%n]+p[(i*n+j)*n+(k-1)%n]-6*p[bb])

def pb_arm():
 from jaxpb import pb as P
 source=Path(P.__file__).read_text(); start=source.index('    lap = (',source.index('def rf_energy(')); end=source.index('    sv = s0 - qb / epsin',start)
 old=source[start:end]
 new='''    if jax.default_backend() == "cpu":
        b = bidx; bb = jnp.maximum(b, 0)
        lap = _boundary_laplacian(phi, bidx)
        s0 = jnp.where(b >= 0, -(h / (4 * jnp.pi * EPKT)) * lap, 0.0)
        qb = jnp.where(b >= 0, rho.reshape(-1)[bb], 0.0)
    else:
'''+''.join('    '+l+'\n' for l in old.rstrip().splitlines())
 source=source[:start]+new+source[end:]
 return module('jaxpb._opt4_post',source,dict(_boundary_laplacian=boundary_laplacian))
