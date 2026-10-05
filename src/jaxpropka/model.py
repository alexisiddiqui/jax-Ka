"""JAX-only sequence evaluation, titration curves, and differentiable midpoints."""
from __future__ import annotations
from dataclasses import replace
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from .cache import StructureCache
from .parameters import (ALPHABET, GROUPS, GROUP_AA, MODEL_PKA, Q_DEPROT,
                         FORMAL_CHARGE, ModelConfig)


class LocalTerms(NamedTuple):
    intrinsic: jax.Array       # [N,9]
    field0: jax.Array          # [N,9]
    coupling: jax.Array        # [N,K,9,9], includes neighbor identity probability
    weights: jax.Array         # [N,9]
    burial: jax.Array          # [N,9]


class CurveResult(NamedTuple):
    ph: jax.Array
    protonated: jax.Array      # [H,R,9], conditional on identity, not multiplied by P
    site_charge: jax.Array     # [H,R,9], already multiplied by identity probability
    residue_charge: jax.Array  # [H,R], includes termini at their owner residue
    chain_charge: jax.Array    # [H,C], ALWAYS all residues in the full environment
    total_charge: jax.Array    # [H], ALWAYS full environment
    probability: jax.Array     # [R,9]
    intrinsic_pka: jax.Array    # [R,9]
    effective_pka: jax.Array    # [H,R,9], pH-dependent field, NOT midpoint pKa
    residual: jax.Array        # [H], undamped full-state fixed-point residual
    weighted_residual: jax.Array  # [H]
    converged: jax.Array       # [H]


class PkaResult(NamedTuple):
    value: jax.Array           # [R,G] or [Q] for pka_sites()
    valid: jax.Array           # bracket, slope, convergence, crossing all pass
    bracketed: jax.Array
    probability: jax.Array     # identity probability; zero does not make conditional pKa invalid
    slope: jax.Array           # dh/dpH at the root
    crossing_error: jax.Array  # |h - 0.5|
    residual: jax.Array        # full-state fixed-point residual at root pH



class GridPkaResult(NamedTuple):
    value: jax.Array           # [R,G], midpoint of interpolated titration curve
    valid: jax.Array           # bracketed, sampled-monotone, converged grid
    bracketed: jax.Array
    probability: jax.Array
    slope: jax.Array           # local piecewise-linear dh/dpH
    bracket_width: jax.Array   # resolution, not an independently measured error
    sampled_monotone: jax.Array
    grid_residual: jax.Array   # maximum full-state residual across the pH grid


class SiteCurveResult(NamedTuple):
    """Full numerical curves; residue labels and chain aggregation stay outside JIT."""
    ph: jax.Array
    protonated: jax.Array
    site_charge: jax.Array
    residue_charge: jax.Array
    total_charge: jax.Array
    probability: jax.Array
    intrinsic_pka: jax.Array
    effective_pka: jax.Array
    residual: jax.Array
    weighted_residual: jax.Array
    converged: jax.Array


def _local_terms(d, p, cfg):
    n = d["group_mask"].shape[0]
    if p.shape != (n,20) or not jnp.issubdtype(p.dtype,jnp.floating):
        raise ValueError(f"expected floating P[{n},20]")
    native = jax.nn.one_hot(d["native_index"],20,dtype=p.dtype)
    p = jnp.where(d["frozen"][:,None],native,p)
    gm, idx = d["group_mask"], d["neighbors"]
    # Cast structural floats to the sequence dtype for both float32 and float64.
    f = lambda name: d[name].astype(p.dtype)
    # Mask the small [N,Ke,A] sequence factor, not the [N,Ke,9,A] tensors:
    # jnp.where on the tensors is not fused into the contraction on CPU and
    # materialized a full temporary copy of each (~1 GB at 845 residues).
    pn = p[d["env_neighbors"]]*d["env_mask"][:,:,None].astype(p.dtype)
    if "identity_columns" in d:
        # Compact restricted cache: gather each source residue's stored identities.
        pn = jnp.take_along_axis(pn,d["identity_columns"][d["env_neighbors"]],axis=-1)
    volume = f("bb_volume") + jnp.einsum("nkga,nka->ng",f("volume"),pn)
    mass = f("bb_mass") + jnp.einsum("nkga,nka->ng",f("mass"),pn)
    hb = f("bb_hbond") + jnp.einsum("nga,na->ng",f("local_hbond"),p) + jnp.einsum("nkga,nka->ng",f("hbond"),pn)
    burial = jnp.clip((mass-cfg.nmin)/(cfg.nmax-cfg.nmin),0,1)
    desolv = (jnp.asarray(FORMAL_CHARGE,dtype=p.dtype) * cfg.desolv_prefactor * volume
              * (cfg.surface_scale+(1-cfg.surface_scale)*burial))
    intrinsic = (jnp.asarray(MODEL_PKA,dtype=p.dtype)+cfg.desolv_scale*desolv
                 +cfg.hbond_scale*(hb+f("reorganization")*burial))
    intrinsic = jnp.where(gm,intrinsic,0)
    weights = jnp.concatenate((p[:,GROUP_AA],jnp.ones((p.shape[0],2),dtype=p.dtype)),axis=-1)*gm
    pair_mass = mass[:,None,:,None]+mass[idx][:,:,None,:]
    pair_burial = jnp.clip((pair_mass-2*cfg.nmin)/(2*(cfg.nmax-cfg.nmin)),0,1)
    eps = cfg.dielectric_surface-(cfg.dielectric_surface-cfg.dielectric_buried)*pair_burial
    # Retain the COO--TYR burial eligibility exception and optional smooth gate.
    coo = jnp.asarray([True,True,False,False,False,False,False,False,True])
    tyr = jnp.arange(9)==4
    exception = (coo[:,None]&tyr[None,:]) | (tyr[:,None]&coo[None,:])
    if cfg.gate_width > 0:
        eligibility = jax.nn.sigmoid((pair_mass-cfg.nmin)/cfg.gate_width)
    else:
        eligibility = (pair_mass>=cfg.nmin).astype(p.dtype)
    eligibility = jnp.where(exception,1,eligibility)
    c = cfg.coulomb_scale * f("coulomb_geometry")/eps * eligibility
    c = jnp.where(d["pair_mask"],c,0)
    hd = cfg.hbond_scale*jnp.where(d["pair_mask"],f("hb_donor"),0)
    hr = cfg.hbond_scale*jnp.where(d["pair_mask"],f("hb_reverse"),0)
    wn = weights[idx][:,:,None,:]
    # E_HB = -Hd*h_i*(1-h_j) - Hr*(1-h_i)*h_j.
    # dE/dh_i = -Hd + (Hd+Hr)*h_j. This is not a signed PROPKA determinant.
    q0 = jnp.asarray(Q_DEPROT,dtype=p.dtype)
    field0 = jnp.sum((c*q0[None,None,None,:]-hd)*wn,axis=(1,3))
    return LocalTerms(intrinsic,field0,(c+hd+hr)*wn,weights,burial)


def _field(d, terms, h):
    return terms.field0+jnp.einsum("nkgt,nkt->ng",terms.coupling,h[d["neighbors"]])


def pack_interaction_edges(pair_mask, neighbors):
    """Pack a fixed padded interaction graph without pruning sequence identities."""
    pair_mask, neighbors = np.asarray(pair_mask), np.asarray(neighbors)
    if pair_mask.ndim != 4 or pair_mask.shape[2:] != (9,9):
        raise ValueError("pair_mask must have shape [N,Kc,9,9]")
    if neighbors.shape != pair_mask.shape[:2]:
        raise ValueError("neighbors must have shape [N,Kc]")
    i,k,g,t = np.nonzero(pair_mask)
    return tuple(np.asarray(x,dtype=np.int32) for x in
                 (i,k,g,t,i*9+g,neighbors[i,k]*9+t))


def _packed_field(terms, h, edges):
    i,k,g,t,dest,src = edges
    values = terms.coupling[i,k,g,t]*h.reshape(-1)[src]
    return terms.field0+jax.ops.segment_sum(
        values,dest,num_segments=h.size).reshape(h.shape)


def _solve_with_field(d, terms, ph, cfg, field):
    log10 = jnp.log(jnp.asarray(10,dtype=terms.intrinsic.dtype))
    def target(h):
        return jnp.where(d["group_mask"],jax.nn.sigmoid(log10*(terms.intrinsic-ph-field(h))),0)
    h = jnp.where(d["group_mask"],jax.nn.sigmoid(log10*(terms.intrinsic-ph-terms.field0)),0)
    def step(_, old):
        return old+cfg.damping*(target(old)-old)
    h = jax.lax.fori_loop(0,cfg.steps,step,h)
    error = jnp.abs(target(h)-h)
    return h,jnp.max(error),jnp.max(error*terms.weights)


def _solve(d, terms, ph, cfg):
    return _solve_with_field(d,terms,ph,cfg,lambda h:_field(d,terms,h))


def _over_ph(fn, xs, ph_batch):
    # ph_batch=None keeps the original all-pH vmap. An int maps sequentially in
    # chunks, bounding solver working memory to ph_batch pH states.
    return jax.vmap(fn)(xs) if ph_batch is None else jax.lax.map(fn,xs,batch_size=ph_batch)


def _curve_result(d, probabilities, ph, terms, solve, field, config, ph_batch=None):
    ph = jnp.asarray(ph,dtype=probabilities.dtype)
    h,residual,wresidual = _over_ph(solve,ph,ph_batch)
    charge = terms.weights[None,:,:]*(jnp.asarray(Q_DEPROT,dtype=probabilities.dtype)[None,None,:]+h)
    residue_charge = charge.sum(-1)
    effective = _over_ph(lambda x:terms.intrinsic-field(x),h,ph_batch)
    return SiteCurveResult(ph,h,charge,residue_charge,residue_charge.sum(-1),
                           terms.weights,terms.intrinsic,effective,
                           residual,wresidual,residual<config.residual_tolerance)


@partial(jax.jit, static_argnames=("config","ph_batch"))
def curve_kernel(arrays, probabilities, ph, *, config, ph_batch=None):
    """Shared all-site curves with structure and pH values as dynamic inputs.

    Inputs must be validated outside JIT. Use batching.pack_inputs for padded
    multi-structure evaluation; padded residues have zero charge and residual.
    """
    terms = _local_terms(arrays,probabilities,config)
    field = lambda h:_field(arrays,terms,h)
    return _curve_result(arrays,probabilities,ph,terms,
        lambda x:_solve_with_field(arrays,terms,x,config,field),field,config,ph_batch)


@partial(jax.jit, static_argnames=("config","ph_batch"))
def packed_curve_kernel(arrays, probabilities, ph, edges, *, config, ph_batch=None):
    """Curve kernel using a host-packed fixed interaction graph."""
    terms = _local_terms(arrays,probabilities,config)
    field = lambda h:_packed_field(terms,h,edges)
    return _curve_result(arrays,probabilities,ph,terms,
        lambda x:_solve_with_field(arrays,terms,x,config,field),field,config,ph_batch)


@partial(jax.jit, static_argnames=("config","ph_batch"))
def grid_pka_kernel(arrays, probabilities, ph, *, config, ph_batch=None):
    """Shared grid interpolation; ph must be finite, increasing, and length >= 2."""
    out = curve_kernel(arrays,probabilities,ph,config=config,ph_batch=ph_batch)
    return _grid_pka_result(arrays,out,ph,config)


def _grid_pka_result(arrays, out, ph, config):
    y, x = out.protonated, out.ph
    # First sampled value <= 1/2; clamp to a safe interval before masking.
    upper = jnp.clip(jnp.argmax(y<=.5,axis=0),1,len(ph)-1)
    lower = upper-1
    y0 = jnp.take_along_axis(y,lower[None,:,:],axis=0)[0]
    y1 = jnp.take_along_axis(y,upper[None,:,:],axis=0)[0]
    width = x[upper]-x[lower]
    delta = y1-y0
    slope = delta/width
    safe = slope < -config.slope_min
    bracketed = arrays["group_mask"] & (y[0]>=.5) & (y[-1]<=.5)
    sampled_monotone = jnp.all(jnp.diff(y,axis=0)<=1e-6,axis=0)
    good = bracketed & safe
    value = x[lower]+(.5-y0)*width/jnp.where(safe,delta,-1)
    residual = jnp.max(out.residual)
    valid = good & sampled_monotone & jnp.all(out.converged)
    return GridPkaResult(jnp.where(good,value,0),valid,bracketed,out.probability,slope,
                         jnp.where(bracketed,width,0),sampled_monotone,
                         jnp.broadcast_to(residual,value.shape))


@partial(jax.jit, static_argnames=("config","ph_batch"))
def packed_grid_pka_kernel(arrays, probabilities, ph, edges, *, config, ph_batch=None):
    """Packed equivalent of grid_pka_kernel."""
    out = packed_curve_kernel(arrays,probabilities,ph,edges,config=config,ph_batch=ph_batch)
    return _grid_pka_result(arrays,out,ph,config)


def one_hot(sequence, dtype=np.float32):
    """Encode a one-letter sequence or a vector of indices, without implicit gaps."""
    if isinstance(sequence,str):
        try:
            indices = [ALPHABET.index(x) for x in sequence]
        except ValueError as exc:
            raise ValueError(f"sequence must use alphabet {ALPHABET}") from exc
    else:
        indices = np.asarray(sequence)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype,np.integer):
            raise TypeError("sequence indices must be a rank-one integer array")
        if np.any((indices<0) | (indices>=20)):
            raise ValueError("identity outside alphabet")
    return np.eye(20,dtype=dtype)[indices]


class TitrationModel:
    """One frozen structure; construct JIT readouts once, then pass only P[N,20].

    Structure selection affects outputs only. All chains and all residues remain
    in the electrostatic/environment solve. The input contract is a finite row-
    stochastic probability matrix; call validate_probabilities() outside JIT or
    use probabilities_from_logits(). Fixed-disulfide identities are explicitly
    clamped to their native identity before all terms are evaluated. The opt-in
    ``backend="packed"`` changes only the fixed interaction-field contraction;
    ``"dense"`` remains the default.
    """
    def __init__(self, cache: StructureCache, config: ModelConfig | None = None,
                 *, backend: str = "dense", ph_batch: int | None = None):
        self.cache = cache.validate()
        self.config = config or ModelConfig()
        if backend not in ("dense","packed"):
            raise ValueError("backend must be 'dense' or 'packed'")
        if ph_batch is not None and (not isinstance(ph_batch,int) or ph_batch < 1):
            raise ValueError("ph_batch must be None or a positive Python int")
        # Execution option only (not part of ModelConfig or its hash): pH states
        # solved concurrently in curves()/pka_from_grid(); None = all at once.
        self.ph_batch = ph_batch
        self.backend = backend
        self._d = {k:jnp.asarray(v) for k,v in vars(cache).items()
                   if isinstance(v,np.ndarray) and k != "chain_index"}
        self._native = jnp.asarray(one_hot(cache.native_index))
        self._gm = self._d["group_mask"]
        self._edges = tuple(jnp.asarray(x) for x in pack_interaction_edges(
            cache.pair_mask,cache.neighbors)) if backend == "packed" else None
        self._root = self._make_root()
        restricted = cache.metadata.get("allowed_identities")
        # Identity-restricted caches (build_cache(identities=...)) have zero
        # environmental columns for disallowed identities.
        self.allowed = (None if restricted is None else
                        np.array([[a in row for a in ALPHABET] for row in restricted]))

    def release_host_arrays(self):
        """Free the NumPy copies of large structural tensors after device transfer.

        On CPU the device arrays are separate copies, so the host cache doubles
        peak memory. Large fields are replaced by zero-memory broadcast views of
        the same shape and dtype; keys, masks and indices are kept. Readouts are
        unaffected. The host cache no longer holds the kernels: take
        cache.fingerprint() or cache.save() BEFORE calling this, and drop any
        other reference to the original cache for the memory to be returned.
        """
        names = ("volume","mass","hbond","local_hbond","coulomb_geometry","hb_donor",
                 "hb_reverse","pair_mask")
        self.cache = replace(self.cache, **{k: np.broadcast_to(
            np.zeros((),getattr(self.cache,k).dtype),getattr(self.cache,k).shape) for k in names})
        return self

    @property
    def arrays(self):
        """Device structural arrays, for passing explicitly through a caller's jit."""
        return self._d

    @property
    def native_probabilities(self):
        return self._native

    def validate_probabilities(self, probabilities, atol=1e-5):
        p = np.asarray(probabilities)
        if p.shape != (self.cache.n_residues,20) or not np.issubdtype(p.dtype,np.floating):
            raise ValueError(f"expected floating P[{self.cache.n_residues},20]")
        if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1,atol=atol):
            raise ValueError("P must be finite, nonnegative and sum to one in each row")
        if self.allowed is not None:
            free = ~np.asarray(self.cache.frozen)  # frozen rows are clamped to native
            if np.any(p[free][~self.allowed[free]] > atol):
                raise ValueError("P puts mass on identities excluded from this restricted cache")
        return p

    def probabilities_from_logits(self, logits, allowed=None):
        """JAX-transformable. An optional STATIC allowed[N,20] mask is validated."""
        if logits.shape != (self.cache.n_residues,20):
            raise ValueError("wrong logits shape")
        if self.allowed is not None:
            allowed = self.allowed if allowed is None else np.asarray(allowed,dtype=bool) & self.allowed
        if allowed is not None:
            allowed = np.asarray(allowed,dtype=bool)
            if allowed.shape != logits.shape or not allowed.any(-1).all():
                raise ValueError("allowed must leave at least one identity at every position")
            logits = jnp.where(jnp.asarray(allowed),logits,-jnp.inf)
        p = jax.nn.softmax(logits,axis=-1)
        return jnp.where(self._d["frozen"][:,None],self._native.astype(p.dtype),p)

    def _local_terms(self, p):
        return _local_terms(self._d,p,self.config)

    def local_terms(self):
        """Return a JIT-compiled P-only term inspector."""
        return jax.jit(self._local_terms)

    def _field(self, terms, h):
        return (_field(self._d,terms,h) if self.backend == "dense"
                else _packed_field(terms,h,self._edges))

    def _solve(self, terms, ph):
        return _solve_with_field(self._d,terms,ph,self.config,
                                 lambda h:self._field(terms,h))

    def curves(self, ph, residues=None):
        """Compile conditional occupancies and physical charges on a static pH grid."""
        grid = np.atleast_1d(np.asarray(ph,dtype=float))
        if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all():
            raise ValueError("ph must be a finite nonempty scalar or vector")
        sel = jnp.asarray(self.cache.select(residues))
        # Segment reduction includes all positions, not only the output selection.
        chain_onehot = jnp.asarray(np.eye(len(self.cache.chain_ids))[self.cache.chain_index])
        # Structural arrays are explicit jit ARGUMENTS. Closing over them would
        # embed every cache array as an HLO constant, and compile-time memory
        # would then grow with the cache (the production OOM mechanism). A caller
        # that wraps evaluate() in its own jit re-captures them as constants; pass
        # model.arrays through that jit and call the module kernels instead.
        @jax.jit
        def kernel(d, edges, p):
            if self.backend == "dense":
                out = curve_kernel(d,p,jnp.asarray(grid,dtype=p.dtype),config=self.config,
                                   ph_batch=self.ph_batch)
            else:
                out = packed_curve_kernel(d,p,jnp.asarray(grid,dtype=p.dtype),
                                          edges,config=self.config,ph_batch=self.ph_batch)
            return CurveResult(out.ph,out.protonated[:,sel],out.site_charge[:,sel],out.residue_charge[:,sel],
                               out.residue_charge@chain_onehot.astype(p.dtype),out.total_charge,
                               out.probability[sel],out.intrinsic_pka[sel],out.effective_pka[:,sel],
                               out.residual,out.weighted_residual,out.converged)
        def evaluate(p):
            return kernel(self._d,self._edges,p)
        evaluate.lower = lambda p: kernel.lower(self._d,self._edges,p)
        return evaluate

    def charge(self, ph=7.0, residues=None):
        """Compile residue charges only: shape [R] at scalar pH, [H,R] for a grid."""
        curves = self.curves(ph,residues)
        scalar = np.ndim(ph)==0
        def evaluate(p):
            q = curves(p).residue_charge
            return q[0] if scalar else q
        return evaluate

    def _site_fraction(self, terms, ph, index):
        h,_,_ = self._solve(terms,ph)
        return h.reshape(-1)[index]

    def _bracket(self, terms, index):
        cfg = self.config
        lo = jnp.asarray(cfg.ph_min,dtype=terms.intrinsic.dtype)
        hi = jnp.asarray(cfg.ph_max,dtype=terms.intrinsic.dtype)
        fl = self._site_fraction(terms,lo,index)-0.5
        fh = self._site_fraction(terms,hi,index)-0.5
        return lo,hi,(fl>=0)&(fh<=0)&self._gm.reshape(-1)[index]

    def _make_root(self):
        cfg = self.config
        @jax.custom_jvp
        def root(terms, index):
            lo,hi,bracketed = self._bracket(terms,index)
            def step(_, bounds):
                a,b = bounds
                mid = (a+b)*0.5
                higher = self._site_fraction(terms,mid,index)>0.5
                return jnp.where(higher,mid,a),jnp.where(higher,b,mid)
            lo,hi = jax.lax.fori_loop(0,cfg.root_steps,step,(lo,hi))
            return jnp.where(bracketed,(lo+hi)*0.5,0)

        @root.defjvp
        def root_jvp(primals,tangents):
            terms,index = primals
            terms_dot,_ = tangents  # integer query index has float0 tangent
            value = root(terms,index)
            _,_,bracketed = self._bracket(terms,index)
            f = lambda t,x:self._site_fraction(t,x,index)
            numerator = jax.jvp(lambda t:f(t,value),(terms,),(terms_dot,))[1]
            denominator = jax.jvp(lambda x:f(terms,x),(value,),(jnp.ones_like(value),))[1]
            safe = bracketed & (denominator < -cfg.slope_min)
            # Safe denominator is required *inside* masking, not only at output.
            divisor = jnp.where(safe,denominator,-1)
            tangent = jnp.where(safe,-numerator/divisor,0)
            return value,tangent
        return root

    def _pka_readout(self, flat_indices, shape):
        query = jnp.asarray(flat_indices,dtype=jnp.int32)
        cfg = self.config
        @jax.jit
        def evaluate(p):
            terms = self._local_terms(p)
            def one(index):
                value = self._root(terms,index)
                _,_,bracketed = self._bracket(terms,index)
                h,residual,_ = self._solve(terms,value)
                crossing = jnp.abs(h.reshape(-1)[index]-0.5)
                slope = jax.jvp(lambda x:self._site_fraction(terms,x,index),
                                (value,),(jnp.ones_like(value),))[1]
                valid = (bracketed & (residual<cfg.residual_tolerance)
                         & (crossing<cfg.root_tolerance) & (slope < -cfg.slope_min))
                probability = terms.weights.reshape(-1)[index]
                return PkaResult(value,valid,bracketed,probability,slope,crossing,residual)
            # Sequential chunks bound working occupancy storage to O(B*N*G).
            # Reverse-mode through the custom JVP does not retain bisection history.
            out = jax.lax.map(one,query,batch_size=cfg.root_batch_size)
            return jax.tree.map(lambda x:x.reshape(shape),out)
        return evaluate

    def pka(self, residues=None, groups=None):
        """Compile midpoint pKas [R,Gselected]; defaults to all residues / 9 channels.

        A scalar pKa for a mixture of acid/base identities is not defined here.
        Keep type channels; select a desired type or use pka_sites(). An absent
        terminal returns finite zero with valid=False and probability=0.
        """
        sel = self.cache.select(residues)
        group_indices = list(range(9)) if groups is None else [GROUPS.index(g) for g in groups]
        if not group_indices or len(set(group_indices)) != len(group_indices):
            raise ValueError("groups must be nonempty and unique")
        flat = (sel[:,None]*9+np.asarray(group_indices)[None,:]).reshape(-1)
        return self._pka_readout(flat,(len(sel),len(group_indices)))

    def pka_from_grid(self, ph, residues=None, groups=None):
        """Fast all-site midpoint interpolation from shared full-structure solves.

        Piecewise differentiable linear interpolation of a static pH grid; this
        is NOT the direct root from pka(). Gradients are of the interpolated
        observable, including the selected occupancies. A change of interval can
        introduce a derivative kink. Interval width is reported as resolution;
        a root on the chosen continuous branch lies in the bracket, but sampled
        monotonicity alone cannot establish global uniqueness. Unbracketed/flat outputs
        are finite zero; other invalid estimates are retained with valid=False. Only the sequence P is a traced input.
        """
        grid=np.asarray(ph,dtype=float)
        if grid.ndim!=1 or len(grid)<2 or not np.isfinite(grid).all() or not np.all(np.diff(grid)>0):
            raise ValueError("midpoint grid must be finite and strictly increasing, with at least two points")
        indices=list(range(9)) if groups is None else [GROUPS.index(g) for g in groups]
        if not indices or len(set(indices))!=len(indices):
            raise ValueError("groups must be nonempty and unique")
        selected=self.cache.select(residues)
        group_indices=jnp.asarray(indices)
        @jax.jit
        def kernel(d, edges, p):  # arrays as arguments; see curves()
            if self.backend == "dense":
                out=grid_pka_kernel(d,p,jnp.asarray(grid,dtype=p.dtype),config=self.config,
                                    ph_batch=self.ph_batch)
            else:
                out=packed_grid_pka_kernel(d,p,jnp.asarray(grid,dtype=p.dtype),
                                           edges,config=self.config,ph_batch=self.ph_batch)
            if residues is None and groups is None:
                return out
            return jax.tree.map(lambda x:x[selected][:,group_indices],out)
        def evaluate(p):
            return kernel(self._d,self._edges,p)
        return evaluate

    def pka_sites(self, sites):
        """Compile an arbitrary static [(residue_key_or_index, group_name), ...] list."""
        indices = []
        for residue,group in sites:
            i = int(self.cache.select([residue])[0])
            indices.append(9*i+GROUPS.index(group))
        if not indices or len(set(indices))!=len(indices):
            raise ValueError("site selection must be nonempty and unique")
        return self._pka_readout(indices,(len(indices),))
