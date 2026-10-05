"""Stage-resolved wall time and peak RSS for the production JAX-Ka readout.

Diagnostic only; no solver or model changes. The driver picks production
complexes nearest to target residue counts and runs each (complex, steps) in a
fresh subprocess, so peak RSS is clean and an OOM kill only loses one record.
Each child checkpoints after every stage; a killed child still reports the last
stage it reached and the peak RSS observed so far.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def rss_mib():
    with open('/proc/self/status') as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])/1024
    return 0.


class Sampler:
    """Peak resident memory per stage, sampled every 20 ms."""
    def __init__(self):
        self.peak = 0.; self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, rss_mib()); time.sleep(.02)

    def reset(self):
        value, self.peak = self.peak, rss_mib()
        return value


def child(cif, variant, out):
    # variant: dash-separated tokens, e.g. '1024', 'native-64', 'native-optx',
    # 'native-1024-b8-slim'. native = identity-restricted cache; optx = optimistix
    # solver; b<k> = pH chunk size; slim = free host cache + candidates after transfer.
    tokens = variant.split('-'); variant_name = variant
    native = 'native' in tokens; optx = 'optx' in tokens; slim = 'slim' in tokens
    ph_batch = next((int(t[1:]) for t in tokens if t.startswith('b') and t[1:].isdigit()), None)
    steps = next((int(t) for t in tokens if t.isdigit()), 64)
    sampler = Sampler(); record = dict(cif=str(cif), steps=steps, variant=variant, stages=[], status='running')
    last = [time.perf_counter()]
    def stage(name, **extra):
        now = time.perf_counter()
        record['stages'].append(dict(name=name, seconds=now-last[0],
                                     peak_rss_mib=max(sampler.reset(), rss_mib()),
                                     end_rss_mib=rss_mib(), **extra))
        record['stage'] = name
        out.write_text(json.dumps(record, indent=1))
        print(cif.parent.name, variant_name, name, round(now-last[0], 1), flush=True)
        last[0] = time.perf_counter()
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jaxpropka import TitrationModel
    from jaxpropka.parameters import ModelConfig, GROUP_AA
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache, native_identities
    from jaxpropka.model import _grid_pka_result
    from pkabench.prep import read_cif
    from pkabench.schema import PH
    jax.config.update('jax_enable_compilation_cache', False)
    stage('import')
    atoms = read_cif(cif); stage('read_cif', atoms=int(len(atoms)))
    topology = load_topology(atoms, gap_policy='cap', freeze_disulfides=True)
    stage('topology', n_residues=int(topology.n_residues))
    candidates = build_candidates(topology, missing_sidechain='error'); stage('candidates')
    cache = build_cache(topology, candidates,
                        identities=native_identities(topology) if native else None)
    n, ke = cache.env_neighbors.shape; kc = cache.neighbors.shape[1]
    weights = np.concatenate((np.eye(20)[cache.native_index][:, GROUP_AA],
                              np.ones((n,2))), axis=1)*cache.group_mask
    stage('build_cache', Ke=int(ke), Kc=int(kc), cache_mib=cache.nbytes/2**20,
          env_tensor_mib=sum(getattr(cache,k).nbytes for k in ('volume','mass','hbond'))/2**20,
          pair_tensor_mib=sum(getattr(cache,k).nbytes for k in
                              ('pair_mask','coulomb_geometry','hb_donor','hb_reverse'))/2**20,
          env_mask_fill=float(cache.env_mask.mean()), pair_entries_active=int(cache.pair_mask.sum()),
          group_channels=int(cache.group_mask.sum()), native_active_sites=int((weights>0).sum()))
    model = TitrationModel(cache, config=ModelConfig(steps=steps), backend='packed', ph_batch=ph_batch)
    jax.block_until_ready(model._d)
    p = model.native_probabilities
    if optx:
        from jaxpropka.optx_solver import SolverConfig, active_channels, optx_curve_kernel
        active = jnp.asarray(active_channels(cache, np.asarray(p)))
    if slim:
        import gc
        model.release_host_arrays(); del cache, candidates, topology, atoms; gc.collect()
    stage('model_init', packed_edges=int(len(model._edges[0])), ph_batch=ph_batch, slim=slim)
    if optx:
        grid = jnp.asarray(PH, p.dtype)
        lowered = optx_curve_kernel.lower(model.arrays, p, grid, active, model._edges,
                                          config=model.config, solver_config=SolverConfig())
        record['active_channels'] = int(active.size)
    else:
        lowered = model.curves(PH).lower(p)
    stage('lower')
    compiled = lowered.compile(); memory = {}
    try:
        m = compiled.memory_analysis()
        memory = {k: getattr(m, k)/2**20 for k in ('temp_size_in_bytes','argument_size_in_bytes',
                  'output_size_in_bytes','generated_code_size_in_bytes') if hasattr(m, k)}
    except Exception as exc:  # backend-dependent API
        memory = dict(error=repr(exc))
    stage('compile', xla_memory_mib=memory)
    run = (lambda: compiled(model.arrays, p, grid, active, model._edges)) if optx else (lambda: compiled(model.arrays, model._edges, p))
    curves = jax.block_until_ready(run()); stage('execute')
    curves = jax.block_until_ready(run()); stage('execute_warm')
    if optx:
        curves, extra = curves
        record['newton_steps_max'] = int(np.max(extra['newton_steps']))
        record['newton_steps_mean'] = float(np.mean(extra['newton_steps']))
        record['optx_success_all'] = bool(np.all(extra['optx_success']))
        record['per_ph'] = dict(ph=[float(x) for x in PH],
            steps=np.asarray(extra['newton_steps']).tolist(),
            success=np.asarray(extra['optx_success']).tolist(),
            residual=np.asarray(curves.residual).tolist())
    mid = _grid_pka_result(model._d, curves, curves.ph, model.config)
    jax.block_until_ready(mid)
    stage('grid_readout', max_residual=float(np.max(curves.residual)),
          converged=bool(np.all(curves.converged)), valid_sites=int(np.sum(mid.valid)))
    record['status'] = 'complete'; out.write_text(json.dumps(record, indent=1))


def driver(campaign, targets, variants, out, complexes=None):
    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    out.mkdir(parents=True, exist_ok=True)
    sizes = []
    for path in sorted((campaign/'structures').glob('*/AB.cif')):
        sizes.append((sum(1 for line in path.open() if ' CA ' in line), path))
    chosen = [x for x in sizes if x[1].parent.name in complexes] if complexes else []
    for target in ([] if complexes else targets):
        size, path = min((x for x in sizes if x[1] not in [c[1] for c in chosen]),
                         key=lambda x: abs(x[0]-target))
        chosen.append((size, path))
    summary = []
    for size, path in chosen:
        for s in variants:
            target = out/f'{path.parent.name}-{s}.json'
            started = time.perf_counter()
            proc = subprocess.run([sys.executable, __file__, 'child', str(path), s, str(target)],
                                  timeout=5400)
            record = json.loads(target.read_text()) if target.exists() else dict(stages=[])
            record.update(complex_id=path.parent.name, ca_count=size, variant=s, returncode=proc.returncode,
                          process_seconds=time.perf_counter()-started)
            if proc.returncode and record.get('status') != 'complete':
                record['status'] = f'killed_after_{record.get("stage", "start")}'
            atomic_json(target, record); summary.append(record)
    atomic_json(out/'summary.json', dict(job=os.environ.get('SLURM_JOB_ID'),
                node=os.uname().nodename, targets=targets, variants=variants, records=summary))
    for r in summary:
        top = sorted(r['stages'], key=lambda x: -x['seconds'])[:3]
        print(r['complex_id'], r['ca_count'], r['variant'], r['status'],
              ' '.join(f"{x['name']}={x['seconds']:.0f}s" for x in top),
              f"peak={max([x['peak_rss_mib'] for x in r['stages']] or [0]):.0f}MiB", flush=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['child']:
        child(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--campaign', type=Path, required=True)
        parser.add_argument('--targets', type=int, nargs='+', default=[280, 450, 650, 900])
        parser.add_argument('--variants', nargs='+', default=['64', '1024'],
                            help="'<steps>', 'native-<steps>' or 'native-optx'")
        parser.add_argument('--complexes', nargs='+', help='explicit complex IDs (overrides --targets)')
        parser.add_argument('--output', type=Path, required=True)
        a = parser.parse_args()
        driver(a.campaign, a.targets, a.variants, a.output, a.complexes)
