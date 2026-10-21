"""Standalone oGQT preparation and mixed-size batch prediction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time


def _atomic_json(path, value):
    from pkabench.runtime import atomic_json
    atomic_json(path, value)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ogqt')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare','predict'):
        p = sub.add_parser(name)
        p.add_argument('structures', nargs='*', help='PDB/mmCIF/BCIF paths')
        p.add_argument('--input-list', type=Path, help='one structure path per line, relative to the list file')
        p.add_argument('--cache', type=Path, required=True, help='reusable prepared-feature directory')
        p.add_argument('--gap-policy', choices=('cap','error','free'), default='cap')
        p.add_argument('--ignore-nonprotein', action='store_true')
        p.add_argument('--output', type=Path, required=True)
        if name == 'predict':
            p.add_argument('--model', type=Path, help='export containing model.json and params.npz (default: the pinned '
                           'all-residue model, models/ogqt-allres15-siamese-50, or GQT_DEFAULT_MODEL)')
            p.add_argument('--state', choices=('bound','free','both'), default='both',
                           help='all-residue models: bound = the structure as given; free = every chain alone (edges '
                                'between chains removed); both adds the binding shift and free-state interface scores')
            p.add_argument('--no-residues', action='store_true', help='all-residue models: omit per-residue records')
            p.add_argument('--batch-size', type=int, default=4)
            p.add_argument('--device', choices=('cpu','gpu'), default='cpu')
            p.add_argument('--backend', choices=('jax','triton'), default='jax')
            p.add_argument('--max-input-mib', type=int, default=512)
            cache = p.add_mutually_exclusive_group()
            cache.add_argument('--compilation-cache', type=Path, help='persistent JAX executable cache (default: JAX_COMPILATION_CACHE_DIR or XDG cache)')
            cache.add_argument('--no-compilation-cache', action='store_true', help='disable the persistent executable cache for this process')
    args = parser.parse_args(argv)
    paths = [Path(p).resolve() for p in args.structures]
    if args.input_list:
        paths += [(args.input_list.parent / line.strip()).resolve()
                  for line in args.input_list.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith('#')]
    if not paths: parser.error('provide structures or --input-list')
    if any(not p.is_file() for p in paths): parser.error('one or more input files do not exist')
    if args.command == 'predict':
        if args.batch_size < 1 or args.max_input_mib < 1: parser.error('batch and memory limits must be positive')
        import jax
        if args.no_compilation_cache:
            jax.config.update('jax_enable_compilation_cache', False)
        else:
            from .compilation import configure_compilation_cache
            configure_compilation_cache(args.compilation_cache)
    from .features import prepare_file
    began = time.perf_counter()
    records = [prepare_file(p, args.cache, gap_policy=args.gap_policy,
                            ignore_nonprotein=args.ignore_nonprotein) for p in paths]
    prep_seconds = time.perf_counter()-began
    if args.command == 'prepare':
        _atomic_json(args.output, dict(version=1, preparation_seconds=prep_seconds,
            records=[dict(input=str(p), cache=str(cache.resolve()), metadata=meta)
                     for p,(cache,meta) in zip(paths,records)]))
        return
    if args.model is None:
        from pkatrain.production_defaults import default_model_path
        args.model = default_model_path()
    from .allres import is_allres_export
    if is_allres_export(args.model):
        return _predict_allres(args, paths, records, prep_seconds, began)
    from .inference import Predictor
    predictor = Predictor(args.model, device=args.device, backend=args.backend,
                          batch_size=args.batch_size, max_input_bytes=args.max_input_mib*1024**2,
                          compilation_cache=False if args.no_compilation_cache else args.compilation_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Spool bucket-ordered predictions, then stream them out in original input
    # order without retaining all per-site predictions or graph tensors in RAM.
    offsets = {}
    with tempfile.TemporaryFile() as spool:
        for index, shift, pka in predictor.predict(records):
            row = dict(input_index=index, source=str(paths[index]),
                model_sha256=predictor.model_info['params_sha256'],
                sites=[dict(**site, shift=float(ds), pka=float(value))
                       for site,ds,value in zip(records[index][1]['sites'],shift,pka)])
            payload = (json.dumps(row, allow_nan=False)+'\n').encode()
            offsets[index] = (spool.tell(),len(payload)); spool.write(payload)
        fd, temporary = tempfile.mkstemp(dir=args.output.parent)
        try:
            with os.fdopen(fd,'wb') as stream:
                for index in range(len(paths)):
                    start,size = offsets[index]; spool.seek(start); stream.write(spool.read(size))
            os.replace(temporary,args.output)
        finally:
            Path(temporary).unlink(missing_ok=True)
    summary = dict(predictor.last_stats, preparation_seconds=prep_seconds,
        total_seconds=time.perf_counter()-began, output=str(args.output),
        model_sha256=predictor.model_info['params_sha256'], batch_size=args.batch_size,
        shape_capacities=[list(c) for c in predictor.executables])
    _atomic_json(args.output.with_suffix(args.output.suffix+'.summary.json'),summary)
    print(json.dumps(summary,indent=2))


def _predict_allres(args, paths, records, prep_seconds, began):
    """All-residue export: per-site pKa (bound / free / binding shift) and, unless --no-residues, per-residue pKa for
    every output, titratable gate and burial probabilities and the free-state interface logit."""
    import numpy as np
    from .allres import AllResiduePredictor, residue_rows
    states = ('bound', 'free') if args.state == 'both' else (args.state,)
    predictor = AllResiduePredictor(args.model, device=args.device, backend=args.backend, batch_size=args.batch_size,
                                    compilation_cache=False if args.no_compilation_cache else args.compilation_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    offsets = {}
    with tempfile.TemporaryFile() as spool:
        for index, out in predictor.predict(records, states):
            metadata = records[index][1]; sites = []
            for k, site in enumerate(metadata['sites']):
                r, g = int(out['query_residue'][k]), int(out['query_group'][k]); col = 0 if g < 7 else g - 6
                row = dict(site)
                for state in states:
                    suffix = '' if len(states) == 1 else f'_{state}'
                    o = out[state]
                    row[f'shift{suffix}'] = float(o['query_shift'][k]); row[f'pka{suffix}'] = float(o['query_pka'][k])
                    row[f'gate{suffix}'] = float(1 / (1 + np.exp(-o['gate_logit'][r, col])))
                if len(states) == 2: row['binding_shift'] = row['pka_bound'] - row['pka_free']
                sites.append(row)
            record = dict(input_index=index, source=str(paths[index]), model=predictor.model_info['run'],
                          model_sha256=predictor.model_info['params_sha256'], states=list(states), sites=sites)
            if not args.no_residues: record['residues'] = residue_rows(metadata, out, states)
            payload = (json.dumps(record, allow_nan=False) + '\n').encode()
            offsets[index] = (spool.tell(), len(payload)); spool.write(payload)
        fd, temporary = tempfile.mkstemp(dir=args.output.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                for index in range(len(paths)):
                    start, size = offsets[index]; spool.seek(start); stream.write(spool.read(size))
            os.replace(temporary, args.output)
        finally:
            Path(temporary).unlink(missing_ok=True)
    summary = dict(predictor.last_stats, preparation_seconds=prep_seconds, total_seconds=time.perf_counter() - began,
                   output=str(args.output), model=predictor.model_info['run'], model_sha256=predictor.model_info['params_sha256'],
                   batch_size=args.batch_size, shape_capacities=[[list(c), s] for c, s in predictor.executables])
    _atomic_json(args.output.with_suffix(args.output.suffix + '.summary.json'), summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
