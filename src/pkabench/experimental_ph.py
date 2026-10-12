"""Versioned, structure-level experimental pH metadata for optional auxiliary supervision.

This module does not change features, pKa targets, validation splits or loss functions.
Quality weights are deterministic binary eligibility, not calibrated confidence.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
FORMAT = 'experimental-ph-v1'
DATASETS = {'pinder': 'pretraining/pinder-pkai-v1', 'pkpdb': 'pretraining/pkpdb-full-v1'}
NUM = '(?:\\d+(?:[.,]\\d+)?|[.,]\\d+)'
MISSING = {'', 'none', '.', '?', 'null', 'unk', 'unknown', 'not applicable', 'n.a.', '/', 'unbuffered', 'not buffered', 'no buffer added', 'no buffer'}

def value(v):
    s = str(v).strip()
    if s.lower() in MISSING:
        return None
    if not re.fullmatch(NUM, s):
        raise ValueError('unparsed numerical value: ' + s)
    n = float(s.replace(',', '.'))
    if not 0 <= n <= 14:
        raise ValueError('pH outside 0–14: ' + s)
    return n

def parse_range(v):
    s = str('' if v is None else v).strip()
    if s.lower() in MISSING:
        return ('missing', None)
    s = re.sub('^pH\\s*[:=]?\\s*', '', s, flags=re.I)
    try:
        n = value(s)
        if n is not None:
            return ('point', n)
    except ValueError:
        pass
    match = re.fullmatch('(' + NUM + ')\\s*(?:[-–—]|to)\\s*(' + NUM + ')', s, re.I)
    if match:
        try:
            a, b = map(value, match.groups())
        except ValueError:
            return ('unparsed', str(v))
        if a > b:
            return ('unparsed', str(v))
        return ('point', a) if a == b else ('range', [a, b])
    parts = re.split('\\s*[;/]\\s*', s)
    if len(parts) > 1:
        try:
            vals = [value(x) for x in parts]
        except ValueError:
            return ('unparsed', str(v))
        if all((x is not None for x in vals)):
            return ('points', vals)
    return ('unparsed', str(v))
TEXT_RE = re.compile('\\bpH\\s*(?:(?:values?|of|is|was|at)\\s*)?[:=]?\\s*(' + NUM + ')(?:\\s*(?:[-–—]|to)\\s*(?:pH\\s*)?(' + NUM + '))?', re.I)

def text_candidates(text):
    out = []
    for m in TEXT_RE.finditer(text):
        try:
            a = value(m.group(1))
            b = value(m.group(2)) if m.group(2) else None
        except ValueError:
            continue
        if b is not None and b < a:
            continue
        out.append({'low': a, 'high': b if b is not None else a, 'context': text[max(0, m.start() - 80):min(len(text), m.end() + 100)]})
    return out

def audit_record(r):
    points = []
    ranges = []
    sources = set()
    warnings = []
    notes = []
    rawpoint = []
    rawrange = []
    raw_nmr = r.get('pdbx_nmr_exptl_sample_conditions') or []
    for c in r.get('exptl_crystal_grow') or []:
        rawpoint.append(c.get('pH'))
        rawrange.append(c.get('pdbx_pH_range'))
        notes.append(c.get('pdbx_details') or '')
        try:
            n = value(c.get('pH'))
        except ValueError as e:
            n = None
            warnings.append(str(e))
        if n is not None:
            points.append(n)
            sources.add('crystallization_point')
        kind, v = parse_range(c.get('pdbx_pH_range'))
        if kind in ('point', 'points'):
            points.extend([v] if kind == 'point' else v)
            sources.add('crystallization_range_field_point')
        elif kind == 'range':
            ranges.append(v)
            sources.add('crystallization_range')
        elif kind == 'unparsed':
            warnings.append('unparsed_crystallization_range: ' + str(v))
    for c in raw_nmr:
        try:
            n = value(c.get('pH'))
        except ValueError as e:
            n = None
            warnings.append(str(e))
        if n is None:
            continue
        units = str(c.get('pH_units') or '').strip()
        if units.lower() in ('ph', '', '.', '?'):
            points.append(n)
            sources.add('nmr')
            if units.lower() != 'ph':
                warnings.append('nmr_units_unspecified')
        else:
            warnings.append('nmr_other_units: ' + units)
    for v in r.get('em_buffer_ph', []):
        try:
            n = value(v)
        except ValueError as e:
            n = None
            warnings.append(str(e))
        if n is not None:
            points.append(n)
            sources.add('em_buffer')
    points = sorted(set(points))
    ranges = sorted({tuple(x) for x in ranges})
    text = text_candidates('\n'.join(notes))
    fatal = [w for w in warnings if w != 'nmr_units_unspecified']
    if 0.0 in points:
        fatal.append('zero_pH_requires_verification')
        warnings.append('zero_pH_requires_verification')
    if len(points) > 1:
        fatal.append('multiple_structured_point_values')
        warnings.append('multiple_structured_point_values')
    if points and any((not lo - 1e-06 <= points[0] <= hi + 1e-06 for lo, hi in ranges)):
        fatal.append('point_outside_reported_range')
        warnings.append('point_outside_reported_range')
    if ranges and max((lo for lo, hi in ranges)) > min((hi for lo, hi in ranges)):
        fatal.append('disjoint_reported_ranges')
        warnings.append('disjoint_reported_ranges')
    text_disagreement = bool(len(points) == 1 and any((not c['low'] - 1e-06 <= points[0] <= c['high'] + 1e-06 for c in text)))
    candidate = None
    if fatal:
        status = 'structured_metadata_needs_review'
    elif len(points) == 1:
        if text_disagreement:
            status = 'structured_point_with_other_text_values'
            warnings.append('other_pH_values_in_crystallization_notes')
        else:
            status = 'single_consistent_structured_point'
            candidate = points[0]
    elif ranges:
        status = 'structured_range_only'
    elif text:
        status = 'text_only_candidate'
    else:
        status = 'no_recorded_pH'
    return {'status': status, 'candidate_ph': candidate, 'sources': sorted(sources), 'structured_points': points, 'structured_ranges': ranges, 'text_candidates': text, 'warnings': sorted(set(warnings)), 'experimental_methods': sorted({x['method'] for x in r.get('exptl') or []}), 'raw_crystal_ph': rawpoint, 'raw_crystal_range': rawrange, 'raw_nmr': raw_nmr, 'raw_em_buffer_ph': r.get('em_buffer_ph', []), 'crystal_notes': notes}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def annotation_dir(root, dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    return Path(root) / 'annotations' / FORMAT / dataset

def quality_record(record, audit, multiplicity):
    """Strict eligibility requires one point and no unresolved warning."""
    eligible = audit['status'] == 'single_consistent_structured_point' and (not audit['warnings'])
    value = audit['candidate_ph']
    return {'id': record['id'], 'pdb_id': record['pdb_id'].upper(), 'split': record.get('split', 'train'), 'ph_value': value, 'ph_target': value if eligible else None, 'ph_mask': eligible, 'ph_quality_weight': float(eligible), 'ph_status': audit['status'], 'ph_sources': audit['sources'], 'ph_warnings': audit['warnings'], 'parent_entry_count': multiplicity, 'parent_weight': 1.0 / multiplicity, 'validation_chain_similarity': str(record.get('chain_70_to_validation', '0')) == '1'}

def build(root, dataset):
    """Install an immutable, hash-verified metadata sidecar from our audited snapshot."""
    from pkabench.dataset_transfer import locked
    root = Path(root)
    source = root / 'audits/ph-coverage'
    inputs = [source / 'metadata.json', source / 'em-metadata.json']
    meta = json.loads(inputs[0].read_text())
    em = json.loads(inputs[1].read_text())
    for name in ('recovered-metadata.json', 'extra-metadata.json'):
        path = source / name
        if path.exists():
            inputs.append(path)
            extra = json.loads(path.read_text())
            for pdb_id, record in extra.items():
                meta[pdb_id] = record
    pool_path = root / DATASETS[dataset] / 'pool-v3.tsv'
    inputs.append(pool_path)
    with pool_path.open() as f:
        pool = list(csv.DictReader(f, delimiter='\t'))
    records = [{**r, 'split': 'train'} for r in pool]
    if dataset == 'pinder':
        cohort_path = root / 'training/ogqt-pinder-factorial-v1/cohort.json'
        inputs.append(cohort_path)
        cohort = json.loads(cohort_path.read_text())
        records += [{'id': r['id'], 'pdb_id': r.get('pdb_id', r.get('pdb', r['id'][:4])), 'split': 'val'} for r in cohort['records'] if r['split'] == 'val']
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate or train/val overlap in annotation IDs')
    pdb_ids = {r['pdb_id'].upper() for r in records}
    missing = pdb_ids - meta.keys()
    if missing:
        raise ValueError(f'metadata not audited for {len(missing)} PDBs: {sorted(missing)[:8]}')
    evidence = {}
    for i in sorted(pdb_ids):
        raw = dict(meta[i])
        raw['em_buffer_ph'] = raw.get('em_buffer_ph', em.get(i, {}).get('em_buffer_ph', []))
        evidence[i] = {'pdb_id': i, 'audit': audit_record(raw), 'raw': raw}
    counts = collections.Counter(((r['split'], r['pdb_id'].upper()) for r in records))
    rows = [quality_record(r, evidence[r['pdb_id'].upper()]['audit'], counts[r['split'], r['pdb_id'].upper()]) for r in sorted(records, key=lambda r: (r['split'], r['id']))]
    provenance = {str(p.relative_to(root)): digest(p) for p in inputs}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    dest = annotation_dir(root, dataset)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with locked(dest.parent / f'.{dataset}.lock'):
        if dest.exists():
            manifest = json.loads((dest / 'manifest.json').read_text())
            if manifest['input_fingerprint'] != fingerprint:
                raise ValueError('annotation inputs changed; create a new annotation version')
            load_annotations(root, dataset)
            return manifest
        temp = Path(tempfile.mkdtemp(prefix=f'.{dataset}.pending-', dir=dest.parent))
        try:
            for name, values in [('records.jsonl', rows), ('pdb-evidence.jsonl', list(evidence.values()))]:
                with (temp / name).open('w') as f:
                    for row in values:
                        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            manifest = {'format': FORMAT, 'dataset': dataset, 'input_fingerprint': fingerprint, 'inputs': provenance, 'source': 'Deposited RCSB crystallization/NMR metadata and EM mmCIF headers', 'policy': 'One consistent structured point; zero warnings; pH 0.0, ranges, notes-only and unspecified NMR units masked', 'unit': 'structure/complex; shared across sites, not independent per-residue labels', 'quality_weight': 'binary eligibility (0 or 1), not calibrated confidence', 'counts': {split: {'entries': sum((r['split'] == split for r in rows)), 'eligible': sum((r['split'] == split and r['ph_mask'] for r in rows))} for split in sorted({r['split'] for r in rows})}, 'files': {name: digest(temp / name) for name in ('records.jsonl', 'pdb-evidence.jsonl')}}
            (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            os.rename(temp, dest)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    load_annotations(root, dataset)
    return manifest

def load_annotations(root, dataset):
    """Verify the sidecar, its frozen pool and unique identifiers before joining."""
    folder = annotation_dir(root, dataset)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['format'] != FORMAT or manifest['dataset'] != dataset:
        raise ValueError('annotation format mismatch')
    pool_rel = f'{DATASETS[dataset]}/pool-v3.tsv'
    if digest(Path(root) / pool_rel) != manifest['inputs'][pool_rel]:
        raise ValueError('annotation pool hash mismatch')
    if dataset == 'pinder':
        cohort_rel = 'training/ogqt-pinder-factorial-v1/cohort.json'
        if digest(Path(root) / cohort_rel) != manifest['inputs'][cohort_rel]:
            raise ValueError('annotation cohort hash mismatch')
    for name, expected in manifest['files'].items():
        if digest(folder / name) != expected:
            raise ValueError(f'annotation file hash mismatch: {name}')
    rows = {}
    with (folder / 'records.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['id'] in rows:
                raise ValueError('duplicate annotation ID')
            if row['ph_mask'] != (row['ph_target'] is not None):
                raise ValueError('invalid target/mask')
            if row['ph_quality_weight'] != float(row['ph_mask']):
                raise ValueError('invalid quality weight')
            if row['ph_mask'] and (not 0 < row['ph_target'] <= 14 or row['ph_warnings']):
                raise ValueError('invalid eligible pH')
            rows[row['id']] = row
    return rows

def targets_for_records(annotations, names, *, exclude_validation_similar=False):
    """Arrays in explicit record order; masked targets are finite zero fillers.

    Optional parent weights de-duplicate experiments within a split. They are
    separate from quality weights and are not automatically applied to any loss.
    """
    import numpy as np
    if (missing := (set(names) - annotations.keys())):
        raise KeyError(f'annotation IDs missing: {sorted(missing)[:8]}')
    rows = [annotations[name] for name in names]
    mask = np.asarray([r['ph_mask'] and (not (exclude_validation_similar and r['validation_chain_similarity'])) for r in rows], dtype=bool)
    return {'ph_target': np.asarray([r['ph_target'] if m else 0.0 for r, m in zip(rows, mask)], dtype=np.float32), 'ph_mask': mask, 'ph_quality_weight': mask.astype(np.float32), 'ph_parent_weight': np.asarray([r['parent_weight'] for r in rows], dtype=np.float32)}

def targets_for_feature_records(annotations, names, *, exclude_validation_similar=False):
    """Align the existing pKAI store's explicit ``train-ID`` / ``val-ID`` names."""
    by_feature_name = {f"{r['split']}-{r['id']}": r for r in annotations.values()}
    return targets_for_records(by_feature_name, names,
                               exclude_validation_similar=exclude_validation_similar)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify'])
    parser.add_argument('dataset', choices=sorted(DATASETS))
    parser.add_argument('--root', default=os.environ.get('PKABENCH_RUNTIME'))
    args = parser.parse_args()
    if not args.root:
        parser.error('--root or PKABENCH_RUNTIME required')
    result = build(args.root, args.dataset) if args.action == 'build' else {'verified_entries': len(load_annotations(args.root, args.dataset))}
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
