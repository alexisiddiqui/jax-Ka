"""Combined candidate-universe clustering and split readiness checks."""
import json
from collections import Counter
from pathlib import Path
from .runtime import require_compute, atomic_json, digest


def run(campaigns, out):
    require_compute()
    import pyarrow.parquet as pq
    from .pool import sequence
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    candidates = {}
    sources = []
    eligible = {}
    accepted = set()
    for campaign in map(Path, campaigns):
        manifest = json.loads((campaign / 'manifest.json').read_text())
        report = json.loads((campaign / 'report.json').read_text())
        if report['pipeline_errors']:
            raise ValueError(f'Unresolved pipeline errors: {campaign}')
        masks_path = campaign / 'site_masks.parquet'
        if digest(masks_path) != report['mask_sha256']:
            raise ValueError('Mask hash mismatch')
        sources.append({'campaign': str(campaign), 'manifest_sha256': digest(campaign / 'manifest.json'),
                        'mask_sha256': digest(masks_path), 'report_sha256': digest(campaign / 'report.json')})
        for row in manifest['candidates']:
            cid = row['complex_id']
            if cid in candidates:
                raise ValueError(f'Duplicate candidate across campaigns: {cid}')
            candidates[cid] = row
        seen = set()
        for row in pq.read_table(masks_path).to_pylist():
            key = tuple(row[k] for k in ('complex_id', 'chain', 'resnum', 'icode', 'group'))
            if key in seen:
                raise ValueError(f'Duplicate site: {key}')
            seen.add(key)
            if row['eval_mask'] and not row['train_mask']:
                raise ValueError('Evaluation mask not a subset of training mask')
            if (row['train_mask'] or row['eval_mask']) and row['natural_gap_tier'] not in ('clean', 'uncertain'):
                raise ValueError('Ineligible natural-gap tier retained')
            cid = row['complex_id']
            accepted.add(cid)
            counts = eligible.setdefault(cid, Counter())
            for role in ('train', 'eval'):
                if row[role + '_mask']:
                    counts[role + '_sites'] += 1
                    counts[role + '_interface_sites'] += int(row['interface'])
    atomic_json(out / 'index.json', {'candidates': list(candidates.values()), 'sources': sources, 'production_allowed': False})
    sequence(out)
    diversity = json.loads((out / 'sequence/diversity.json').read_text())
    assignments = diversity['assignments']['cdr_partner']
    groups = {}
    for cid, component in assignments.items():
        group = groups.setdefault(component, {'component_id': component, 'candidates': 0, 'accepted': 0,
                                              'train_pairs': 0, 'eval_pairs': 0, 'train_interface_pairs': 0,
                                              'eval_interface_pairs': 0, 'train_sites': 0, 'eval_sites': 0})
        group['candidates'] += 1
        group['accepted'] += int(cid in accepted)
        for role in ('train', 'eval'):
            counts = eligible.get(cid, {})
            group[role + '_pairs'] += int(counts.get(role + '_sites', 0) > 0)
            group[role + '_interface_pairs'] += int(counts.get(role + '_interface_sites', 0) > 0)
            group[role + '_sites'] += counts.get(role + '_sites', 0)
    ordered = sorted(groups.values(), key=lambda g: (-g['candidates'], g['component_id']))
    atomic_json(out / 'components.json', ordered)
    largest = ordered[0]['candidates'] if ordered else 0
    missing = diversity['missing_cdr']
    blockers = ['Forced-test experimental/PKAD-3 sequence inventory has not been supplied or verified.']
    if largest > .05 * len(assignments):
        blockers.append('Largest CDR-mode connected component exceeds the specified 5% limit.')
    if missing:
        blockers.append('Candidates with missing CDR annotations require explicit resolution or exclusion.')
    summary = {'candidates': len(candidates), 'accepted_with_site_rows': len(accepted),
               'assigned_candidates': len(assignments), 'components': len(groups),
               'largest_component': largest, 'largest_component_fraction': largest / max(1, len(assignments)),
               'missing_cdr': len(missing), 'usable_pairs_missing_cdr': sum(bool(eligible.get(c, {}).get('train_sites')) for c in missing),
               'totals': {k: sum(g[k] for g in ordered) for k in ('train_pairs', 'eval_pairs', 'train_interface_pairs', 'eval_interface_pairs', 'train_sites', 'eval_sites')},
               'top_components': ordered[:10], 'freeze_blockers': blockers, 'frozen': False,
               'limitations': 'Whole sampled candidate graph, including rejected bridges. MMseqs2 heuristic 30% identity/80% bidirectional coverage. CDR replacement follows shared specification. No production split or teacher predictions created.'}
    atomic_json(out / 'readiness.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
