"""Independent invariants for the versioned preparation/mask migration."""
import json
import sys
from collections import Counter
from pathlib import Path
from .runtime import require_compute, atomic_json, digest


def run(out):
    require_compute()
    import pyarrow.parquet as pq
    from .schema import KEY
    from .glycan_buffer_policy import BUFFERS
    out = Path(out); report = json.loads((out/'report.json').read_text())
    identity = lambda r: tuple(r[k] for k in KEY)
    lost_sites = 0; gained_sites = 0; unchanged_evaluation = 0; reused = 0; glycan_pairs = 0; errors = []
    buffer_ids = set(); oldaccepted = set(); newglycanids = set()
    for source in report['source_campaigns']:
        campaign = Path(source['campaign']); manifest = json.loads((campaign/'manifest.json').read_text()); previous = Path(manifest['source_campaign'])
        assert digest(previous/'manifest.json') == manifest['source_manifest_sha256']
        assert digest(previous/'site_masks.parquet') == manifest['source_masks_sha256']
        before = {identity(r): r for r in pq.read_table(previous/'site_masks.parquet').to_pylist()}
        after = {identity(r): r for r in pq.read_table(campaign/'site_masks.parquet').to_pylist()}
        assert before.keys() <= after.keys(), 'Previously prepared site disappeared'
        structures = {r['complex_id']: r for r in pq.read_table(campaign/'structures.parquet').to_pylist()}
        oldstructures = {r['complex_id']: r for r in pq.read_table(previous/'structures.parquet').to_pylist()}
        assert oldstructures.keys() <= structures.keys(), 'Previously accepted preparation rejected'
        for cid, structure in structures.items():
            removal = json.loads((campaign/'structures'/cid/'removed_components.json').read_text())['components']
            has_buffer = any(c['name'] in BUFFERS for c in removal)
            if has_buffer:
                buffer_ids.add(cid)
            for c in removal:
                expected = (20., 25.) if c['policy_class'] in ('glycan', 'buffer') else (25., 25.) if c['policy_class'] == 'metal' else (15., 25.)
                assert (c['train_radius_A'], c['eval_radius_A']) == expected
            if cid in oldstructures:
                oldaccepted.add(cid); reused += 1
                assert structure['content_sha256'] == oldstructures[cid]['content_sha256'], 'Protein content hash changed'
                assert (campaign/'structures'/cid/'AB.cif').resolve() == (previous/'structures'/cid/'AB.cif').resolve(), 'Expected immutable reused geometry'
            else:
                assert any(c['policy_class'] == 'glycan' for c in removal), 'New preparation without glycan rescue'
                glycan_pairs += 1; newglycanids.add(cid)
        for key, old in before.items():
            new = after[key]
            assert old['eval_mask'] == new['eval_mask'], 'Old evaluation mask changed'
            unchanged_evaluation += 1
            assert not new['train_mask'] or old['train_mask'], 'Old training mask became less restrictive'
            if old['train_mask'] != new['train_mask']:
                assert key[0] in buffer_ids, 'Training mask changed without a buffer'
                lost_sites += 1
            assert old['natural_gap_tier'] == new['natural_gap_tier'] and old['interface'] == new['interface'], 'Old gap/interface annotation changed'
        gained_sites += sum(r['train_mask'] for key, r in after.items() if key not in before)
    current = pq.read_table(out/'eligibility-with-existing-assignments.parquet').to_pylist()
    proposal = Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/universe/combined-split-v1/usable-proposal-v2/proposal.parquet')
    assert digest(proposal) == report['original_proposal_sha256']
    previous = {r['complex_id']: r for r in pq.read_table(proposal).to_pylist()}
    lost = []; gained = []
    for r in current:
        old = previous[r['complex_id']]
        assert all(r[k] == old[k] for k in ('split', 'component_id', 'benchmark_eligible'))
        if r['split'] == 'train' and r['benchmark_eligible']:
            if old['train_interface_sites'] > 0 and r['train_interface_sites'] == 0:
                assert r['complex_id'] in buffer_ids and r['complex_id'] in oldaccepted
                lost.append(r['complex_id'])
            if old['train_interface_sites'] == 0 and r['train_interface_sites'] > 0:
                assert r['complex_id'] in newglycanids
                gained.append(r['complex_id'])
    result = {'checks_passed': True, 'reused_preparations_unchanged': reused, 'new_glycan_preparations': glycan_pairs,
              'existing_site_evaluation_masks_checked_unchanged': unchanged_evaluation,
              'existing_training_sites_lost_to_buffer_radius': lost_sites,
              'new_training_sites_from_glycan_preparations': gained_sites,
              'site_count_scope': 'All prepared candidates, before benchmark role eligibility and split assignment.',
              'training_interface_pairs_lost_to_buffer_radius': len(lost), 'training_interface_pairs_gained_from_glycans': len(gained),
              'lost_training_interface_pair_ids': lost, 'gained_training_interface_pair_ids': gained,
              'old_proposal_hash_unchanged': True, 'split_component_and_role_assignments_unchanged': True}
    atomic_json(out/'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('_ids')}, indent=2), flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
