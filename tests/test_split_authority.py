from dataclasses import replace
import json
from pathlib import Path

import pytest

from pkabench.role_engine import Evidence, PUBLISHED, STRICT, Unit, assign, hard_test_status, quarantine_group_conflicts
from pkabench.source_registry import SourceRegistry, SourceSpec
from pkabench.split_authority import SplitAuthority


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    return SplitAuthority.load(ROOT / 'config/v7-split-authority-v2.json')


@pytest.fixture
def registry():
    return SourceRegistry.load(ROOT / 'config/v7-source-registry.json')


def unit(uid='u', source='pkpdb', split='train', **kw):
    return Unit(uid, source, split, 'declared_scope', **kw)


def edges(uid, components=(), axis='antigen'):
    return [Evidence(uid, PUBLISHED, 'antibody_CDR' if axis == 'antigen' else 'antigen', 'assessed', 'no_hit'),
            Evidence(uid, PUBLISHED, axis, 'published', 'hit' if components else 'no_hit', tuple(components))]


def roles(units, evidence, policy, registry):
    return assign(units, evidence, {'tr': 'train', 'va': 'test'}, policy, registry)


def test_authority_is_strict_and_preserves_native_mapping(policy):
    assert policy.role('train') == 'train'
    assert policy.role('test') == 'validation'
    data = policy.to_dict()
    data['authority']['operational_roles']['test'] = 'hard_test'
    with pytest.raises(ValueError):
        SplitAuthority.from_dict(data)
    data = policy.to_dict(); data['extra'] = True
    with pytest.raises(ValueError):
        SplitAuthority.from_dict(data)
    assert SplitAuthority.from_dict(policy.to_dict()).sha256 == policy.sha256


@pytest.mark.parametrize('source', ['pkpdb', 'pinder'])
@pytest.mark.parametrize('native', ['train', 'validation'])
@pytest.mark.parametrize('axis', ['antibody_CDR', 'antigen'])
def test_same_rules_for_original_training_and_validation(source, native, axis, policy, registry):
    a = roles([unit(source=source, split=native)], edges('u', ['va'], axis), policy, registry)[0]
    assert a.operational_role == 'validation'
    assert a.native_source_split == native


def test_pkpdb_validation_does_not_override_authoritative_training(policy, registry):
    assert roles([unit(split='validation')], edges('u', ['tr']), policy, registry)[0].operational_role == 'train'


def test_bridge_quarantines_even_authoritative_unit_without_moving_component(policy, registry):
    u = unit(source='sabdab', native_component='tr')
    assert roles([u], edges('u', ['va']), policy, registry)[0].operational_role == 'quarantine'


def test_strict_hits_do_not_veto_published_training(policy, registry):
    ee = edges('u', ['tr']) + [Evidence('u', STRICT, 'generic', 'legacy70_80', 'hit', ('old-heldout',))]
    assert roles([unit()], ee, policy, registry)[0].operational_role == 'train'


def test_unassessed_and_unmapped_antibody_fail_closed(policy, registry):
    assert roles([unit()], [], policy, registry)[0].operational_role == 'pending'
    assert roles([unit()], edges('u'), policy, registry)[0].operational_role == 'pending'
    ee = edges('u', ['tr']) + [Evidence('u', PUBLISHED, 'antibody_CDR', 'failed', 'unassessed')]
    assert roles([unit()], ee, policy, registry)[0].operational_role == 'pending'
    assert roles([unit(declared_general=True)], edges('u'), policy, registry)[0].operational_role == 'train'


def test_no_antigen_is_reference_only(policy, registry):
    u = unit(source='sabdab', native_component='tr', supervised=False)
    assert roles([u], edges('u'), policy, registry)[0].operational_role == 'diagnostic'


def test_precedence_does_not_change_assignments_except_policy_hash(policy, registry):
    data = policy.to_dict(); data['source_precedence'].reverse()
    altered = SplitAuthority.from_dict(data)
    a = roles([unit()], edges('u', ['va']), policy, registry)[0]
    b = roles([unit()], edges('u', ['va']), altered, registry)[0]
    assert replace(a, authority_sha256='') == replace(b, authority_sha256='')


def test_new_vhh_source_uses_same_role_engine(policy, registry):
    registry = SourceRegistry(registry.sources + (SourceSpec('vhh', True, 'paired', 'split', 'cluster', 'scope', 'vhh_metadata'),))
    data = policy.to_dict(); data['source_precedence'].append('vhh')
    assert roles([unit(source='vhh')], edges('u', ['tr']), SplitAuthority.from_dict(data), registry)[0].operational_role == 'train'


def test_crop_or_parent_identity_never_proves_ownership(policy, registry):
    base = dict(source_native_id='one', eligible=True, operational_role='train', pdb_id='same')
    records = [dict(base, source=s) for s in ['pkpdb', 'sabdab']]
    with pytest.raises(ValueError):
        registry.choose_owner(records, policy.source_precedence)
    records = [dict(r, prepared_key='same', teacher_context_id='same') for r in records]
    assert registry.choose_owner(records, policy.source_precedence)['source'] == 'pkpdb'
    records[1]['teacher_context_id'] = 'crop'
    with pytest.raises(ValueError):
        registry.choose_owner(records, policy.source_precedence)
    records[1]['teacher_context_id'] = 'same'; records[1]['operational_role'] = 'validation'
    with pytest.raises(ValueError):
        registry.choose_owner(records, policy.source_precedence)


def test_hard_test_requires_both_roles_all_axes_and_prior_use():
    checks = [(axis, role, 'no_hit') for axis in ['antibody_CDR', 'antigen', 'generic', 'interface'] for role in ['train', 'validation']]
    assert hard_test_status(checks, prior_use='never_used') == 'candidate_not_frozen'
    assert hard_test_status(checks) == 'diagnostic_prior_use_unresolved'
    assert hard_test_status(checks[:-1], prior_use='never_used') == 'pending_novelty_checks'
    assert hard_test_status(checks[:-1] + [('antigen', 'train', 'hit')]) == 'diagnostic_overlap'


def test_unknown_components_and_duplicate_ids_rejected(policy, registry):
    with pytest.raises(ValueError):
        roles([unit()], edges('u', ['unknown']), policy, registry)
    with pytest.raises(ValueError):
        roles([unit(), unit()], [], policy, registry)


def test_group_guard_does_not_move_authoritative_components(policy, registry):
    units = [unit('s', 'sabdab', 'test', native_component='va'), unit('p', split='train')]
    aa = roles(units, edges('s') + edges('p', ['tr']), policy, registry)
    out = {a.unit_id: a for a in quarantine_group_conflicts(aa, {'s': ['exact:x'], 'p': ['exact:x']})}
    assert out['s'].operational_role == 'validation'
    assert out['p'].operational_role == 'quarantine'


def test_group_guard_rejects_authoritative_scope_collision(policy, registry):
    units = [unit('s', 'sabdab', 'test', native_component='va'), unit('t', 'sabdab', 'train', native_component='tr')]
    aa = roles(units, edges('s') + edges('t'), policy, registry)
    with pytest.raises(ValueError, match='authoritative'):
        quarantine_group_conflicts(aa, {'s': ['exact:x'], 't': ['exact:x']})


def test_pinder_removed_first_when_that_resolves_cross_source_conflict(policy, registry):
    uu = [unit('p', 'pkpdb'), unit('r', 'pinder', 'validation')]
    aa = roles(uu, edges('p', ['tr']) + edges('r', ['va']), policy, registry)
    out = {a.unit_id: a for a in quarantine_group_conflicts(aa, {'p': ['exact:x'], 'r': ['exact:x']})}
    assert out['p'].operational_role == 'train'
    assert out['r'].operational_role == 'quarantine'


def test_compatible_pkpdb_anchor_survives_opposing_pinder_record(policy, registry):
    uu = [unit('s', 'sabdab', 'test', native_component='va'),
          unit('p', 'pkpdb', 'validation'), unit('r', 'pinder')]
    aa = roles(uu, edges('s') + edges('p', ['va']) + edges('r', ['tr']), policy, registry)
    out = {a.unit_id: a for a in quarantine_group_conflicts(aa, {u.unit_id: ['exact:x'] for u in uu})}
    assert out['s'].operational_role == out['p'].operational_role == 'validation'
    assert out['r'].operational_role == 'quarantine'


def test_deterministic_under_input_and_evidence_order(policy, registry):
    uu = [unit('b', split='validation'), unit('a')]
    ee = edges('b', ['tr']) + edges('a', ['va'])
    assert roles(uu, ee, policy, registry) == roles(list(reversed(uu)), list(reversed(ee)), policy, registry)
