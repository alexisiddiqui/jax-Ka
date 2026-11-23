from dataclasses import replace

import pytest

from pkabench.group_adjudication import GroupPolicy, assert_boundaries, make_groups, propagate_family_anchors, resolve_groups
from pkabench.role_engine import Assignment, quarantine_group_conflicts


def a(uid, role='train', source='pkpdb', mapped=False):
    return Assignment(uid, source, role, '', 'scope', role, (), ('component',) if mapped else (), 'authority', 'registry')


def unit(g='g', source='pkpdb', general=True):
    return dict(source=source, source_groups=[g], declared_general=general)


def test_explanation_preserves_historical_result_and_records_group():
    aa = [a('t'), a('v', 'validation', 'pinder')]
    groups = {'t': {'exact:x'}, 'v': {'exact:x'}}
    old = quarantine_group_conflicts(aa, groups)
    explained, log = quarantine_group_conflicts(aa, groups, explain=True)
    assert explained == old
    assert log[0]['group_id'] == 'exact:x'
    assert log[0]['removed_members'] == ['v']


def test_removed_unit_does_not_trigger_secondary_group_quarantine():
    aa = [a('s', 'validation', 'sabdab'), a('t'), a('v', 'validation')]
    groups = {'s': {'exact:anchor'}, 't': {'exact:anchor', 'family:legacy'}, 'v': {'family:legacy'}}
    old = {x.unit_id: x.operational_role for x in quarantine_group_conflicts(aa, groups)}
    out, log = resolve_groups(aa, groups)
    new = {x.unit_id: x.operational_role for x in out}
    assert old['v'] == 'quarantine'
    assert new == {'s': 'validation', 't': 'quarantine', 'v': 'validation'}
    assert len(log) == 1
    assert_boundaries(out, groups)


def test_mapped_family_exemption_retains_exact_input_constraints():
    aa = [a('t', mapped=True), a('v', 'validation', mapped=True), a('g')]
    units = {u.unit_id: unit() for u in aa}
    members = [dict(unit_id=u, sequence_key='same') for u in ['t', 'v']]
    groups = make_groups(units, members, aa, exempt_mapped=True)
    assert groups['t'] == {'exact:same'}
    assert groups['g'] == {'family:pkpdb:g'}
    out, _ = resolve_groups(aa, groups)
    assert all(x.operational_role == 'quarantine' for x in out if x.unit_id in {'t', 'v'})


def test_unmapped_general_family_inherits_only_agreeing_anchors():
    aa = [a('t', mapped=True), a('g', 'validation'), a('ab', 'pending')]
    units = {u.unit_id: unit(g='g', general=u.unit_id != 'ab') for u in aa}
    out, _ = propagate_family_anchors(aa, units)
    assert {x.unit_id: x.operational_role for x in out} == {'t': 'train', 'g': 'train', 'ab': 'pending'}
    assert out[1].native_source_split == 'validation'
    assert out[1].matched_components == ()


def test_opposing_anchors_do_not_move_mapped_units():
    aa = [a('t', mapped=True), a('v', 'validation', mapped=True), a('g')]
    out, _ = propagate_family_anchors(aa, {x.unit_id: unit() for x in aa})
    assert [x.operational_role for x in out] == ['train', 'validation', 'quarantine']


def test_multiple_family_anchors_cannot_bridge_through_general_record():
    aa = [a('t', mapped=True), a('v', 'validation', mapped=True), a('g')]
    units = {'t': unit('one'), 'v': unit('two'), 'g': dict(source='pkpdb', source_groups=['one', 'two'], declared_general=True)}
    out, _ = propagate_family_anchors(aa, units)
    assert out[-1].operational_role == 'quarantine'


def test_staged_removal_is_deterministic_and_prefers_pinder():
    aa = [a('t'), a('v', 'validation', 'pinder')]
    groups = {'t': {'exact:x', 'family:g'}, 'v': {'exact:x', 'family:g'}}
    out, log = resolve_groups(aa, groups)
    other, other_log = resolve_groups(list(reversed(aa)), {k: set(reversed(sorted(v))) for k, v in reversed(list(groups.items()))})
    assert {x.unit_id: x for x in out} == {x.unit_id: x for x in other}
    assert log == other_log
    assert out[0].operational_role == 'train' and out[1].operational_role == 'quarantine'


def test_authoritative_exact_collision_is_not_reassigned():
    with pytest.raises(ValueError, match='authoritative'):
        resolve_groups([a('t', source='sabdab'), a('v', 'validation', 'sabdab')], {'t': {'exact:x'}, 'v': {'exact:x'}})


def test_unassessed_pending_and_bridged_units_never_propagate():
    aa = [a('anchor', mapped=True), a('missing', 'pending'), a('bridge', 'quarantine', mapped=True)]
    out, _ = propagate_family_anchors(aa, {x.unit_id: unit() for x in aa})
    assert out == aa


def test_policy_requires_explicit_compatible_options():
    with pytest.raises(ValueError):
        GroupPolicy('invalid', anchor_propagation=True)
    assert GroupPolicy('one').sha256 != GroupPolicy('two').sha256
