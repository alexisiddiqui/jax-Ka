from dataclasses import replace
from pathlib import Path

import pytest

from pkabench.source_policy import SourcePolicy


@pytest.fixture
def policy():
    return SourcePolicy.load(Path(__file__).parents[1]/'config/v7-source-policy.json')


def candidate(source, id='one', **kwargs):
    return dict(source=source, source_native_id=id, prepared_key='same-prepared-complex',
                teacher_context_id='native-fullchain-teacher-v1', eligible=True, **kwargs)


def test_first_party_sources_have_configurable_ownership(policy):
    rows = [candidate(s) for s in ['pinder', 'sabdab', 'pkpdb']]
    assert policy.choose_owner(rows)['source'] == 'pkpdb'
    changed = replace(policy, source_precedence=('sabdab', 'pkpdb', 'pinder'))
    assert changed.choose_owner(rows)['source'] == 'sabdab'
    assert changed.sha256 != policy.sha256
    assert policy.choose_owner(rows)['source'] == 'pkpdb'


def test_quality_comes_before_precedence(policy):
    high = candidate('pkpdb'); high['eligible'] = False
    assert policy.choose_owner([high, candidate('sabdab'), candidate('pinder')])['source'] == 'sabdab'


@pytest.mark.parametrize('field,value', [('prepared_key',None),('prepared_key','different-conformation'),
                                       ('teacher_context_id','cropped-Fv-teacher')])
def test_sequence_or_provenance_alias_cannot_authorise_reuse(policy,field,value):
    a,b=candidate('pkpdb'),candidate('sabdab');b[field]=value
    with pytest.raises(ValueError):policy.choose_owner([a,b])


@pytest.mark.parametrize('source',['pkpdb','sabdab','pinder'])
@pytest.mark.parametrize('role',['external_heldout','sabdab_test','validation'])
def test_precedence_never_overrides_protected_evaluation(policy,source,role):
    assert policy.training_action(source,role) == 'exclude_training'


def test_preserve_pkpdb_requires_a_different_explicit_evaluation_policy(policy):
    alternative=replace(policy,evaluation_policy='preserve_pkpdb_report_clean_test_subset')
    assert alternative.sha256 != policy.sha256
    assert alternative.training_action('pkpdb','sabdab_test') == 'keep_training_exclude_affected_test_component'
    assert alternative.training_action('pkpdb','external_heldout') == 'exclude_training'
    assert alternative.training_action('pkpdb','validation') == 'exclude_training'
    assert alternative.training_action('pinder','sabdab_test') == 'exclude_training'


def test_partial_or_duplicate_precedence_is_rejected(policy):
    with pytest.raises(ValueError):replace(policy,source_precedence=('pkpdb','sabdab','sabdab'))


def test_unknown_policy_does_not_silently_fall_back(policy):
    with pytest.raises(ValueError):replace(policy,evaluation_policy='prefer_any_training')
