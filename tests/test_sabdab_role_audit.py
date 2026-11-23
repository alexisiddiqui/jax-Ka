from dataclasses import replace

from pkabench.role_engine import Assignment
import sabdab_v7_roles as audit


def test_expected_antigen_is_separate_from_resolved_scope():
    rows = [dict(agchains='A', agtypes='PROTEIN', agresolvedseqs='ACDE',
                 agexpectedseqs='ACDEFG', ab_ag_cluster='one')]
    sequences = {}
    observed = audit.target_index(rows, sequences)
    union = audit.target_index(rows, sequences, expected=True)
    assert {sequences[t['sequence_key']] for t in observed} == {'ACDE'}
    assert {sequences[t['sequence_key']] for t in union} == {'ACDE', 'ACDEFG'}


def test_unresolved_reference_not_substituted_for_native_input():
    kinds = ['native_observed_coordinates', 'numbering_reference_including_unresolved',
             'source_sequence_inventory', 'expected_antigen', 'resolved_antigen',
             'published_variable_domain', 'Hseq', 'Hseq_expected']
    assert {m['sequence_kind'] for m in audit.selected_members([dict(sequence_kind=k) for k in kinds])} == {
        'native_observed_coordinates', 'source_sequence_inventory', 'resolved_antigen', 'published_variable_domain'}


def test_partial_ig_without_cdr3_is_not_assessed_as_clean():
    raw = {'partial': [dict(chain_type='H', numbering=[[27, '', 'A']])],
           'complete': [dict(chain_type='H', numbering=[[105, '', 'A']])]}
    attempted = {'partial', 'complete', 'negative'}
    assert not audit.numbering_assessed('partial', 'protein', raw, attempted)
    assert audit.numbering_assessed('complete', 'antibody_H', raw, attempted)
    assert not audit.numbering_assessed('negative', 'antibody', raw, attempted)
    assert audit.numbering_assessed('negative', 'protein', raw, attempted)
    assert not audit.numbering_assessed('unknown', 'protein', raw, attempted)


def assigned(uid, role, components=()):
    return Assignment(uid, 'pkpdb', role, '', 'scope', role, (), components, 'authority', 'registry')


def test_general_holdout_guard_protects_mapped_roles_and_ignores_ig_frameworks(monkeypatch, tmp_path):
    assignments = [assigned('tr', 'train'), assigned('val', 'validation', ('v',)), assigned('ig', 'train')]
    units = {a.unit_id: dict(declared_general=True) for a in assignments}
    members = [dict(unit_id=u, sequence_key=k, annotation_role='protein') for u, k in [('tr', 'q'), ('val', 't'), ('ig', 'h')]]
    def fake(folder, queries, targets, executable, workers):
        assert 'h' not in queries  # Conserved Ig framework cannot trigger generic guard.
        assert set(targets) == {'t'}
        return {'q': {'t'}}
    monkeypatch.setattr(audit, 'mmsearch', fake)
    out, _ = audit.general_holdout_guard(assignments, units, members, {'h': [dict(chain_type='H')]},
        {'q': 'AAAA', 't': 'AAAT', 'h': 'CCCC'}, 'unused', tmp_path / 'mmseqs', 1)
    assert {a.unit_id: a.operational_role for a in out} == {'tr': 'quarantine', 'val': 'validation', 'ig': 'train'}


def test_generic_similarity_alone_does_not_override_two_mapped_roles(monkeypatch, tmp_path):
    assignments = [assigned('tr', 'train', ('tr',)), assigned('val', 'validation', ('va',))]
    units = {a.unit_id: dict(declared_general=True) for a in assignments}
    members = [dict(unit_id=u, sequence_key=u, annotation_role='protein') for u in units]
    monkeypatch.setattr(audit, 'mmsearch', lambda *args: {'tr': {'val'}})
    out, evidence = audit.general_holdout_guard(assignments, units, members, {},
        {'tr': 'AAAA', 'val': 'AAAT'}, 'unused', tmp_path / 'mmseqs', 1)
    assert out == assignments
    assert evidence[0]['action'] == 'mapped_family_similarity_diagnostic'


def test_general_guard_prefers_removing_pinder_when_sufficient(monkeypatch, tmp_path):
    assignments = [assigned('tr', 'train'), replace(assigned('val', 'validation', ('va',)), source='pinder')]
    units = {a.unit_id: dict(declared_general=True) for a in assignments}
    members = [dict(unit_id=u, sequence_key=u, annotation_role='protein') for u in units]
    monkeypatch.setattr(audit, 'mmsearch', lambda *args: {'tr': {'val'}})
    out, _ = audit.general_holdout_guard(assignments, units, members, {},
        {'tr': 'AAAA', 'val': 'AAAT'}, 'unused', tmp_path / 'mmseqs', 1)
    assert {a.unit_id: a.operational_role for a in out} == {'tr': 'train', 'val': 'quarantine'}
