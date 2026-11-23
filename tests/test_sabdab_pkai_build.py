import copy
import pytest

from sabdab_v7_pkai_build import canary,validate_labels


def labels():
    return dict(errors=[],**{model:{state:[['A',1,'','ASP',3.8]] for state in ['AB','A','B']}
        for model in ['pkai','pkai_plus']})


def test_validate_both_models_and_all_states():
    validate_labels(labels())
    for mutation in ['nonfinite_plus','duplicate','missing_state','reported_error']:
        value=labels()
        if mutation=='nonfinite_plus':value['pkai_plus']['B'][0][4]=float('nan')
        if mutation=='duplicate':value['pkai']['AB']*=2
        if mutation=='missing_state':del value['pkai_plus']['A']
        if mutation=='reported_error':value['errors']=[['pkai','A','failed']]
        with pytest.raises(ValueError):validate_labels(value)


def test_canary_reproducible_and_covers_reuse_both_roles_and_size_tails():
    rows=[dict(unit_id=str(i),n_res=i+100,operational_role='train' if i%2 else 'validation',
        teacher_action='reuse_verified_same_native_pilot' if i in [8,21] else 'build_native_pkai_pkai_plus_AB_A_B') for i in range(100)]
    selected=canary(rows,16)
    assert selected==canary(list(reversed(rows)),16)
    ids={r['unit_id'] for r in selected}
    assert {'8','21','0','1','98','99'}<=ids
    assert len(ids)==len(selected)
