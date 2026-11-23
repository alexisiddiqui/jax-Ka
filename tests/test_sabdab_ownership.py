from sabdab_v7_ownership import select_representatives


def pair(a,b,exact=False):
    return dict(unit_id=a,peer=b,redundant_native_copy=True,exact_teacher_equivalence=exact)


def test_near_copies_do_not_claim_teacher_reuse():
    result=select_representatives(['c','a','b'],[pair('a','b',True),pair('a','c'),pair('b','c')])
    assert result['a']['membership_action']=='include_native'
    assert result['b']['membership_action']=='alias_exact_context'
    assert result['c']['membership_action']=='withhold_redundant_native_copy'
    assert not result['c']['teacher_reuse_allowed']


def test_no_tolerance_chaining():
    result=select_representatives(['a','b','c'],[pair('a','b'),pair('b','c')])
    assert result['a']['representative_id']=='a'
    assert result['b']['representative_id']=='a'
    assert result['c']['membership_action']=='include_native'
