import numpy as np
from pkabench.long_gap_policy import LONG_GAP_RADII_A, long_gap_radius, site_usable
from pkabench.pkpdb_mask_all import heldout_70

SITE = np.array([[0., 0., 0.]])


def gap(kind, length, centres):
    return dict(kind=f'{kind}_gap', length=length, centres=[list(c) for c in centres], extent=8.)


def test_radii_match_approved_table():
    assert LONG_GAP_RADII_A == {('terminal', 10): 30., ('terminal', 20): 35., ('terminal', 30): 40., ('terminal', 50): 45.,
                                ('internal', 5): 20., ('internal', 10): 20., ('internal', 20): 25.}
    assert long_gap_radius('terminal', 11) == 35. and long_gap_radius('terminal', 50) == 45. and long_gap_radius('terminal', 51) is None
    assert long_gap_radius('internal', 4) == 20. and long_gap_radius('internal', 21) is None


def test_long_terminal_gap_uses_flank_ca_radius():
    far = [(0., 0., 35.)]; near = [(0., 0., 34.9)]
    assert site_usable(SITE, True, [(gap('terminal', 15, far), None, None)]) == (True, 'usable')
    assert site_usable(SITE, True, [(gap('terminal', 15, near), None, None)]) == (False, 'near_long_gap')
    assert site_usable(SITE, True, [(gap('terminal', 80, far), None, None)]) == (False, 'untested_long_gap')


def test_internal_gap_needs_both_flanks_clear():
    assert site_usable(SITE, True, [(gap('internal', 8, [(0., 0., 20.), (0., 20., 0.)]), None, None)])[0]
    assert not site_usable(SITE, True, [(gap('internal', 8, [(0., 0., 20.), (0., 19., 0.)]), None, None)])[0]


def test_short_tail_tightened_and_uncalibrated_short_gap_excludes():
    anchor = np.array([[0., 0., 25.]])
    assert site_usable(SITE, True, [(gap('terminal', 8, [(0., 0., 25.)]), 20., anchor)]) == (False, 'near_gap_short_tail_pkai')
    assert site_usable(SITE, True, [(gap('terminal', 4, [(0., 0., 25.)]), 15., anchor)])[0]
    assert site_usable(SITE, True, [(gap('terminal', 4, [(0., 0., 25.)]), None, None)]) == (False, 'uncalibrated_short_gap')


def test_envelope_clearance_and_eligibility_unchanged():
    assert site_usable(SITE, True, [(gap('terminal', 3, [(0., 0., 60.)]), None, None)])[0]  # 60 - 3*3.8 - 8 >= 20
    assert site_usable(SITE, False, []) == (False, 'ineligible')


def test_heldout_70_requires_both_coverages():
    assert heldout_70(.7, .8, .8, ['benchmark']) and not heldout_70(.69, .9, .9, ['benchmark'])
    assert not heldout_70(.95, .9, .5, ['benchmark']) and not heldout_70(.95, .9, .9, ['experimental'])
