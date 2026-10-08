import numpy as np
from pkabench.site_weights import burial_weight, interface_weight


def test_burial_weight_is_smooth_bounded_and_clipped():
    w = burial_weight([0., .5, 1., 1.3, -.1])
    assert np.allclose(w, [1., .7, .4, .4, 1.])
    assert np.all(np.diff(burial_weight(np.linspace(0, 1, 11))) < 0)


def test_interface_weight_contact_decay_and_floor():
    w = interface_weight([0., 4., 7., 10., 30.])
    assert np.allclose(w[:2], 1.) and np.isclose(w[2], np.exp(-1.)) and np.isclose(w[3], np.exp(-2.)) and w[4] == .05
    assert np.all(np.diff(interface_weight(np.linspace(0, 40, 81))) <= 0)
