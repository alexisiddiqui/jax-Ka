import numpy as np

from pkatrain.pkai_shift_compare import inverse_frequency_weights, shift_bin


def test_shift_bins_and_inverse_frequency_weights():
    assert [shift_bin(value) for value in (0.0, 0.499, 0.5, -0.999, 1.0, 1.999, 2.0)] == [0, 0, 1, 1, 2, 2, 3]
    counts = np.asarray([12, 6, 3, 1])
    weights = inverse_frequency_weights(counts)
    np.testing.assert_allclose(counts * weights, np.repeat(counts.sum() / 4, 4))
    assert np.isclose(np.average(weights, weights=counts), 1.0)
