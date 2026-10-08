import numpy as np
from pkatrain.graph_pkmod_long import continuation_lr


def test_cosine_continuation_endpoints_and_monotonicity():
    values=[continuation_lr(epoch,batch,10) for epoch in range(21,101) for batch in range(1,11)]
    assert values[0]<1e-3 and values[0]>9.99e-4
    assert np.all(np.diff(values)<=0)
    np.testing.assert_allclose(values[-1],1e-5,rtol=1e-6)
