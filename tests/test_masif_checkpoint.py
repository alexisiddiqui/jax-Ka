"""Catch the real failure: finite descriptors can hide nonfinite model weights."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from masif_checkpoint import expected_shapes, validate_arrays, normalize_paper_electrostatics


def weights():
    return {name: np.ones(shape,np.float32) for name,shape in expected_shapes().items()}


def test_valid_complete_model():
    report=validate_arrays(weights())
    assert report['model_parameters']==66080
    assert report['parameters_finite']


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_reject_nonfinite_branch_before_inference(value):
    arrays=weights();arrays['W_conv_0'][0,0]=value
    with pytest.raises(ValueError,match='nonfinite tensors: W_conv_0'):
        validate_arrays(arrays)


def test_reject_nonfinite_optimizer_slot():
    arrays=weights();arrays['mu_rho_0/Adam']=np.array([np.nan])
    with pytest.raises(ValueError,match='mu_rho_0/Adam'):
        validate_arrays(arrays)


def test_reject_wrong_model_architecture():
    arrays=weights();arrays['fully_connected/weights']=np.ones((320,80),np.float32)
    with pytest.raises(ValueError,match='incompatible tensor shapes'):
        validate_arrays(arrays)
    del arrays['fully_connected/weights']
    with pytest.raises(ValueError,match='missing tensors'):
        validate_arrays(arrays)


def test_original_paper_clips_both_potential_signs_without_mutating_input():
    values=np.array([-30.,-3.,-1.5,0.,1.5,3.,30.])
    original=values.copy()
    np.testing.assert_allclose(normalize_paper_electrostatics(values),[-1.,-1.,-.5,0.,.5,1.,1.])
    np.testing.assert_array_equal(values,original)
