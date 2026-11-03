"""Paired ANM samples preserve original components and prescribed amplitudes."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from augmentation_modes10 import paired_directions


def test_modes_preserve_old_random_draws_and_fixed_rmsd():
    vectors=np.linalg.qr(np.random.default_rng(8).normal(size=(30,10)))[0];values=np.arange(1,11.)
    five,ten,fraction,basis=paired_directions(vectors[:,:5],values[:5],vectors,values,17)
    rng=np.random.default_rng(17);expected=(rng.normal(size=(8,5))/np.sqrt(values[:5]))@vectors[:,:5].T
    expected=expected.reshape(8,10,3);expected/=np.sqrt(np.mean(np.sum(expected**2,axis=-1),axis=-1))[:,None,None]
    np.testing.assert_allclose(five,expected)
    for directions in (five,ten):np.testing.assert_allclose(np.sqrt(np.mean(np.sum(directions**2,axis=-1),axis=-1)),1.)
    assert np.all((fraction>0)&(fraction<1))
    np.testing.assert_allclose(basis.T@basis,np.eye(10),atol=1e-12)
    assert not np.allclose(ten,five)


def test_changed_lowest_mode_subspace_is_rejected():
    vectors=np.linalg.qr(np.random.default_rng(9).normal(size=(30,11)))[0];values=np.arange(1,11.)
    old=vectors[:,[0,1,2,3,10]]
    with pytest.raises(ValueError,match='subspace changed'):paired_directions(old,values[:5],vectors[:,:10],values,17)
