"""Physical invariants and exact feature compatibility of the plane-tilt encoding."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from pkatrain import pkai_scratch as ps
from pkatrain.pkai_orientation import backbone_frames, orientation_features
from pkabench.pkai_backbone_pinder_eval import _features


def structure():
    names = ["N", "CA", "C", "O"] * 2
    return SimpleNamespace(hetero=np.zeros(8, bool), atom_name=np.asarray(names),
        chain_id=np.full(8, "A"), res_id=np.repeat([1, 2], 4), ins_code=np.full(8, ""),
        res_name=np.repeat(["ASP", "ALA"], 4),
        coord=np.asarray([[0,1,0],[0,0,0],[1,0,0],[1,1,0], [2,1,0],[2,0,0],[3,0,0],[3,1,0]], np.float64))


def encode(atoms):
    return orientation_features(atoms, [("A", 1, "", "ASP")])[0]


def test_orientation_preserves_base_and_has_expected_aligned_normal():
    atoms = structure(); x = encode(atoms); base, _ = _features(atoms, [("A",1,"","ASP")], "atom16aa20")
    blocks = x[:, :ps.SLOTS * 39].reshape(1, ps.SLOTS, 39)
    assert np.array_equal(blocks[..., :36].reshape(1, -1), base[:, :ps.SLOTS * 36])
    assert np.array_equal(x[:, -8:], base[:, -8:])
    np.testing.assert_allclose(blocks[0,:2,36:], [[0,0,1/5],[0,0,1/10]], atol=1e-8)
    assert not blocks[0,2:].any()


def test_orientation_rigid_motion_invariance():
    atoms = structure(); expected = encode(atoms)
    rotation, _ = np.linalg.qr(np.random.default_rng(17).normal(size=(3,3)))
    if np.linalg.det(rotation) < 0: rotation[:, 0] *= -1
    atoms.coord = atoms.coord @ rotation.T + [4.3,-7.1,2.2]
    np.testing.assert_allclose(encode(atoms), expected, rtol=0, atol=1e-7)
    for frame in backbone_frames(atoms).values():
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1)


def test_orientation_responds_to_plane_tilt_without_changing_distance_channels():
    atoms = structure(); before = encode(atoms).reshape(-1)[:ps.SLOTS*39].reshape(ps.SLOTS,39)
    atoms.coord[6] = [2,0,1]  # C does not enter the N/O context; rotates the neighbour frame.
    after = encode(atoms).reshape(-1)[:ps.SLOTS*39].reshape(ps.SLOTS,39)
    assert np.array_equal(before[:,:36], after[:,:36])
    np.testing.assert_allclose(after[:2,36:], [[-1/5,0,0],[-1/10,0,0]], atol=1e-8)
    x = encode(atoms)
    assert np.array_equal(ps.expand(ps.compact(x,"atom16aa20ori"),"atom16aa20ori"), x)


@pytest.mark.parametrize("which", ["missing", "degenerate", "query"])
def test_unavailable_frames_are_explicit_zero_vectors(which):
    atoms = structure()
    if which == "missing": atoms.atom_name[6] = "CB"
    elif which == "degenerate": atoms.coord[6] = atoms.coord[5]
    else: atoms.coord[2] = atoms.coord[1]
    blocks = encode(atoms)[:, :ps.SLOTS*39].reshape(1,ps.SLOTS,39)
    assert blocks[...,:36].any()
    assert not blocks[...,36:].any()


def test_orientation_ignores_sidechains():
    atoms = structure(); expected = encode(atoms)
    for name in ("hetero", "atom_name", "chain_id", "res_id", "ins_code", "res_name"):
        value = getattr(atoms,name)
        tail = {"hetero":False,"atom_name":"CB","chain_id":"A","res_id":2,"ins_code":"","res_name":"ALA"}[name]
        setattr(atoms,name,np.concatenate((value, np.asarray([tail], dtype=value.dtype))))
    atoms.coord = np.concatenate((atoms.coord, [[50,50,50]]))
    assert np.array_equal(encode(atoms), expected)
    atoms.coord[-1] = [0.1,0.1,0.1]
    assert np.array_equal(encode(atoms), expected)
