"""Deferred interpolation preserves meshing precision and skips remeshing."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import augmentation_masif_bridge as bridge


def test_deferred_electrostatics_samples_original_mesh_coordinates(tmp_path,monkeypatch):
    precise=np.array([[100.123456789,2.,3.],[4.,5.123456789,6.]])
    mesh=tmp_path/'features.npz'
    np.savez(mesh,vertices=precise.astype(np.float32),electrostatics_vertices=precise)
    original=mesh.read_bytes()
    def potentials(vertices,pdb):
        np.testing.assert_array_equal(vertices,precise)
        return vertices[:,0].copy(),'fixed-charge-inventory'
    module=ModuleType('masif_modules.read_data_from_surface')
    module.normalize_electrostatics=lambda v:np.clip(v,-1,1)
    monkeypatch.setitem(sys.modules,'masif_modules.read_data_from_surface',module)
    monkeypatch.setattr(bridge,'electrostatics',potentials)
    job=dict(mesh=str(mesh),protonated=str(tmp_path/'protonated.pdb'),output=str(tmp_path/'electrostatics.npz'),
             receipt=str(tmp_path/'receipt.json'),signature='stage-test')
    bridge.deferred_electrostatics(job)
    with np.load(job['output']) as z:np.testing.assert_array_equal(z['electrostatics'],np.clip(precise[:,0]/10,-1,1).astype(np.float32))
    receipt=json.loads(Path(job['receipt']).read_text())
    assert receipt['charge_identity_sha256']=='fixed-charge-inventory'
    assert mesh.read_bytes()==original


def test_incomplete_electrostatics_cannot_enter_full_filter():
    import pytest
    from augmentation_surface_filter import bad_fraction,KEYS
    from augmentation_surface_staged import CHEAP_KEYS
    row={k:0. for k in KEYS};row['electrostatics_rmse']=None
    limits={k:1. for k in KEYS}
    assert bad_fraction([row],1,limits,CHEAP_KEYS)==0
    with pytest.raises(TypeError):bad_fraction([row],1,limits)
