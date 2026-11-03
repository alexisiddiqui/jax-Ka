"""Physical controls for uncapped, nested geodesic surface comparisons."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from augmentation_patch_scale import graph, supports, surface_metrics


def test_unique_mesh_edges_do_not_double_geodesic_distances():
    from scipy.sparse.csgraph import dijkstra
    v=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)
    f=np.array([[0,1,2],[0,2,3]])
    g=graph(v,f)
    assert g[0,2]==np.sqrt(2)
    np.testing.assert_allclose(dijkstra(g,indices=0),[0,1,np.sqrt(2),1])


def test_supports_are_nested_uncapped_and_exclude_disconnected_components():
    n=300
    angles=np.arange(n)*2*np.pi/n
    v=np.vstack([[0,0,0],np.c_[np.cos(angles),np.sin(angles),np.zeros(n)],[[100,0,0]]])
    f=np.array([[0,i+1,(i+1)%n+1] for i in range(n)])
    frame=dict(vertices=v,faces=f,anchor_valid=np.array([True]),anchor_vertex_ids=np.array([0]))
    s=supports(frame,[.5,1.1,2.])[0]
    assert len(s[.5][0])==1 and len(s[1.1][0])==301
    assert set(s[.5][0])<=set(s[1.1][0])<=set(s[2.][0])
    assert 301 not in s[2.][0]


def test_rigid_surface_and_chemistry_are_invariant_but_exposure_changes_are_detected():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]],float);translation=np.array([10,2,3])
    ref=dict(vertices=v,vertex_area=np.ones(4),owner=np.arange(4),raw_features=np.arange(16).reshape(4,4)/16)
    var=dict(ref,vertices=(v-translation)@rotation.T)
    ids=np.arange(4)
    s=surface_metrics(ref,var,ids,ids,rotation,translation,4)
    assert s['distance']<1e-12 and s['exposure_tv']==0
    np.testing.assert_allclose(s['rmse'],0,atol=1e-12)
    changed=dict(var,vertex_area=np.array([2.,1.,1.,1.]))
    s=surface_metrics(ref,changed,ids,ids,rotation,translation,4)
    assert np.isclose(s['exposure_tv'],.15) and np.isclose(s['area_relative_change'],.25)
