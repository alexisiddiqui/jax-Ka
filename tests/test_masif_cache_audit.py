import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from masif_cache_audit import inspect_arrays,extent_metrics


def tetrahedron():
    v=np.array([[0.,0,0],[2,0,0],[0,2,0],[0,0,2]])
    return dict(vertices=v,faces=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]]),
        normals=np.ones((4,3)),owner=np.zeros(4,int),descriptors=np.ones((4,80)),
        raw_features=np.ones((4,4)),canonical_residue_ids=np.array([0]))


def test_cache_sweep_rejects_finite_but_degenerate_and_disconnected_geometry():
    a=tetrahedron();hard,review,stats=inspect_arrays(a)
    assert not hard and not review
    a['faces']=np.vstack([a['faces'],[0,0,1]])
    assert 'repeated_index_faces' in inspect_arrays(a)[0]


def test_convex_hull_check_catches_loss_hidden_by_axis_aligned_bbox():
    # [2,2,2] is inside the tetrahedron's bbox but outside its actual convex hull.
    v=tetrahedron()['vertices'];points=np.array([[.2,.2,.2],[2.,2.,2.]])
    m=extent_metrics(v,points)
    # Use a larger hull to put the point beyond the conservative 2 A tolerance.
    m=extent_metrics(v*10,points*10)
    assert m['atoms_outside_bbox_plus_2A']==0 and m['atoms_outside_convex_hull_plus_2A']==1
