import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import masif_pkpdb_recovery as recovery
from masif_topology_audit import topology


def test_recovery_inventory_rejects_membership_and_chain_identity_changes(tmp_path,monkeypatch):
    rows=[dict(id='1abc',split='train',group='g',n_chains='2')]
    monkeypatch.setattr(recovery,'membership',lambda runtime:(rows,{'pool':'hash'}))
    u=dict(id='pkpdb/1abc/chain-0001',cid='1abc',dataset='pkpdb',split='train',group='g',expected_chains=2,
        chain_index=1,unit_policy=recovery.POLICY,n=10,source_files_sha256={k:'hash' for k in ['structure','defects.json','receipt.json']})
    path=tmp_path/'inventory.json';meta=dict(membership_sha256={'pool':'hash'})
    recovery.write_inventory(path,[u],meta)
    assert recovery.load_inventory(path,tmp_path)[0]==[u]
    recovery.write_inventory(path,[dict(u,split='val')],meta)
    with pytest.raises(ValueError,match='membership'):recovery.load_inventory(path,tmp_path)
    recovery.write_inventory(path,[dict(u,chain_index=0)],meta)
    with pytest.raises(ValueError,match='ID mismatch'):recovery.load_inventory(path,tmp_path)
    recovery.write_inventory(path,[u],meta)
    data=json.loads(path.read_text());data['units'][0]['n']=11;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash'):recovery.load_inventory(path,tmp_path)


def test_topology_locality_uses_edge_paths_not_nearby_disconnected_surface():
    # Three faces share edge 0--1. A spatially close disconnected triangle
    # must remain unaffected by a graph-geodesic radius.
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1],[.1,.1,.1],[.2,.1,.1],[.1,.2,.1]])
    f=np.array([[0,1,2],[1,0,3],[0,1,4],[5,6,7]])
    stats,d=topology(v,f)
    assert stats['nonmanifold_edges']==1 and stats['max_edge_face_incidence']==3
    assert stats['vertices_within_12A_graph']==5
    assert np.isfinite(d[:5]).all() and np.isinf(d[5:]).all()


def test_topology_closed_tetrahedron_has_no_flagged_neighbourhoods():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    f=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
    stats,d=topology(v,f)
    assert stats['nonmanifold_edges']==stats['boundary_edges']==0
    assert np.isinf(d).all()
