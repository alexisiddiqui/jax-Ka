"""Synthetic graph regression: joint H/L free branches retain H--L edges."""
import numpy as np
from biotite.structure.io import pdbx

from pkabench.sabdab_units import prepare
from pkatrain.native_paired_graph import build
from test_sabdab_units import cif,row


def test_free_branch_keeps_joint_antibody_and_removes_only_other_partner():
    file=cif();cat=file.block['atom_site']
    for name in ['label_comp_id','auth_comp_id']:
        if name in cat:cat[name]=pdbx.CIFColumn(np.full(cat.row_count,'GLU'))
    _,topology,side,meta=prepare(file,row())
    sites=[dict(chain=c,resnum=1,icode='',group='GLU',partner=p,train_mask=True,eval_mask=True,
        interface=True,w_burial=1.,w_interface=1.,partner_distance_A=4.,rsa_free=.5) for c,p in [('A','A'),('C','B')]]
    labels=dict(pkai=dict(AB=[['A',1,'','GLU',4.],['C',1,'','GLU',5.]],
        A=[['A',1,'','GLU',4.2]],B=[['C',1,'','GLU',5.3]]))
    graph,detail=build(topology,meta['side_by_chain'],sites,labels,[],'train')
    assert detail['free_edges_between_same_side_physical_chains']>0
    assert detail['free_edges_between_opposite_sides']==0
    free=graph['branch_edge_mask'][1]
    assert (side[graph['neighbors']][free]==np.broadcast_to(side[:,None],free.shape)[free]).all()
    assert np.array_equal(graph['targets'],np.array([[4.,5.],[4.2,5.3]],np.float32))
    assert not graph['nodes'][:,20:22].any()  # no source-proven true termini
    endpoint_graph,_=build(topology,meta['side_by_chain'],sites,labels,[],'train',
        node_termini=(topology.nterm,topology.cterm))
    assert endpoint_graph['nodes'][:,20:22].any()
    assert np.array_equal(endpoint_graph['targets'],graph['targets'])
    assert np.array_equal(endpoint_graph['query_group'],graph['query_group'])
    assert np.array_equal(endpoint_graph['query_residue'],graph['query_residue'])
