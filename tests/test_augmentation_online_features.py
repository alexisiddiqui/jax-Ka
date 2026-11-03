"""Standalone CPU regressions run inside the same one-GPU pilot allocation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

if os.environ.get('ANM_TRAINING_SOURCE'):
    sys.path.insert(0,os.environ['ANM_TRAINING_SOURCE'])
    import pkatrain
    pkatrain.__path__.insert(0,str(Path(os.environ['ANM_PILOT_OVERLAY'])/'pkatrain'))

import numpy as np
from jaxpropka.parameters import THREE_TO_INDEX
from pkanet.graph import geometry
from pkatrain.site_graph_data import build_site_graph,candidate_sites
from pkatrain.production_inputs import _sequence_index
from pkatrain.augmentation_graphs import backbone_table,INPUT_FIELDS
from pkatrain.augmentation_cache_pilot import FeatureSource,unit_graph
from pkatrain.augmentation_store import AugmentationStore,ReleaseWriter
from pkatrain.augmentation_online_features import CompactSource,DERIVED,build_compact,rebuild_features


def fixture():
    ca=np.array([[0,0,0],[3.8,0,0],[7.6,1,0],[11.4,1,1],
                 [0,4,0],[3.8,4,0],[7.6,5,0],[11.4,5,1]],np.float32)
    bb=ca[:,None]+np.array([[-.9,.7,.2],[0,0,0],[1.3,.1,0],[1.9,.5,.3]],np.float32)
    chain=np.array([0]*4+[1]*4,np.int32);nodes=np.zeros((8,24),np.float32)
    for i,aa in enumerate(('ASP','LYS','ALA','GLU','HIS','GLY','TYR','CYS')):nodes[i,THREE_TO_INDEX[aa]]=1
    nodes[[0,4],20]=1;nodes[[3,7],21]=1;nodes[:,23]=1
    sites=candidate_sites(nodes);qr=sites[:,0];qg=sites[:,1]
    graph,_=geometry(bb,chain);graph.update(nodes=nodes,node_mask=np.ones(8,bool),query_residue=qr,query_group=qg)
    graph.update(build_site_graph(bb,chain,nodes,qr,qg))
    return graph,bb,chain


class OnlineFeatureTests(unittest.TestCase):
    def test_active_geometry_matches_existing_numpy_builders(self):
        native,bb,chain=fixture();graph={k:v for k,v in native.items()}
        graph['nodes']=np.concatenate((graph['nodes'],backbone_table(bb,chain,_sequence_index(chain))),axis=-1)
        compact={k:v for k,v in graph.items() if k not in DERIVED};rebuilt=rebuild_features(compact)
        for name in DERIVED:
            mask=graph['site_edge_mask' if name.startswith('site_') else 'edge_mask']
            np.testing.assert_allclose(rebuilt[name][mask],graph[name][mask],atol=2e-5,rtol=2e-5)
        for name in compact:np.testing.assert_array_equal(rebuilt[name],compact[name])

    def test_real_compact_records_and_partner_loader_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'pilot';compact=Path(temporary)/'compact';capacities={}
            native,bb,chain=fixture();seq=_sequence_index(chain)
            with ReleaseWriter(root/'features',{},zstd_level=1) as features,ReleaseWriter(root/'controls',{},zstd_level=1) as controls:
                for dataset in ('pkpdb','pinder'):
                    control={k:v.copy() for k,v in native.items()}
                    control['nodes']=np.concatenate((control['nodes'],backbone_table(bb,chain,seq)),axis=-1)
                    partners=chain if dataset=='pinder' else np.zeros(8,np.int32)
                    control.update(partners=partners,
                        free_edge_mask=native['edge_mask']&(partners[native['neighbors']]==partners[:,None]),
                        free_site_edge_mask=native['site_edge_mask']&(partners[native['site_residue']][native['site_neighbors']]==partners[native['site_residue']][:,None]))
                    key=dataset+'/fixture';parent=key+('/A' if dataset=='pinder' else '/assembly')
                    controls.add(key,{},control,summary=dict(dataset=dataset))
                    ids=np.arange(4 if dataset=='pinder' else 8,dtype=np.int32)
                    for c in (0,1):
                        moved=bb[ids].copy();moved[-1]+=[c*.6,c*.2,0]
                        g=unit_graph(native,moved,chain[ids],seq[ids],ids)
                        dims=[8,max(native['neighbors'].shape[1],g['neighbors'].shape[1]),len(native['query_residue']),len(native['site_residue']),max(native['site_neighbors'].shape[1],g['site_neighbors'].shape[1])]
                        meta=dict(dataset=dataset,source_parent=parent,control_key=key,candidate=c,dims=dims)
                        features.add(parent+f'/c{c:02d}',meta,g,summary=meta)
                        capacities[dataset+':128']=[128,*dims[1:]]
                features.finish();controls.finish()
            (root/'build-summary.json').write_text(json.dumps(dict(capacities=capacities,native_capacities=capacities)))
            storage=build_compact(root,compact)
            self.assertEqual(storage['records'],4)
            with AugmentationStore(compact/'features') as store:
                for row in store.rows:self.assertFalse(set(store.raw(row['id'])['arrays'])&DERIVED)
            for dataset in ('pkpdb','pinder'):
                old=FeatureSource(root,2,2,dataset=dataset);new=CompactSource(root,compact,2,2,dataset=dataset)
                try:
                    spec=old.plans('128','augmented',1)[0];a,mask,_=old.load(spec);b,other,_=new.load(spec)
                    np.testing.assert_array_equal(mask,other)
                    for name in b:np.testing.assert_array_equal(a[name],b[name])
                    for slot in range(2):
                        for branch in range(b['nodes'].shape[1]):
                            rebuilt=rebuild_features({k:v[slot,branch] for k,v in b.items()})
                            for name in DERIVED:
                                active=a['site_edge_mask' if name.startswith('site_') else 'edge_mask'][slot,branch]
                                np.testing.assert_allclose(rebuilt[name][active],a[name][slot,branch][active],atol=2e-5,rtol=2e-5)
                finally:old.close();new.close()


if __name__=='__main__':unittest.main()
