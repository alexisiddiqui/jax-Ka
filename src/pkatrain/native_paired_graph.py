"""Existing paired graph tensors for source-qualified biological assemblies."""
import numpy as np


def build(topology, side_by_chain, sites, labels, real_termini, split, *, node_termini=None):
    from jaxpropka.parameters import GROUPS
    from pkanet.graph import geometry
    from .site_graph_data import build_site_graph
    if split not in ('train','val'):raise ValueError('Unknown operational split')
    side=np.asarray([{'A':0,'B':1}[side_by_chain[k.chain]] for k in topology.keys],np.int8)
    if set(side.tolist())!={0,1}:raise ValueError('Paired graph requires both biological sides')
    real={tuple(k) for k in real_termini};lookup={(k.chain,k.number,k.insertion):i for i,k in enumerate(topology.keys)}
    nterm=np.asarray([(*k,'NTERM') in real for k in lookup]);cterm=np.asarray([(*k,'CTERM') in real for k in lookup])
    # Student endpoint features and chemical teacher termini are independent.
    # Existing native caches encode observed topology/segment endpoints. A
    # source-aware builder can retain that convention without granting an
    # artificial endpoint a teacher target (sites retain their frozen masks).
    if node_termini is not None:
        nterm,cterm=(np.asarray(x,bool) for x in node_termini)
        if nterm.shape!=(topology.n_residues,) or cterm.shape!=nterm.shape:
            raise ValueError('Student endpoint flags differ from native topology')
    graph,valid=geometry(topology.backbone,topology.chain_index,radius=20.)
    nodes=np.concatenate((np.eye(20,dtype=np.float32)[topology.native_index],
        np.stack((nterm,cterm,np.zeros(topology.n_residues,bool),valid),axis=-1)),axis=-1).astype(np.float32)
    graph.update(nodes=nodes,node_mask=np.ones(topology.n_residues,bool))
    values={state:{tuple(r[:4]):r[4] for r in rows} for state,rows in labels['pkai'].items()}
    mask='train_mask' if split=='train' else 'eval_mask';rows=[];queries=[];targets=[]
    for site in sites:
        key=(site['chain'],site['resnum'],site['icode'],site['group'])
        if not site[mask]:continue
        if site['group']=='ARG':raise ValueError('Existing site graph does not support ARG queries')
        if key[:3] not in lookup or key not in values['AB'] or key not in values[site['partner']]:
            raise ValueError('Qualified paired site lacks graph correspondence or teacher')
        if site['partner']!=side_by_chain[key[0]]:raise ValueError('Site free teacher differs from biological partner')
        rows.append(site);queries.append((lookup[key[:3]],GROUPS.index(key[3])))
        targets.append((values['AB'][key],values[site['partner']][key]))
    detail=dict(qualified_queries=len(rows),qualified_interface_queries=sum(r['interface'] for r in rows))
    if not rows or not detail['qualified_interface_queries']:
        return None,dict(detail,reason='no qualified graph-mapped paired interface sites')
    if len(set(queries))!=len(queries):raise ValueError('Duplicate qualified paired query')
    query=np.asarray(queries,np.int32);graph.update(query_residue=query[:,0],query_group=query[:,1])
    sg=build_site_graph(topology.backbone,topology.chain_index,nodes,query[:,0],query[:,1])
    free=graph['edge_mask'] & (side[graph['neighbors']]==side[:,None])
    site_side=side[sg['site_residue']]
    site_free=sg['site_edge_mask'] & (site_side[sg['site_neighbors']]==site_side[:,None])
    graph.update(sg,branch_edge_mask=np.stack((graph['edge_mask'],free)),
        branch_site_edge_mask=np.stack((sg['site_edge_mask'],site_free)),targets=np.asarray(targets,np.float32).T,
        **{name:np.asarray([r[name] for r in rows],bool if name=='interface' else np.float32)
           for name in ['w_burial','w_interface','interface','partner_distance_A','rsa_free']})
    cross_physical=topology.chain_index[graph['neighbors']]!=topology.chain_index[:,None]
    detail.update(free_edges_between_same_side_physical_chains=int(np.sum(free & cross_physical)),
        free_edges_between_opposite_sides=int(np.sum(free & (side[graph['neighbors']]!=side[:,None]))))
    if not np.isfinite(graph['targets']).all():raise ValueError('Nonfinite paired targets')
    return graph,detail
