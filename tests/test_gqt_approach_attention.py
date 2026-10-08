import numpy as np

from pkabench.gqt_approach_attention import (
    build_clusters, choose_separation_direction, components, edge_categories,
    perturb_identities, same_partner_chain_masks,
)


def test_components_and_cross_partner_cluster_definitions():
    assert components([1,2,3,4],[(1,2),(3,4)]) == [[1,2],[3,4]]
    sites=[
        dict(chain="A",resnum=1,icode="",group="ASP",coords=[[0,0,0]]),
        dict(chain="B",resnum=2,icode="",group="LYS",coords=[[0,0,9]]),
        dict(chain="A",resnum=3,icode="",group="GLU",coords=[[50,0,0]]),
    ]
    pairs=[dict(site_i="A|1||ASP",site_j="B|2||LYS",strength_kbt=1.2)]
    rows,membership=build_clusters(sites,pairs,{"A":0,"B":1})
    assert {r["cluster_definition"] for r in rows} == {"geometric","coupling"}
    assert ("A",1,"","ASP") in membership["geometric"]


def test_edge_categories_distinguish_chain_from_partner():
    chains=np.array(["H","L","A"]);partners=np.array([0,0,1])
    neighbors=np.array([[0,1,2],[1,0,2],[2,0,1]])
    got=edge_categories(chains,partners,neighbors,np.ones_like(neighbors,bool))
    np.testing.assert_array_equal(got,[[0,1,2],[0,1,2],[0,2,2]])


def test_separation_is_monotonic_and_rigid():
    backbone=np.zeros((4,4,3),np.float32)
    backbone[:2,:,0]=np.array([0,1])[:,None];backbone[2:,:,0]=np.array([4,5])[:,None]
    direction,minimum=choose_separation_direction(backbone,np.array([0,0,1,1]))
    assert np.all(np.diff(minimum)>=0);np.testing.assert_allclose(direction,[1,0,0])
    before=np.linalg.norm(backbone[2,1]-backbone[3,1])
    moved=backbone.copy();moved[2:]+=12*direction
    assert np.linalg.norm(moved[2,1]-moved[3,1]) == before


def test_identity_interventions_are_deterministic_and_preserve_composition():
    nodes=np.zeros((4,24),np.float32);nodes[np.arange(4),[0,1,2,3]]=1
    partners=np.array([0,0,1,1]);chains=np.array(["H","L","A","A"])
    a,names=perturb_identities(nodes,partners,partners,7,shuffles=2)
    b,_=perturb_identities(nodes,partners,partners,7,shuffles=2)
    np.testing.assert_array_equal(a,b)
    for x,name in zip(a,names):
        if name.startswith("shuffle"):
            np.testing.assert_array_equal(np.sort(x[:2,:20],axis=0),np.sort(nodes[:2,:20],axis=0))
            np.testing.assert_array_equal(np.sort(x[2:,:20],axis=0),np.sort(nodes[2:,:20],axis=0))
    masked,mnames=same_partner_chain_masks(nodes,chains,partners,chains)
    h=masked[mnames.index("mask_same_partner_other_chain_for_H")]
    assert h[0,:20].sum()==1 and h[1,:20].sum()==0 and h[2,:20].sum()==1


def test_identity_intervention_assignment_does_not_restore_masked_features():
    padded=np.zeros((4,24),np.float32);padded[:,23]=1
    premask=padded.copy();premask[:,22]=1;premask[np.arange(4),np.arange(4)]=1
    variants,_=perturb_identities(premask,np.array([0,0,1,1]),np.array([0,1]),7,shuffles=1)
    changed=padded.copy();changed[:,:20]=variants[0,:,:20]
    assert not changed[:,22].any()
    np.testing.assert_array_equal(changed[:,23],1)


def test_trace_matches_standard_prediction():
    import jax
    import jax.numpy as jnp
    from pkanet.model import initialize,predict_pkpdb,predict_pkpdb_with_trace
    n,k,q=5,3,2;rng=np.random.default_rng(3)
    neighbors=np.stack([np.array([i,(i+1)%n,(i+2)%n]) for i in range(n)]).astype(np.int32)
    graph=dict(nodes=jnp.asarray(rng.normal(size=(n,24)),jnp.float32),node_mask=jnp.ones(n,bool),
        neighbors=jnp.asarray(neighbors),edge=jnp.asarray(rng.normal(size=(n,k,20)),jnp.float32),
        edge_mask=jnp.ones((n,k),bool),switch=jnp.ones((n,k),jnp.float32),
        query_residue=jnp.array([0,3],jnp.int32),query_group=jnp.array([0,5],jnp.int32))
    params=initialize(jax.random.PRNGKey(0))
    expected=predict_pkpdb(params,graph);trace=predict_pkpdb_with_trace(params,graph)
    np.testing.assert_allclose(trace["predicted_pka"],expected,rtol=1e-6,atol=1e-6)
    for layer in (*trace["encoder"],trace["query"]):
        np.testing.assert_allclose(np.asarray(layer["weights"]).sum(axis=1),1,rtol=1e-6,atol=1e-6)
