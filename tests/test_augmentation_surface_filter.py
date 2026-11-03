"""Selection and missing-patch handling without statistical model training."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from augmentation_surface_filter import evaluation_ids,bad_fraction,farthest,KEYS


def test_frozen_design_is_disjoint_and_balanced():
    parents=[dict(parent=f'{d}/{b}/{k}',dataset=d,size_bin=b,sample_kind=k)
             for d in ('atlas','pinder','pkpdb') for b in range(5) for k in ('typical','tail')]
    heldout=evaluation_ids(parents)
    assert len(heldout)==12 and len(set(r['parent'] for r in parents)-heldout)==18
    assert all(sum(p.startswith(d+'/') for p in heldout)==4 for d in ('atlas','pinder','pkpdb'))
    assert heldout==evaluation_ids(list(reversed(parents)))


def test_missing_anchors_count_as_bad_and_one_bad_feature_is_enough():
    limits={k:1 for k in KEYS};good={k:.5 for k in KEYS};bad=dict(good,electrostatics_rmse=2)
    assert bad_fraction([good,bad],4,limits)==.75
    assert bad_fraction([],0,limits)==1


def test_farthest_selects_by_aligned_deformation_not_translation():
    ref=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    one=ref.copy();one[3,2]+=1
    two=ref.copy();two[2,1]+=1
    coords={'reference':ref,'1':ref+10,'2':one,'3':two}
    chosen=farthest([1,2,3],coords,2)
    assert set(chosen)=={2,3}
    assert farthest([],coords)==[]


def test_surface_parity_accepts_vertex_permutations_but_rejects_changed_geometry():
    import pytest
    from augmentation_surface_filter_cpu import canonical_errors
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    ref=dict(vertices=v,faces=np.array([[0,1,2],[0,2,3]]),owner=np.arange(4),
             raw_features=np.arange(16).reshape(4,4).astype(float),anchor_vertex_ids=np.array([0,2,-1]))
    permutation=np.array([2,0,3,1]);inverse=np.argsort(permutation)
    var=dict(vertices=v[permutation],faces=inverse[ref['faces']],owner=ref['owner'][permutation],
             raw_features=ref['raw_features'][permutation],anchor_vertex_ids=np.array([inverse[0],inverse[2],-1]))
    assert max(canonical_errors(ref,var).values())==0
    var['vertices']=var['vertices'].copy();var['vertices'][0,0]+=.01
    with pytest.raises(ValueError,match='geometry changed'):canonical_errors(ref,var)


def test_deferred_electrostatics_matches_exhaustive_selection(tmp_path):
    from augmentation_surface_filter_summary import staged_selection
    ref=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    coordinates={'reference':ref};candidates=[];patches=[]
    for c in range(1,6):
        xyz=ref.copy();xyz[3,2]+=c*.5;coordinates[str(c)]=xyz
        candidates.append(dict(candidate=c,status='measured'))
        p={k:.5 for k in KEYS};p.update(parent='p',role=f'c{c:02d}')
        if c==4:p['electrostatics_rmse']=2
        if c==5:p['surface_distance_A']=2
        patches.append(p)
    np.savez(tmp_path/'candidate-ca.npz',**coordinates)
    result=dict(parent='p',dataset='atlas',size_bin=0,candidates=candidates,patches=patches,path=str(tmp_path),expected_anchors=1)
    out=staged_selection(result,{k:1 for k in KEYS},.2)
    assert set(out['selected'])=={1,2,3}
    assert 5 not in out['queried_candidates'] and 4 in out['queried_candidates']
    assert out['electrostatic_candidate_queries']==4
