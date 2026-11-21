import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from masif_release_plan import route_parent


def test_missing_antigen_withholds_whole_assembly():
    row=dict(dataset='pinder',id='complex',split='val',group='cluster',n_res='500')
    req=[dict(id='complex/Ab',role='Ab'),dict(id='complex/Ag',role='Ag')]
    routed=route_parent(row,req,{'complex/Ab':object()})
    assert routed['status']=='held' and routed['missing_units']==['complex/Ag']
    assert routed['split']=='val' and routed['group']=='cluster'


def test_complete_intact_antibody_does_not_require_obsolete_chain_surfaces():
    row=dict(dataset='pkpdb',id='fab',split='train',group='ig',n_res='440')
    req=[dict(id='fab/intact-Ab',chains=[0,1],role='Ab')]
    assert route_parent(row,req,{'fab/intact-Ab':object()})['status']=='covered'


def test_empty_protein_inventory_and_forced_hold_cannot_be_training_ready():
    row=dict(dataset='pkpdb',id='empty',split='train',group='g',n_res='0')
    assert route_parent(row,[],{})['status']=='held'
    assert route_parent(row,[dict(id='surface')],{'surface':object()},['source_review'])['status']=='held'
