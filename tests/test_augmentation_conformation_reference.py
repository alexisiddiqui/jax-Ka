"""Controls for correspondence, intact-surface alignment and parent weighting."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from augmentation_conformation_reference import correspondence,state,make_surface,measure,atom_mapping
from augmentation_conformation_reference_report import balanced


def test_sequence_correspondence_allows_missing_terminus_but_rejects_mutation():
    names='ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()
    mapping,quality=correspondence(names,names[1:])
    assert mapping=={i:i-1 for i in range(1,20)}
    assert quality['identity']==1. and quality['reference_coverage']==.95
    changed=names.copy();changed[10]='VAL'
    with pytest.raises(ValueError,match='identity/coverage'):correspondence(names,changed)
    with pytest.raises(ValueError,match='identity/coverage'):correspondence(names,names[5:])


def fixture():
    ca=np.array([[0,0,0],[3,0,0],[0,3,0],[0,0,3],[3,3,0]],float)
    offsets=np.array([[0,1,0],[0,0,0],[1,0,0],[1.5,0,0],[0,0,1]],float)
    xyz=(ca[:,None,:]+offsets).reshape(-1,3)
    owner=np.repeat(np.arange(5),5);atoms=['N','CA','C','O','CB']*5
    return state(xyz,atoms,owner,['ALA','ARG','ASN','ASP','CYS'],list(range(5)))


def test_experimental_pair_is_rigid_invariant_with_permuted_atom_inventory():
    a=fixture();rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
    order=np.arange(len(a['xyz']))[::-1]
    b=state((a['xyz']@rotation+[12,8,-3])[order],[a['atom_names'][i] for i in order],
            a['owner'][order],a['names'],a['keys'])
    sa,sb=make_surface(a,128),make_surface(b,128)
    defs=[dict(anchor=0,center=sa['points'][0],radius=12.,members=np.arange(5))]
    w,patches=measure(a,b,{i:i for i in range(5)},sa,sb,defs,'control','experimental',-1,0.,128)
    assert w['global_ca_rmsd_A']<1e-12 and w['global_surface_distance_A']<1e-11
    assert len(patches)==1
    assert patches[0]['local_surface_distance_A']<1e-11
    assert patches[0]['local_chemistry_mismatch_fraction']==0
    assert abs(patches[0]['atom_exposure_change'])<1e-12


def test_missing_atoms_are_identified_without_truncating_the_reference():
    a=fixture();keep=np.arange(len(a['xyz'])-1)
    b=state(a['xyz'][keep],[a['atom_names'][i] for i in keep],a['owner'][keep],a['names'],a['keys'])
    pairs,bad_a,bad_b=atom_mapping(a,b,{i:i for i in range(5)})
    assert bad_a==bad_b=={4} and len(pairs)==24 and len(a['xyz'])==25


def test_parent_and_conformer_weight_do_not_depend_on_number_of_patches():
    rows=[dict(parent='a',candidate=1,value=0.)]*100
    rows += [dict(parent='a',candidate=2,value=2.)]
    rows += [dict(parent='b',candidate=1,value=4.)]*2
    x,w=balanced(rows,'value')
    assert np.isclose(w.sum(),1.)
    assert np.isclose(w[x==0].sum(),.25)
    assert np.isclose(w[x==2].sum(),.25)
    assert np.isclose(w[x==4].sum(),.5)
