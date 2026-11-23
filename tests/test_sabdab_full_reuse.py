import numpy as np
import pytest

from sabdab_v7_full_reuse import compare_contexts


def atoms():
    from biotite.structure import AtomArray
    a=AtomArray(8)
    a.chain_id=np.array(['H']*4+['G']*4)
    a.res_id=np.array([1]*4+[20]*4)
    a.res_name=np.array(['ALA']*4+['GLY']*4)
    a.atom_name=np.array(['N','CA','C','O']*2)
    a.element=np.array(['N','C','C','O']*2)
    a.coord=np.array([[0,0,0],[1,0,0],[1,1,0],[1,1,1],
        [4,1,0],[5,2,0],[6,2,1],[7,2,2]],dtype=np.float32)
    return a


def test_rigid_transform_and_renumbering_preserve_context():
    left=atoms();right=left.copy()
    right.chain_id=np.where(right.chain_id=='H','R','L');right.res_id+=7
    right.coord=right.coord@np.array([[0,-1,0],[1,0,0],[0,0,1]])+[3,4,5]
    result=compare_contexts(left,right,{'H':'A','G':'B'},{'R':'B','L':'A'})
    assert result['status']=='rigid_heavy_atom_identity'
    assert result['biological_partition_match'] is True


@pytest.mark.parametrize('change',['deformation','reflection','atom'])
def test_non_equivalent_context_is_not_reusable(change):
    left=atoms();right=left.copy()
    if change=='deformation':right.coord[0,0]+=0.1
    if change=='reflection':right.coord[:,0]*=-1
    if change=='atom':right.atom_name[0]='NZ'
    assert compare_contexts(left,right,{}, {})['status']!='rigid_heavy_atom_identity'


def test_whole_partner_partition_must_match():
    left=atoms();right=left.copy()
    result=compare_contexts(left,right,{'H':'A','G':'B'},{'H':'A','G':'A'})
    assert result['status']=='same_geometry_different_biological_partition'
