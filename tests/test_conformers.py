import numpy as np
import pytest
from biotite.structure.io import pdbx
from pkabench.conformers import resolve
from pkabench.prep import Rejection


def source():
    # A and B have opposite per-atom preferences: an atomwise maximum would
    # create a hybrid. A occurs first and must win.
    columns={'label_asym_id':['A']*6,'label_seq_id':['1']*6,'auth_seq_id':['10']*6,
        'pdbx_PDB_ins_code':['?']*6,'label_comp_id':['ASP']*6,
        'label_atom_id':['CA','OD1','OD2','OD1','OD2','O'],
        'label_alt_id':['.','A','A','B','B','.'], 'occupancy':['1','.8','.2','.2','.8','0'],
        'type_symbol':['C','O','O','O','O','O'],
        'Cartn_x':['0','1','2','11','12','100'],'Cartn_y':['0']*6,'Cartn_z':['0']*6}
    cif=pdbx.CIFFile(); block=pdbx.CIFBlock(); cif['test']=block
    cat=pdbx.CIFCategory()
    for name,values in columns.items(): cat[name]=pdbx.CIFColumn(values)
    block['atom_site']=cat
    return cif


def test_coherent_choice_preserves_alternatives_and_excludes_zero_occupancy():
    resolved,evidence=resolve(source(),['A'])
    cat=resolved.block['atom_site']
    assert cat['Cartn_x'].as_array(float).tolist()==[0.,1.,2.]
    assert evidence['zero_occupancy_rows_removed']==1
    record=evidence['records'][0]
    assert record['selected_alt_id']=='A'
    assert len(record['variants'])==2
    assert record['variants'][1]['coordinates'][-1]==[12.,0.,0.]
    assert evidence['defects']==[]
    assert record['key']==['A',10,'']


def test_altloc_selection_uses_first_deposited_id():
    cif=source(); cat=cif.block['atom_site']; reverse=pdbx.CIFCategory()
    for name in cat: reverse[name]=pdbx.CIFColumn(cat[name].as_array(str)[::-1])
    cif.block['atom_site']=reverse
    resolved,evidence=resolve(cif,['A'])
    assert evidence['records'][0]['selected_alt_id']=='B'
    assert set(resolved.block['atom_site']['Cartn_x'].as_array(float))=={0.,11.,12.}


def test_first_conformer_is_not_replaced_by_higher_occupancy():
    cif=source(); cif.block['atom_site']['occupancy']=pdbx.CIFColumn(['1','.1','.1','.9','.9','0'])
    resolved,evidence=resolve(cif,['A'])
    assert evidence['records'][0]['selected_alt_id']=='A'
    assert set(resolved.block['atom_site']['Cartn_x'].as_array(float))=={0.,1.,2.}


def test_multiple_models_and_missing_occupancy_are_explicit():
    cif=source(); cif.block['atom_site']['pdbx_PDB_model_num']=pdbx.CIFColumn(['1','1','1','2','2','2'])
    with pytest.raises(Rejection,match='per-model'): resolve(cif,['A'])
    cif=source(); cif.block['atom_site']['occupancy']=pdbx.CIFColumn(['?']*6)
    with pytest.raises(Rejection,match='occupancy'): resolve(cif,['A'])


def test_duplicate_shared_and_alternate_atom_is_not_silently_combined():
    cif=source(); cif.block['atom_site']['label_atom_id']=pdbx.CIFColumn(['OD1','OD1','OD2','OD1','OD2','O'])
    with pytest.raises(Rejection,match='overlap'): resolve(cif,['A'])
