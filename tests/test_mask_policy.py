from pathlib import Path
from jaxpropka.topology import load_topology
from pkabench.mask_policy import masks
from pkabench.coverage_audit import map_positions


def test_radius_boundaries_repaired_atoms_and_whole_gaps_only():
    atoms=load_topology(Path(__file__).parent/'data/peptide.pdb').atoms
    point=atoms.coord[atoms.atom_name=='N'][0]
    site={'complex_id':'x','chain':str(atoms.chain_id[0]),'resnum':int(atoms.res_id[0]),'icode':'','group':'NTERM',
        'functional_atoms_complete':True,'is_break_terminus':False,'residue_delta_sasa':0,'was_completed':True}
    evidence={'sequences':[{'sequence_source':'entity_poly_canonical'}],
        'defects':[{'kind':'missing_atoms','centres':[point.tolist()],'extent':8}]}
    assert masks([site],atoms,evidence)[0]['eval_mask']
    evidence['defects']=[{'kind':'terminal_gap','length':1,'centres':[(point+[13.8,0,0]).tolist()],'extent':0}]
    result=masks([site],atoms,evidence)[0]
    assert result['train_mask'] and not result['eval_mask']
    site['is_break_terminus']=True
    assert not masks([site],atoms,evidence)[0]['train_mask']


def test_construct_mapping_requires_identity_and_equal_spans():
    mapping={'start':{'residue_number':1},'end':{'residue_number':3},'unp_start':4,'unp_end':6}
    assert map_positions('ACD',mapping,'MMMACDKK')=={1:4,2:5,3:6}
    assert map_positions('ACE',mapping,'MMMACDKK') is None
    mapping['unp_end']=7
    assert map_positions('ACD',mapping,'MMMACDKK') is None
