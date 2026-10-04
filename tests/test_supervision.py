import numpy as np
import pytest
from pkabench.supervision import clearance, annotate_masks
from pkabench.curation import validate_partners
from pkabench.prep import Rejection, export_pdb
from jaxpropka.topology import load_topology
from pathlib import Path

DATA=Path(__file__).parent/'data'


def test_two_partners_can_contain_multiple_chains():
    atoms=load_topology(DATA/'two_chains.pdb').atoms
    names=list(dict.fromkeys(map(str,atoms.chain_id)))
    extra=atoms[atoms.chain_id==names[0]].copy(); extra.chain_id[:]='Z'
    combined=atoms+extra
    context=validate_partners(combined,{'A':[names[0],'Z'],'B':[names[1]]})
    assert context['omitted_protein_chains']==[]
    assert validate_partners(combined,{'A':[names[0]],'B':[names[1]]})['omitted_protein_chains']==['Z']
    with pytest.raises(Rejection): validate_partners(combined,{'A':[names[0]],'B':[names[0]]})
    with pytest.raises(Rejection): validate_partners(combined,{'A':[names[0]],'B':[names[1]]},cap=1)


def test_gap_extent_is_not_just_flanking_points():
    defect={'kind':'internal_gap','centres':[[0,0,0],[12,0,0]],'length':5}
    assert clearance([[6,0,0]],defect)==0
    assert clearance([[100,0,0]],defect)>60
    assert clearance([[10,0,0]],{'kind':'terminal_gap','centres':[[0,0,0]],'length':5})==0
    assert clearance([[30,0,0]],{'kind':'terminal_gap','centres':[[0,0,0]],'length':5,'extent':8})==3


def test_exposed_tail_radius_does_not_relax_buried_or_unknown_defects():
    atoms=load_topology(DATA/'peptide.pdb').atoms
    key=(str(atoms.chain_id[0]),int(atoms.res_id[0]),str(atoms.ins_code[0]).strip())
    point=atoms.coord[atoms.atom_name=='N'][0]
    site={'chain':key[0],'resnum':key[1],'icode':key[2],'group':'NTERM','is_break_terminus':False,'functional_atoms_complete':True}
    defect={'kind':'terminal_gap','length':1,'centres':[(point+[13.8,0,0]).tolist()],'extent':0.,'smaller_tail_radius_candidate':True}
    evidence={'atoms':[],'artificial_terminal_keys':[],'defects':[defect]}
    result=annotate_masks([dict(site)],atoms,evidence)[0]
    assert not result['supervision_mask_15']
    assert result['supervision_mask_adaptive_15']
    defect['smaller_tail_radius_candidate']=False
    assert not annotate_masks([dict(site)],atoms,evidence)[0]['supervision_mask_adaptive_15']


def test_altloc_uncertainty_masks_observed_targets():
    atoms=load_topology(DATA/'peptide.pdb').atoms
    key=(str(atoms.chain_id[0]),int(atoms.res_id[0]),str(atoms.ins_code[0]).strip())
    site={'chain':key[0],'resnum':key[1],'icode':key[2],'group':'NTERM','is_break_terminus':False,'functional_atoms_complete':True}
    evidence={'atoms':[],'artificial_terminal_keys':[],
        'defects':[{'kind':'alternate_conformation','key':key,'centres':[atoms.coord[0].tolist()],'extent':0}]}
    result=annotate_masks([site],atoms,evidence)[0]
    assert result['coordinates_observed']
    assert result['alternate_conformation_uncertain']
    assert not result['supervision_mask']


def test_reconstructed_residue_has_no_supervised_target():
    atoms=load_topology(DATA/'peptide.pdb').atoms
    key=(str(atoms.chain_id[0]),int(atoms.res_id[0]),str(atoms.ins_code[0]).strip())
    site={'chain':key[0],'resnum':key[1],'icode':key[2],'group':'NTERM','is_break_terminus':False,'functional_atoms_complete':True}
    evidence={'atoms':[{'key':list(key),'missing_atoms':['CB']}], 'artificial_terminal_keys':[], 'defects':[]}
    result=annotate_masks([site],atoms,evidence)[0]
    assert result['coordinates_observed'] is False
    assert result['supervision_mask'] is False


def test_masked_targets_are_excluded_from_scores_and_linkage():
    from pkabench.score import paired
    from pkabench.linkage import linkage
    site={'complex_id':'x','chain':'A','resnum':1,'icode':'','group':'ASP','partner':'A',
        'is_break_terminus':False,'min_partner_distance':3.,'supervision_mask':False}
    rows=[{**site,'method':'test','state':s,'pka':v,'status':'ok','curve':[.5]*73} for s,v in [('AB',6.),('A',4.)]]
    assert paired(rows,[site],'test')=={}
    assert linkage(rows,[site])['status']=='masked_uncertain_charge_coverage'


def test_pdb_export_splits_physical_gap_without_losing_mapping(tmp_path):
    from biotite.structure import get_residue_starts
    from biotite.structure.io.pdb import PDBFile
    atoms=load_topology(DATA/'peptide.pdb').atoms.copy()
    starts=get_residue_starts(atoms)
    atoms.coord[starts[1]:,0]+=30
    mapping=export_pdb(atoms,tmp_path/'gap.pdb')
    exported=PDBFile.read(tmp_path/'gap.pdb').get_structure(model=1)
    assert len(set(exported.chain_id))>=2
    assert len(mapping)==len(starts)
    assert {k[0] for k in mapping.values()}==set(atoms.chain_id)
