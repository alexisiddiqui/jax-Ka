import numpy as np
import pytest
from biotite.structure.io import pdbx
from pkabench.radial import radial_bin, statistics, deletion_mask


def test_radial_bins_use_known_deleted_coordinates_without_exclusion():
    assert [radial_bin(d) for d in (0,4.99,5,10,15,20,30,100)]==['0-5','0-5','5-10','10-15','15-20','20-30','30+','30+']
    with pytest.raises(ValueError): radial_bin(-1)


def test_whole_residue_deletion_preserves_chain_and_insertion_identity():
    cif=pdbx.CIFFile(); cif['x']=pdbx.CIFBlock(); cat=pdbx.CIFCategory()
    for name,values in {'label_asym_id':['A','A','A','B'],'auth_seq_id':['3']*4,
        'pdbx_PDB_ins_code':['?','?','X','?']}.items(): cat[name]=pdbx.CIFColumn(values)
    cif.block['atom_site']=cat
    assert deletion_mask(cif,[('A',3,'')]).tolist()==[False,False,True,True]


def test_error_summary_retains_teacher_coverage_failures():
    rows=[{'complex_id':'a','status':'ok','delta_pka_error':.2,'ab_pka_error':.5,'free_pka_error':.3},
        {'complex_id':'b','status':'coverage_failure'}]
    result=statistics(rows)
    assert result['eligible_sites']==2 and result['paired_sites']==1
    assert result['complexes']==1 and result['mae']==pytest.approx(.2)
    assert result['ab_mae']==pytest.approx(.5) and result['free_mae']==pytest.approx(.3)


def test_complex_balanced_mae_does_not_overweight_more_sites():
    small={'complex_id':'a','status':'ok','delta_pka_error':1.,'ab_pka_error':1.,'free_pka_error':0.}
    large={'complex_id':'b','status':'ok','delta_pka_error':0.,'ab_pka_error':0.,'free_pka_error':0.}
    result=statistics([small]+[large]*9)
    assert result['mae']==pytest.approx(.1)
    assert result['mean_complex_mae']==pytest.approx(.5)


def test_inclusive_prep_repairs_before_missing_backbone_rejection(monkeypatch,tmp_path):
    from pathlib import Path
    from jaxpropka.topology import load_topology
    import pkabench.prep as prep
    atoms=load_topology(Path(__file__).parent/'data/two_chains.pdb').atoms
    source=atoms.copy(); source=source[np.arange(len(source))!=np.flatnonzero(source.atom_name=='N')[0]]
    cif=pdbx.CIFFile(); pdbx.set_structure(cif,source)
    calls=[]
    def repair(observed,executable,out):
        calls.append(len(observed)); return atoms.copy()
    monkeypatch.setattr(prep,'complete',repair)
    original=prep.annotate
    def annotation(t,p):
        rows,delta,summary=original(t,p)
        summary.update(half_sum_buried_area=600,interface_residues=12)
        return rows,delta,summary
    monkeypatch.setattr(prep,'annotate',annotation)
    chains=list(dict.fromkeys(map(str,atoms.chain_id)))
    row={'complex_id':'x','pdb_id':'test','source_sha256':'test','partner_A_chains':[chains[0]],'partner_B_chains':[chains[1]]}
    prep.prepare_pair(cif,row,tmp_path,'repair',audit_allow_subcomplex=True,pdb2pqr_acceptance=True)
    assert calls==[len(atoms)-1]


def test_diagnostic_perturbation_does_not_censor_reduced_interface(monkeypatch,tmp_path):
    from pathlib import Path
    from jaxpropka.topology import load_topology
    import pkabench.prep as prep
    atoms=load_topology(Path(__file__).parent/'data/two_chains.pdb').atoms
    cif=pdbx.CIFFile(); pdbx.set_structure(cif,atoms)
    monkeypatch.setattr(prep,'complete',lambda observed,executable,out:observed)
    original=prep.annotate
    def low_area(t,p):
        rows,delta,summary=original(t,p); summary.update(half_sum_buried_area=1,interface_residues=1)
        return rows,delta,summary
    monkeypatch.setattr(prep,'annotate',low_area)
    chains=list(dict.fromkeys(map(str,atoms.chain_id)))
    row={'complex_id':'x','pdb_id':'test','source_sha256':'test','partner_A_chains':[chains[0]],'partner_B_chains':[chains[1]]}
    with pytest.raises(prep.Rejection,match='buried_area'):
        prep.prepare_pair(cif,row,tmp_path,'repair',audit_allow_subcomplex=True,pdb2pqr_acceptance=True)
    prep.prepare_pair(cif,row,tmp_path,'repair',audit_allow_subcomplex=True,pdb2pqr_acceptance=True,enforce_geometry=False)
