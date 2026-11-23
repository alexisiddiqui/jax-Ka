import numpy as np
import pytest
import biotite.structure as struc
from biotite.structure.io import pdbx

from pkabench.sabdab_units import author_roles, prepare


def row(**extra):
    return dict(INSTANCE='test',Hchain='Q',Lchain='R',agchains='S/T',agtypes='PROTEIN/PEPTIDE',
        ab_ag_split='train',ab_ag_cluster='component',type='FV',construct='',**extra)


def cif():
    atoms=struc.AtomArray(36)
    labels=[];numbers=[];names=[];coords=[]
    for c,chain in enumerate(['A','B','C','D','Z']):
        for resid in range(1,3 if chain!='Z' else 2):
            for j,name in enumerate(['N','CA','C','O']):
                labels.append(chain);numbers.append(resid);names.append(name)
                coords.append([4.1*(resid-1)+[0,1.4,2.8,3.3][j],10*c+[0,.3,1.,1.8][j],0.])
    atoms.chain_id[:]=labels;atoms.res_id[:]=numbers;atoms.res_name[:]='ALA'
    atoms.atom_name[:]=names;atoms.element[:]=[n[0] for n in names];atoms.coord[:]=coords
    file=pdbx.CIFFile();pdbx.set_structure(file,atoms)
    cat=file.block['atom_site'];author={'A':'Q','B':'R','C':'S','D':'T','Z':'Q'}
    cat['occupancy']=pdbx.CIFColumn(np.ones(len(labels)))
    cat['auth_asym_id']=pdbx.CIFColumn(np.asarray([author[c] for c in labels]))
    cat['label_entity_id']=pdbx.CIFColumn(np.asarray([str('ABCDZ'.index(c)+1) for c in labels]))
    cat['label_seq_id']=pdbx.CIFColumn(np.asarray(numbers))
    file.block['entity_poly']=pdbx.CIFCategory(dict(entity_id=['1','2','3','4'],
        type=['polypeptide(L)']*4,pdbx_seq_one_letter_code_can=['AA']*4))
    return file


def test_whole_antibody_and_multichain_antigen_partition_excludes_free_amino_acid():
    atoms,topology,side,meta=prepare(cif(),row())
    assert topology.n_residues==8
    assert {c for c,s in meta['side_by_chain'].items() if s=='A'}=={'A','B'}
    assert {c for c,s in meta['side_by_chain'].items() if s=='B'}=={'C','D'}
    assert 'Z' not in set(atoms.chain_id)
    assert list(side).count(0)==4 and list(side).count(1)==4
    assert meta['atoms_added'] is False


def test_shared_heavy_light_author_does_not_duplicate_a_physical_chain():
    r=row();r['Lchain']='Q';r['construct']='SCFV'
    _,topology,_,meta=prepare(cif(),r)
    assert topology.n_residues==6
    assert [c for c,s in meta['side_by_chain'].items() if s=='A']==['A']


def test_overlapping_biological_roles_are_rejected():
    r=row();r['agchains']='Q'
    with pytest.raises(ValueError,match='overlapping'):
        author_roles(r)


def test_scfv_domain_aliases_on_one_physical_chain_retain_unique_nodes():
    file=cif();cat=file.block['atom_site']
    labels=cat['label_asym_id'].as_array(str)
    positions=cat['label_seq_id'].as_array(int)
    positions[labels=='B']+=2
    cat['label_seq_id']=pdbx.CIFColumn(positions)
    labels[labels=='B']='A'
    cat['label_asym_id']=pdbx.CIFColumn(labels)
    r=row();r['construct']='SCFV'
    atoms,topology,side,meta=prepare(file,r)
    assert topology.n_residues==8
    assert len({(k.chain,k.number,k.insertion) for k in topology.keys})==8
    assert meta['side_by_chain']=={'A':'A','C':'B','D':'B'}
    assert meta['chains'][0]['source_author_chains']==['Q','R']
    assert sorted({r['source_author_chain'] for r in meta['residue_map'] if r['chain']=='A'})==['Q','R']
