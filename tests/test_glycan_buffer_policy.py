import numpy as np
import pytest
import biotite.structure as struc
from biotite.structure.io import pdbx
from biotite.structure.info import residue
from pkabench.glycan_buffer_policy import glycan_inventory, masks, buffer_screen
from pkabench.prep import Rejection


def test_masks_use_all_functional_atoms_and_all_removed_species():
    atoms = struc.AtomArray(2)
    atoms.chain_id[:] = 'A'; atoms.res_id[:] = 1; atoms.ins_code[:] = ''
    atoms.res_name[:] = 'ASP'; atoms.atom_name[:] = ['OD1', 'OD2']; atoms.element[:] = 'O'
    atoms.coord[:] = [[20, 0, 0], [21, 0, 0]]
    site = dict(complex_id='test', chain='A', resnum=1, icode='', group='ASP', functional_atoms_complete=True, is_break_terminus=False)
    glycan = dict(coordinates=[[0, 0, 0]], train_radius_A=20., eval_radius_A=25.)
    result = masks([site], atoms, [glycan])[0]
    assert result['component_train_mask'] and not result['component_eval_mask']
    ligand = dict(coordinates=[[7, 0, 0]], train_radius_A=15., eval_radius_A=25.)
    assert not masks([site], atoms, [glycan, ligand])[0]['component_train_mask']
    site['is_break_terminus'] = True
    assert not masks([site], atoms, [])[0]['component_train_mask']


def linked_nag():
    sugar = residue('NAG')
    c1 = sugar.coord[sugar.atom_name == 'C1'][0]
    o1 = sugar.coord[sugar.atom_name == 'O1'][0]
    direction = (o1-c1)/np.linalg.norm(o1-c1)
    sugar = sugar[(sugar.element != 'H') & (sugar.atom_name != 'O1')]
    sugar.chain_id[:] = 'G'; sugar.res_id[:] = 1; sugar.ins_code[:] = ''
    attachment = struc.AtomArray(1)
    attachment.chain_id[:] = 'A'; attachment.res_id[:] = 10; attachment.ins_code[:] = ''
    attachment.res_name[:] = 'ASN'; attachment.atom_name[:] = 'ND2'; attachment.element[:] = 'N'
    attachment.coord[:] = c1+1.45*direction
    # Drop CCD bonds here: production source checks geometry against the CCD.
    sugar.bonds = None
    atoms = sugar+attachment
    cif = pdbx.CIFFile(); cif['test'] = pdbx.CIFBlock()
    components = [dict(chain='G', resnum=1, name='NAG', code='glycan', start=0, end=len(sugar))]
    return atoms, cif, components


def test_glycan_attachment_and_nonleaving_atom_completeness():
    atoms, cif, components = linked_nag()
    result = glycan_inventory(atoms, cif, components, ['A'])
    assert len(result) == 1 and result[0]['attachment']['protein_atom'] == 'ND2'
    assert len(result[0]['coordinates']) == components[0]['end']
    wrong = atoms.copy(); wrong.atom_name[-1] = 'CA'
    with pytest.raises(Rejection, match='attachment chemistry'):
        glycan_inventory(wrong, cif, components, ['A'])
    incomplete = atoms[atoms.atom_name != 'C6']
    components[0]['end'] -= 1
    with pytest.raises(Rejection, match='missing non-leaving'):
        glycan_inventory(incomplete, cif, components, ['A'])


def test_buffer_bridge_is_not_classified_as_exposed_nonbridging():
    atoms = struc.AtomArray(3)
    atoms.chain_id[:] = ['A', 'B', 'L']; atoms.res_id[:] = [1, 2, 3]
    atoms.res_name[:] = ['ALA', 'ALA', 'GOL']; atoms.atom_name[:] = ['CA', 'CA', 'C1']; atoms.element[:] = 'C'
    atoms.coord[:] = [[10, 0, 0], [-10, 0, 0], [0, 0, 0]]
    component = dict(start=2, end=3, name='GOL')
    assert buffer_screen(atoms, component, {'A': ['A'], 'B': ['B']})['exposed_fraction'] >= .99
    atoms.coord[:2] = [[4, 0, 0], [-4, 0, 0]]
    result = buffer_screen(atoms, component, {'A': ['A'], 'B': ['B']})
    assert result['bridges_partners'] and not result['exposed_nonbridging_annotation']
