from types import SimpleNamespace
import numpy as np
import biotite.structure as struc
from pkabench import audit
from pkabench.component_mask_policy import POLICY, CLASSES, classify, class_trees, nearest_by_class, clear_of_components


def test_radii_match_approved_policy():
    assert POLICY['train_radii_A'] == {'ligand': 15., 'buffer': 15., 'glycan': 20., 'exposed_metal': 25., 'bound_metal': 30.}
    assert POLICY['eval_radii_A'] == {'ligand': 25., 'buffer': 25., 'glycan': 25., 'exposed_metal': 25., 'bound_metal': 30.}
    assert set(POLICY['train_radii_A']) == set(CLASSES) == set(POLICY['eval_radii_A'])
    for radii in (POLICY['train_radii_A'], POLICY['eval_radii_A']):
        assert min(radii['exposed_metal'], radii['bound_metal']) >= 25.
        assert all(radii[c] >= POLICY['train_radii_A'][c] for c in CLASSES)


def test_boundary_is_inclusive_of_radius():
    radii = POLICY['train_radii_A']
    assert clear_of_components({'ligand': 15.0, 'bound_metal': 30.0}, radii)
    assert not clear_of_components({'ligand': 14.99}, radii)
    assert not clear_of_components({'bound_metal': 29.99}, radii)
    assert clear_of_components({}, radii)


def _atoms(rows):
    atoms = struc.AtomArray(len(rows))
    atoms.chain_id = np.array([r[0] for r in rows]); atoms.res_id = np.array([r[1] for r in rows])
    atoms.res_name = np.array([r[2] for r in rows]); atoms.atom_name = np.array([r[3] for r in rows])
    atoms.element = np.array([r[4] for r in rows]); atoms.coord = np.array([r[5] for r in rows], float)
    return atoms


def test_every_component_is_masked_never_rejected(monkeypatch):
    rows = [('A', 1, 'CYS', 'SG', 'S', (0., 0., 0.)), ('A', 1, 'CYS', 'CB', 'C', (1.5, 0., 0.)),
            ('B', 1, 'ZN', 'ZN', 'ZN', (0., 2.3, 0.)),                       # coordinated by Cys SG
            ('C', 1, 'GOL', 'C1', 'C', (20., 0., 0.)), ('C', 1, 'GOL', 'O1', 'O', (21., 0., 0.)),
            ('D', 1, 'LIG', 'C1', 'C', (0., 0., 1.5)),                       # <1.9 A protein contact
            ('E', 1, 'NAG', 'C1', 'C', (-20., 0., 0.)),
            ('F', 1, 'HEM', 'FE', 'FE', (0., -20., 0.)), ('F', 1, 'HEM', 'C1', 'C', (0., -21.5, 0.)),
            ('G', 1, 'NA', 'NA', 'NA', (40., 40., 40.))]                    # isolated ion
    atoms = _atoms(rows)
    comps = [dict(chain='B', resnum=1, name='ZN', code='metal', start=2, end=3),
             dict(chain='C', resnum=1, name='GOL', code='ligand', start=3, end=5),
             dict(chain='D', resnum=1, name='LIG', code='ligand', start=5, end=6),
             dict(chain='E', resnum=1, name='NAG', code='glycan', start=6, end=7),
             dict(chain='F', resnum=1, name='HEM', code='metal', start=7, end=9)]
    monkeypatch.setattr(audit, 'component_inventory', lambda *args: [dict(c) for c in comps])
    result, without_heavy = classify(atoms, SimpleNamespace(block={}), ['A'])
    classes = {c['name']: c['cls'] for c in result}
    assert without_heavy == 0
    assert classes == {'ZN': 'bound_metal', 'GOL': 'buffer', 'LIG': 'ligand', 'NAG': 'glycan', 'HEM': 'bound_metal', 'NA': 'exposed_metal'}
    assert [c['covalent'] for c in result if c['name'] == 'LIG'] == [True]
    trees = class_trees(result)
    distances = nearest_by_class(atoms.coord[:1], trees)
    assert distances['bound_metal'] < 30 and not clear_of_components(distances, POLICY['train_radii_A'])


def test_declared_connection_makes_buffer_a_ligand_and_ion_bound(monkeypatch):
    rows = [('A', 1, 'ALA', 'CA', 'C', (0., 0., 0.)), ('C', 1, 'GOL', 'C1', 'C', (10., 0., 0.)),
            ('G', 1, 'NA', 'NA', 'NA', (40., 40., 40.))]
    atoms = _atoms(rows)
    monkeypatch.setattr(audit, 'component_inventory', lambda *args: [dict(chain='C', resnum=1, name='GOL', code='ligand', start=1, end=2)])
    column = lambda values: SimpleNamespace(as_array=lambda dtype: np.array(values))
    conn = {'conn_type_id': column(['covale', 'metalc']), 'ptnr1_label_asym_id': column(['C', 'G']), 'ptnr1_label_comp_id': column(['GOL', 'NA']),
            'ptnr2_label_asym_id': column(['A', 'A']), 'ptnr2_label_comp_id': column(['ALA', 'ALA'])}
    result, _ = classify(atoms, SimpleNamespace(block={'struct_conn': conn}), ['A'])
    assert {c['name']: c['cls'] for c in result} == {'GOL': 'ligand', 'NA': 'bound_metal'}


def test_component_without_heavy_atoms_is_counted(monkeypatch):
    atoms = _atoms([('A', 1, 'ALA', 'CA', 'C', (0., 0., 0.)), ('B', 1, 'UNK', 'H1', 'H', (4., 0., 0.))])
    monkeypatch.setattr(audit, 'component_inventory', lambda *args: [dict(chain='B', resnum=1, name='UNK', code='ligand', start=1, end=2)])
    result, without_heavy = classify(atoms, SimpleNamespace(block={}), ['A'])
    assert result == [] and without_heavy == 1
