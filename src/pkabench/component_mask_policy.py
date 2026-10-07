"""Mask-all component policy for pretraining/augmentation datasets (pKPDB, extracted PINDER pairs).

Approved 2026-10-06. Every non-protein component with heavy-atom coordinates is stripped and masks nearby
supervision; component chemistry is never an entry rejection. The frozen antibody benchmark keeps its
`buffer-15-20-v3` policy (glycan_buffer_policy.py); this module does not change it.
"""
import numpy as np
from scipy.spatial import cKDTree

from .prep import CANONICAL

CLASSES = ('ligand', 'buffer', 'glycan', 'exposed_metal', 'bound_metal')
POLICY = {
    'version': 'mask-all-v1',
    'train_radii_A': {'ligand': 15., 'buffer': 15., 'glycan': 20., 'exposed_metal': 25., 'bound_metal': 30.},
    'eval_radii_A': {'ligand': 25., 'buffer': 25., 'glycan': 25., 'exposed_metal': 25., 'bound_metal': 30.},
    'ligand_rule': 'Noncovalent and covalent/declared-connected ligands are stripped with ligand radii; no rejection.',
    'buffer_rule': 'Known buffer/additive names without a declared connection or <1.9 A protein contact; 15/25 A.',
    'glycan_rule': 'Any glycan residue, whatever its chemistry or attachment; 20/25 A (shielding).',
    'metal_rule': 'Exposed ions keep the existing eligibility test (monatomic, >=70% exposed, no N/O/S/Se donor '
                  'within 3 A, no declared connection) and use 25/25 A. Every other metal-containing component '
                  '(coordinated, buried, metal complexes such as heme) uses 30/30 A. Metals never use 15 A.',
    'unchanged': 'Noncanonical peptide residues and missing coordinates keep their existing gap/defect fallbacks.',
}
BUFFERS = {'GOL', 'EDO', 'PEG', 'PGE', 'PG4', '1PE', 'MPD', 'DMS', 'SO4', 'PO4', 'ACT', 'FMT', 'TRS', 'MES', 'HEP', 'BME'}
WATER = ['HOH', 'WAT', 'H2O', 'DOD']


def classify(atoms, cif, selected):
    """Class and heavy-atom coordinates of every component; returns (components, components_without_heavy_atoms)."""
    import biotite.structure as struc
    from .audit import component_inventory
    components = component_inventory(atoms, cif, selected)
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for s, e in zip(starts[:-1], starts[1:]):
        if e-s == 1 and str(atoms.res_name[s]) in ('NA', 'K', 'CL'):
            components.append({'chain': str(atoms.chain_id[s]), 'resnum': int(atoms.res_id[s]), 'name': str(atoms.res_name[s]),
                               'code': 'metal', 'start': int(s), 'end': int(e)})
    declared = set(); conn = cif.block.get('struct_conn')
    if conn is not None:
        for i, kind in enumerate(conn['conn_type_id'].as_array(str)):
            if str(kind).lower().startswith(('covale', 'metalc')):
                for p in ('ptnr1', 'ptnr2'):
                    declared.add((str(conn[f'{p}_label_asym_id'].as_array(str)[i]), str(conn[f'{p}_label_comp_id'].as_array(str)[i])))
    heavy = ~np.isin(np.char.upper(atoms.element), ['H', 'D']); water = np.isin(atoms.res_name, WATER)
    protein = heavy & np.isin(atoms.res_name, list(CANONICAL))
    ptree = cKDTree(atoms.coord[protein]) if protein.any() else None
    result = []; without_heavy = 0
    for c in components:
        a = atoms[c['start']:c['end']]; a = a[~np.isin(np.char.upper(a.element), ['H', 'D'])]
        if len(a) == 0:
            without_heavy += 1; continue
        contact = float(ptree.query(a.coord)[0].min()) if ptree is not None else float('inf')
        connected = (c['chain'], c['name']) in declared
        if c['code'] == 'metal':
            cls = 'exposed_metal' if _exposed_ion(atoms, c, a, heavy, water, connected) else 'bound_metal'
        elif c['code'] == 'glycan':
            cls = 'glycan'
        elif c['code'] == 'ligand':
            cls = 'buffer' if c['name'] in BUFFERS and not connected and contact >= 1.9 else 'ligand'
        else:
            cls = c['code']
        result.append(dict(c, cls=cls, covalent=bool(connected or contact < 1.9), coordinates=a.coord.astype(float)))
    return result, without_heavy


def _exposed_ion(atoms, component, a, heavy, water, connected):
    import biotite.structure as struc
    if len(a) != 1 or connected:
        return False
    donor = heavy & ~water & np.isin(np.char.upper(atoms.element), ['N', 'O', 'S', 'SE'])
    donor[component['start']:component['end']] = False
    if donor.any() and float(cKDTree(atoms.coord[donor]).query(a.coord)[0].min()) <= 3:
        return False
    env = heavy & ~water; ids = np.flatnonzero(env)
    positions = np.flatnonzero((ids >= component['start']) & (ids < component['end']))
    try:
        isolated = float(struc.sasa(a, ignore_ions=False, vdw_radii='Single').sum())
        exposed = float(struc.sasa(atoms[env], ignore_ions=False, vdw_radii='Single')[positions].sum())/isolated
    except (KeyError, ValueError, ZeroDivisionError):
        return False
    return bool(np.isfinite(exposed) and exposed >= .7)


def class_trees(components):
    return {cls: cKDTree(np.concatenate([c['coordinates'] for c in components if c['cls'] == cls]))
            for cls in {c['cls'] for c in components}}


def nearest_by_class(points, trees):
    return {cls: (float(tree.query(points)[0].min()) if len(points) else 0.) for cls, tree in trees.items()}


def clear_of_components(distances, radii):
    """True when the site is at least the class radius from every component of every class."""
    return all(d >= radii[cls] for cls, d in distances.items())
