"""Canonical, source-pinned SAbDab antibody-versus-antigen input assemblies.

Physical chains and gaps are retained. Author identifiers are projected through
polypeptide entities so a free amino acid sharing an antibody author ID cannot
become an extra protein residue. This module never invents coordinates.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np

from .runtime import config_hash


PREPARATION_POLICY = 'sabdab-native-ab-ag-v1'


def load_prepared(unit):
    """Verify immutable pilot inputs and recover their canonical residue order."""
    import gzip
    from pathlib import Path
    from biotite.structure.io import pdbx
    from jaxpropka.topology import load_topology
    from .runtime import digest
    entry = Path(unit['entry'])
    for name, field in [('AB.cif.gz', 'cif_sha256'), ('structure.json', 'structure_sha256')]:
        if digest(entry / name) != unit[field]:
            raise ValueError('Prepared SAbDab input changed: ' + str(entry / name))
    meta = json.loads((entry / 'structure.json').read_text())
    if unit['source'] != 'sabdab' or unit['unit_id'] != 'sabdab:' + meta['instance']:
        raise ValueError('Prepared SAbDab source identity differs from inventory')
    with gzip.open(entry / 'AB.cif.gz', 'rt') as stream:
        atoms = pdbx.get_structure(pdbx.CIFFile.read(stream), model=1, use_author_fields=True)
    topology = load_topology(atoms, gap_policy='cap', freeze_disulfides=True)
    keys = [(k.chain, k.number, k.insertion) for k in topology.keys]
    if keys != [(r['chain'], r['number'], r['insertion']) for r in meta['residue_map']]:
        raise ValueError('Prepared SAbDab residue order differs from inverse map')
    if topology.n_residues != unit['n'] or meta['native_source_split'] != unit['native_source_split'] or meta['native_component'] != unit['component']:
        raise ValueError('Prepared SAbDab identity/scope differs from inventory')
    if meta['operational_role'] != unit['split']:
        raise ValueError('Prepared SAbDab operational role differs from inventory')
    return topology, meta


def surface_inventory(path):
    """Native Ab, Ag and bound surfaces, with source identity and stable node IDs."""
    from pathlib import Path
    from .runtime import digest
    path = Path(path); data = json.loads(path.read_text()); units = []
    if data.get('format') != 'sabdab-native-annotation-pilot-v1':
        raise ValueError('Unknown SAbDab native inventory')
    seen = set()
    for unit in data['units']:
        if unit['source'] != 'sabdab' or unit['unit_id'] in seen:
            raise ValueError('Invalid or duplicate SAbDab native unit')
        seen.add(unit['unit_id'])
        topology, meta = load_prepared(unit)
        sides = [meta['side_by_chain'][k.chain] for k in topology.keys]
        for state, role in [('A', 'Ab'), ('B', 'Ag'), ('AB', 'bound')]:
            canonical = [i for i, side in enumerate(sides) if state == 'AB' or side == state]
            units.append(dict(id='sabdab/' + meta['instance'] + '/' + state, dataset='sabdab',
                cid=meta['instance'], state=state, role=role, split={'train':'train', 'validation':'val'}[unit['split']],
                group=unit['component'], family='sabdab_' + role, n=len(canonical),
                canonical_residue_ids=canonical, prepared_unit=unit, unit_policy=PREPARATION_POLICY,
                native_source_split=unit['native_source_split']))
    return units, dict(inventory_sha256=digest(path), source='SAbDab published antigen-aware native pilot',
        unit_policy=PREPARATION_POLICY, source_inventory=str(path))


def author_roles(row):
    antibody = {row.get('Hchain', ''), row.get('Lchain', '')} - {'', '+'}
    antigens = {chain for chain, typ in zip(row['agchains'].split('/'), row['agtypes'].split('/'))
                if typ in {'PROTEIN', 'PEPTIDE'}}
    if not antibody or not antigens or antibody & antigens:
        raise ValueError('Missing or overlapping antibody/antigen author roles')
    return {**{c: 'A' for c in antibody}, **{c: 'B' for c in antigens}}


def prepare(file, row):
    """Return canonical atoms, topology and provenance for the actual ML scope."""
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from jaxpropka.parameters import ALPHABET, THREE
    from jaxpropka.topology import load_topology
    from .conformers import resolve

    roles = author_roles(row)
    poly = file.block.get('entity_poly')
    if poly is None:
        raise ValueError('Protein entity annotation missing')
    protein_entities = {str(e) for e, t in zip(poly['entity_id'].as_array(str), poly['type'].as_array(str))
                        if str(t).strip().lower() == 'polypeptide(l)'}
    cat = file.block['atom_site']
    if 'label_entity_id' not in cat:
        raise ValueError('Atom-to-polymer-entity annotation missing')
    original_labels = cat['label_asym_id'].as_array(str)
    original_authors = cat['auth_asym_id'].as_array(str)
    original_entities = cat['label_entity_id'].as_array(str)
    selected = sorted({str(l) for l, a, e in zip(original_labels, original_authors, original_entities)
                       if str(a) in roles and str(e) in protein_entities})
    if not selected:
        raise ValueError('No selected protein label chains')
    file, conformers = resolve(file, selected)
    cat = file.block['atom_site']
    label = pdbx.get_structure(file, model=1, altloc='occupancy', use_author_fields=False, include_bonds=True)
    author = pdbx.get_structure(file, model=1, altloc='occupancy', use_author_fields=True, include_bonds=True)
    if len(label) != cat.row_count or len(label) != len(author) or not np.array_equal(label.coord, author.coord):
        raise ValueError('Ambiguous source atom/author mapping')
    entity = cat['label_entity_id'].as_array(str)
    chosen = np.isin(label.chain_id, selected) & np.isin(entity, sorted(protein_entities))
    unsupported = set(label.res_name[chosen]) - set(THREE)
    if unsupported:
        raise ValueError('Unsupported selected polymer chemistry: ' + repr(sorted(unsupported)))
    heavy = chosen & ~np.isin(np.char.upper(label.element), ['H', 'D'])
    atoms = label[heavy].copy()
    auth = author[heavy]
    source_positions = label.res_id[heavy].copy()
    # Canonical node keys are consecutive observed residues per PHYSICAL label
    # chain. Source author/domain aliases and numbering remain in the inverse
    # map; scFv aliases must not duplicate or merge observed residues.
    chain_meta = []; side_by_chain = {}; residue_map = []; aa = dict(zip(THREE, ALPHABET))
    for chain in dict.fromkeys(map(str, atoms.chain_id)):
        mask = atoms.chain_id == chain
        authors = set(map(str, auth.chain_id[mask]))
        if not authors <= roles.keys() or len({roles[a] for a in authors}) != 1:
            raise ValueError('Ambiguous physical-chain role')
        side = roles[next(iter(authors))]
        side_by_chain[chain] = side
        sub = atoms[mask]; source_auth = auth[mask]; positions = source_positions[mask]
        indices = np.flatnonzero(mask)
        starts = struc.get_residue_starts(source_auth, add_exclusive_stop=True)
        sequence = ''; previous = None; breaks = []; seen = set()
        for ordinal, (s, e) in enumerate(zip(starts[:-1], starts[1:]), 1):
            residue = sub[s:e]
            original_key = (str(source_auth.chain_id[s]), int(source_auth.res_id[s]), str(source_auth.ins_code[s]).strip())
            key = (chain, ordinal, '')
            if original_key in seen:
                raise ValueError('Repeated physical residue identity')
            seen.add(original_key)
            names = {str(n): i for i, n in enumerate(residue.atom_name)}
            if not {'N', 'CA', 'C', 'O'} <= names.keys():
                raise ValueError('Incomplete native backbone: ' + repr(key))
            if previous is not None and np.linalg.norm(previous - residue.coord[names['N']]) > 1.9:
                breaks.append(dict(before_number=key[1], before_insertion=key[2]))
            previous = residue.coord[names['C']]
            sequence += aa[str(residue.res_name[0])]
            atoms.res_id[indices[s:e]] = ordinal
            atoms.ins_code[indices[s:e]] = ''
            residue_map.append(dict(chain=chain, number=key[1], insertion=key[2],
                source_label_chain=chain, source_author_chain=original_key[0],
                source_author_number=original_key[1], source_author_insertion=original_key[2],
                source_label_position=int(positions[s]), side=side))
        chain_meta.append(dict(chain=chain, source_author_chain=next(iter(authors)) if len(authors)==1 else None,
            source_author_chains=sorted(authors), side=side,
            residues=len(sequence), observed_sequence=sequence, backbone_breaks=breaks))
    if set(side_by_chain.values()) != {'A', 'B'}:
        raise ValueError('Prepared input must contain both biological partners')
    topology = load_topology(atoms, gap_policy='cap', freeze_disulfides=True)
    side = np.asarray([0 if side_by_chain[k.chain] == 'A' else 1 for k in topology.keys], np.int8)
    atom_hash = hashlib.sha256()
    for field in ['chain_id', 'res_id', 'ins_code', 'res_name', 'atom_name', 'element']:
        atom_hash.update(json.dumps(getattr(atoms, field).tolist(), separators=(',', ':')).encode())
    atom_hash.update(np.asarray(atoms.coord, np.float32).tobytes())
    metadata = dict(policy=PREPARATION_POLICY, instance=row['INSTANCE'],
        native_source_split=row['ab_ag_split'], native_component=row['ab_ag_cluster'],
        operational_role={'train': 'train', 'test': 'validation'}[row['ab_ag_split']],
        antibody_type=row['type'], construct=row.get('construct', ''),
        side_by_chain=side_by_chain, chains=chain_meta, residue_map=residue_map,
        native_coordinate_sha256=atom_hash.hexdigest(), context_identity_established=False,
        teacher_compatibility_established=False, atoms_added=False,
        dropped_nonselected_or_nonprotein_atoms=int((~heavy).sum()),
        n_res=topology.n_residues, n_atoms=len(atoms), conformer_selection=conformers['selection'])
    if 'VH_imgt_keep_interval' in row or 'VL_imgt_keep_interval' in row:
        from .sabdab_scope import crop_record, retained_intervals
        metadata['antibody_crop_metadata'] = crop_record(row)
        _,metadata['antibody_scope'] = retained_intervals(metadata['antibody_crop_metadata'],residue_map,
            [aa[str(topology.residue(i).res_name[0])] for i in range(topology.n_residues)],
            side_by_chain,row['INSTANCE'])
    metadata['preparation_sha256'] = config_hash(metadata)
    return atoms, topology, side, metadata
