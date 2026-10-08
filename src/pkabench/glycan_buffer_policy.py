"""Versioned glycan/buffer stripping and immutable-source retention recount."""
import copy
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from .runtime import require_compute, atomic_json, digest

BUFFERS = {'GOL', 'EDO', 'PEG', 'PGE', 'PG4', '1PE', 'MPD', 'DMS', 'SO4', 'PO4', 'ACT', 'FMT', 'TRS', 'MES', 'HEP', 'BME'}
SUGARS = {'NAG', 'NDG', 'MAN', 'BMA', 'FUC', 'FUL', 'GAL', 'GLC', 'BGC', 'A2G', 'NGA', 'XYS', 'XYP'}
POLICY = {
    'version': 'glycan-buffer-v2', 'glycan_train_A': 20., 'glycan_eval_A': 25.,
    'buffer_train_A': 20., 'buffer_eval_A': 25., 'other_ligand_train_A': 15., 'other_ligand_eval_A': 25.,
    'metal_radius_A': 25., 'buffer_exposed_annotation_fraction': .7, 'buffer_bridge_distance_A': 4.,
    'buffer_candidates': sorted(BUFFERS), 'neutral_sugars': sorted(SUGARS),
    'glycan_rule': 'Whole resolved neutral glycan trees, one canonical ASN/SER/THR N/O attachment, CCD-complete non-leaving atoms and checked intra-residue bonds. All removed heavy atoms contribute to masks.',
    'buffer_rule': 'Known buffer/additive candidates retain existing ligand chemical eligibility: no covalent/metal connection or <1.9 A protein contact. Apply 20/25 A to all eligible instances. Exposure >=70% and nonbridging (>4 A from at least one partner) identify a low-concern annotation only, not a new rejection gate.',
    'metal_rule': 'Unchanged: monatomic, >=70% exposed, no declared connection or nonwater N/O/S/Se donor within 3 A; 25 A both masks.',
    'scope': 'Stripped protein-only labels; operational uncertainty masks. Existing natural-gap tiers and structural gates remain. No teacher run, split reassignment or physical-error guarantee.'}


def implementation():
    root = Path(__file__).parent
    return {name: digest(root/name) for name in ('glycan_buffer_policy.py', 'stripped_policy.py', 'curation.py', 'prep.py',
                                                'conformers.py', 'anchor_tiers.py', 'supervision.py', 'annotate.py')}


def initialise(source, out, smoke=False):
    require_compute()
    source = Path(source).resolve(); out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    old = json.loads((source/'manifest.json').read_text())
    candidates = old['candidates']
    if smoke:
        buckets = defaultdict(list)
        for row in candidates:
            r = json.loads((source/'rows'/f"{row['complex_id']}.json").read_text())
            if r['status'] == 'accepted':
                cc = json.loads((source/'structures'/row['complex_id']/'removed_components.json').read_text())['components']
                kind = 'buffer' if any(c['name'] in BUFFERS for c in cc) else 'accepted_other'
            else:
                kind = 'glycan' if r.get('code') == 'glycan' else 'rejected_other'
            if len(buckets[kind]) < (12 if kind in ('glycan', 'buffer') else 4):
                buckets[kind].append(row)
            if all(len(buckets[k]) >= n for k, n in [('glycan', 12), ('buffer', 12), ('accepted_other', 4), ('rejected_other', 4)]):
                break
        candidates = [r for k in ('glycan', 'buffer', 'accepted_other', 'rejected_other') for r in buckets[k]]
    manifest = {'candidates': candidates, 'source_campaign': str(source), 'source_audit': old['source_audit'],
                'source_manifest_sha256': digest(source/'manifest.json'), 'source_masks_sha256': digest(source/'site_masks.parquet'),
                'source_report_sha256': digest(source/'report.json'), 'policy': POLICY, 'implementation': implementation(),
                'smoke': smoke, 'production_allowed': False}
    atomic_json(out/'manifest.json', manifest)
    print(json.dumps({'candidates': len(candidates), 'source': str(source), 'smoke': smoke}), flush=True)


def radii(kind):
    return {'glycan': (20., 25.), 'buffer': (20., 25.), 'ligand': (15., 25.), 'metal': (25., 25.)}[kind]


def masks(sites, atoms, removals):
    import numpy as np
    import biotite.structure as struc
    from scipy.spatial import cKDTree
    from .annotate import SITE_ATOMS
    from .schema import KEY
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    residues = {(str(atoms.chain_id[s]), int(atoms.res_id[s]), str(atoms.ins_code[s]).strip()): atoms[s:e]
                for s, e in zip(starts[:-1], starts[1:])}
    trees = [(cKDTree(c['coordinates']), c) for c in removals]
    result = []
    for site in sites:
        a = residues[site['chain'], site['resnum'], site['icode']]
        points = a.coord[np.isin(a.atom_name, SITE_ATOMS[site['group']])]
        distances = [float(tree.query(points)[0].min()) if len(points) else 0. for tree, _ in trees]
        eligible = bool(site['functional_atoms_complete'] and not site['is_break_terminus'])
        result.append({k: site[k] for k in KEY} | {
            'component_train_mask': bool(eligible and all(d >= c['train_radius_A'] for d, (_, c) in zip(distances, trees))),
            'component_eval_mask': bool(eligible and all(d >= c['eval_radius_A'] for d, (_, c) in zip(distances, trees))),
            'nearest_removed_A': min(distances, default=None)})
    return result


def glycan_inventory(atoms, cif, components, selected):
    """Check sugar chemistry and complete trees in the full source assembly."""
    import numpy as np
    from scipy.spatial import cKDTree
    from biotite.structure.info import get_from_ccd, bonds_in_residue
    from .prep import CANONICAL, Rejection
    sugars = [c for c in components if c['code'] == 'glycan']
    if not sugars:
        return []
    heavy = ~np.isin(np.char.upper(atoms.element), ['H', 'D'])
    indexes = []; residue_for_atom = {}; data = []
    for i, c in enumerate(sugars):
        name = c['name']
        if name not in SUGARS:
            raise Rejection('unsupported_glycan', f'Neutral glycan whitelist excludes {name}')
        ids = np.arange(c['start'], c['end']); ids = ids[heavy[ids]]; a = atoms[ids]
        cat = get_from_ccd('chem_comp_atom', name)
        names = cat['atom_id'].as_array(str); elements = dict(zip(names, cat['type_symbol'].as_array(str)))
        charges = dict(zip(names, cat['charge'].as_array(int)))
        leaving = dict(zip(names, cat['pdbx_leaving_atom_flag'].as_array(str)))
        observed = {str(n): int(j) for n, j in zip(a.atom_name, ids)}
        required = {n for n in names if elements[n] != 'H' and leaving[n] != 'Y'}
        if len(observed) != len(ids) or not required <= observed.keys():
            raise Rejection('incomplete_glycan', f'{name} missing non-leaving CCD atoms {sorted(required-observed.keys())}')
        if any(elements.get(str(atom.atom_name)) != str(atom.element).upper() or charges.get(str(atom.atom_name)) != 0 for atom in a):
            raise Rejection('unsupported_glycan', f'{name} CCD element/charge mismatch')
        bonds = bonds_in_residue(name)
        expected = {tuple(sorted((observed[x], observed[y]))) for x, y in bonds if x in observed and y in observed}
        actual = {tuple(sorted((int(ids[x]), int(ids[y])))) for x, y in cKDTree(a.coord).query_pairs(2.)}
        if actual != expected:
            raise Rejection('glycan_bond_geometry', f'{name} observed intra-sugar bond graph differs from CCD')
        for j in ids:
            residue_for_atom[int(j)] = i
        indexes.extend(map(int, ids)); data.append((c, ids))
    parents = list(range(len(sugars)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    indexes = np.asarray(indexes)
    for x, y in cKDTree(atoms.coord[indexes]).query_pairs(1.9):
        ia, ib = int(indexes[x]), int(indexes[y]); ra, rb = residue_for_atom[ia], residue_for_atom[ib]
        if ra == rb:
            continue
        pair = [(str(atoms.atom_name[j]), str(atoms.element[j]).upper()) for j in (ia, ib)]
        if not (('C1', 'C') in pair and any(element == 'O' for _, element in pair)):
            raise Rejection('glycan_linkage', 'Unsupported inter-sugar link')
        parents[find(ra)] = find(rb)
    groups = defaultdict(list)
    for i in range(len(sugars)):
        groups[find(i)].append(i)
    pidx = np.flatnonzero(heavy & np.isin(atoms.res_name, list(CANONICAL)))
    ptree = cKDTree(atoms.coord[pidx]); result = []
    # Declared nonprotein/metal links cannot bypass the geometric attachment gate.
    conn = cif.block.get('struct_conn'); glycan_keys = {(c['chain'], c['name']) for c in sugars}
    if conn is not None:
        for i, kind in enumerate(conn['conn_type_id'].as_array(str)):
            if not str(kind).lower().startswith(('covale', 'metalc')):
                continue
            ends = [(str(conn[f'{p}_label_asym_id'].as_array(str)[i]), str(conn[f'{p}_label_comp_id'].as_array(str)[i])) for p in ('ptnr1', 'ptnr2')]
            if any(end in glycan_keys for end in ends) and (str(kind).lower().startswith('metalc') or any(end not in glycan_keys and end[1] not in CANONICAL for end in ends)):
                raise Rejection('glycan_linkage', 'Glycan has unsupported declared nonprotein/metal connection')
    for members in groups.values():
        ids = np.concatenate([data[i][1] for i in members]); attachments = []
        for gi, neighbours in enumerate(ptree.query_ball_point(atoms.coord[ids], 1.9)):
            for pi in neighbours:
                sugar = atoms[int(ids[gi])]; protein = atoms[int(pidx[pi])]
                attachments.append({'protein_chain': str(protein.chain_id), 'protein_resnum': int(protein.res_id),
                    'protein_resname': str(protein.res_name), 'protein_atom': str(protein.atom_name), 'sugar_atom': str(sugar.atom_name),
                    'distance_A': float(np.linalg.norm(protein.coord-sugar.coord)), 'selected_partner': str(protein.chain_id) in selected})
        if len(attachments) != 1:
            raise Rejection('glycan_attachment', f'Resolved tree has {len(attachments)} protein attachments; one required')
        link = attachments[0]
        if (link['protein_resname'], link['protein_atom']) not in {('ASN', 'ND2'), ('SER', 'OG'), ('THR', 'OG1')} or link['sugar_atom'] != 'C1':
            raise Rejection('glycan_attachment', 'Unsupported glycan attachment chemistry')
        result.append({'code': 'glycan', 'policy_class': 'glycan', 'name': '+'.join(data[i][0]['name'] for i in members),
                       'members': [data[i][0] for i in members], 'attachment': link,
                       'coordinates': atoms.coord[ids].astype(float).tolist(), 'train_radius_A': 20., 'eval_radius_A': 25.})
    return result


def buffer_screen(atoms, component, partners):
    import numpy as np
    import biotite.structure as struc
    from scipy.spatial import cKDTree
    from .prep import CANONICAL, Rejection
    heavy = ~np.isin(np.char.upper(atoms.element), ['H', 'D'])
    water = np.isin(atoms.res_name, ['HOH', 'WAT', 'H2O', 'DOD'])
    ids = np.arange(component['start'], component['end']); ids = ids[heavy[ids]]
    a = atoms[ids]
    distances = {p: float(cKDTree(atoms.coord[heavy & np.isin(atoms.chain_id, chains) & np.isin(atoms.res_name, list(CANONICAL))]).query(a.coord)[0].min()) for p, chains in partners.items() if chains}
    bridging = len(distances)>1 and max(distances.values()) <= 4.
    env = heavy & ~water; positions = np.searchsorted(np.flatnonzero(env), ids)
    iso = float(struc.sasa(a, probe_radius=1.4, point_number=1000, ignore_ions=False, vdw_radii='Single').sum())
    bound = float(struc.sasa(atoms[env], probe_radius=1.4, point_number=1000, ignore_ions=False, vdw_radii='Single')[positions].sum())
    if not np.isfinite(iso+bound) or iso <= 0:
        raise ValueError('Invalid buffer SASA calculation')
    return {'sasa_A2': bound, 'isolated_sasa_A2': iso, 'exposed_fraction': bound/iso, 'partner_distances_A': distances,
            'bridges_partners': bridging, 'exposed_nonbridging_annotation': bound/iso >= .7 and not bridging}


def classify_components(atoms, cif, partners):
    import numpy as np
    import biotite.structure as struc
    from scipy.spatial import cKDTree
    from .audit import component_inventory
    from .prep import CANONICAL, Rejection
    selected = partners['A']+partners['B']; components = component_inventory(atoms, cif, selected)
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for s, e in zip(starts[:-1], starts[1:]):
        if e-s == 1 and str(atoms.res_name[s]) in ('NA', 'K', 'CL'):
            components.append({'chain': str(atoms.chain_id[s]), 'resnum': int(atoms.res_id[s]), 'name': str(atoms.res_name[s]), 'code': 'metal', 'start': int(s), 'end': int(e)})
    removals = glycan_inventory(atoms, cif, components, selected)
    protected = set(); conn = cif.block.get('struct_conn')
    if conn is not None:
        for i, kind in enumerate(conn['conn_type_id'].as_array(str)):
            if str(kind).lower().startswith(('covale', 'metalc')):
                for p in ('ptnr1', 'ptnr2'):
                    protected.add((str(conn[f'{p}_label_asym_id'].as_array(str)[i]), str(conn[f'{p}_label_comp_id'].as_array(str)[i])))
    heavy = ~np.isin(np.char.upper(atoms.element), ['H', 'D']); water = np.isin(atoms.res_name, ['HOH', 'WAT', 'H2O', 'DOD'])
    protein_tree = cKDTree(atoms.coord[heavy & np.isin(atoms.res_name, list(CANONICAL))])
    for c in components:
        if c['code'] == 'glycan':
            continue
        if c['code'] not in ('ligand', 'metal'):
            raise Rejection(c['code'], f"{c['name']} remains outside stripped policy")
        a = atoms[c['start']:c['end']]; a = a[~np.isin(np.char.upper(a.element), ['H', 'D'])]
        if (c['chain'], c['name']) in protected:
            raise Rejection('connected_component', f"{c['name']} has a declared covalent/metal connection")
        if len(a) == 0:
            raise Rejection('component_without_heavy_atoms', f"{c['name']} has no heavy atoms for distance/SASA screening")
        distance = float(protein_tree.query(a.coord)[0].min())
        record = dict(c, coordinates=a.coord.astype(float).tolist(), min_protein_distance_A=distance, declared_connection=False)
        if c['code'] == 'metal':
            if len(a) != 1:
                raise Rejection('metal_complex', f"{c['name']} is not monatomic")
            donor = heavy & ~water & np.isin(np.char.upper(atoms.element), ['N', 'O', 'S', 'SE']); donor[c['start']:c['end']] = False
            nearest = float(cKDTree(atoms.coord[donor]).query(a.coord)[0].min()) if donor.any() else None
            if nearest is not None and nearest <= 3:
                raise Rejection('coordinated_metal', f"{c['name']} donor distance {nearest}")
            env = heavy & ~water; positions = np.flatnonzero((np.flatnonzero(env) >= c['start']) & (np.flatnonzero(env) < c['end']))
            try:
                iso = float(struc.sasa(a, ignore_ions=False, vdw_radii='Single').sum())
                exposure = float(struc.sasa(atoms[env], ignore_ions=False, vdw_radii='Single')[positions].sum())/iso
            except (KeyError, ValueError):
                raise Rejection('metal_sasa_unavailable', c['name'])
            if not np.isfinite(exposure) or exposure < .7:
                raise Rejection('buried_metal', f"{c['name']} exposed fraction {exposure}")
            record.update(policy_class='metal', train_radius_A=25., eval_radius_A=25., exposed_fraction=exposure, nearest_donor_A=nearest)
        else:
            if distance < 1.9:
                raise Rejection('connected_component', f"{c['name']} short protein contact {distance}")
            kind = 'buffer' if c['name'] in BUFFERS else 'ligand'
            if kind == 'buffer':
                record.update(buffer_screen(atoms, c, partners))
            train, evaluation = radii(kind)
            record.update(policy_class=kind, train_radius_A=train, eval_radius_A=evaluation)
        removals.append(record)
    return removals, components


def process(row, manifest, campaign):
    import numpy as np
    from biotite.structure.io import pdbx
    from .prep import read_cif, Rejection
    from .conformers import resolve
    from .curation import prepare_revised, validate_partners
    from .dataset_audit import filtered_cif
    from .schema import read_table, write_table
    cid = row['complex_id']; source_campaign = Path(manifest['source_campaign'])
    old_path = source_campaign/'rows'/f'{cid}.json'; old = json.loads(old_path.read_text())
    root = campaign/'structures'/cid; previous = source_campaign/'structures'/cid
    result = {**row, 'previous_status': old['status'], 'previous_code': old.get('code'), 'source_row_sha256': digest(old_path)}
    if old['status'] == 'rejected' and old.get('code') != 'glycan':
        return {**result, 'status': 'rejected', 'code': old['code'], 'detail': old['detail'], 'stage': old.get('stage', 'prep'), 'execution': 'reused_unchanged_rejection'}
    root.mkdir(parents=True, exist_ok=True)
    try:
        partners = {p: row[f'partner_{p}_chains'] for p in ('A', 'B')}
        if old['status'] == 'accepted':
            record = json.loads((previous/'removed_components.json').read_text())
            if record['source_sha256'] != row['source_sha256'] or record['resolved_sha256'] != digest(previous/'original-resolved.cif'):
                raise ValueError('Previous preparation source hash mismatch')
            removals = copy.deepcopy(record['components'])
            if any(c['name'] in BUFFERS for c in removals):
                cif = pdbx.CIFFile.read(previous/'original-resolved.cif')
                atoms = pdbx.get_structure(cif, model=1, altloc='occupancy', use_author_fields=False)
                for c in removals:
                    kind = 'metal' if c['code'] == 'metal' else 'buffer' if c['name'] in BUFFERS else 'ligand'
                    if kind == 'buffer':
                        c.update(buffer_screen(atoms, c, partners))
                    c.update(policy_class=kind)
                    c['train_radius_A'], c['eval_radius_A'] = radii(kind)
            else:
                for c in removals:
                    c['policy_class'] = c['code']
            # Immutable files are links; every revised file is written separately.
            for path in previous.iterdir():
                if path.is_file() and path.name not in {'removed_components.json', 'component_masks.json', 'provenance.json'}:
                    (root/path.name).symlink_to(path.resolve())
            structure = copy.deepcopy(old['structure']); sites = read_table(root/'sites.parquet')
            result['execution'] = 'reused_identical_protein_preparation'
        else:
            source = Path(manifest['source_audit'])/'sources'/f"{row['pdb_id']}.cif"
            if digest(source) != row['source_sha256']:
                raise ValueError('Original candidate source hash mismatch')
            cif, _ = resolve(pdbx.CIFFile.read(source), partners['A']+partners['B'])
            resolved = root/'original-resolved.cif'; cif.write(resolved)
            atoms = pdbx.get_structure(cif, model=1, altloc='occupancy', use_author_fields=False)
            validate_partners(atoms, partners)
            removals, components = classify_components(atoms, cif, partners)
            remove_keys = {(c['chain'], c['name']) for c in components}
            cat = cif.block['atom_site']; chains = cat['label_asym_id'].as_array(str); names = cat['label_comp_id'].as_array(str)
            keep = np.array([(str(c), str(n)) not in remove_keys for c, n in zip(chains, names)])
            modified = root/'stripped-source.cif'; filtered_cif(resolved, keep).write(modified)
            structure, sites, _ = prepare_revised(modified, {**row, 'source_sha256': digest(modified)}, root)
            write_table(root/'sites.parquet', 'sites', sites)
            result['execution'] = 'new_glycan_preparation'
        fixed = read_cif(root/'AB.cif'); component_masks = masks(sites, fixed, removals)
        atomic_json(root/'removed_components.json', {'policy': POLICY, 'source_sha256': row['source_sha256'],
                    'resolved_sha256': digest(root/'original-resolved.cif'), 'components': removals})
        atomic_json(root/'component_masks.json', component_masks)
        provenance = json.loads(structure['provenance'])
        provenance.update(stripping_policy=POLICY, removed_components_sha256=digest(root/'removed_components.json'),
                          original_source_sha256=row['source_sha256'], label_scope='stripped protein-only reference',
                          source_campaign=str(source_campaign), reused_protein_preparation=result['execution'].startswith('reused'))
        structure['provenance'] = json.dumps(provenance)
        atomic_json(root/'provenance.json', provenance)
        result.update(status='accepted', structure=structure, sites=len(sites), components_removed=len(removals),
                      component_classes=dict(Counter(c['policy_class'] for c in removals)))
    except Rejection as exc:
        result.update(status='rejected', code=exc.code, detail=str(exc), stage=exc.stage)
    except Exception as exc:
        import traceback
        result.update(status='pipeline_error', detail=str(exc), traceback=traceback.format_exc())
    return result


def scan(campaign, shard, shards):
    require_compute(); campaign = Path(campaign)
    manifest = json.loads((campaign/'manifest.json').read_text())
    if manifest['policy'] != POLICY or manifest['implementation'] != implementation():
        raise ValueError('Frozen campaign implementation changed')
    if digest(Path(manifest['source_campaign'])/'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Source campaign manifest changed')
    counts = Counter()
    for row in manifest['candidates'][shard::shards]:
        receipt = campaign/'rows'/f"{row['complex_id']}.json"
        if receipt.exists():
            continue
        result = process(row, manifest, campaign)
        result.update(job=os.environ['SLURM_JOB_ID'], node=os.environ['SLURMD_NODENAME'])
        atomic_json(receipt, result); counts[result.get('code', result['status'])] += 1
        atomic_json(campaign/'progress'/f'{shard}.json', {'shard': shard, 'completed': sum(counts.values()), 'counts': dict(counts)})
    print(json.dumps({'shard': shard, 'counts': dict(counts)}), flush=True)


def collect(campaign):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .schema import KEY, read_table, write_table
    from .anchor_tiers import apply
    campaign = Path(campaign); manifest = json.loads((campaign/'manifest.json').read_text()); source = Path(manifest['source_campaign'])
    if digest(source/'site_masks.parquet') != manifest['source_masks_sha256'] or digest(source/'report.json') != manifest['source_report_sha256']:
        raise ValueError('Source campaign changed during execution')
    rows = [json.loads((campaign/'rows'/f"{r['complex_id']}.json").read_text()) for r in manifest['candidates']]
    failures = [r for r in rows if r['status'] == 'pipeline_error']
    if failures:
        atomic_json(campaign/'pipeline-errors.json', failures)
        raise ValueError(f'{len(failures)} pipeline errors; inspect pipeline-errors.json')
    accepted = [r for r in rows if r['status'] == 'accepted']; new = [r for r in accepted if r['execution'] == 'new_glycan_preparation']
    reused_ids = {r['complex_id'] for r in accepted if r['execution'] == 'reused_identical_protein_preparation'}
    write_table(campaign/'structures.parquet', 'structures', [r['structure'] for r in accepted])
    sites = [s for r in accepted for s in read_table(campaign/'structures'/r['complex_id']/'sites.parquet')]
    write_table(campaign/'sites.parquet', 'sites', sites)
    key = lambda r: tuple(r[k] for k in KEY)
    gap_rows = [r for r in pq.read_table(source/'natural-gap-tiers/site_tiers.parquet').to_pylist() if r['complex_id'] in reused_ids]
    if new:
        small = campaign/'new-glycan-gap-input'; small.mkdir(); (small/'structures').symlink_to(campaign/'structures', target_is_directory=True)
        write_table(small/'structures.parquet', 'structures', [r['structure'] for r in new])
        apply(small, campaign/'new-glycan-gap-tiers')
        gap_rows.extend(pq.read_table(campaign/'new-glycan-gap-tiers/site_tiers.parquet').to_pylist())
    gaps = {key(r): r for r in gap_rows}
    assert len(gaps) == len(gap_rows) == len(sites)
    natural = campaign/'natural-gap-tiers'; natural.mkdir()
    pq.write_table(pa.Table.from_pylist(gap_rows), natural/'site_tiers.parquet')
    final = []
    for r in accepted:
        for s in json.loads((campaign/'structures'/r['complex_id']/'component_masks.json').read_text()):
            g = gaps[key(s)]
            final.append(s | {'natural_gap_tier': g['tier'], 'interface': g['interface'],
                             'train_mask': s['component_train_mask'] and g['provisional_retained'],
                             'eval_mask': s['component_eval_mask'] and g['provisional_retained']})
    assert len({key(r) for r in final}) == len(final)
    assert all(not r['eval_mask'] or r['train_mask'] for r in final)
    assert all(r['natural_gap_tier'] in ('clean', 'uncertain') for r in final if r['train_mask'])
    pq.write_table(pa.Table.from_pylist(final), campaign/'site_masks.parquet')
    report = {'candidates': len(rows), 'accepted': len(accepted), 'new_glycan_preparations': len(new), 'reused_preparations': len(reused_ids),
              'histogram': dict(Counter(r.get('code', r['status']) for r in rows)), 'pipeline_errors': [], 'policy': POLICY,
              'transitions': dict(Counter(f"{r['previous_status']}:{r.get('previous_code')} -> {r['status']}:{r.get('code')}" for r in rows)),
              'training_sites': sum(r['train_mask'] for r in final), 'evaluation_sites': sum(r['eval_mask'] for r in final),
              'training_interface_sites': sum(r['train_mask'] and r['interface'] for r in final),
              'evaluation_interface_sites': sum(r['eval_mask'] and r['interface'] for r in final),
              'complexes_with_training': len({r['complex_id'] for r in final if r['train_mask']}),
              'mask_sha256': digest(campaign/'site_masks.parquet'), 'production_allowed': False,
              'limits': POLICY['scope']}
    atomic_json(campaign/'report.json', report)
    atomic_json(natural/'report.json', {'tier_counts': dict(Counter(r['tier'] for r in gap_rows)), 'reused_source': str(source), 'new_preparations': len(new)})
    print(json.dumps(report, indent=2), flush=True)


def recount(campaigns, out):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    runtime = Path(os.environ['PKABENCH_RUNTIME']); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    proposal = runtime/'universe/combined-split-v1/usable-proposal-v2/proposal.parquet'
    original_hash = digest(proposal); previous = pq.read_table(proposal).to_pylist(); assigned = {r['complex_id']: r for r in previous}
    counts = defaultdict(Counter); accepted = {}; before_accepted = set(); manifests = []
    for campaign in map(Path, campaigns):
        manifest = json.loads((campaign/'manifest.json').read_text()); report = json.loads((campaign/'report.json').read_text())
        assert not report['pipeline_errors'] and report['mask_sha256'] == digest(campaign/'site_masks.parquet')
        manifests.append({'campaign': str(campaign), 'manifest_sha256': digest(campaign/'manifest.json'), 'mask_sha256': report['mask_sha256']})
        for candidate in manifest['candidates']:
            cid = candidate['complex_id']; row = json.loads((campaign/'rows'/f'{cid}.json').read_text())
            assert cid not in accepted
            accepted[cid] = {k: row.get(k) for k in ('status', 'code', 'execution', 'previous_status', 'previous_code')}
            if row['previous_status'] == 'accepted':
                before_accepted.add(cid)
        for r in pq.read_table(campaign/'site_masks.parquet').to_pylist():
            for role in ('train', 'eval'):
                if r[role+'_mask']:
                    counts[r['complex_id']][role+'_sites'] += 1
                    counts[r['complex_id']][role+'_interface_sites'] += int(r['interface'])
    assert set(assigned) == set(accepted), 'Candidate universe changed'
    revised = []; transitions = Counter()
    for old in previous:
        cid = old['complex_id']; new = dict(old)
        for k in ('train_sites', 'eval_sites', 'train_interface_sites', 'eval_interface_sites'):
            new[k] = counts[cid][k]
        new.update(preparation_status=accepted[cid]['status'], preparation_code=accepted[cid]['code'])
        revised.append(new)
        role = 'train' if old['split'] == 'train' else 'eval'
        if old['benchmark_eligible']:
            a = old[role+'_interface_sites'] > 0; b = new[role+'_interface_sites'] > 0
            transitions[f"{old['split']}:{'usable' if a else 'unusable'}->{'usable' if b else 'unusable'}"] += 1
    pq.write_table(pa.Table.from_pylist(revised), out/'eligibility-with-existing-assignments.parquet')
    summary = {}
    for split in ('train', 'val', 'test'):
        def summarize(data):
            rr = [r for r in data if r['split'] == split and r['benchmark_eligible']]
            role = 'train' if split == 'train' else 'eval'
            return {'interface_pairs': sum(r[role+'_interface_sites'] > 0 for r in rr),
                    'interface_sites': sum(r[role+'_interface_sites'] for r in rr),
                    'pairs_with_sites': sum(r[role+'_sites'] > 0 for r in rr),
                    'sites': sum(r[role+'_sites'] for r in rr),
                    'usable_components': len({r['component_id'] for r in rr if r[role+'_interface_sites'] > 0})}
        summary[split] = {'before': summarize(previous), 'after': summarize(revised)}
    # Previously inactive family reservations must not silently become training data.
    reservations = runtime/'audits/set2-screen-v1/matched-candidate-chains.json'
    overlap = json.loads(reservations.read_text()) if reservations.exists() else []
    blocked_groups = {r['component_id'] for r in overlap if r['system'] == 'protein_g_fc' and r['split'] != 'test'}
    newly_active = [r for r in revised if r['component_id'] in blocked_groups and r['split'] != 'test' and r['benchmark_eligible'] and r[('train' if r['split'] == 'train' else 'eval')+'_interface_sites'] > 0]
    atomic_json(out/'prospective-reservation-review.json', {'protein_g_fc_groups': sorted(blocked_groups), 'newly_usable_non_test_pairs': newly_active,
                'note': 'Only a readiness check. No split reassignment or training. FcRn remains on hold; 1AXT exception unchanged.'})
    result = {'candidate_pairs': len(revised), 'prepared_before': len(before_accepted), 'prepared_after': sum(r['status'] == 'accepted' for r in accepted.values()),
              'by_split': summary, 'usability_transitions': dict(transitions), 'source_campaigns': manifests,
              'original_proposal_sha256': original_hash, 'split_assignments_changed': False, 'frozen': False,
              'newly_usable_pairs_in_prospective_reserved_groups': len(newly_active),
              'limits': 'Eligibility recount only. Newly usable candidates retain previous graph components and assignments; targets were not rebalanced. Experimental reservation checks and teacher coverage precede production use.'}
    assert digest(proposal) == original_hash
    atomic_json(out/'report.json', result); print(json.dumps(result, indent=2), flush=True)
