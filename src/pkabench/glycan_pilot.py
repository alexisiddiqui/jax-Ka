"""Fixed-geometry glycan removal pilot; never changes production eligibility."""
import contextlib
import csv
import json
import os
import subprocess
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from .runtime import require_compute, atomic_json, digest

NEUTRAL_SUGARS = {'NAG', 'NDG', 'MAN', 'BMA', 'FUC', 'FUL', 'GAL', 'GLC', 'BGC', 'A2G', 'NGA', 'XYS', 'XYP'}
KEY = ('complex_id', 'chain', 'resnum', 'icode', 'group')


def ccd_prediction(path):
    """Standard PROPKA parameters with explicit CCD sugar hybridization.

    Only ligand atom typing is supplied. No coordinates or pKa parameters change.
    Geometry-derived connectivity must agree with CCD for every observed sugar.
    """
    from propka.lib import loadOptions
    from propka.input import read_parameter_file, read_molecule_file
    from propka.parameters import Parameters
    from propka.molecular_container import MolecularContainer
    from propka.hydrogens import setup_bonding, set_ligand_atom_names
    from propka.ligand import set_type
    from biotite.structure.info import bonds_in_residue, get_from_ccd
    options = loadOptions([str(path)])
    parameters = read_parameter_file(options.parameters, Parameters())
    model = MolecularContainer(parameters, options)
    typing_evidence = []

    def preparation(molecule):
        maker = setup_bonding(molecule)
        for conf in molecule.conformations.values():
            byres = defaultdict(list)
            for atom in conf.atoms:
                if atom.chain_id == 'z' and atom.element != 'H':
                    byres[atom.res_num].append(atom)
            for number, atoms in byres.items():
                name = atoms[0].res_name.strip()
                if name not in NEUTRAL_SUGARS:
                    raise ValueError('CCD typing requested for unsupported sugar')
                bonds = bonds_in_residue(name)
                cat = get_from_ccd('chem_comp_atom', name)
                ids = cat['atom_id'].as_array(str)
                elements = dict(zip(ids, cat['type_symbol'].as_array(str)))
                charges = dict(zip(ids, cat['charge'].as_array(int)))
                leaving = dict(zip(ids, cat['pdbx_leaving_atom_flag'].as_array(str)))
                observed = {a.name: a for a in atoms}
                required = {n for n in ids if elements[n] != 'H' and leaving[n] != 'Y'}
                if not required <= observed.keys():
                    raise ValueError(f'Incomplete glycan {name}: missing non-leaving CCD atoms {sorted(required-observed.keys())}')
                for atom in atoms:
                    if elements.get(atom.name) != atom.element or charges.get(atom.name) != 0:
                        raise ValueError('Sugar element/charge incompatible with neutral CCD typing')
                    expected = set()
                    orders = []
                    for (a, b), order in bonds.items():
                        other = b if a == atom.name else a if b == atom.name else None
                        if other in observed:
                            expected.add(other); orders.append((other, int(order)))
                    actual = {b.name for b in atom.get_bonded_heavy_atoms() if b.chain_id == 'z' and b.res_num == number}
                    if actual != expected:
                        raise ValueError(f'Glycan geometry/CCD bond mismatch {name}:{atom.name}')
                    double = any(order == 2 for _, order in orders)
                    if any(order not in (1, 2) for _, order in orders):
                        raise ValueError('Unsupported sugar bond order')
                    if atom.element == 'C':
                        kind = 'C.2' if double else 'C.3'
                    elif atom.element == 'O':
                        kind = 'O.2' if double else 'O.3'
                    elif atom.element == 'N':
                        amide = any(observed[n].element == 'C' and any(int(order) == 2 and
                            ((a == n and elements.get(b) == 'O') or (b == n and elements.get(a) == 'O'))
                            for (a, b), order in bonds.items()) for n in expected)
                        if not amide:
                            raise ValueError('Neutral sugar nitrogen is not a CCD amide')
                        kind = 'N.am'
                    else:
                        raise ValueError('Unsupported neutral sugar element')
                    set_type(atom, kind)
                    typing_evidence.append({'resnum': number, 'resname': name, 'atom': atom.name, 'sybyl': kind})
        # All glycan atoms are preassigned; the normal routine preserves them.
        set_ligand_atom_names(molecule)
        maker.add_pi_electron_information(molecule)
    model.version.molecular_preparation_method = preparation
    model = read_molecule_file(str(path), model)
    model.calculate_pka()
    return model, typing_evidence


def initialise_ccd(out, source):
    require_compute()
    out = Path(out); source = Path(source)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source/'manifest.json').read_text())
    manifest.update(typing_mode='ccd', native_control=str(source), native_manifest_sha256=digest(source/'manifest.json'), code_sha256=digest(Path(__file__)))
    for task in manifest['tasks']:
        task['typing_mode'] = 'ccd'
    atomic_json(out/'manifest.json', manifest)
    print(json.dumps({'tasks': len(manifest['tasks']), 'typing_mode': 'ccd'}), flush=True)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometry(source, row):
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from scipy.spatial import cKDTree
    from .audit import component_inventory
    from .prep import CANONICAL
    cif = pdbx.CIFFile.read(source)
    raw = pdbx.get_structure(cif, model=1, altloc='occupancy', use_author_fields=False)
    auth = pdbx.get_structure(cif, model=1, altloc='occupancy', use_author_fields=True)
    assert len(raw) == len(auth) and np.array_equal(raw.coord, auth.coord)
    selected = row['partner_A_chains'] + row['partner_B_chains']
    components = component_inventory(raw, cif, selected)
    if not components or any(c['code'] != 'glycan' for c in components):
        raise ValueError('Mixed nonprotein chemistry outside neutral-glycan pilot')
    if any(c['name'] not in NEUTRAL_SUGARS for c in components):
        raise ValueError('Sugar identity outside neutral pilot whitelist')
    # Legacy inventory omits these ions. Do not silently accept mixed contexts.
    if np.isin(raw.res_name, ['NA', 'K', 'CL']).any():
        raise ValueError('Additional ions outside pure-glycan pilot')
    heavy = ~np.isin(np.char.upper(raw.element), ['H', 'D'])
    protein_mask = np.isin(raw.chain_id, selected) & np.isin(raw.res_name, list(CANONICAL)) & heavy
    protein = raw[protein_mask].copy()
    protein.res_id = auth.res_id[protein_mask]
    protein.ins_code = auth.ins_code[protein_mask]
    if not 50 <= len(struc.get_residue_starts(protein)) <= 1500:
        raise ValueError('Selected protein size outside pilot range')
    sugar_mask = np.zeros(len(raw), bool)
    for c in components:
        sugar_mask[c['start']:c['end']] = True
    sugar_mask &= heavy
    sugars = raw[sugar_mask].copy()
    sugars.res_id = auth.res_id[sugar_mask]
    sugars.ins_code = auth.ins_code[sugar_mask]
    starts = struc.get_residue_starts(sugars, add_exclusive_stop=True)
    residue_index = np.empty(len(sugars), int)
    for i, (s, e) in enumerate(zip(starts[:-1], starts[1:])):
        residue_index[s:e] = i
    parents = list(range(len(starts)-1))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    # Whole covalently connected resolved trees, never single-sugar cuts.
    for i, j in cKDTree(sugars.coord).query_pairs(1.9):
        a, b = find(int(residue_index[i])), find(int(residue_index[j]))
        parents[a] = b
    groups = defaultdict(list)
    for i in range(len(sugars)):
        groups[find(int(residue_index[i]))].append(i)
    all_context = protein + sugars
    context_sasa = struc.sasa(all_context, probe_radius=1.4, point_number=1000, ignore_ions=False, vdw_radii='Single')
    trees = []
    protein_tree = cKDTree(protein.coord)
    for indexes in sorted(groups.values(), key=lambda v: min(v)):
        a = sugars[indexes]
        attachments = []
        for gi, neighbours in enumerate(protein_tree.query_ball_point(a.coord, 1.9)):
            for pi in neighbours:
                atom = protein[pi]
                attachments.append({'glycan_atom_index': gi, 'protein_key': [str(atom.chain_id), int(atom.res_id), str(atom.ins_code).strip()],
                                    'protein_atom': str(atom.atom_name), 'protein_resname': str(atom.res_name),
                                    'sugar_atom': str(a.atom_name[gi]), 'distance_A': float(np.linalg.norm(a.coord[gi]-atom.coord))})
        if len(attachments) != 1:
            raise ValueError(f'Expected one unambiguous selected-protein attachment per tree; found {len(attachments)}')
        link = attachments[0]
        if (link['protein_resname'], link['protein_atom']) not in {('ASN', 'ND2'), ('SER', 'OG'), ('THR', 'OG1')} or link['sugar_atom'] != 'C1':
            raise ValueError('Attachment outside N/O-linked glycan pilot')
        owner = 'A' if link['protein_key'][0] in row['partner_A_chains'] else 'B'
        iso = float(struc.sasa(a, probe_radius=1.4, point_number=1000, ignore_ions=False, vdw_radii='Single').sum())
        bound = float(context_sasa[len(protein)+np.asarray(indexes)].sum())
        if not np.isfinite(iso+bound) or iso <= 0 or not -.001 <= bound/iso <= 1.001:
            raise ValueError('Invalid glycan SASA')
        distances = {p: float(cKDTree(protein.coord[np.isin(protein.chain_id, row[f'partner_{p}_chains'])]).query(a.coord)[0].min()) for p in ('A', 'B')}
        trees.append({'atoms': a, 'owner': owner, 'attachment': link, 'sugar_residues': len(struc.get_residue_starts(a)),
                      'names': sorted(set(map(str, a.res_name))), 'sasa_A2': bound, 'isolated_sasa_A2': iso,
                      'exposed_fraction': bound/iso, 'partner_distances_A': distances, 'bridges_partners': max(distances.values()) <= 4})
    return cif, protein, trees


def initialise(out, count=30):
    require_compute()
    import numpy as np
    runtime = Path(os.environ['PKABENCH_RUNTIME'])
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    candidates = []
    for name in ('stripped-components-v1', 'stripped-components-v2-5000'):
        campaign = runtime/'campaigns'/name
        for path in sorted((campaign/'rows').glob('*.json')):
            row = json.loads(path.read_text())
            if row.get('code') == 'glycan':
                candidates.append({'row': row, 'campaign': str(campaign), 'row_path': str(path)})
    # One pair per deposited assembly; deterministic hash order, no selection on error.
    candidates.sort(key=lambda t: t['row']['complex_id'])
    seen = set(); inventory = []; rejected = []
    for task in candidates:
        row = task['row']; pdb = row['pdb_id']
        if pdb in seen:
            continue
        source = Path(task['campaign'])/'structures'/row['complex_id']/'original-resolved.cif'
        try:
            _, protein, trees = geometry(source, row)
            task.update(source=str(source), source_sha256=digest(source), row_sha256=digest(task['row_path']),
                        glycan_trees=len(trees), glycan_residues=sum(t['sugar_residues'] for t in trees),
                        exposure=float(np.mean([t['exposed_fraction'] for t in trees])),
                        bridges=any(t['bridges_partners'] for t in trees),
                        fc_context=any('gamma' in c.get('description', '').lower() and 'region' in c.get('description', '').lower() for c in row['chains']))
            inventory.append(task); seen.add(pdb)
        except Exception as exc:
            rejected.append({'complex_id': row['complex_id'], 'pdb_id': pdb, 'reason': str(exc)})
        atomic_json(out/'inventory-progress.json', {'inventoried': len(inventory), 'screened_out': len(rejected), 'target_inventory': count*3})
        if len(inventory) >= count*3:
            break
    if len(inventory) < 3:
        raise ValueError('Insufficient supported glycan structures')
    # Exposure-spanning sample; preserve multi-sugar and Fc examples where available.
    ordered = sorted(inventory, key=lambda t: t['exposure'])
    positions = np.unique(np.linspace(0, len(ordered)-1, min(count, len(ordered))).round().astype(int))
    selected = [ordered[i] for i in positions]
    for predicate in (lambda t: t['fc_context'], lambda t: t['glycan_residues'] >= 6):
        if not any(predicate(t) for t in selected):
            replacement = next((t for t in ordered if predicate(t)), None)
            if replacement is not None:
                selected[len(selected)//2] = replacement
    selected = sorted({t['row']['complex_id']: t for t in selected}.values(), key=lambda t: t['exposure'])
    atomic_json(out/'inventory.json', {'supported': inventory, 'screened_out': rejected})
    manifest = {'tasks': selected, 'count': len(selected), 'code_sha256': digest(Path(__file__)),
                'scope': 'Neutral resolved N/O-linked glycans; one pair per assembly, exposure-spanning convenience pilot. Excludes mixed species, ambiguous attachments, unsupported sugars. Protein-only production rules unchanged.',
                'distance': 'Minimum titratable functional-atom to any removed glycan heavy-atom distance.',
                'sasa': 'Resolved whole-tree SASA in selected proteins plus all selected glycans divided by isolated whole-tree SASA; 1.4 A probe, 1000 points, Single element radii.',
                'production_allowed': False}
    atomic_json(out/'manifest.json', manifest)
    print(json.dumps({'selected': len(selected), 'inventory': len(inventory), 'screened_out': len(rejected),
                      'exposure_range': [selected[0]['exposure'], selected[-1]['exposure']], 'fc_contexts': sum(t['fc_context'] for t in selected)}), flush=True)


def run(out, shard, shards):
    require_compute()
    out = Path(out).resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    for task in manifest['tasks'][shard::shards]:
        cid = task['row']['complex_id']; work = out/'structures'/cid
        if (work/'result.json').exists():
            previous = json.loads((work/'result.json').read_text())
            if previous['status'] == 'complete' or previous['code_sha256'] == digest(Path(__file__)):
                continue
            import shutil
            archive = out/'failed-attempts'/f"{cid}-job{previous['job']}"
            archive.parent.mkdir(exist_ok=True)
            shutil.move(str(work), str(archive))
        work.mkdir(parents=True, exist_ok=True)
        try:
            result = run_pair(task, work)
        except Exception as exc:
            result = {'complex_id': cid, 'pdb_id': task['row']['pdb_id'], 'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()}
        result.update(job=os.environ['SLURM_JOB_ID'], node=os.environ['SLURMD_NODENAME'], code_sha256=digest(Path(__file__)))
        atomic_json(work/'result.json', result)
        print(json.dumps({k: result.get(k) for k in ('complex_id', 'pdb_id', 'status', 'error', 'observations')}), flush=True)


def run_pair(task, work):
    import numpy as np
    import biotite.structure as struc
    from propka.run import single
    from .prep import read_cif, export_pdb
    from .curation import prepare_revised
    from .dataset_audit import filtered_cif
    from .schema import write_table
    from .annotate import SITE_ATOMS
    row = task['row']; cid = row['complex_id']; source = Path(task['source'])
    if digest(source) != task['source_sha256'] or digest(task['row_path']) != task['row_sha256']:
        raise ValueError('Pilot source changed')
    cif, _, trees = geometry(source, row)
    cat = cif.block['atom_site']; remove = np.isin(cat['label_comp_id'].as_array(str), list(NEUTRAL_SUGARS))
    modified = work/'protein-only-source.cif'; filtered_cif(source, ~remove).write(modified)
    structure, sites, extra = prepare_revised(modified, {**row, 'source_sha256': digest(modified)}, work)
    write_table(work/'sites.parquet', 'sites', sites)
    atomic_json(work/'prepared-structure.json', structure)
    atoms = read_cif(work/'AB.cif')
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    residues = {(str(atoms.chain_id[s]), int(atoms.res_id[s]), str(atoms.ins_code[s]).strip()): atoms[s:e] for s, e in zip(starts[:-1], starts[1:])}
    mappings = {}; texts = {}; glycan_records = {}; attachment_keys = {}
    for state in ('AB', 'A', 'B'):
        a = atoms if state == 'AB' else atoms[np.isin(atoms.chain_id, row[f'partner_{state}_chains'])]
        mapping = export_pdb(a, work/f'{state}-protein.pdb')
        if any(k[0] == 'z' for k in mapping):
            raise ValueError('Reserved glycan chain collision')
        mappings[state] = mapping
        texts[state] = '\n'.join(l for l in (work/f'{state}-protein.pdb').read_text().splitlines() if l.startswith(('ATOM', 'TER'))) + '\n'
    # Stable glycan PDB residue IDs shared across every state and deletion.
    number = 0
    for i, tree in enumerate(trees):
        a = tree['atoms']; ss = struc.get_residue_starts(a, add_exclusive_stop=True); records = []
        for s, e in zip(ss[:-1], ss[1:]):
            number += 1
            for j in range(s, e):
                atom = a[j]
                records.append((number, str(atom.atom_name), str(atom.res_name), str(atom.element), atom.coord))
                if j == tree['attachment']['glycan_atom_index']:
                    attachment_keys[i] = ('z', number, str(atom.atom_name))
        glycan_records[i] = records
    atom_owner = {('z', n, name): i for i, records in glycan_records.items() for n, name, _, _, _ in records}
    def predict(state, removed, tag):
        text = texts[state]; included = []; serial = 80000
        for i, tree in enumerate(trees):
            if i in removed or (state != 'AB' and tree['owner'] != state):
                continue
            included.append(i)
            for n, name, residue, element, xyz in glycan_records[i]:
                serial += 1; x, y, z = xyz
                # PROPKA infers element from columns 13–14, not the element field.
                atom_field = f' {name:<3s}' if len(element) == 1 and len(name) < 4 else f'{name:<4s}'
                text += f'HETATM{serial:5d} {atom_field} {residue:>3s} z{n:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{0.:6.2f}          {element:>2s}\n'
        path = work/f'{state}-{tag}.pdb'; path.write_text(text+'END\n')
        with (work/f'{state}-{tag}.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if task.get('typing_mode') == 'ccd':
                model, ccd_evidence = ccd_prediction(path)
            else:
                model = single(str(path), write_pka=False)
                ccd_evidence = []
        conf = model.conformations[model.conformation_names[0]]
        heavy = {(a.chain_id, a.res_num, a.name): a for a in conf.atoms if a.element != 'H'}
        expected = {key for key, i in atom_owner.items() if i in included}
        retained = {key for key in heavy if key[0] == 'z'}
        if retained != expected:
            raise ValueError('PROPKA glycan atom retention mismatch')
        for i in included:
            for n, name, _, element, _ in glycan_records[i]:
                if heavy['z', n, name].element.upper() != element.upper():
                    raise ValueError('PROPKA glycan element parsing mismatch')
        typing = []; predictions = {}; attachment_checks = []
        for group in conf.groups:
            atom = group.atom
            if atom.chain_id == 'z':
                typing.append({'tree': atom_owner[(atom.chain_id, atom.res_num, atom.name)], 'resnum': atom.res_num,
                               'atom': atom.name, 'type': group.type, 'charge': group.charge, 'sybyl': atom.sybyl_type})
            else:
                key = mappings[state].get((atom.chain_id, atom.res_num))
                name = {'N+': 'NTERM', 'C-': 'CTERM'}.get(group.residue_type.strip(), group.residue_type.strip())
                if key is not None and name in SITE_ATOMS and np.isfinite(group.pka_value):
                    predictions[(*key, name)] = float(group.pka_value)
        for i in included:
            if not any(t['tree'] == i for t in typing):
                raise ValueError('PROPKA retained glycan atoms but assigned no interacting groups')
            if any(t['tree'] == i and t['charge'] != 0 for t in typing):
                raise ValueError('Unexpected charged group in neutral glycan: typing gate failed')
            link = trees[i]['attachment']; inverse = {v: k for k, v in mappings[state].items()}
            pk = inverse[tuple(link['protein_key'])] + (link['protein_atom'],)
            sugar = heavy[attachment_keys[i]]; protein = heavy[pk]
            if protein not in sugar.bonded_atoms:
                raise ValueError('PROPKA failed to recognize glycan-protein covalent attachment')
            attachment_checks.append({'tree': i, 'protein': pk, 'sugar': attachment_keys[i], 'bond_recognized': True,
                                      'protein_hydrogens': len(protein.get_bonded_elements('H'))})
        gate = {'retained_glycan_heavy_atoms': len(retained), 'typing': typing, 'attachments': attachment_checks,
                'typing_mode': task.get('typing_mode', 'native'), 'ccd_atom_types': ccd_evidence,
                'ligand_typing': conf.parameters.ligand_typing,
                'cutoffs': {k: getattr(conf.parameters, k, None) for k in ('coulomb_cutoff1', 'coulomb_cutoff2', 'buried_cutoff', 'desolv_cutoff')}}
        atomic_json(work/f'{state}-{tag}-gate.json', gate)
        return predictions
    baseline = {state: predict(state, set(), 'full') for state in ('AB', 'A', 'B')}
    # Repeat one unmodified prediction: distinguish reproducibility from removal.
    repeat = predict('AB', set(), 'repeat')
    repeat_error = max((abs(v-repeat[k]) for k, v in baseline['AB'].items()), default=0.)
    comparisons = []; variants = []
    deletions = [(f'tree{i}', {i}) for i in range(len(trees))]
    if len(trees) > 1:
        deletions.append(('all', set(range(len(trees)))))
    for tag, removed in deletions:
        changed = {state: predict(state, removed, tag) if state == 'AB' or any(trees[i]['owner'] == state for i in removed) else baseline[state]
                   for state in ('AB', 'A', 'B')}
        removed_atoms = np.concatenate([trees[i]['atoms'].coord for i in sorted(removed)])
        sasa = sum(trees[i]['sasa_A2'] for i in removed)
        isolated = sum(trees[i]['isolated_sasa_A2'] for i in removed)
        all_removed = len(removed) == len(trees)
        variants.append({'variant': tag, 'all_glycans_removed': all_removed, 'trees_removed': sorted(removed),
                         'sasa_A2': sasa, 'exposed_fraction': sasa/isolated})
        for site in sites:
            key = tuple(site[k] for k in ('chain', 'resnum', 'icode', 'group'))
            partner = site['partner']
            if any(key not in x for x in (baseline['AB'], baseline[partner], changed['AB'], changed[partner])):
                continue
            a = residues[key[:3]]; points = a.coord[np.isin(a.atom_name, SITE_ATOMS[key[3]])]
            if not len(points):
                continue
            distance = float(np.linalg.norm(points[:, None]-removed_atoms[None, :], axis=-1).min())
            delta0 = baseline['AB'][key]-baseline[partner][key]
            delta1 = changed['AB'][key]-changed[partner][key]
            comparisons.append({k: site[k] for k in KEY} | {'pdb_id': row['pdb_id'], 'variant': tag, 'all_glycans_removed': all_removed,
                'interface': site['residue_delta_sasa'] > 10, 'native_eligible': bool(site['functional_atoms_complete'] and not site['is_break_terminus']),
                'distance_A': distance, 'sasa_A2': sasa, 'exposed_fraction': sasa/isolated,
                'ab_pka_change': changed['AB'][key]-baseline['AB'][key], 'delta_pka_change': delta1-delta0})
    if not comparisons:
        raise ValueError('No matched pKa observations')
    write_csv(work/'site_changes.csv', comparisons)
    metadata = [{k: v for k, v in t.items() if k != 'atoms'} for t in trees]
    atomic_json(work/'glycans.json', metadata)
    return {'complex_id': cid, 'pdb_id': row['pdb_id'], 'status': 'complete', 'observations': len(comparisons),
            'typing_mode': task.get('typing_mode', 'native'),
            'glycans': metadata, 'variants': variants, 'repeat_max_abs_pka_change': repeat_error,
            'source_sha256': digest(source), 'protein_sha256': digest(work/'AB.cif'), 'fc_context': task['fc_context']}


def collect(out):
    require_compute()
    import numpy as np
    import pyarrow.parquet as pq
    from .schema import write_table
    from .anchor_tiers import apply
    out = Path(out); manifest = json.loads((out/'manifest.json').read_text()); tasks = manifest['tasks']
    results = [json.loads((out/'structures'/t['row']['complex_id']/'result.json').read_text()) for t in tasks]
    successful = [r for r in results if r['status'] == 'complete']
    if not successful:
        atomic_json(out/'report.json', {'statuses': dict(Counter(r['status'] for r in results)), 'failures': results, 'sensitivity_supported': False})
        raise ValueError('No supported glycan comparisons; do not present zero errors')
    structures = [json.loads((out/'structures'/r['complex_id']/'prepared-structure.json').read_text()) for r in successful]
    write_table(out/'structures.parquet', 'structures', structures)
    gaproot = out/'natural-gap-tiers'
    apply(out, gaproot)
    tiers = {tuple(r[k] for k in KEY): r for r in pq.read_table(gaproot/'site_tiers.parquet').to_pylist()}
    rows = []
    for result in successful:
        with (out/'structures'/result['complex_id']/'site_changes.csv').open() as stream:
            for r in csv.DictReader(stream):
                for field in ('distance_A', 'sasa_A2', 'exposed_fraction', 'ab_pka_change', 'delta_pka_change'):
                    r[field] = float(r[field])
                r['resnum'] = int(r['resnum'])
                for field in ('interface', 'native_eligible', 'all_glycans_removed'):
                    r[field] = r[field] == 'True'
                tier = tiers[tuple(r[k] for k in KEY)]
                r.update(natural_gap_tier=tier['tier'], existing_mask_retained=tier['provisional_retained'])
                rows.append(r)
    write_csv(out/'site_changes.csv', rows)
    # All-deletion observations give unique sites and simultaneous-removal effects.
    all_rows = [r for r in rows if r['all_glycans_removed'] and r['native_eligible']]
    assert len({tuple(r[k] for k in KEY) for r in all_rows}) == len(all_rows)
    retention = []; summaries = []
    for radius in range(0, 41):
        for mask in ('glycan_only', 'combined_existing_gaps'):
            rr = [r for r in all_rows if r['distance_A'] >= radius and (mask == 'glycan_only' or r['existing_mask_retained'])]
            interface = [r for r in rr if r['interface']]
            retention.append({'radius_A': radius, 'mask': mask, 'sites': len(rr), 'interface_sites': len(interface),
                              'pairs_with_sites': len({r['complex_id'] for r in rr}), 'pairs_with_interface_sites': len({r['complex_id'] for r in interface})})
        if radius not in (0, 10, 15, 20, 25, 30):
            continue
        for selection in ('all_native', 'existing_masks', 'interface_existing_masks'):
            rr = [r for r in all_rows if r['distance_A'] >= radius and
                  (selection == 'all_native' or r['existing_mask_retained']) and
                  (selection != 'interface_existing_masks' or r['interface'])]
            for metric in ('ab_pka_change', 'delta_pka_change'):
                values = np.abs([r[metric] for r in rr])
                summaries.append({'radius_A': radius, 'selection': selection, 'metric': metric, 'n': len(rr),
                                  'complexes': len({r['complex_id'] for r in rr}),
                                  'median': float(np.median(values)) if len(values) else None,
                                  'p95': float(np.quantile(values, .95)) if len(values) else None,
                                  'max': float(np.max(values)) if len(values) else None,
                                  'fraction_over_0_1': float(np.mean(values > .1)) if len(values) else None})
    write_csv(out/'retention.csv', retention); write_csv(out/'distance_summary.csv', summaries)
    for mask in ('glycan_only', 'combined_existing_gaps'):
        rr = [r for r in retention if r['mask'] == mask]
        for field in ('sites', 'interface_sites', 'pairs_with_sites', 'pairs_with_interface_sites'):
            assert all(a[field] >= b[field] for a, b in zip(rr, rr[1:])), 'Nonmonotonic retention'
        assert all(r['interface_sites'] <= r['sites'] and r['pairs_with_interface_sites'] <= r['pairs_with_sites'] <= len(successful) for r in rr)
    report = {'statuses': dict(Counter(r['status'] for r in results)), 'failures': [r for r in results if r['status'] != 'complete'],
              'typing_mode': successful[0].get('typing_mode', 'native'),
              'completed_complexes': len(successful), 'resolved_glycan_trees': sum(len(r['glycans']) for r in successful),
              'unique_native_matched_sites': len(all_rows), 'repeat_max_abs_pka_change': max(r['repeat_max_abs_pka_change'] for r in successful),
              'fc_contexts': sum(r['fc_context'] for r in successful), 'summaries': summaries,
              'retention_at_thresholds': [r for r in retention if r['radius_A'] in (0, 10, 15, 20, 25)],
              'limits': 'PROPKA fixed-coordinate sensitivity, not measured physical error. Native finite interaction cutoffs constrain distance interpretation. Resolved neutral glycans only; unobserved sugars/relaxation not modeled. Exposure-spanning convenience sample, not population prevalence. Protein partners and attached glycans separated together. Repeated sites/trees correlated; any bootstrap uses complexes. Retention counts require matched PROPKA coverage, functional atom completeness and natural termini; combined counts also apply existing natural-gap tiers. No production mask or split changes.'}
    from importlib.metadata import version
    from biotite.structure.info import ccd
    report['software'] = {'propka': version('propka'), 'biotite': version('biotite'),
                          'ccd_sha256': digest(ccd._CCD_FILE), 'pilot_code_sha256': digest(Path(__file__))}
    if manifest.get('typing_mode') == 'ccd':
        native = Path(manifest['native_control']); changes = []
        for r in successful:
            native_result = json.loads((native/'structures'/r['complex_id']/'result.json').read_text())
            if native_result['status'] == 'complete':
                assert r['protein_sha256'] == native_result['protein_sha256'], 'Native/CCD protein geometry differs'
            before_path = native/'structures'/r['complex_id']/'AB-full-gate.json'
            after_path = out/'structures'/r['complex_id']/'AB-full-gate.json'
            if not before_path.exists():
                continue
            before = json.loads(before_path.read_text()); after = json.loads(after_path.read_text())
            expected = {(a['resnum'], a['atom']): a for a in after['ccd_atom_types']}
            for atom in before['typing']:
                key = atom['resnum'], atom['atom']
                if key in expected and atom['sybyl'] != expected[key]['sybyl']:
                    changes.append({'complex_id': r['complex_id'], 'pdb_id': r['pdb_id'], 'resnum': key[0], 'atom': key[1],
                                    'native_sybyl': atom['sybyl'], 'ccd_sybyl': expected[key]['sybyl']})
        write_csv(out/'native_typing_disagreements.csv', changes)
        report['native_typing_disagreements'] = {'interacting_atoms': len(changes), 'complexes': len({r['complex_id'] for r in changes})}
    atomic_json(out/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('summaries', 'failures', 'retention_at_thresholds')}, indent=2), flush=True)


def plots(out):
    require_compute()
    runtime = Path(os.environ['PKABENCH_RUNTIME'])
    subprocess.run([str(runtime/'envs/radial-plots/bin/python'), '-m', 'pkabench.glycan_plots', str(out)], check=True)


def verify(out):
    require_compute()
    import numpy as np
    out = Path(out); manifest = json.loads((out/'manifest.json').read_text())
    native = Path(manifest['native_control'])
    def load(path):
        with path.open() as stream:
            return {tuple(r[k] for k in (*KEY, 'variant')): r for r in csv.DictReader(stream)}
    before = load(native/'site_changes.csv'); after = load(out/'site_changes.csv')
    assert before.keys() == after.keys(), 'Native/CCD matched-site coverage differs'
    changes = {}
    for metric in ('ab_pka_change', 'delta_pka_change'):
        difference = np.array([abs(float(after[k][metric])-float(before[k][metric])) for k in after])
        changes[metric] = {'max_absolute_difference': float(difference.max()), 'observations_changed': int((difference > 1e-12).sum())}
    assert (out/'retention.csv').read_bytes() == (native/'retention.csv').read_bytes(), 'Typing changed geometric retention'
    outliers = [r for r in after.values() if r['all_glycans_removed'] == 'True' and r['native_eligible'] == 'True' and
                float(r['distance_A']) >= 15 and max(abs(float(r[m])) for m in ('ab_pka_change', 'delta_pka_change')) > .1]
    write_csv(out/'outliers_beyond_15A.csv', outliers)
    for name in ('error_vs_distance', 'error_vs_sasa', 'error_vs_exposure', 'retained_sites_vs_radius'):
        for extension in ('png', 'pdf', 'svg'):
            assert (out/'plots'/f'{name}.{extension}').stat().st_size > 1000
    result = {'native_ccd_matched_observations': len(after), 'sensitivity_differences': changes,
              'identical_retention': True, 'outliers_beyond_15A': len(outliers), 'plot_artifacts_checked': 12,
              'production_policy_changed': False, 'job': os.environ['SLURM_JOB_ID']}
    atomic_json(out/'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)
