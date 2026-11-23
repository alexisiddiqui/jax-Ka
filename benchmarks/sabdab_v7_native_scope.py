"""Read-only prepared-input scope checks for adjudication and a small native pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, config_hash, digest, require_compute
from sabdab_v7_exposure import read, skey
from sabdab_v7_roles import write_rows


def source(root, rel):
    path = root / rel
    return path if path.exists() else root / 'overlay' / rel


def read_cif(path):
    from biotite.structure.io import pdbx
    with gzip.open(path, 'rt') as f:
        return pdbx.CIFFile.read(f)


def atom_signature(atoms):
    fields = ['chain_id', 'res_id', 'ins_code', 'res_name', 'atom_name', 'element']
    h = hashlib.sha256()
    for field in fields:
        h.update(json.dumps(getattr(atoms, field).tolist(), separators=(',', ':')).encode())
    h.update(np.asarray(atoms.coord, np.float32).tobytes())
    return h.hexdigest()


def parquet_rows(rows):
    columns = sorted({k for row in rows for k in row})
    return [dict({k: r.get(k) for k in columns},
        input_pins=[dict(path=p, sha256=h) for p, h in sorted(r['input_pins'].items())]) for r in rows]


def describe(atoms):
    import biotite.structure as struc
    from jaxpropka.parameters import THREE, ALPHABET
    aa = dict(zip(THREE, ALPHABET))
    chains = []
    for cid in dict.fromkeys(map(str, atoms.chain_id)):
        subset = atoms[atoms.chain_id == cid]
        starts = struc.get_residue_starts(subset, add_exclusive_stop=True)
        sequence = ''; keys = []; breaks = 0; prev = None
        for s, e in zip(starts[:-1], starts[1:]):
            residue = subset[s:e]; names = {str(n): i for i, n in enumerate(residue.atom_name)}
            if not {'N', 'CA', 'C'} <= names.keys():
                raise ValueError('Incomplete model backbone at ' + cid + ':' + str(residue.res_id[0]))
            if prev is not None and np.linalg.norm(prev - residue.coord[names['N']]) > 1.9:
                breaks += 1
            prev = residue.coord[names['C']]
            sequence += aa[str(residue.res_name[0])]
            keys.append(dict(chain=cid, number=int(residue.res_id[0]), insertion=str(residue.ins_code[0]).strip()))
        chains.append(dict(chain_id=cid, sequence_key=skey(sequence), sequence=sequence,
            residues=len(sequence), backbone_breaks=breaks, residue_keys=keys))
    return chains


def check_one(task):
    runtime, unit, inventory_keys, reasons = task
    runtime = Path(runtime); uid = unit['unit_id']; cid = unit['source_native_id']
    started = time.monotonic(); pins = {}; free_check = None
    result = dict(unit_id=uid, source=unit['source'], source_native_id=cid,
        selected_reasons=reasons, input_scope_id=unit['input_scope_id'])
    try:
        import biotite.structure as struc
        from biotite.structure.io import pdbx
        from jaxpropka.parameters import THREE
        if unit['source'] == 'pkpdb':
            from pkabench.conformers import resolve
            entry = source(runtime, 'pretraining/pkpdb-full-v1/entries/' + cid)
            path = source(runtime, 'pretraining/pkpdb-v1/structures/' + cid[1:3] + '/' + cid + '.cif.gz')
            receipt = read(entry / 'receipt.json'); defects = read(entry / 'defects.json')
            for name in ['receipt.json', 'defects.json', 'conformers.json']:
                pins[str(entry / name)] = digest(entry / name)
            pins[str(path)] = digest(path)
            if receipt.get('status') != 'accepted' or pins[str(path)] != receipt['source_sha256']:
                raise ValueError('Native source differs from accepted graph receipt')
            selected = [r['chain'] for r in defects['sequences']]
            file, conformers = resolve(read_cif(path), selected)
            if json.loads(json.dumps(conformers)) != read(entry / 'conformers.json'):
                raise ValueError('Conformer replay differs from graph receipt')
            label = pdbx.get_structure(file, model=1, altloc='occupancy', use_author_fields=False)
            author = pdbx.get_structure(file, model=1, altloc='occupancy', use_author_fields=True)
            if len(label) != len(author) or not np.array_equal(label.coord, author.coord):
                raise ValueError('Label/author atom mapping differs')
            atoms = label.copy(); atoms.res_id = author.res_id.copy(); atoms.ins_code = author.ins_code.copy()
            keep = np.isin(atoms.res_name, list(THREE)) & np.isin(atoms.chain_id, selected) & ~np.isin(np.char.upper(atoms.element), ['H', 'D'])
            atoms = atoms[keep]
            n = struc.get_residue_count(atoms)
            if n != receipt['n']:
                raise ValueError('Replayed model residue count differs from graph receipt')
            result['model_receipt_residues'] = n
            result['scope_replay'] = 'production_graphs._pkpdb_one_selection_and_conformers'
            result['teacher_context_status'] = 'historical_label_context_not_reconstructed'
        else:
            from jaxpropka.topology import load_topology
            fv = unit['input_scope_id'] == 'v6c-native-observed-Fv-coordinates'
            original = source(runtime, 'pretraining/pinder-pkai-fv-v1/entries/' + cid) if fv else None
            entry = (Path(unit['_baseline_root']) / 'fv-build/pretraining/pinder-pkai-fv-v1/labels-v3/entries' / cid
                     if fv else source(runtime, 'pretraining/pinder-pkai-v1/entries/' + cid))
            path = entry / 'AB.cif.gz'
            for name in ['AB.cif.gz', 'A.cif.gz', 'B.cif.gz', 'meta.json', 'structure.json', 'labels.json']:
                if (entry / name).exists():
                    pins[str(entry / name)] = digest(entry / name)
            arrays = {}
            for state in (['AB'] if fv else ['AB', 'A', 'B']):
                file = read_cif(entry / (state + '.cif.gz'))
                aa = pdbx.get_structure(file, model=1, altloc='occupancy', use_author_fields=True, include_bonds=True)
                topology = load_topology(aa, gap_policy='cap', freeze_disulfides=True)
                arrays[state] = topology.atoms
            atoms = arrays['AB']
            if fv:
                original_meta = read(original / 'structure.json')
                pins[str(original / 'structure.json')] = digest(original / 'structure.json')
                meta = read(entry / 'meta.json')
                for name, expected in meta['outputs'].items():
                    actual = digest(entry / name); pins[str(entry / name)] = actual
                    if actual != expected:
                        raise ValueError('Curated Fv teacher output pin differs: ' + name)
                if pins[str(path)] != original_meta['cif_sha256']:
                    raise ValueError('Curated Fv coordinates differ from the source assembly')
                side = np.asarray([0 if key.chain in {'H', 'L'} else 1 if key.chain == 'G' else -1 for key in topology.keys])
                if np.any(side < 0) or not np.array_equal(side, original_meta['side_by_residue']):
                    raise ValueError('Whole-Fv H+L versus antigen partition differs from teacher provenance')
                arrays['A'] = atoms[np.isin(atoms.chain_id, ['H', 'L'])]
                arrays['B'] = atoms[atoms.chain_id == 'G']
                result['free_view_source'] = 'derived_from_pinned_whole_Fv_side_partition'
            else:
                result['free_view_source'] = 'stored_separate_partner_states'
            def atoms_dict(arr):
                return {(str(c), int(r), str(i).strip(), str(rn), str(an)): xyz
                        for c, r, i, rn, an, xyz in zip(arr.chain_id, arr.res_id, arr.ins_code, arr.res_name, arr.atom_name, arr.coord)}
            bound, free = atoms_dict(atoms), {}
            for state in ['A', 'B']:
                per = atoms_dict(arrays[state])
                if free.keys() & per.keys():
                    raise ValueError('Separated partners overlap in atom identity')
                free.update(per)
            free_check = bound.keys() == free.keys() and all(np.array_equal(bound[k], free[k]) for k in bound)
            if not free_check:
                raise ValueError('Native A/B coordinates differ from partner removal of AB')
            side_chains={state:set(map(str,arrays[state].chain_id)) for state in ['A','B']}
            result['side_by_chain']=([dict(chain=c,side=state) for state in ['A','B'] for c in sorted(side_chains[state])]
                if not side_chains['A'] & side_chains['B'] else [])
            result['side_chain_partition_unambiguous']=not bool(side_chains['A'] & side_chains['B'])
            result['scope_replay'] = 'production_graphs._pinder_one_topology'
            result['teacher_context_status'] = 'existing_bound_free_coordinates_pinned_labels_not_rebuilt'
        chains = describe(atoms)
        observed = {c['sequence_key'] for c in chains}
        result.update(status='checked', coordinate_scope_matches_inventory=observed == set(inventory_keys),
            old_sequence_keys=sorted(inventory_keys), observed_sequence_keys=sorted(observed),
            chains=chains, prepared_student_context_sha256=atom_signature(atoms),
            bound_free_coordinate_identity=free_check, n_res=sum(c['residues'] for c in chains),
            n_atoms=len(atoms), new_labels_or_features=False)
    except Exception as exc:
        result.update(status='unassessed', error=repr(exc), missing_path=getattr(exc, 'filename', None), new_labels_or_features=False)
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError('Native input changed while reading: ' + path)
    result.update(input_pins=pins, wall_s=time.monotonic()-started)
    return result


def run(args):
    require_compute(threads=args.workers, allow_comp1400=True)
    if args.output.exists():
        raise FileExistsError('Choose a new scope-check output root')
    args.output.mkdir(parents=True)
    root = args.root; data = read(root / 'inputs.json')
    units = {u['unit_id']: u for u in pq.read_table(root / 'units.parquet').to_pylist()}
    old = pq.read_table(root / 'report/proposed-assignments.parquet').to_pylist()
    chosen = defaultdict(set)
    for row in old:
        for reason in ['bridges_authoritative_train_validation', 'general_protein_holdout_70_80', 'unmapped_antibody_or_unsupported_source_split']:
            if reason in row['reason_codes']:
                chosen[row['unit_id']].add(reason)
    # A generic overlap cannot be adjudicated from just the removed member.
    # Inspect the opposing native context as well; SAbDab counterparts remain
    # explicitly at their published metadata scope until the native ML pilot.
    general_removed = {a['unit_id'] for a in old if 'general_protein_holdout_70_80' in a['reason_codes']}
    for pair in pq.read_table(root / 'report/general-holdout-evidence.parquet').to_pylist():
        if {pair['train_unit'], pair['validation_unit']} & general_removed:
            for uid in [pair['train_unit'], pair['validation_unit']]:
                if units[uid]['source'] != 'sabdab':
                    chosen[uid].add('generic_overlap_counterpart')
    pilot = pq.read_table(args.adjudication / 'cf2-published-mapped-records/proposed-assignments.parquet').to_pylist()
    previous = {a['unit_id']: a for a in old}
    strata = defaultdict(list)
    for a in pilot:
        if a['source'] == 'sabdab' or a['operational_role'] not in {'train', 'validation'}:
            continue
        recovered = previous[a['unit_id']]['operational_role'] == 'quarantine'
        kind = 'Fv' if 'observed-Fv' in a['input_scope_id'] else ('mapped' if a['matched_components'] else 'general')
        strata[(a['source'], a['operational_role'], recovered, kind)].append(a['unit_id'])
    for stratum, ids in sorted(strata.items()):
        for uid in sorted(ids, key=lambda k: config_hash(k))[:2]:
            chosen[uid].add('native_scope_reuse_pilot:' + repr(stratum))
    member_keys = defaultdict(set)
    for m in pq.read_table(root / 'scope-membership.parquet').to_pylist():
        member_keys[m['unit_id']].add(m['sequence_key'])
    manifest = read(Path(data['inventory']) / 'inventory-manifest.json')
    tasks = [(manifest['runtime'], dict(units[uid], _baseline_root=manifest['baseline']), sorted(member_keys[uid]), sorted(reasons))
             for uid, reasons in sorted(chosen.items())]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(check_one, tasks))
    # Explicit fields keep integer residue numbers typed correctly; arbitrary
    # path keys are records rather than thousands of sparse struct columns.
    write_rows(args.output / 'native-scope-checks.parquet', parquet_rows(rows))
    observed = {c['sequence_key']: c['sequence'] for r in rows if r['status'] == 'checked' for c in r['chains']}
    cached = read(Path(data['inventory']) / 'numbering-cached.json')
    fresh = read(Path(data['inventory']) / 'numbering-new.json')
    attempted = set(cached['attempted']) | set(fresh['numbered'])
    if args.numbering_cache:
        attempted.update(read(args.numbering_cache)['numbered'])
    todo = {k: seq for k, seq in observed.items() if k not in attempted}
    atomic_json(args.output / 'observed-sequences.json', observed)
    atomic_json(args.output / 'numbering-input.json', todo)
    counts = defaultdict(Counter)
    for r in rows:
        counts[r['source']][r['status'] + ('/inventory_equal' if r.get('coordinate_scope_matches_inventory') else '/inventory_differs')] += 1
    summary = dict(status='native_scope_checks_complete_no_membership_changes', release_ready=False,
        selected_units=len(rows), counts={k: dict(v) for k, v in counts.items()},
        new_observed_sequences_requiring_numbering=len(todo),
        unresolved=[dict(unit_id=r['unit_id'], error=r.get('error')) for r in rows if r['status'] != 'checked'],
        structures_modified=False, labels_or_features_built=False,
        scope_receipt_sha256=digest(args.output / 'native-scope-checks.parquet'),
        code_sha256=digest(__file__),
        reused_numbering_cache_sha256=digest(args.numbering_cache) if args.numbering_cache else None,
        limitations=['An observed sequence difference does not by itself resolve a split conflict.',
            'Student-coordinate identity does not establish equivalent teacher-label context.',
            'No duplicate ownership, loader activation or hard-test independence has been established.'])
    atomic_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'adjudication', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--numbering-cache', type=Path)
    run(p.parse_args())
