"""pKPDB 5k pilot rebuilt under the mask-all-v1 component policy (component_mask_policy.py).

Same cohort rules, label mapping and backbone inputs as pkpdb_pilot.py / pkpdb_pilot_clean.py. Differences:
- components: no component chemistry rejections, class radii from component_mask_policy.POLICY (2026-10-06);
- gaps: long_gap_policy.site_usable replaces clean/uncertain anchor tiers as the usable-site rule (2026-10-08);
- leakage: in addition to the pilot rule, a held-out benchmark hit at >= 70% identity and >= 80% coverage of both
  sequences excludes the entry, as does the precomputed list against the PINDER held-out reference
  (audits/seq-overlap-v1/pkpdb_heldout_exclusions_70.tsv) (2026-10-08);
- antibody path: entries excluded only through antibody chains whose CDRs are < 70% identical to held-out and
  experimental antibody CDRs are released (audits/pkpdb-ab-path-v1/pkpdb_ab_path_v2.tsv, as PINDER
  pinder-heldout-exclusions-v3; v2 uses complete held-out CDR sets, including held-out antibodies without SAbDab
  CDR annotations) (2026-10-09).
Kept as a separate module so the existing pkpdb-5k-v2 protocol hashes are unchanged.
Usage (compute node): python -m pkabench.pkpdb_mask_all <out> [--smoke] [--full]
--full processes every pKPDB entry (no 5,000 cap); pipeline errors are then recorded in audit.json instead of aborting.
--revise reruns a verified build whose protocol changed: the previous protocol and outputs move to revisions/<n>/, cached
entry receipts are reused (clean() is unchanged) and only newly passing entries are cleaned.
"""
import concurrent.futures
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx
from .runtime import atomic_json, digest, require_compute, config_hash
from .prep import CANONICAL, Rejection
from .annotate import SITE_ATOMS
from .supervision import inventory
from .pkpdb_pilot_refs import read, reference_inventory, cif
from .pkpdb_pilot_clean import metadata, gap_context, gap_tier
from .pkpdb_pilot import labels, search
from .component_mask_policy import POLICY, classify, class_trees, nearest_by_class, clear_of_components
from .long_gap_policy import POLICY as GAP_POLICY, site_usable

EXCLUSIONS_70 = 'audits/seq-overlap-v1/pkpdb_heldout_exclusions_70.tsv'
ANTIBODY_PATH = 'audits/pkpdb-ab-path-v1/pkpdb_ab_path_v2.tsv'
REVISION_OUTPUTS = ('protocol.json', 'pilot.json', 'verification.json', 'audit.json', 'status.json', 'report.md')


def heldout_70(identity, qcov, tcov, kinds):
    """PINDER leakage rule for benchmark held-out chains: >= 70% identity over >= 80% of both sequences."""
    return 'benchmark' in kinds and identity >= .7 and min(qcov, tcov) >= .8


def excluded_70(out, batch_number, references, candidates):
    """Entries excluded by heldout_70, re-read from the pilot search hits (computed at >= 30% identity, any coverage)."""
    owners = defaultdict(set)
    for r in candidates:
        for c in r['chains']: owners['q'+config_hash(c['sequence'])[:24]].add(r['pdb_id'])
    hits = out/'sequence'/f'batch-{batch_number:03d}'/'hits.tsv'; excluded = set()
    if hits.exists():
        for line in hits.open():
            q, t, i, qc, tc = line.strip().split('\t')
            if heldout_70(float(i), float(qc), float(tc), references['references'][t]['kinds']): excluded |= owners[q]
    return excluded


def antibody_released(path):
    """Entries whose sequence-overlap exclusion comes only from antibody chains that pass the CDR check (status released*)."""
    import csv
    return {r['pdb'] for r in csv.DictReader(open(path), delimiter='\t') if r['status'].startswith('released')}


def revise_outputs(out):
    """Move a previous protocol and its outputs to revisions/<n>/; returns the previous protocol sha256."""
    previous = digest(out/'protocol.json'); n = len(list((out/'revisions').glob('*'))) if (out/'revisions').exists() else 0
    dest = out/'revisions'/f'{n:02d}'; dest.mkdir(parents=True)
    for name in REVISION_OUTPUTS:
        if (out/name).exists(): (out/name).rename(dest/name)
    return previous


def clean(task):
    root, out, row = task; root = Path(root); out = Path(out); pdb = row['pdb_id']; dest = out/'entries'/pdb
    dest.mkdir(parents=True, exist_ok=True); resultfile = dest/'receipt.json'
    if resultfile.exists(): return json.loads(resultfile.read_text())
    try:
        path = root/'pretraining/pkpdb-v1/structures'/pdb[1:3]/f'{pdb}.cif.gz'
        from .conformers import resolve
        from pkanet.graph import geometry
        from jaxpropka.parameters import THREE_TO_INDEX, GROUPS
        source_receipt = json.loads((path.parent/f'{pdb}.json').read_text()); assert digest(path) == source_receipt['sha256']
        selected = [r['chain'] for r in row['chains']]; partners = {'A': selected, 'B': []}
        file, conformers = resolve(cif(path), selected)
        evidence = inventory(file, partners)
        label = pdbx.get_structure(file, model=1, altloc='occupancy', use_author_fields=False)
        author = pdbx.get_structure(file, model=1, altloc='occupancy', use_author_fields=True)
        assert len(label) == len(author) and np.array_equal(label.coord, author.coord)
        working = label.copy(); working.res_id = author.res_id.copy(); working.ins_code = author.ins_code.copy()
        cat = file.block['chem_comp']; types = dict(zip(cat['id'].as_array(str), cat['type'].as_array(str)))
        # Noncanonical peptide residues are absent from the backbone model and become sequence gaps (unchanged).
        modified = {name for name, kind in types.items() if 'PEPTIDE' in str(kind).upper() and name not in CANONICAL}
        components, without_heavy = classify(working[~np.isin(working.res_name, list(modified))], file, selected)
        if without_heavy:
            raise Rejection('component_without_heavy_atoms', f'{without_heavy} component(s) have no heavy atoms to define a mask')
        trees = class_trees(components)
        keep = np.isin(working.res_name, list(CANONICAL)) & np.isin(working.chain_id, selected) & ~np.isin(np.char.upper(working.element), ['H', 'D'])
        starts = struc.get_residue_starts(working, add_exclusive_stop=True)
        residues = {}; nodes = []; backbone = []; chainindex = []; lookup = defaultdict(list); original = []; positions = []
        polylen = {r['chain']: len(r['sequence']) for r in row['chains']}
        for s, e in zip(starts[:-1], starts[1:]):
            if not keep[s]: continue
            a = working[s:e]; k = (str(a.chain_id[0]), int(a.res_id[0]), str(a.ins_code[0]).strip())
            if k in residues: raise Rejection('ambiguous_residue_key', str(k))
            residues[k] = a
            names = {str(n): i for i, n in enumerate(a.atom_name)}
            if not {'N', 'CA', 'C'} <= names.keys(): raise Rejection('incomplete_backbone', 'Backbone input requires N, CA and C; no coordinates invented')
            seqpos = int(label.res_id[s]); authchain = str(author.chain_id[s]); n = len(nodes)
            bb = np.stack([a.coord[names[name]] if name in names else a.coord[names['C']] for name in ('N', 'CA', 'C', 'O')])
            backbone.append(bb); chainindex.append(selected.index(k[0])); positions.append(seqpos)
            aa = np.eye(20, dtype=np.float32)[THREE_TO_INDEX[str(a.res_name[0])]]
            nodes.append(np.concatenate((aa, [seqpos == 1, seqpos == polylen[k[0]], False, True])))
            lookup[authchain, k[1]].append((n, k, str(a.res_name[0]))); original.append(dict(chain=authchain, resnum=k[1], icode=k[2], label_chain=k[0], label_seq_id=seqpos))
        if not nodes: raise Rejection('no_backbone', 'No canonical protein backbone')
        graphs, valid = geometry(np.asarray(backbone), np.asarray(chainindex)); nodes = np.asarray(nodes, np.float32); nodes[:, 23] = valid
        graphs.update(nodes=nodes, node_mask=np.ones(len(nodes), bool))
        gaps, known = gap_context(evidence, residues); breaks = {tuple(k) for k in evidence['artificial_terminal_keys']}
        db = sqlite3.connect(f'file:{out}/labels.sqlite?mode=ro', uri=True)
        rows = db.execute('select chain,kind,number,pka from labels where pdb=?', (pdb,)).fetchall(); db.close()
        counts = Counter(); mapped = []; queries = []; values = []; seen = Counter((chain, kind, number) for chain, kind, number, _ in rows)
        for chain, kind, number, value in rows:
            counts['deposited_labels'] += 1
            if seen[chain, kind, number] > 1: counts['duplicate_label_key'] += 1; continue
            match = re.fullmatch(r'(-?\d+)([A-Za-z]?)', number)
            if not match: counts['unparseable_residue_number'] += 1; continue
            num = int(match[1]); ins = match[2]; choices = lookup.get((chain, num), [])
            if not ins and any(k[2] for _, k, _ in choices): counts['ambiguous_insertion_code'] += 1; continue
            choices = [r for r in choices if r[1][2] == ins]
            if len(choices) != 1: counts['unmapped_or_ambiguous_site'] += 1; continue
            n, k, resname = choices[0]; group = {'NTR': 'NTERM', 'CTR': 'CTERM'}.get(kind, kind)
            if group not in GROUPS or (group not in ('NTERM', 'CTERM') and group != resname): counts['residue_type_mismatch'] += 1; continue
            if value is None or not np.isfinite(value): counts['nonfinite_label'] += 1; continue
            a = residues[k]; points = a.coord[np.isin(a.atom_name, SITE_ATOMS[group])]
            functional = set(SITE_ATOMS[group]) <= set(a.atom_name)
            terminal = (group == 'NTERM' and positions[n] != 1) or (group == 'CTERM' and positions[n] != polylen[k[0]])
            eligible = known and functional and not terminal and not (group in ('NTERM', 'CTERM') and k in breaks)
            tier = gap_tier(points, eligible, gaps); usable, gap_reason = site_usable(points, eligible, gaps)
            distances = nearest_by_class(points, trees)
            train = usable and clear_of_components(distances, POLICY['train_radii_A'])
            evaluation = usable and clear_of_components(distances, POLICY['eval_radii_A'])
            mapped.append(dict(complex_id=pdb, **original[n], group=group, pka=float(value), train_mask=bool(train), eval_mask=bool(evaluation),
                natural_gap_tier=tier, gap_rule=gap_reason, functional_atoms_complete=functional, nearest_component_A=min(distances.values(), default=None),
                nearest_component_by_class={c: round(d, 3) for c, d in distances.items()}))
            queries.append((n, GROUPS.index(group))); values.append(value); counts['raw_sites'] += 1; counts['clean_sites'] += bool(train); counts['eval_sites'] += bool(evaluation)
        if not values: raise Rejection('no_mapped_labels', json.dumps(counts))
        if not counts['clean_sites']: raise Rejection('no_clean_sites', json.dumps(counts))
        q = np.asarray(queries, np.int32); graphs.update(query_residue=q[:, 0], query_group=q[:, 1])
        np.savez_compressed(dest/'graph.npz', **graphs, labels=np.asarray(values, np.float32))
        atomic_json(dest/'sites.json', mapped); atomic_json(dest/'conformers.json', conformers); atomic_json(dest/'defects.json', evidence)
        atomic_json(dest/'removed_components.json', dict(policy=POLICY['version'], components=[
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in components]))
        result = dict(pdb_id=pdb, complex_id=pdb, status='accepted', policy=POLICY['version'], counts=dict(counts), n=len(nodes), k=graphs['neighbors'].shape[1], q=len(values),
            component_classes=dict(Counter(c['cls'] for c in components)),
            component_id='exact-'+config_hash(sorted(r['sequence'] for r in row['chains']))[:20],
            sha256=digest(dest/'graph.npz'), sites_sha256=digest(dest/'sites.json'), source_sha256=digest(path),
            noncanonical_residue_names=sorted(modified), keys=[[pdb, r['chain'], r['resnum'], r['icode'], r['group']] for r in mapped])
    except Rejection as exc: result = dict(pdb_id=pdb, status='rejected', reason=exc.code, detail=str(exc))
    except Exception as exc:
        import traceback
        result = dict(pdb_id=pdb, status='pipeline_error', reason=type(exc).__name__, detail=repr(exc), traceback=traceback.format_exc())
    atomic_json(resultfile, result); return result


def run(root, out, smoke=False, full=False, revise=False):
    target = None if full else 5000
    threads = int(os.environ['SLURM_CPUS_PER_TASK']); require_compute(threads=threads)
    out.mkdir(parents=True, exist_ok=True); began = time.time()
    verified = (out/'verification.json').exists() and read(out/'verification.json')['passed']
    if verified and not revise: return
    refs = read(out/'references.json') if (out/'references.json').exists() else reference_inventory(root, out)
    if refs['unresolved']: raise RuntimeError(f'Unresolved experimental reserve sequences: {refs["unresolved"]}')
    labels(root, out)
    index = root/'pretraining/pkpdb-v1/index.json'; order = read(index)['pdb_ids']; random.Random(20261006).shuffle(order)
    here = Path(__file__).parent
    manifest = dict(target=target if target else 'all', selection_seed=20261006, label_index_sha256=digest(index), references_sha256=digest(out/'references.json'),
        validation_test_identity_cutoff=.9, validation_test_shorter_coverage=.8, experimental_identity_cutoff=.3, experimental_bidirectional_coverage=.8,
        heldout_identity_cutoff_70=.7, heldout_bidirectional_coverage_70=.8, heldout_exclusions_70_sha256=digest(root/EXCLUSIONS_70), gap_policy=GAP_POLICY,
        antibody_path=('entries excluded only through antibody chains (aligned to SAbDab-annotated V domains) whose concatenated CDRs are < 70% identical, '
                       'same chain type, to every held-out and experimental antibody CDR set are released'),
        antibody_path_sha256=digest(root/ANTIBODY_PATH),
        selection=('every eligible structure' if full else 'first 5000 eligible structures in deterministic shuffled order')+'; same cohort for raw and cleaned labels',
        component_policy=POLICY,
        masks='long-gap-v1 gap rule (anchor tiers + calibrated long-gap radii); components never reject; train/eval radii ligand 15/25, buffer 15/25, glycan 20/25, exposed ion 25/25, bound metal/complex 30/30 A',
        scope='Temporary pilot; deposited asymmetric units, 30-1500 declared protein residues, no nonprotein polymer. No pKa recalculation or structural reconstruction.',
        code={p.name: digest(p) for p in sorted(here.glob('pkpdb_pilot*.py'))+
              [here/name for name in ('pkpdb_mask_all.py', 'component_mask_policy.py', 'long_gap_policy.py', 'audit.py', 'conformers.py', 'supervision.py', 'anchor_tiers.py')]})
    if (out/'protocol.json').exists():
        stored = read(out/'protocol.json')
        if {k: v for k, v in stored.items() if k != 'revises_protocol_sha256'} == manifest:
            if verified: return
            manifest = stored
        else:
            assert revise, 'protocol changed; rerun with --revise to supersede the previous build in place'
            manifest['revises_protocol_sha256'] = revise_outputs(out); atomic_json(out/'protocol.json', manifest)
    else: atomic_json(out/'protocol.json', manifest)
    accepted = []; audit = []; reserved = set(refs['reserved_pdb_ids']); batch_size = 40 if smoke else 2000
    import csv
    listed_70 = {r['pdb'] for r in csv.DictReader(open(root/EXCLUSIONS_70), delimiter='\t')}
    released = antibody_released(root/ANTIBODY_PATH)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(16, threads//2)) as pool:
        for batch_number, start in enumerate(range(0, len(order), batch_size)):
            ids = order[start:start+batch_size]; tasks = []
            for pdb in ids:
                if pdb in reserved: audit.append(dict(pdb_id=pdb, status='rejected', reason='reserved_pdb_id')); continue
                path = root/'pretraining/pkpdb-v1/structures'/pdb[1:3]/f'{pdb}.cif.gz'; tasks.append((pdb, str(path)))
            meta = list(pool.map(metadata, tasks, chunksize=4)); audit.extend(r for r in meta if r['status'] != 'candidate')
            candidates = [r for r in meta if r['status'] == 'candidate']
            atomic_json(out/'status.json', dict(stage='sequence screening', batch=batch_number, accepted=len(accepted), candidates=len(candidates), target=target or 'all'))
            excluded = (set(search(root, out, candidates, refs, batch_number, threads)) | excluded_70(out, batch_number, refs, candidates) | (listed_70 & {r['pdb_id'] for r in candidates})) - released
            passing = []
            for r in candidates:
                if r['pdb_id'] in excluded: audit.append(dict(pdb_id=r['pdb_id'], status='rejected', reason='sequence_overlap'))
                else: passing.append(r)
            results = list(pool.map(clean, [(str(root), str(out), r) for r in passing], chunksize=1)); audit.extend(results)
            errors = [r for r in results if r['status'] == 'pipeline_error']
            accepted.extend(r for r in results if r['status'] == 'accepted')
            report = dict(stage='cleaning', processed=len(audit), accepted=len(accepted), target=target or 'all',
                reasons=dict(Counter(r.get('reason', r['status']) for r in audit)), elapsed_seconds=time.time()-began, pipeline_errors=errors)
            atomic_json(out/'status.json', report); atomic_json(out/'audit.json', audit); print(json.dumps({k: v for k, v in report.items() if k != 'pipeline_errors'}), flush=True)
            if errors and not full: raise RuntimeError(f'{len(errors)} unexpected preparation errors; pilot not released')
            if smoke or (target and len(accepted) >= target): break
    if smoke:
        assert accepted, 'Smoke produced no accepted structures'
        atomic_json(out/'smoke.json', dict(passed=True, processed=len(audit), accepted=len(accepted))); return
    if target and len(accepted) < target: raise RuntimeError(f'Only {len(accepted)} accepted structures; refusing to relax leakage or cleaning gates')
    selected = accepted[:target] if target else accepted; assert len({r['pdb_id'] for r in selected}) == len(selected)
    for r in selected:
        assert r['pdb_id'] not in reserved
        path = out/'entries'/r['pdb_id']; assert digest(path/'graph.npz') == r['sha256'] and digest(path/'sites.json') == r['sites_sha256']
    release = dict(target=target or 'all', pipeline_errors=sum(r['status'] == 'pipeline_error' for r in audit), records=selected, component_policy=POLICY['version'], gap_policy=GAP_POLICY['version'], raw_arm='all unambiguously mapped finite scalar labels',
        clean_arm='the same structures/inputs with train_mask applied from sites.json',
        validation_source=str(root/'pretraining/graph-pilot-v1'), validation_manifest_sha256=digest(root/'pretraining/graph-pilot-v1/manifest.json'),
        protocol_sha256=digest(out/'protocol.json'), reference_sha256=digest(out/'references.json'),
        raw_sites=sum(r['counts']['raw_sites'] for r in selected), clean_sites=sum(r['counts']['clean_sites'] for r in selected),
        training_launched=False, notes=['Historical teacher preparation/version remain unknown; direct author chain/residue/type matches only.',
            'Structures must have at least one clean site in both arms; raw/clean comparison is conditional on this cohort.',
            'Within-training component_id groups exact sequence sets only; 90%/70% exclusions are against held-out chains, not a within-training clustering claim.',
            'Gap policy long-gap-v1: calibrated long-gap radii on pKAI deletions; pKPDB labels are PypKa, so these radii are not validated for them.',
            'Component policy mask-all-v1: component chemistry never rejects; masks only.',
            'Antibody path: entries excluded only through antibody chains with CDRs < 70% identical to held-out/experimental antibody CDRs are released.'])
    if manifest.get('revises_protocol_sha256'): release['revises_protocol_sha256'] = manifest['revises_protocol_sha256']
    atomic_json(out/'pilot.json', release); atomic_json(out/'verification.json', dict(passed=True, structures=len(selected), raw_sites=release['raw_sites'], clean_sites=release['clean_sites'], pilot_sha256=digest(out/'pilot.json')))
    text = [f'# pKPDB {"full build" if full else "5k pilot"} (mask-all-v1, long-gap-v1, 70% held-out, antibody path)', '', f'{len(selected):,} structures; {release["raw_sites"]:,} raw mapped sites; {release["clean_sites"]:,} clean training sites; {release["pipeline_errors"]} pipeline errors recorded.',
        'Cohort rules as pkpdb-5k-v2; leakage adds the 70%/80%-both held-out rule and the CDR-based antibody path. Components never reject (component_mask_policy); gaps per long_gap_policy.',
        'Raw and clean arms share identical inputs and structure membership. No training launched.', '', '| Audit reason | Structures |', '|---|---:|']
    text.extend(f'| {k} | {v} |' for k, v in sorted(Counter(r.get('reason', r['status']) for r in audit).items()))
    text += ['', *release['notes']]; (out/'report.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    root = Path(os.environ['PKABENCH_RUNTIME']); out = Path(sys.argv[1]); run(root, out, '--smoke' in sys.argv, '--full' in sys.argv, '--revise' in sys.argv)
