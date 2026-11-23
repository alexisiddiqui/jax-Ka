"""Inactive SAbDab-authoritative metadata audit. All science runs in Slurm.

Published train/test map to operational train/validation. Legacy cohort evidence
is diagnostic, never a veto. This program cannot activate pools or build labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import csv
import gzip
import json
import os
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, digest, require_compute
from pkabench.role_engine import Evidence, PUBLISHED, Unit, assign, hard_test_status, quarantine_group_conflicts
from pkabench.source_registry import SourceRegistry
from pkabench.split_authority import SplitAuthority
from sabdab_pool_overlap import domain, official_h3_links, published_cdr_string
from sabdab_v7_antigen import compare_chunk, initialise
from sabdab_v7_exposure import fasta, read, skey
from sabdab_v7_screen import mmsearch


def write_rows(path, rows):
    pq.write_table(pa.Table.from_pylist(list(rows)), path, compression='zstd')


def add_sequence(sequences, sequence):
    k = skey(sequence)
    if k in sequences and sequences[k] != sequence:
        raise ValueError('Sequence digest collision')
    sequences[k] = sequence
    return k


def target_index(rows, sequences, expected=False):
    grouped = defaultdict(set)
    for r in rows:
        if not r['agchains']:
            continue
        types = r['agtypes'].split('/')
        resolved, full = r['agresolvedseqs'].split('/'), r['agexpectedseqs'].split('/')
        if not (len(types) == len(resolved) == len(full)):
            raise ValueError('Misaligned antigen metadata')
        for typ, obs, exp in zip(types, resolved, full):
            if typ not in {'PROTEIN', 'PEPTIDE'}:
                continue
            for seq in sorted({obs, exp} if expected else {obs}):
                if seq:
                    grouped[(add_sequence(sequences, seq), typ)].add(r['ab_ag_cluster'])
    return [dict(sequence_key=k, antigen_type=t, components=sorted(v), reference_ids=[])
            for (k, t), v in sorted(grouped.items())]


def selected_members(members):
    # Expected/reference chains remain separate sensitivity evidence, never
    # silently substituted for the coordinate/source input scope.
    return [m for m in members if m['sequence_kind'] not in {
        'numbering_reference_including_unresolved', 'Hseq_expected', 'Lseq_expected',
        'expected_antigen', 'Hseq', 'Lseq'}]


def numbering_assessed(key, annotation_role, numbered, attempted):
    domains = numbered.get(key, [])
    if key not in attempted:
        return False
    if annotation_role.startswith('antibody') and not domains:
        return False
    for d in domains:
        if 'numbering' not in d or not any(105 <= int(p) <= 117 and aa != '-'
                                           for p, _, aa in d['numbering']):
            return False
    # An attempted/no-HKL result allows a generic context screen; it is not
    # a claim that the biological chain is definitively non-antibody.
    return True


def prepare(args):
    require_compute(threads=args.workers, allow_comp1400=True)
    out = args.output
    if out.exists():
        raise FileExistsError('Choose a new metadata-audit root')
    out.mkdir(parents=True)
    authority, registry = SplitAuthority.load(args.authority), SourceRegistry.load(args.registry)
    registry.validate_precedence(authority.source_precedence)
    manifest = read(args.inventory / 'inventory-manifest.json')
    for key, pin in manifest['input_pins'].items():
        if digest(pin['path']) != pin['sha256']:
            raise ValueError('Frozen input changed: ' + key)
    if read(args.screen / 'numbering-verification.json')['status'] != 'passed':
        raise ValueError('Numbering cache verification did not pass')
    units = pq.read_table(args.inventory / 'units.parquet').to_pylist()
    members = pq.read_table(args.inventory / 'sequence-membership.parquet').to_pylist()
    with Path(manifest['input_pins']['sabdab_csv']['path']).open() as f:
        official = list(csv.DictReader(f))
    sequences = fasta(args.inventory / 'sequences.fasta')
    component_splits = {}
    for r in official:
        c, s = r['ab_ag_cluster'], r['ab_ag_split']
        if component_splits.setdefault(c, s) != s:
            raise ValueError('Authoritative component spans source splits')
    cached = read(args.inventory / 'numbering-cached.json')
    numbered = cached['numbered']
    new = read(args.inventory / 'numbering-new.json')
    numbered.update(new['numbered'])
    attempted = set(cached['attempted']) | set(new['numbered'])
    retained = selected_members(members)
    by_unit = defaultdict(list)
    for m in retained:
        by_unit[m['unit_id']].append(m)
    # Only protein/peptide-antigen units receive supervised candidate roles.
    specs, query_keys, missing, scope_counts = [], set(), [], Counter()
    for u in units:
        uid = u['unit_id']; native = u['source'] != 'sabdab'
        general = native and u['stratum'] != 'Ab/Ag' and u['source_annotation'] not in {'accepted', 'antibody_only', 'paired_Fv', 'AbAg'}
        scope = registry.get(u['source']).input_scope_id
        if native and any(m['sequence_kind'] == 'native_observed_coordinates' for m in by_unit[uid]):
            scope = 'v6c-native-observed-Fv-coordinates'
        specs.append(dict(asdict(Unit(uid, u['source'], u['original_role'], scope,
            u['sabdab_component'], bool(u['protein_partner']), general)),
            source_native_id=u['source_native_id'], source_groups=u['source_groups'], pdb_id=u['pdb_id']))
        if not native:
            continue
        for m in by_unit[uid]:
            k = m['sequence_key']; ds = numbered.get(k, [])
            if k not in attempted:
                missing.append(dict(unit_id=uid, sequence_key=k, reason='numbering_not_attempted'))
            # Antibody chains do not become antigen queries through numbering
            # failure. Unannotated no-domain sequences are screened as context.
            if not m['annotation_role'].startswith('antibody') and (
                    not ds or m['annotation_role'] in {'antigen', 'antigen_context'}):
                query_keys.add(k)
            scope_counts[m['sequence_kind']] += 1
    targets = target_index(official, sequences)
    union_targets = target_index(official, sequences, expected=True)
    # Include legacy reference chains for an explicit, incomplete hard-test
    # feasibility bound. Final strict novelty still requires both complete
    # development sets, generic screening, interfaces and prior-use history.
    refs = read(manifest['input_pins']['reference_sequences']['path'])
    external_keys = sorted({add_sequence(sequences, s) for s in refs['heldout']})
    query_keys.update(k for k in external_keys if not numbered.get(k))
    inputs = dict(sequences=sequences, targets=targets, union_targets=union_targets,
        query_keys=sorted(query_keys), external_keys=external_keys,
        component_splits=component_splits, official=official,
        inventory=str(args.inventory), screen=str(args.screen),
        scope_counts=dict(scope_counts), missing_numbering=missing,
        input_pins={str(p): digest(p) for p in [
            args.inventory / 'inventory-manifest.json', args.inventory / 'units.parquet',
            args.inventory / 'sequence-membership.parquet', args.inventory / 'sequences.fasta',
            args.inventory / 'numbering-cached.json', args.inventory / 'numbering-new.json',
            args.screen / 'numbering-verification.json',
            Path(manifest['input_pins']['sabdab_csv']['path']),
            Path(manifest['input_pins']['reference_sequences']['path'])]},
        original_input_pins=manifest['input_pins'])
    atomic_json(out / 'authority.json', authority.to_dict())
    atomic_json(out / 'registry.json', registry.to_dict())
    atomic_json(out / 'inputs.json', inputs)
    write_rows(out / 'units.parquet', specs)
    write_rows(out / 'scope-membership.parquet', retained)
    print(json.dumps(dict(units=len(specs), antigen_queries=len(query_keys),
        resolved_targets=len(targets), union_targets=len(union_targets),
        external_chains=len(external_keys), missing_numbering=len(missing))), flush=True)


def align(args):
    require_compute(threads=args.workers, allow_comp1400=True)
    data = read(args.root / 'inputs.json')
    out = args.root / ('align-' + args.scope + '-' + str(args.shard))
    if out.exists():
        raise FileExistsError('Alignment shard already exists')
    out.mkdir()
    keys = data['query_keys'][args.shard::args.shards]
    targets = data['targets' if args.scope == 'resolved' else 'union_targets']
    started = time.monotonic(); pairs = processed = hits = 0
    chunks = [keys[i:i+32] for i in range(0, len(keys), 32)]
    with gzip.open(out / 'hits.jsonl.gz', 'wt') as f:
        with ProcessPoolExecutor(args.workers, initializer=initialise,
                initargs=(data['sequences'], targets)) as pool:
            for i, r in enumerate(pool.map(compare_chunk, chunks)):
                pairs += r['pairs']; processed += r['queries']; hits += len(r['hits'])
                for row in r['hits']:
                    f.write(json.dumps(row) + '\n')
                if i % 25 == 0:
                    print(json.dumps(dict(processed=processed, total=len(keys), wall_s=time.monotonic()-started)), flush=True)
    atomic_json(out / 'summary.json', dict(status='complete', scope=args.scope,
        shard=args.shard, shards=args.shards, queries=processed, hit_queries=hits, pairs=pairs,
        wall_s=time.monotonic()-started, inputs_sha256=digest(args.root / 'inputs.json'),
        hits_sha256=digest(out / 'hits.jsonl.gz'), code_sha256=digest(__file__),
        slurm_job_id=os.environ['SLURM_JOB_ID']))


def collect(root, scope, inputs_sha):
    folders = sorted(root.glob('align-' + scope + '-*'))
    receipts = [read(p / 'summary.json') for p in folders]
    if not receipts or len({r['shards'] for r in receipts}) != 1:
        raise ValueError('Missing alignment receipts')
    n = receipts[0]['shards']
    if len(receipts) != n or {r['shard'] for r in receipts} != set(range(n)):
        raise ValueError('Incomplete or duplicate alignment shards')
    result = {}
    for p, r in zip(folders, receipts):
        if (r['status'] != 'complete' or r['scope'] != scope or r['inputs_sha256'] != inputs_sha or
                r['hits_sha256'] != digest(p / 'hits.jsonl.gz')):
            raise ValueError('Alignment provenance mismatch')
        with gzip.open(p / 'hits.jsonl.gz', 'rt') as f:
            for line in f:
                row = json.loads(line); k = row['sequence_key']
                if k in result:
                    raise ValueError('Duplicate sequence across shards')
                result[k] = row['hits']
    return result, receipts


def cdr_components(numbered, keys, official, workers):
    by_cdr = defaultdict(set); by_h3 = defaultdict(set)
    for r in official:
        for role in ['H', 'L']:
            c = r['CDR' + role + '123']
            if c:
                by_cdr[(role, c)].add(r['ab_ag_cluster'])
        if r['CDRH3']:
            by_h3[r['CDRH3']].add(r['ab_ag_cluster'])
    all_h3 = {published_cdr_string(domain(d['numbering']), (105, 117))
              for k in keys for d in numbered.get(k, [])
              if d['chain_type'] == 'H' and 'numbering' in d}
    all_h3.discard('')
    with ProcessPoolExecutor(workers) as pool:
        links = dict(pool.map(official_h3_links, ((q, sorted(by_h3)) for q in sorted(all_h3)), chunksize=16))
    out = {}
    for k in sorted(keys):
        cc = set()
        for d in numbered.get(k, []):
            if 'numbering' not in d:
                continue
            role = 'H' if d['chain_type'] == 'H' else 'L'
            cd = domain(d['numbering'])
            cc.update(by_cdr[(role, published_cdr_string(cd))])
            if role == 'H':
                for h in links.get(published_cdr_string(cd, (105, 117)), []):
                    cc.update(by_h3[h])
        out[k] = sorted(cc)
    return out


def family_guard(assignments, units, members):
    groups = defaultdict(set)
    for u in units.values():
        if u['source'] != 'sabdab':
            groups[u['unit_id']].update('family:' + u['source'] + ':' + g for g in u['source_groups'])
    for m in members:
        groups[m['unit_id']].add('exact:' + m['sequence_key'])
    return quarantine_group_conflicts(assignments, groups)


def general_holdout_guard(assignments, units, members, numbered, sequences, executable, output, workers):
    """Screen inherited general-protein splits without overriding SAb authority.

    The old 70/80 protein guard applies to unmapped general-protein records.
    Numbered Ig chains are excluded from this generic comparison. Mapped
    authoritative families keep their assigned role; conflicting unmapped
    records are quarantined. SAbDab train/validation is not subjected to a new
    stricter CDR or framework gate.
    """
    active = {a.unit_id: a for a in assignments if a.operational_role in {'train', 'validation'}}
    seq_units = defaultdict(set)
    for m in members:
        if m['unit_id'] in active and not numbered.get(m['sequence_key']) and not m['annotation_role'].startswith('antibody'):
            seq_units[m['sequence_key']].add(m['unit_id'])
    targets = {k: sequences[k] for k, uu in seq_units.items() if any(active[u].operational_role == 'validation' for u in uu)}
    queries = {k: sequences[k] for k, uu in seq_units.items() if any(active[u].operational_role == 'train' for u in uu)}
    hits = mmsearch(output, queries, targets, executable, workers)
    conflicts = set(); diagnostics = []
    for q, tt in hits.items():
        for t in tt:
            for qu in seq_units[q]:
                if active[qu].operational_role != 'train':
                    continue
                for tu in seq_units[t]:
                    if active[tu].operational_role != 'validation':
                        continue
                    qa, ta = active[qu], active[tu]
                    qfree = not qa.matched_components and units[qu]['declared_general']
                    tfree = not ta.matched_components and units[tu]['declared_general']
                    if qfree or tfree:
                        pinder = {a.unit_id for a in [qa, ta] if a.source == 'pinder'}
                        if pinder:
                            conflicts.update(pinder)
                        elif qfree:
                            conflicts.add(qu)
                        else:
                            conflicts.add(tu)
                    diagnostics.append(dict(train_unit=qu, validation_unit=tu,
                        query=q, target=t, action='quarantine_unmapped_general' if qfree or tfree else 'mapped_family_similarity_diagnostic'))
    diagnostics.sort(key=lambda r: (r['train_unit'], r['validation_unit'], r['query'], r['target']))
    return [replace(a, operational_role='quarantine', reason_codes=a.reason_codes + ('general_protein_holdout_70_80',))
            if a.unit_id in conflicts else a for a in assignments], diagnostics


def report(args):
    require_compute(threads=args.workers, allow_comp1400=True)
    started = time.monotonic(); root = args.root
    out = root / 'report'
    if out.exists():
        raise FileExistsError('Report already exists')
    out.mkdir()
    data = read(root / 'inputs.json'); input_sha = digest(root / 'inputs.json')
    for path, expected in data['input_pins'].items():
        if digest(path) != expected:
            raise ValueError('Input changed: ' + path)
    for key, pin in data['original_input_pins'].items():
        if digest(pin['path']) != pin['sha256']:
            raise ValueError('Original input changed: ' + key)
    authority, registry = SplitAuthority.load(root / 'authority.json'), SourceRegistry.load(root / 'registry.json')
    gh_union, receipts = collect(root, 'union', input_sha)
    resolved_targets = {(t['sequence_key'], t['antigen_type']): t for t in data['targets']}
    gh = {k: [dict(h, components=resolved_targets[(h['target'], h['antigen_type'])]['components'])
              for h in hh if (h['target'], h['antigen_type']) in resolved_targets]
          for k, hh in gh_union.items()}
    gh = {k: hh for k, hh in gh.items() if hh}
    if sum(r['queries'] for r in receipts) != len(data['query_keys']):
        raise ValueError('Query coverage mismatch')
    inventory = Path(data['inventory'])
    units = pq.read_table(root / 'units.parquet').to_pylist(); by_id = {u['unit_id']: u for u in units}
    members = pq.read_table(root / 'scope-membership.parquet').to_pylist()
    cached = read(inventory / 'numbering-cached.json'); numbered = cached['numbered']
    new = read(inventory / 'numbering-new.json'); numbered.update(new['numbered'])
    attempted = set(cached['attempted']) | set(new['numbered'])
    keys = {m['sequence_key'] for m in members} | set(data['external_keys'])
    cc = cdr_components(numbered, keys, data['official'], args.workers)
    atomic_json(out / 'sequence-component-evidence.json', dict(
        profile=PUBLISHED, cdr_components=cc,
        antigen_components={k: sorted({c for h in hh for c in h['components']}) for k, hh in gh.items()},
        authority_sha256=authority.sha256, inputs_sha256=input_sha))
    by_unit = defaultdict(list)
    for m in members:
        by_unit[m['unit_id']].append(m)
    rows = []
    for u in units:
        rows.append(Unit(**{k: u[k] for k in Unit.__dataclass_fields__}))
    # Standalone authoritative records inherit published membership. Extra
    # cross-role component links found at the declared ML scope quarantine;
    # they never rewrite the published component catalogue.
    own_by_id = {'sabdab:' + r['INSTANCE']: r for r in data['official']}
    evidence = []
    for u in rows:
        ms = by_unit[u.unit_id]; own = u.source == 'sabdab'
        ab = set(); ag = set(); incomplete = False
        for m in ms:
            k = m['sequence_key']
            if not own:
                ab.update(cc.get(k, []))
                if not numbering_assessed(k, m['annotation_role'], numbered, attempted):
                    incomplete = True
            if not own and k in data['query_keys']:
                ag.update(c for h in gh.get(k, []) for c in h['components'])
        if own:
            r = own_by_id[u.unit_id]
            # Official annotations, rather than ANARCI, are authoritative for
            # its own scope. VNAR metadata can be retained without inventing
            # a compatible external HKL numbering scheme.
            ab.add(u.native_component)
            if by_id[u.unit_id]['supervised']:
                ag.add(u.native_component)
        for axis, matches in [('antibody_CDR', ab), ('antigen', ag)]:
            evidence.append(Evidence(u.unit_id, PUBLISHED, axis, 'published_global_scope',
                'unassessed' if incomplete and axis == 'antibody_CDR' and not matches else
                ('hit' if matches else 'no_hit'), tuple(sorted(matches))))
        if incomplete and ab:
            evidence.append(Evidence(u.unit_id, PUBLISHED, 'antibody_CDR', 'additional_scope_unassessed', 'unassessed'))
    baseline = assign(rows, evidence, data['component_splits'], authority, registry)
    proposed = family_guard(baseline, by_id, members)
    original_manifest = read(inventory / 'inventory-manifest.json')
    executable = Path(original_manifest['runtime']) / 'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
    proposed, general_hits = general_holdout_guard(proposed, by_id, members, numbered,
        data['sequences'], executable, out / 'general-holdout', args.workers)
    if general_hits:
        write_rows(out / 'general-holdout-evidence.parquet', general_hits)
    write_rows(out / 'published-evidence.parquet', [asdict(e) for e in evidence])
    evidence_sha = digest(out / 'published-evidence.parquet')
    write_rows(out / 'component-assignments.parquet', [dict(asdict(a), evidence_sha256=evidence_sha) for a in baseline])
    write_rows(out / 'proposed-assignments.parquet', [dict(asdict(a), release_active=False,
        evidence_sha256=evidence_sha, source_native_id=by_id[a.unit_id]['source_native_id'],
        prepared_identity_established=False, scope_requires_preparation_check=True) for a in proposed])
    # Prove that changing ownership priority does not move any component role.
    changed = authority.to_dict(); changed['source_precedence'].reverse()
    alternate = assign(rows, evidence, data['component_splits'], SplitAuthority.from_dict(changed), registry)
    if [(a.unit_id, a.operational_role, a.reason_codes) for a in baseline] != [(a.unit_id, a.operational_role, a.reason_codes) for a in alternate]:
        raise ValueError('Source precedence changed roles')
    # Scope sensitivity changes reference antigen sequences only. The unit
    # input scope stays fixed; full-source query scope needs native checks.
    union_by_unit = defaultdict(set)
    for m in members:
        for h in gh_union.get(m['sequence_key'], []):
            union_by_unit[m['unit_id']].update(h['components'])
    union_evidence = [Evidence(e.unit_id, e.profile, e.axis, e.rule_id,
        'hit' if union_by_unit[e.unit_id] else 'no_hit', tuple(sorted(union_by_unit[e.unit_id])))
        if e.axis == 'antigen' else e for e in evidence]
    union_proposed = family_guard(assign(rows, union_evidence, data['component_splits'], authority, registry), by_id, members)
    # Compare scope inheritance before independent generic holdout quarantine,
    # so the reported effect is attributable to antigen reference scope alone.
    resolved_component_guard = family_guard(baseline, by_id, members)
    scope_changes = [dict(unit_id=a.unit_id, source=a.source,
        resolved_role=a.operational_role, union_role=b.operational_role)
        for a, b in zip(resolved_component_guard, union_proposed) if a.operational_role != b.operational_role]
    if scope_changes:
        write_rows(out / 'antigen-scope-sensitivity.parquet', scope_changes)
    # A sequence-only overlap bound, explicitly not a finished strict test.
    # Published component matches are already sufficient to reject candidates
    # under the adopted, at-least-as-strict hard-test sequence profile.
    scopes = defaultdict(set)
    for scope in ['train', 'validation']:
        for a in proposed:
            if a.operational_role == scope:
                scopes[scope].update(m['sequence_key'] for m in by_unit[a.unit_id])
    external = []
    splits = {c: authority.role(s) for c, s in data['component_splits'].items()}
    represented = defaultdict(set)
    for a in proposed:
        if a.operational_role in {'train', 'validation'}:
            represented[a.operational_role].update(a.matched_components)
    for k in data['external_keys']:
        ab = set(cc.get(k, [])); ag = {c for h in gh_union.get(k, []) for c in h['components']}
        checks = []
        for role in ['train', 'validation']:
            for axis, matches in [('antibody_CDR', ab), ('antigen', ag)]:
                hit = any(splits[c] == role and c in represented[role] for c in matches)
                checks.append((axis, role, 'hit' if hit else 'unassessed'))
            checks.append(('generic', role, 'hit' if k in scopes[role] else 'unassessed'))
            checks.append(('interface', role, 'unassessed'))
        external.append(dict(sequence_key=k, status=hard_test_status(checks),
            published_antibody_components=sorted(ab), published_antigen_components=sorted(ag),
            exact_train=k in scopes['train'], exact_validation=k in scopes['validation'],
            interface_status='unassessed', prior_use='unknown', scope='legacy_reference_chain_not_whole_complex'))
    write_rows(out / 'external-chain-feasibility.parquet', external)
    headers_path = Path(original_manifest['runtime']) / 'audits/ab-cdr-gap-v1/heldout.fasta'
    ext_by_key = {r['sequence_key']: r for r in external}
    complex_keys = defaultdict(set)
    for name, seq in fasta(headers_path).items():
        complex_keys[name.split('|')[0]].add(skey(seq))
    complex_evidence = []
    for cid, kk in sorted(complex_keys.items()):
        if not kk <= ext_by_key.keys():
            raise ValueError('Legacy heldout headers differ from frozen reference sequences')
        overlap = any(ext_by_key[k]['status'] == 'diagnostic_overlap' for k in kk)
        complex_evidence.append(dict(complex_id=cid, sequence_keys=sorted(kk),
            status='diagnostic_overlap' if overlap else 'pending_novelty_checks', prior_use='unknown',
            rejection_is_chain_evidence_only=True))
    write_rows(out / 'external-complex-feasibility.parquet', complex_evidence)
    counts = defaultdict(Counter); moves = defaultdict(Counter)
    for a in proposed:
        counts[a.source + '/' + a.native_source_split][a.operational_role] += 1
        if a.source != 'sabdab' and a.operational_role != a.native_source_split:
            moves[a.source][a.native_source_split + '->' + a.operational_role] += 1
    summary = dict(status='metadata_role_audit_complete_not_a_release', release_ready=False,
        authority=authority.to_dict(), authority_sha256=authority.sha256, registry_sha256=registry.sha256,
        units=len(proposed), source_role_counts={k: dict(v) for k, v in sorted(counts.items())},
        source_role_changes={k: dict(v) for k, v in sorted(moves.items())},
        family_or_exact_guard_quarantined=sum(a.operational_role != b.operational_role for a, b in zip(resolved_component_guard, baseline)),
        general_holdout_quarantined=sum('general_protein_holdout_70_80' in a.reason_codes for a in proposed),
        generic_holdout_pairs=len(general_hits), mmseqs_sha256=digest(executable),
        antigen_reference_scope_sensitivity=dict(changed_units=len(scope_changes),
            transitions=dict(Counter(r['resolved_role'] + '->' + r['union_role'] for r in scope_changes))),
        precedence_invariance_verified=True,
        alignment=dict(queries=len(data['query_keys']), resolved_targets=len(data['targets']), union_targets=len(data['union_targets']),
            pairs=sum(r['pairs'] for r in receipts), shard_receipts=receipts),
        external_chain_feasibility=dict(Counter(r['status'] for r in external)),
        external_complex_feasibility=dict(Counter(r['status'] for r in complex_evidence)),
        external_antigen_reference_scope='resolved_and_expected_union',
        external_headers_sha256=digest(headers_path),
        hard_test_eligible_complexes=None,
        no_current_release_mutation=True, no_new_teacher_or_features=True,
        original_input_hashes_verified=True,
        code_sha256=digest(__file__), evidence_sha256=evidence_sha,
        code_pins={str(p): digest(p) for p in sorted(
            list(Path(__file__).parent.glob('sabdab*.py')) +
            list((Path(__file__).parent / 'src/pkabench').glob('*.py')))},
        inputs_sha256=input_sha, slurm_job_id=os.environ['SLURM_JOB_ID'],
        limitations=[
            'Proposed metadata roles are not active training memberships.',
            'Existing source inventories are not verified prepared-coordinate scopes except identified native Fv members.',
            'No-HKL numbering is not proof of non-antibody identity; unannotated sequences are screened as context.',
            'Same-parent source overlap is not prepared-context equivalence; no duplicates have been merged.',
            'External complex exclusions aggregate known chain overlaps; remaining complexes are unassessed, not proven novel.',
            'External candidates still require strict CDR/generic checks against all final development inputs, interface evidence and prior-use provenance.',
            'Current decisions use resolved reference antigens; expected-antigen sensitivity is retained as evidence, not as input scope.',
            'Augmentation donor and final loader memberships are not yet audited.',
            'No hard-test membership has been frozen and no model has been trained.'],
        wall_s=time.monotonic()-started)
    atomic_json(out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['source_role_counts', 'source_role_changes', 'family_or_exact_guard_quarantined', 'general_holdout_quarantined', 'external_complex_feasibility']}, indent=2), flush=True)


def verify(args):
    """Check saved artifacts independently of the role-decision implementation."""
    require_compute(threads=args.workers, allow_comp1400=True)
    root = args.root; out = root / 'report'
    if (out / 'verification.json').exists():
        raise FileExistsError('Verification already exists')
    data = read(root / 'inputs.json'); summary = read(out / 'summary.json')
    aa = pq.read_table(out / 'proposed-assignments.parquet').to_pylist()
    baseline = pq.read_table(out / 'component-assignments.parquet').to_pylist()
    ee = pq.read_table(out / 'published-evidence.parquet').to_pylist()
    members = pq.read_table(root / 'scope-membership.parquet').to_pylist()
    assert len(aa) == len({a['unit_id'] for a in aa}) == len(baseline) == summary['units']
    assert summary['release_ready'] is False and summary['no_current_release_mutation'] is True
    role_map = {'train': 'train', 'test': 'validation'}
    splits = {c: role_map[s] for c, s in data['component_splits'].items()}
    expected = defaultdict(set); unassessed = set()
    for e in ee:
        expected[e['unit_id']].update(splits[c] for c in e['components'])
        if e['status'] == 'unassessed':
            unassessed.add(e['unit_id'])
    by_id = {a['unit_id']: a for a in aa}
    groups = defaultdict(set)
    units = {u['unit_id']: u for u in pq.read_table(root / 'units.parquet').to_pylist()}
    evidence_sha = digest(out / 'published-evidence.parquet')
    for a in aa:
        assert a['release_active'] is False and a['prepared_identity_established'] is False
        assert a['authority_sha256'] == summary['authority_sha256']
        assert a['registry_sha256'] == summary['registry_sha256']
        assert a['evidence_sha256'] == evidence_sha
        assert a['native_source_split'] == units[a['unit_id']]['native_split']
        assert a['source'] == units[a['unit_id']]['source']
        if a['operational_role'] in {'train', 'validation'}:
            assert a['unit_id'] not in unassessed
            assert not expected[a['unit_id']] or expected[a['unit_id']] == {a['operational_role']}
            if a['source'] == 'sabdab':
                assert a['operational_role'] == role_map[a['native_source_split']]
            for g in units[a['unit_id']]['source_groups'] if a['source'] != 'sabdab' else []:
                groups[(a['source'], g)].add(a['operational_role'])
    exact = defaultdict(set)
    for m in members:
        a = by_id[m['unit_id']]
        if a['operational_role'] in {'train', 'validation'}:
            exact[m['sequence_key']].add(a['operational_role'])
    assert all(len(v) <= 1 for v in groups.values())
    assert all(len(v) <= 1 for v in exact.values())
    gh, receipts = collect(root, 'union', digest(root / 'inputs.json'))
    for row in pq.read_table(out / 'external-chain-feasibility.parquet').to_pylist():
        assert row['published_antigen_components'] == sorted({c for h in gh.get(row['sequence_key'], []) for c in h['components']})
    for r in receipts:
        assert r['queries'] == len(data['query_keys'][r['shard']::r['shards']])
    # Independent traceback count: compare matched characters in ungapped
    # alignment blocks rather than reusing Alignment.counts().identities.
    import random
    from Bio.Align import PairwiseAligner
    sample = random.Random(20261122).sample(sorted(gh), min(128, len(gh)))
    aligner = PairwiseAligner(mode='global', match_score=1, mismatch_score=0,
                             open_gap_score=-2, extend_gap_score=-1)
    comparisons = 0
    target_lookup = {(t['sequence_key'], t['antigen_type']): t for t in data['union_targets']}
    for k in sample:
        for h in gh[k][:2]:
            q, t = data['sequences'][k], data['sequences'][h['target']]
            alignment = aligner.align(q, t)[0]
            matches = sum(sum(x == y for x, y in zip(q[qa:qb], t[ta:tb]))
                for (qa, qb), (ta, tb) in zip(*alignment.aligned))
            ident = matches / max(len(q), len(t))
            assert abs(ident - h['identity']) < 1e-12
            assert ident >= ({'PROTEIN': .4, 'PEPTIDE': .8}[h['antigen_type']]) - 1e-12
            assert h['components'] == target_lookup[(h['target'], h['antigen_type'])]['components']
            comparisons += 1
    for path, expected_sha in data['input_pins'].items():
        assert digest(path) == expected_sha
    for pin in data['original_input_pins'].values():
        assert digest(pin['path']) == pin['sha256']
    report = dict(status='passed', release_ready=False,
        units=len(aa), original_input_hashes_unchanged=True,
        published_component_roles_verified=True, native_source_splits_preserved=True,
        active_candidates_have_no_unassessed_published_axes=True,
        family_and_exact_cross_role_conflicts=0,
        antigen_tracebacks_verified=comparisons,
        precedence_invariance_verified=summary['precedence_invariance_verified'],
        artifact_pins={p.name: digest(p) for p in sorted(out.glob('*.parquet'))},
        summary_sha256=digest(out / 'summary.json'),
        code_sha256=digest(__file__), slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(out / 'verification.json', report)
    print(json.dumps(report, indent=2), flush=True)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    for name in ['inventory', 'screen', 'authority', 'registry', 'output']:
        q.add_argument('--' + name, type=Path, required=True)
    q.add_argument('--workers', type=int, default=4)
    q = sub.add_parser('align')
    q.add_argument('--root', type=Path, required=True)
    q.add_argument('--scope', choices=['resolved', 'union'], default='resolved')
    q.add_argument('--shard', type=int, default=0); q.add_argument('--shards', type=int, default=1)
    q.add_argument('--workers', type=int, default=32)
    for action in ['report', 'verify']:
        q = sub.add_parser(action)
        q.add_argument('--root', type=Path, required=True); q.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    if args.action == 'align' and not 0 <= args.shard < args.shards:
        p.error('shard must be in [0, shards)')
    {'prepare': prepare, 'align': align, 'report': report, 'verify': verify}[args.action](args)


if __name__ == '__main__':
    cli()
