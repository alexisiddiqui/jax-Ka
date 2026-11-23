"""Bounded reconstruction of conflict evidence; reuse full antigen alignments."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, digest
from sabdab_pool_overlap import domain, official_h3_links, published_cdr_string
from sabdab_v7_antigen import aligner, identity
from sabdab_v7_exposure import read, skey
from sabdab_v7_roles import collect, write_rows


def reconstruct_edges(ctx, output, focus, workers):
    data = ctx['data']; members = [m for m in ctx['members'] if m['unit_id'] in focus]
    keys = {m['sequence_key'] for m in members}
    cached = read(ctx['root'] / 'report/sequence-component-evidence.json')['cdr_components']
    cdr_keys = {k for k in keys if cached.get(k)}
    cdr_refs = defaultdict(lambda: defaultdict(set)); h3_refs = defaultdict(lambda: defaultdict(set))
    ag_refs = defaultdict(set)
    for r in data['official']:
        uid = 'sabdab:' + r['INSTANCE']; comp = r['ab_ag_cluster']
        for role in ['H', 'L']:
            var = r['V' + role + '_numerable_seq']
            if not var:
                continue
            ref = (comp, skey(var))
            if r['CDR' + role + '123']:
                cdr_refs[(role, r['CDR' + role + '123'])][ref].add(uid)
            if role == 'H' and r['CDRH3']:
                h3_refs[r['CDRH3']][ref].add(uid)
        if r['agchains']:
            for typ, seq in zip(r['agtypes'].split('/'), r['agresolvedseqs'].split('/')):
                if typ in {'PROTEIN', 'PEPTIDE'} and seq:
                    ag_refs[(skey(seq), typ, comp)].add(uid)
    domains = {}
    for k in sorted(cdr_keys):
        for i, d in enumerate(ctx['numbered'].get(k, [])):
            if 'numbering' in d:
                domains[(k, i)] = (d, domain(d['numbering']))
    queries = sorted({published_cdr_string(cd, (105, 117)) for d, cd in domains.values() if d['chain_type'] == 'H'} - {''})
    refs = sorted(h3_refs)
    with ProcessPoolExecutor(workers) as pool:
        links = dict(pool.map(official_h3_links, ((q, refs) for q in queries), chunksize=16))
    edges = []; traced = {'antibody_CDR': defaultdict(set), 'antigen': defaultdict(set)}
    component_roles = {c: {'train': 'train', 'test': 'validation'}[s] for c, s in data['component_splits'].items()}
    a = aligner()

    def append(k, axis, rule, comp, target, reference_ids, score, qlen, tlen, **extra):
        traced[axis][k].add(comp)
        edges.append(dict(sequence_key=k, axis=axis, rule_id=rule, status='hit',
            target_component=comp, target_role=component_roles[comp],
            target_sequence_key=target, reference_ids=sorted(reference_ids),
            identity=float(score), query_fragment_length=qlen, target_fragment_length=tlen,
            qcov=None, tcov=None, coverage_rule='global_alignment_no_minimum_coverage',
            reference_scope='published_CDR_or_resolved_antigen',
            identity_kind='sequence', prepared_context_identity='not_established', **extra))

    for (k, i), (d, cd) in sorted(domains.items()):
        role = 'H' if d['chain_type'] == 'H' else 'L'
        query = published_cdr_string(cd)
        for (comp, target), ids in sorted(cdr_refs.get((role, query), {}).items()):
            append(k, 'antibody_CDR', 'published_CDR' + role + '123_100', comp, target, ids,
                1., len(query), len(query), domain_id=k + 'd' + str(i),
                domain_role=role, domain_start=d.get('start'), domain_end=d.get('end'), antigen_type=None)
        if role == 'H':
            query = published_cdr_string(cd, (105, 117))
            for refseq in links.get(query, []):
                score = identity(query, refseq, a)
                for (comp, target), ids in sorted(h3_refs[refseq].items()):
                    append(k, 'antibody_CDR', 'published_CDRH3_global95', comp, target, ids,
                        score, len(query), len(refseq), domain_id=k + 'd' + str(i),
                        domain_role='H', domain_start=d.get('start'), domain_end=d.get('end'), antigen_type=None)
    union_hits, _ = collect(ctx['root'], 'union', digest(ctx['root'] / 'inputs.json'))
    targets = {(t['sequence_key'], t['antigen_type']): t for t in data['targets']}
    query_keys = set(data['query_keys'])
    for k in sorted(keys & query_keys):
        for h in union_hits.get(k, []):
            target = targets.get((h['target'], h['antigen_type']))
            if target is None:
                continue
            for comp in target['components']:
                append(k, 'antigen', 'published_global_' + ('40' if h['antigen_type'] == 'PROTEIN' else '80'),
                    comp, h['target'], ag_refs[(h['target'], h['antigen_type'], comp)], h['identity'],
                    len(data['sequences'][k]), len(data['sequences'][h['target']]),
                    domain_id=None, domain_role=None, domain_start=None, domain_end=None,
                    antigen_type=h['antigen_type'])
    write_rows(output / 'sequence-component-edges.parquet', edges)
    member_keys = defaultdict(set)
    for m in members:
        member_keys[m['unit_id']].add(m['sequence_key'])
    verified = 0
    for row in pq.read_table(ctx['root'] / 'report/published-evidence.parquet').to_pylist():
        uid = row['unit_id']
        if uid not in focus or ctx['units'][uid]['source'] == 'sabdab':
            continue
        observed = {c for k in member_keys[uid] for c in traced[row['axis']].get(k, set())}
        if observed != set(row['components']):
            raise ValueError('Cannot reconstruct component evidence: ' + uid + '/' + row['axis'])
        verified += 1
    # Keep exact generic search scores; the old pair sidecar only retained
    # unit IDs and sequence keys. Scores are recoverable from the pinned TSV.
    pair_rows = []
    hits_path = ctx['root'] / 'report/general-holdout/hits.tsv'
    for line in hits_path.read_text().splitlines():
        q, t, ident, qc, tc = line.split('\t')
        if float(ident) >= .7 and min(float(qc), float(tc)) >= .8:
            pair_rows.append(dict(query_sequence_key=q, target_sequence_key=t,
                identity=float(ident), qcov=float(qc), tcov=float(tc),
                rule_id='generic_70_80_both', prepared_context_identity='not_established'))
    if pair_rows:
        write_rows(output / 'generic-sequence-edges.parquet', pair_rows)
    authoritative = [dict(unit_id=u['unit_id'], native_component=u['native_component'],
        native_source_split=u['native_split'], evidence_kind='published_assignment')
        for u in ctx['units'].values() if u['source'] == 'sabdab' and u['unit_id'] in focus]
    if authoritative:
        write_rows(output / 'authoritative-assignment-edges.parquet', authoritative)
    atomic_json(output / 'edge-verification.json', dict(status='passed',
        native_unit_axes_reconstructed=verified, sequence_component_edges=len(edges),
        positive_CDR_queries_reconstructed=len(cdr_keys), generic_edges=len(pair_rows),
        full_antigen_alignments_recomputed=False,
        input_sha256=digest(ctx['root'] / 'inputs.json'), code_sha256=digest(__file__)))
