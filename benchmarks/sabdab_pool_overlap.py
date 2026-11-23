"""Read-only, antigen-aware SAbDab2/current-pool overlap audit.

Run inside a Slurm CPU allocation with --mem-per-cpu=2G. This produces evidence,
not a new training split. IMGT identities use the existing pool-v6c definition;
published SAbDab component IDs and PINDER cluster IDs are separate evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import pyarrow.dataset as ds

CDRS = {'cdr1': (27, 38), 'cdr2': (56, 65), 'cdr3': (105, 117)}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def key(sequence):
    return 's' + hashlib.sha256(sequence.encode()).hexdigest()[:20]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_rows(path, values):
    values = list(values)
    columns = list(dict.fromkeys(k for value in values for k in value))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, columns, delimiter='\t')
        w.writeheader()
        for value in values:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                        for k, v in value.items()})


def domain(numbering):
    """Ignore gaps, preserve insertion codes, trim spaces consistently."""
    return {(int(p), str(i).strip()): a for p, i, a in numbering if a and a != '-'}


def signature(d, region=None):
    lo, hi = region or (5, 123)
    return tuple((p, i, a) for (p, i), a in sorted(d.items()) if lo <= p <= hi)


def cdr_string(d):
    return ''.join(a for name, (lo, hi) in CDRS.items()
                   for (p, _), a in sorted(d.items()) if lo <= p <= hi)


def published_cdr_string(d, region=None):
    # ANARCI and SAbDab arrays preserve biological residue order. At IMGT 112,
    # insertion codes run in reverse; sorting those codes changes the sequence.
    return ''.join(a for (p, _), a in d.items()
                   if (region[0] <= p <= region[1] if region else
                       any(lo <= p <= hi for lo, hi in CDRS.values())))


def compare(queries, references):
    """All-pair IMGT scores; no nearest-neighbour search sensitivity cutoff."""
    positions = sorted(set().union(*(set(d) for d in queries + references)))
    pi = {p: i for i, p in enumerate(positions)}

    def matrix(domains):
        a = np.zeros((len(domains), len(positions)), dtype=np.uint8)
        for i, d in enumerate(domains):
            for p, aa in d.items():
                a[i, pi[p]] = ord(aa)
        return a

    q, r = matrix(queries), matrix(references)
    masks = {
        'cdr3': np.array([105 <= p <= 117 for p, _ in positions]),
        'cdrs': np.array([any(lo <= p <= hi for lo, hi in CDRS.values()) for p, _ in positions]),
        'variable': np.array([5 <= p <= 123 for p, _ in positions]),
    }
    out = {}
    for name, mask in masks.items():
        qm, rm = q[:, mask], r[:, mask]
        denom = np.maximum(np.count_nonzero(qm, axis=1)[:, None], np.count_nonzero(rm, axis=1)[None, :])
        if name == 'cdrs':
            # Match the pinned v5 function: sum per-loop maximum lengths.
            # max(sum(query_lengths),sum(ref_lengths)) differs when one loop
            # is longer in the query and another is longer in the reference.
            denom = np.zeros_like(denom)
            for lo, hi in CDRS.values():
                part = np.array([lo <= p <= hi for p, _ in positions])
                denom += np.maximum(np.count_nonzero(q[:,part],axis=1)[:,None],
                                    np.count_nonzero(r[:,part],axis=1)[None,:])
        scores = np.zeros((len(q), len(r)), dtype=np.float32)
        for start in range(0, len(q), 48):
            z = qm[start:start+48]
            matches = ((z[:, None, :] == rm[None, :, :]) & (z[:, None, :] != 0)).sum(axis=2)
            scores[start:start+48] = matches / np.maximum(denom[start:start+48], 1)
        out[name] = scores
    return out


def mmseqs_hits(root, queries, references, executable, threads):
    root.mkdir(exist_ok=True)
    if not queries or not references:
        return {}
    for name, data in [('q', queries), ('t', references)]:
        (root / (name + '.fasta')).write_text(''.join(f'>{k}\n{cdr_string(d)}\n' for k, d in data.items()))
    cmd = [str(executable), 'easy-search', str(root/'q.fasta'), str(root/'t.fasta'),
           str(root/'hits.tsv'), str(root/'tmp'), '--min-seq-id', '0.7', '-c', '0.8',
           '--cov-mode', '0', '--alignment-mode', '3', '--max-seqs', '10000', '-s', '7.5',
           '--threads', str(threads), '--format-output', 'query,target,fident,qcov,tcov']
    with (root/'search.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    out = defaultdict(set)
    for line in (root/'hits.tsv').read_text().splitlines():
        q, t, identity, qc, tc = line.split('\t')
        if float(identity) >= .7 and min(float(qc), float(tc)) >= .8:
            out[q].add(t.rsplit('d', 1)[0])
    return out


def official_h3_links(item):
    """Published NW identity: exact matches / aligned span on longer sequence."""
    from Bio.Align import PairwiseAligner
    q, refs = item
    aligner = PairwiseAligner(mode='global', match_score=1, mismatch_score=0,
                              open_gap_score=-2, extend_gap_score=-1)
    hits = []
    for target in refs:
        if min(len(q), len(target)) / max(len(q), len(target), 1) < .95:
            continue
        aln = aligner.align(q, target)[0]
        # Global alignment covers each complete sequence: use the longer span.
        ident = aln.counts().identities / max(len(q), len(target))
        if ident >= .95 - 1e-12:
            hits.append(target)
    return q, hits


def run(args):
    started = time.monotonic()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_MEM_PER_CPU') != '2048':
        raise RuntimeError('Use a real Slurm allocation with --mem-per-cpu=2G')
    base, audit, runtime = args.baseline, args.audit, args.runtime
    out = audit/'results'
    out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():
        raise FileExistsError('Completed audit exists; choose a new audit directory')
    source = audit/'inputs/metadata/splits_final/abag_split.csv'
    with source.open() as f:
        all_sab = list(csv.DictReader(f))
    sab = [r for r in all_sab if r['agchains'].strip()]
    assert len(all_sab) == 16415 and len(sab) == 10878, 'Release count differs'
    assert {r['ab_ag_split'] for r in sab} <= {'train', 'test'}
    print('metadata', len(all_sab), 'with_antigen', len(sab), flush=True)
    pools = {name: rows(base/'pool'/f'{name}.tsv') for name in ['pinder', 'pinder-val', 'pkpdb', 'pkpdb-val']}
    pool_pdbs = {name: {r['pdb_id'][:4].lower() for r in rr} for name, rr in pools.items()}
    seqs = read(base/'work/screen-sequences.json')
    raw = read(base/'work/numbered.json')['numbered']
    # Old caches retain all CDRs but not framework numbering. Enrich separately,
    # asserting exact reproduction of every cached CDR before using the result.
    all_sequences = set(seqs['validation']) | set(seqs['heldout']) | set(seqs['queries'].values())
    needed = {key(s): s for s in all_sequences
              if any('numbering' not in d for d in raw.get(key(s), []))}
    if needed:
        source_enrich = out/'reference-numbering-input.json'
        output_enrich = out/'reference-numbering-enriched.json'
        write_json(source_enrich, needed)
        helper = Path(__file__).with_name('sabdab_reference_numbering.py')
        if not output_enrich.exists() or not set(needed).issubset(read(output_enrich)['numbered']):
            subprocess.run([str(args.anarci_python), str(helper), str(source_enrich), str(output_enrich),
                            '--workers', str(args.workers)], check=True)
        enriched = read(output_enrich)['numbered']
        for sk in needed:
            for d in raw[sk]:
                if 'numbering' in d:
                    continue
                matches = [e for e in enriched[sk] if all(e[k] == d[k] for k in ['chain_type','start','end'])]
                if len(matches) != 1:
                    raise ValueError('Reference numbering changed domain boundaries: '+sk)
                numbering = matches[0]['numbering']
                for c, (lo, hi) in CDRS.items():
                    old = domain(d[c]); new = domain([(p,i,a) for p,i,a in numbering if lo <= p <= hi])
                    if old != new:
                        raise ValueError('Reference numbering changed cached '+c+': '+sk)
                d['numbering'] = numbering
    pairs = read(base/'work/reference-pairs.json')['pairs']
    refs, ref_scope, seqmap = {}, defaultdict(set), {}
    for scope in ['validation', 'heldout']:
        for sequence in seqs[scope]:
            sk = key(sequence)
            seqmap[sk] = sequence
            for i, d in enumerate(raw.get(sk, [])):
                if d['chain_type'] not in {'H', 'K', 'L'} or not d['cdr3']:
                    continue
                dk = sk+'d'+str(i)
                refs[dk] = {'sequence_key': sk, 'role': 'H' if d['chain_type'] == 'H' else 'L',
                            'domain': domain(d['numbering'])}
                ref_scope[dk].add(scope)
    fvroot = runtime/'pretraining/pinder-pkai-fv-v1'
    fv_rows = {}
    fv_keys = defaultdict(list)
    instance_aliases = defaultdict(list)
    for scope, name in [('current_train', 'pinder'), ('current_validation', 'pinder-val')]:
        for r in pools[name]:
            if r.get('unit') != 'Fv':
                continue
            meta = read(fvroot/'entries'/r['id']/'structure.json')
            fv_rows[r['id']] = dict(r, scope=scope)
            role_refs = meta['assembly']['references']
            for role in ['H', 'L']:
                sequence = role_refs[role]['sequence']
                for i, d in enumerate(role_refs[role]['domains']):
                    if ('H' if d['chain_type'] == 'H' else 'L') != role or not d['cdr3']:
                        continue
                    dk = key(sequence)+'d'+str(i)
                    numbered = d.get('numbering')
                    if numbered is None:
                        matched = [e for e in raw[key(sequence)]
                                   if all(e[k] == d[k] for k in ['chain_type','start','end'])]
                        if len(matched) != 1:
                            raise ValueError('Ambiguous Fv domain: '+r['id'])
                        numbered = matched[0]['numbering']
                    refs.setdefault(dk, {'sequence_key': key(sequence), 'role': role, 'domain': domain(numbered)})
                    ref_scope[dk].add(scope)
                    fv_keys[r['id']].append((role, dk))
            assembly = meta['assembly']
            own = meta['chosen_own_family']
            chains = {own: assembly.get('RCSB_own_chain'), 'L' if own == 'H' else 'H': assembly.get('RCSB_partner_chain')}
            if all(chains.values()):
                instance_aliases[(r['pdb_id'], chains['H'], chains['L'])].append((scope, r['id']))
    query_domains, query_key, query_rows = {}, {}, []
    for r in sab:
        for role, col in [('H', 'VH_numbering_list'), ('L', 'VL_numbering_list')]:
            if not r[col]:
                continue
            d = domain([(p, i, aa) for aa, (p, i) in json.loads(r[col])])
            if not d:
                continue
            # VNAR has a distinct domain/numbering convention: retain, do not compare as VH.
            if r['type'] == 'VNAR':
                continue
            sig = tuple(sorted(d.items()))
            qk = role+hashlib.sha256(repr(sig).encode()).hexdigest()[:20]
            query_domains[qk] = dict(role=role, domain=d)
            query_key[(r['INSTANCE'], role)] = qk
    scores, hitmaps, nearest = {}, {}, {}
    for role in ['H', 'L']:
        qks = sorted(k for k, d in query_domains.items() if d['role'] == role)
        rks = sorted(k for k, d in refs.items() if d['role'] == role)
        print('compare', role, len(qks), len(rks), flush=True)
        score = compare([query_domains[k]['domain'] for k in qks], [refs[k]['domain'] for k in rks])
        for scope in ['validation', 'heldout', 'current_train', 'current_validation']:
            indexes = [i for i, k in enumerate(rks) if scope in ref_scope[k]]
            for i, qk in enumerate(qks):
                hits = [rks[j] for j in indexes if score['cdr3'][i, j] >= .90-1e-7 or score['cdrs'][i, j] >= .80-1e-7]
                hitmaps[(scope, qk)] = {refs[k]['sequence_key'] for k in hits}
                detail = {}
                for metric in ['cdr3', 'cdrs', 'variable']:
                    if indexes:
                        j = indexes[int(np.argmax(score[metric][i, indexes]))]
                        detail[metric] = float(score[metric][i, j])
                        detail[metric+'_reference'] = rks[j]
                    else:
                        detail[metric] = None
                detail['cdr_hit_chain_keys'] = sorted(hitmaps[(scope, qk)])
                nearest[(scope, qk)] = detail
        scores[role] = (qks, rks, score)
    # Paired sequence similarities always refer to the same verified antibody.
    pair_domains = defaultdict(list)
    by_chain = defaultdict(list)
    for rk, ref in refs.items():
        by_chain[(ref['sequence_key'],ref['role'])].append(rk)
    for scope in ['validation','heldout']:
        for pair in pairs[scope]:
            hs, ls = by_chain[(pair['H'],'H')], by_chain[(pair['L'],'L')]
            if len(hs) == len(ls) == 1:
                pair_domains[scope].append((pair['id'],hs[0],ls[0]))
    for fid, kk in fv_keys.items():
        hs, ls = [k for role,k in kk if role == 'H'], [k for role,k in kk if role == 'L']
        if len(hs) == len(ls) == 1:
            pair_domains[fv_rows[fid]['scope']].append((fid,hs[0],ls[0]))
    qindexes = {role: {k:i for i,k in enumerate(scores[role][0])} for role in ['H','L']}
    rindexes = {role: {k:i for i,k in enumerate(scores[role][1])} for role in ['H','L']}
    ref_signatures = {k:signature(v['domain']) for k,v in refs.items()}
    query_signatures = {k:signature(v['domain']) for k,v in query_domains.items()}
    pair_arrays = {}
    exact_fv_index = defaultdict(list)
    for scope, pp in pair_domains.items():
        pair_arrays[scope] = dict(
            H=np.array([rindexes['H'][hk] for _,hk,_ in pp],dtype=int),
            L=np.array([rindexes['L'][lk] for _,_,lk in pp],dtype=int),
            nH=np.array([len(ref_signatures[hk]) for _,hk,_ in pp]),
            nL=np.array([len(ref_signatures[lk]) for _,_,lk in pp]))
        if scope.startswith('current_'):
            for fid,hk,lk in pp:
                exact_fv_index[(scope,ref_signatures[hk],ref_signatures[lk])].append(fid)

    def best_pair(hq,lq,scope):
        if not hq or not lq or not pair_domains[scope]:
            return None
        a = pair_arrays[scope]
        h = scores['H'][2]['variable'][qindexes['H'][hq],a['H']]
        l = scores['L'][2]['variable'][qindexes['L'][lq],a['L']]
        nh = np.maximum(len(query_signatures[hq]),a['nH'])
        nl = np.maximum(len(query_signatures[lq]),a['nL'])
        identity = (h*nh+l*nl)/(nh+nl)
        best = int(np.argmax(identity))
        return dict(reference=pair_domains[scope][best][0],identity=float(identity[best]),
                    H=float(h[best]),L=float(l[best]))
    # Carry forward the separate legacy held-out concat-CDR gate, unchanged.
    executable = runtime/'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
    for role in ['H', 'L']:
        mm = mmseqs_hits(out/('heldout-'+role),
            {k: v['domain'] for k, v in query_domains.items() if v['role'] == role},
            {k: v['domain'] for k, v in refs.items() if v['role'] == role and 'heldout' in ref_scope[k]},
            executable, args.workers)
        for qk, hits in mm.items():
            hitmaps[('heldout', qk)].update(hits)
    print('sequence comparisons complete', flush=True)
    # Official clusters are propagated only by matching antibody instances or
    # actual published CDR base-cluster thresholds, never by a parent PDB alone.
    scope_cdrs = {s: {r: set() for r in ['H', 'L']} for s in ['validation', 'heldout', 'current_train', 'current_validation']}
    scope_h3 = {s: set() for s in scope_cdrs}
    for rk, ref in refs.items():
        for scope in ref_scope[rk]:
            scope_cdrs[scope][ref['role']].add(published_cdr_string(ref['domain']))
            if ref['role'] == 'H':
                scope_h3[scope].add(published_cdr_string(ref['domain'], (105,117)))
    h3_queries = sorted({r['CDRH3'] for r in sab if r['CDRH3'] and r['type'] != 'VNAR'})
    h3_refs = sorted(set().union(*scope_h3.values()))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        h3_links = dict(pool.map(official_h3_links, ((q, h3_refs) for q in h3_queries), chunksize=24))
    official_clusters = defaultdict(set)
    official_seeds = defaultdict(list)
    for r in sab:
        inst = r['INSTANCE']; pdb = r['PDB_ID'][-4:].lower()
        direct = instance_aliases.get((pdb, r['Hchain'], r['Lchain']), [])
        for scope in scope_cdrs:
            causes = []
            if any(s == scope for s, _ in direct):
                causes.append('verified_instance_chain_alias')
            if r['type'] != 'VNAR':
                if r['CDRH123'] and r['CDRH123'] in scope_cdrs[scope]['H']:
                    causes.append('published_CDRH123_100pct')
                if r['CDRL123'] and r['CDRL123'] in scope_cdrs[scope]['L']:
                    causes.append('published_CDRL123_100pct')
                if set(h3_links.get(r['CDRH3'], [])).intersection(scope_h3[scope]):
                    causes.append('published_CDRH3_95pct')
            if causes:
                official_clusters[scope].add(r['ab_ag_cluster'])
                official_seeds[scope].append(dict(instance=inst, component=r['ab_ag_cluster'], causes=causes))
    print('official component links complete', flush=True)
    # Map PINDER interface-cluster evidence by exact deposited author chain pairs.
    index_root = runtime/'audits/pinder-index-v1'
    pdbs = sorted({r['PDB_ID'][-4:].lower() for r in sab})
    index = ds.dataset(index_root/'index.parquet').to_table(
        columns=['id','pdb_id','cluster_id','chain_R','chain_L','contains_antibody'],
        filter=ds.field('pdb_id').isin(pdbs) & (ds.field('contains_antibody') == True)).to_pylist()
    ids = [r['id'] for r in index]
    metadata = ds.dataset(index_root/'metadata.parquet').to_table(
        columns=['id','pdb_strand_id_R','pdb_strand_id_L'], filter=ds.field('id').isin(ids)).to_pylist()
    auth = {r['id']: r for r in metadata}
    by_pdb = defaultdict(list)
    for r in index:
        m = auth.get(r['id'], {})
        a, b = m.get('pdb_strand_id_R'), m.get('pdb_strand_id_L')
        if a and b:
            by_pdb[r['pdb_id']].append((r['id'], r['cluster_id'], str(a).strip(), str(b).strip()))
    validation_groups = set()
    for r in pools['pinder-val']:
        if r['stratum'] == 'Ab/Ag':
            validation_groups.add(r['group'])
            if r.get('source_groups'):
                validation_groups.update(json.loads(r['source_groups']))
    train_groups = set()
    for r in pools['pinder']:
        if r['stratum'] == 'Ab/Ag':
            train_groups.add(r['group'])
            if r.get('source_groups'):
                train_groups.update(json.loads(r['source_groups']))
    all_matches = []
    for r in sab:
        inst, pdb = r['INSTANCE'], r['PDB_ID'][-4:].lower()
        ab = {c for c in [r['Hchain'], r['Lchain']] if c}
        ag = set(r['agchains'].split('/'))
        matches = [(pid, g) for pid, g, a, b in by_pdb[pdb]
                   if (a in ab and b in ag) or (b in ab and a in ag)]
        groups = {g for _, g in matches}
        item = dict(instance=inst, pdb_id=pdb, split=r['ab_ag_split'], antibody_type=r['type'],
                    antigen_types=r['agtypes'], sabdab_id=r['SABDAB_ID'], component=r['ab_ag_cluster'],
                    pinder_interface_mapped=bool(matches), pinder_clusters=sorted(groups),
                    pinder_validation_cluster_hit=bool(groups & validation_groups),
                    pinder_training_cluster_hit=bool(groups & train_groups))
        for name, pset in pool_pdbs.items():
            item[name+'_parent_present'] = pdb in pset
        for scope in scope_cdrs:
            hq, lq = query_key.get((inst,'H')), query_key.get((inst,'L'))
            hh = hitmaps.get((scope, hq), set()); lh = hitmaps.get((scope, lq), set())
            verified = pairs.get(scope, [])
            paired = [p['id'] for p in verified if p['H'] in hh and p['L'] in lh]
            exact = []
            if scope in ['validation','heldout']:
                for role in ['H','L']:
                    for name in [role+'seq', role+'seq_expected', 'V'+role+'_numerable_seq']:
                        if r[name] and r[name] in seqs[scope]:
                            exact.append(role)
            item[scope+'_heavy_cdr_hit'] = bool(hh)
            item[scope+'_light_cdr_hit'] = bool(lh)
            item[scope+'_paired_cdr_hit'] = bool(paired)
            item[scope+'_exact_chain_hit'] = sorted(set(exact))
            item[scope+'_legacy_policy_hit'] = bool(hh or paired or exact)
            item[scope+'_official_component_hit'] = r['ab_ag_cluster'] in official_clusters[scope]
            item[scope+'_best_paired_variable'] = best_pair(hq,lq,scope)
        hq, lq = query_key.get((inst, 'H')), query_key.get((inst, 'L'))
        for scope in ['current_train', 'current_validation']:
            exact_fvs = []
            if hq and lq:
                exact_fvs = exact_fv_index.get((scope,query_signatures[hq],query_signatures[lq]),[])
            item[scope+'_exact_variable_pair_ids'] = sorted(exact_fvs)
        item['cdr_comparison_supported'] = r['type'] != 'VNAR' and bool(hq or lq)
        query_rows.append(item)
        for pid, g in matches:
            all_matches.append(dict(instance=inst, pinder_id=pid, cluster=g,
                                    validation_cluster=g in validation_groups, training_cluster=g in train_groups))
    write_rows(out/'instances.tsv', query_rows)
    write_rows(out/'pinder-interface-matches.tsv', all_matches)
    write_json(out/'official-component-seeds.json', official_seeds)
    write_json(out/'reference-pairs.json', pairs)
    write_rows(out/'reference-domains.tsv', [dict(id=k, sequence_key=v['sequence_key'], role=v['role'],
                    scopes=sorted(ref_scope[k]), numbered_non_gap_residues=len(v['domain'])) for k, v in refs.items()])
    with (out/'sequence-evidence.jsonl').open('w') as f:
        for r in sab:
            evidence = {scope: {role: nearest.get((scope, query_key.get((r['INSTANCE'],role)))) for role in ['H','L']}
                        for scope in scope_cdrs}
            f.write(json.dumps(dict(instance=r['INSTANCE'], scores=evidence))+'\n')
    counts = {}
    boolcols = [k for k, v in query_rows[0].items() if isinstance(v, bool)]
    for split in ['train', 'test']:
        rr = [r for r in query_rows if r['split'] == split]
        counts[split] = dict(instances=len(rr), unique_antibodies=len({r['sabdab_id'] for r in rr}),
                            components=len({r['component'] for r in rr}),
                            **{k: sum(r[k] for r in rr) for k in boolcols})
        counts[split]['current_validation_exact_variable_pairs'] = sum(bool(r['current_validation_exact_variable_pair_ids']) for r in rr)
        counts[split]['current_train_exact_variable_pairs'] = sum(bool(r['current_train_exact_variable_pair_ids']) for r in rr)
        # Provisional only: missing interface coverage and preparation still require QC.
        counts[split]['outside_protected_sequence_and_component_union'] = sum(
            not (r['validation_legacy_policy_hit'] or r['heldout_legacy_policy_hit'] or
                 r['validation_official_component_hit'] or r['heldout_official_component_hit'] or
                 r['pinder_validation_cluster_hit']) and r['cdr_comparison_supported'] for r in rr)
    inputs = [source, audit/'inputs/download.json', base/'work/numbered.json',base/'work/screen-sequences.json',
              base/'work/reference-pairs.json',base/'registration.json',
              index_root/'index.parquet',index_root/'metadata.parquet']
    inputs += [base/'pool'/f'{name}.tsv' for name in pools]
    manifest = dict(source='SAbDab2 ML v0.3.0', official_split='ab_ag_split', baseline=str(base),
                    input_sha256={str(p):digest(p) for p in inputs},code_sha256=digest(__file__),
                    slurm_job_id=os.environ['SLURM_JOB_ID'], workers=args.workers,
                    no_pool_membership_changes=True, no_teacher_or_apo_annotation=True,
                    sequence_method='IMGT-position identity; core 5–123 for variable, existing CDR3 0.90 / concat 0.80; heldout concat MMseqs 0.70/0.80both',
                    interface_method='Exact deposited author-chain Ab–Ag pair to existing PINDER cluster_id; unmapped means unknown, not dissimilar',
                    component_method='Official ab_ag_cluster linked by verified chain aliases or published CDR base thresholds; no parent-only or framework-only links',
                    limits=['Official antigen components propagate through CDR/instance anchors, but external validation antigen homologs absent from the official dataset are not newly aligned.',
                            'PINDER cluster evidence covers mapped chain interfaces; no fresh all-vs-all geometric interface clustering.',
                            'VNAR CDR comparison unsupported by existing HKL reference numbering.',
                            'Exact variable-domain sequence pairs do not establish identical coordinates, full constructs or reusable teacher context.',
                            'No v7 membership is approved by this audit.'])
    write_json(out/'manifest.json',manifest)
    summary = dict(status='overlap_audit_complete_not_a_release', source_structures=len(all_sab),
                   antigen_instances=len(sab), excluded_no_antigen=len(all_sab)-len(sab),
                   type_counts=dict(Counter(r['type'] for r in sab)),
                   baseline_counts={n:len(r) for n,r in pools.items()},
                   current_fv_counts=dict(Counter(r['scope'] for r in fv_rows.values())),
                   reference_domain_counts={s:sum(s in ss for ss in ref_scope.values()) for s in scope_cdrs},
                   official_anchored_components={s:len(v) for s,v in official_clusters.items()},
                   split_counts=counts,wall_s=time.monotonic()-started,
                   limitations=manifest['limits'])
    write_json(out/'summary.json',summary)
    print(json.dumps(summary, indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--anarci-python', type=Path, required=True)
    run(parser.parse_args())
