"""pool-v5 (2026-11-16): pool-v4 with antibody chains screened on their CDRs instead of the whole chain.

Rule (controller, brief work/abdata-audit/BRIEF-pool-v5.md):
- Everything as pkabench.training_pools.build_v4 (same candidates, same validation draw, same held-out / exact-reference
  rules, same 70% / 80% MMseqs2 whole-chain screen, same nested min_fraction ranks under the pool-v4 salt).
- Exception: in a PINDER Ab/Ag candidate (stratum Ab/Ag) with exactly one UNDEFINED-UniProt chain (score_masked2 ROLE),
  that chain is the antibody. If ANARCI (IMGT, Ig H/K/L only) numbers it, it is NOT whole-chain screened; it leaks if it is
  exactly equal to a validation chain, or if any of its Ig domains has, against any validation Ig domain of the same chain
  type (H, or L = K/L), CDR3 identity >= 0.90 or CDR1+2+3 identity >= 0.80. If ANARCI cannot number it, the v4 whole-chain
  rule applies to it (counted). The antigen chain, rows without a role and all non-Ab/Ag rows keep the v4 rule.
- Validation Ig domains = every ANARCI-numbered domain of every chain of either validation set (PINDER + pKPDB).
- CDR identity (IMGT, positions with insertion codes): identical residues at the same IMGT position / the longer CDR's
  length; concatenated: summed matches / summed longer lengths over CDR1, CDR2, CDR3.

  python pool_v5.py prepare OUT          (compute node; candidates, validation, MMseqs2 screens -> OUT/state.pkl,
                                          OUT/anarci-input.fasta, OUT/repro-v4/ = the rule switched off, checked
                                          against training/pool-v4)
  python pool_v5.py finish OUT ANARCI    (OUT/pool-v5/{pinder.tsv, pinder-val.tsv, pool-v5.json}, cdr-identity.tsv,
                                          funnel.json)
build_v4 is not called: its canonical branch writes into training/pool-v4. Its body is repeated here with the output
directory and the PINDER training filter as the only differences (see write_pool).
"""
import csv
import hashlib
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pkabench import training_pools as tp
from pkabench.runtime import atomic_json, digest, require_compute

CDR3_MIN = 0.90
CDRS_MIN = 0.80
IMGT = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}
CTYPE = {"H": "H", "K": "L", "L": "L"}
COULSON_ROOT = '/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench'


# ---------------------------------------------------------------- prepare (build_v4 up to the screens)
def prepare(root, workers, out):
    out.mkdir(parents=True, exist_ok=True)
    overlap, anchor = tp.OVERLAP; groups = dict(tp.VALIDATION_GROUPS)
    candidates = {'pinder': tp.pinder_candidates(root, workers), 'pkpdb': tp.pkpdb_candidates(root, workers)}
    validation = {}; cells = {}
    eligible = {'pinder': lambda r: int(r['eval_interface_sites']) > 0, 'pkpdb': lambda r: int(r['labelled_sites']) > 0}
    order = ('pinder', 'pkpdb') if anchor != 'pkpdb' else ('pkpdb', 'pinder')
    first, second = order
    validation[first], cells[first] = tp.choose_validation(candidates[first][0], groups[first], eligible=eligible[first])
    anchor_seqs = {s for r in validation[first] for s in r['_sequences']}
    pool = {tp.sequence_key(s): s for r in candidates[second][0] for s in r['_sequences']}
    matched = {pool[k] for k in tp.screen(root, pool, {tp.sequence_key(s): s for s in anchor_seqs}, workers)} | anchor_seqs
    near_rows = [r for r in candidates[second][0] if any(s in matched for s in r['_sequences'])]
    shared, shared_cells = tp.choose_validation(near_rows, overlap, eligible=eligible[second])
    taken = {r['group'] for r in shared}
    rest, rest_cells = tp.choose_validation([r for r in candidates[second][0] if r['group'] not in taken], groups[second] - overlap,
                                            eligible=eligible[second])
    validation[second] = sorted(shared + rest, key=lambda r: r['id'])
    for r in validation[second]: r['shares_anchor_chain'] = int(r['group'] in taken)
    cells[second] = {'overlap': shared_cells, 'rest': rest_cells}
    val_sequences = {s for rows in validation.values() for r in rows for s in r['_sequences']}
    val_groups = {d: {r['group'] for r in validation[d]} for d in validation}
    queries = {tp.sequence_key(s): s for d in candidates for r in candidates[d][0] if r['group'] not in val_groups[d] for s in r['_sequences']}
    near = {queries[k] for k in tp.screen(root, queries, {tp.sequence_key(s): s for s in val_sequences}, workers)}
    by_source = {v: {s for r in validation[v] for s in r['_sequences']} for v in validation}
    near_by = {v: {queries[k] for k in tp.screen(root, queries, {tp.sequence_key(s): s for s in by_source[v]}, workers)} | by_source[v] for v in by_source}
    held = tp.heldout_pdb_ids(root)
    # receptor / ligand sequences of PINDER rows (prefilter hR / hL; the id is R--L)
    chains, seqs = tp.pinder_chains(root)
    for r in candidates['pinder'][0]: r['_rl'] = (seqs[chains[r['id']][0]], seqs[chains[r['id']][1]])
    state = dict(candidates=candidates, validation=validation, cells=cells, val_sequences=val_sequences, val_groups=val_groups,
                 near=near, near_by=near_by, held=held, workers=workers)
    with open(out / 'state.pkl', 'wb') as f: pickle.dump(state, f, protocol=5)
    # ANARCI input: every validation chain + both chains of every PINDER Ab/Ag candidate outside the validation groups
    numbered = set(val_sequences) | {s for r in candidates['pinder'][0] if r['stratum'] == 'Ab/Ag' and r['group'] not in val_groups['pinder'] for s in r['_rl']}
    with open(out / 'anarci-input.fasta', 'w') as f:
        for s in sorted(numbered): f.write(f'>{tp.sequence_key(s)}\n{s}\n')
    repro = write_pool(root, state, out / 'repro-v4', cdr=None)
    v4 = root / tp.V4_DIR; check = {}
    for name in ('pinder.tsv', 'pinder-val.tsv', 'pkpdb.tsv', 'pkpdb-val.tsv'):
        check[name] = dict(repro=digest(out / 'repro-v4' / name), pool_v4=digest(v4 / name))
        check[name]['equal'] = check[name]['repro'] == check[name]['pool_v4']
    old = json.loads((v4 / 'pool-v4.json').read_text())
    # paths differ between clusters / runs: normalise the output dirs and the runtime root (pool-v4 was built at COULSON_ROOT)
    strip = lambda m: json.loads(json.dumps(m).replace(str(out / 'repro-v4'), 'OUT').replace(str(v4), 'OUT')
                                 .replace(COULSON_ROOT + '/' + tp.V4_DIR, 'OUT').replace(str(root), 'ROOT').replace(COULSON_ROOT, 'ROOT'))
    check['pool-v4.json_equal_except_paths'] = strip(repro) == strip(old)
    check['passed'] = all(v['equal'] for k, v in check.items() if isinstance(v, dict)) and check['pool-v4.json_equal_except_paths']
    atomic_json(out / 'repro-v4-check.json', check)
    return check


# ---------------------------------------------------------------- CDR screen
def uniprots(cid):
    """[(R uniprot), (L uniprot)] from a PINDER id R--L (score_masked2.chain_uniprot without the chain-name sort)."""
    return [part.rsplit('_', 1)[1] for part in cid.split('--')]


def antibody_side(r):
    """0 (receptor) / 1 (ligand) / None: the UNDEFINED-UniProt chain of an Ab/Ag row with exactly one UNDEFINED chain."""
    u = uniprots(r['id'])
    return u.index('UNDEFINED') if r['stratum'] == 'Ab/Ag' and u.count('UNDEFINED') == 1 else None


def cdr_identity(a, b):
    """a, b: {cdr: {(pos, ins): aa}} -> (cdr3 identity, concatenated identity)."""
    out = {}
    for c in IMGT:
        x, y = a[c], b[c]; out[c] = (sum(1 for k, v in x.items() if y.get(k) == v), max(len(x), len(y)))
    m3, n3 = out['cdr3']; ms = sum(v[0] for v in out.values()); ns = sum(v[1] for v in out.values())
    return (m3 / n3 if n3 else 0.0), (ms / ns if ns else 0.0)


def load_anarci(path):
    """{sequence key: [domain {type, cdr1/2/3: {(pos, ins): aa}}]} for chains with >= 1 Ig domain with a non-empty CDR3."""
    raw = json.loads(Path(path).read_text())['numbered']; out = {}
    for key, domains in raw.items():
        keep = []
        for d in domains:
            cdrs = {c: {(int(p), str(i)): a for p, i, a in d[c]} for c in IMGT}
            if cdrs['cdr3']: keep.append(dict(type=CTYPE[d['chain_type']], **cdrs))
        if keep: out[key] = keep
    return out


class CdrScreen:
    def __init__(self, numbered, val_sequences):
        self.numbered = numbered; self.val_sequences = val_sequences
        self.val_domains = [(s, d) for s in sorted(val_sequences) for d in numbered.get(tp.sequence_key(s), [])]
        self.cache = {}

    def nearest(self, sequence):
        """None (not numbered) or dict(cdr3, cdrs, type, leak, val_key...) maximised over the chain's domains."""
        if sequence in self.cache: return self.cache[sequence]
        domains = self.numbered.get(tp.sequence_key(sequence))
        if not domains: self.cache[sequence] = None; return None
        best = dict(cdr3=0.0, cdrs=0.0, cdr3_val=None, cdrs_val=None, types='+'.join(d['type'] for d in domains), leak=False)
        for d in domains:
            for vs, v in self.val_domains:
                if v['type'] != d['type']: continue
                i3, ia = cdr_identity(d, v)
                if i3 > best['cdr3']: best['cdr3'], best['cdr3_val'] = i3, tp.sequence_key(vs)
                if ia > best['cdrs']: best['cdrs'], best['cdrs_val'] = ia, tp.sequence_key(vs)
        best['leak'] = best['cdr3'] >= CDR3_MIN - 1e-12 or best['cdrs'] >= CDRS_MIN - 1e-12
        self.cache[sequence] = best; return best


def row_leaks(r, near, val_sequences, cdr, notes):
    """pool-v4 rule (cdr None) or the pool-v5 rule for one candidate row; notes collects per-row screen details."""
    v4 = any(s in near or s in val_sequences for s in r['_sequences'])
    side = antibody_side(r) if cdr is not None and '_rl' in r else None
    if side is None:
        if cdr is not None and r['stratum'] == 'Ab/Ag': notes[r['id']] = dict(rule='v4 (no single UNDEFINED chain)', v4_leak=v4, leak=v4)
        return v4
    ab, ag = r['_rl'][side], r['_rl'][1 - side]
    ag_leak = ag in near or ag in val_sequences
    hit = cdr.nearest(ab)
    if hit is None:
        ab_leak = ab in near or ab in val_sequences; rule = 'antibody not numbered: whole chain'
    else:
        ab_leak = ab in val_sequences or hit['leak']; rule = 'antibody CDRs'
    notes[r['id']] = dict(rule=rule, side=side, v4_leak=v4, ag_leak=ag_leak, ab_whole_chain_hit=ab in near, ab_exact_val=ab in val_sequences,
                          ab_leak=ab_leak, leak=ag_leak or ab_leak, **({} if hit is None else {k: hit[k] for k in ('types', 'cdr3', 'cdrs', 'cdr3_val', 'cdrs_val')}))
    return ag_leak or ab_leak


# ---------------------------------------------------------------- write (build_v4's loop)
def write_pool(root, state, out, cdr=None, notes=None, version=tp.V4):
    """build_v4's writing loop into `out`; cdr None reproduces pool-v4; otherwise the PINDER training filter is
    row_leaks with the CDR screen (pKPDB unchanged). version only labels the manifest (min_fraction ranks keep the
    pool-v4 salt so every unchanged group keeps its rank)."""
    out.mkdir(parents=True, exist_ok=True); notes = {} if notes is None else notes
    candidates, validation, cells = state['candidates'], state['validation'], state['cells']
    val_sequences, val_groups, near, near_by, held = (state[k] for k in ('val_sequences', 'val_groups', 'near', 'near_by', 'held'))
    V4 = tp.V4
    manifest = dict(version=V4, seed=tp.SEED, created_by='pkabench.training_pools build_v4',
                    rules=dict(validation='one structure per group (first eligible by pool-v4 rank; PINDER >= 1 labelled eval-mask interface site, pKPDB >= 1 clean site), groups stratified by (stratum, size bucket) '
                                          'with proportional largest-remainder shares, chosen by the pool-v4-validation rank',
                               training='candidates outside the validation groups without a chain >= 70% identity over >= 80% of both sequences to, '
                                        'or exactly equal to, any chain of either validation set',
                               size_bounds=list(tp.SIZE_BOUNDS), screen=dict(tool='mmseqs easy-search', args=list(tp.SCREEN_ARGS))),
                    inputs=dict(pinder_exclusions=str(root/tp.PINDER_EXCLUSIONS), pinder_exclusions_sha256=digest(root/tp.PINDER_EXCLUSIONS),
                                pkpdb_pilot_sha256=digest(root/tp.PKPDB/'pilot.json'), pkpdb_protocol_sha256=digest(root/tp.PKPDB/'protocol.json'),
                                retired_validation=str(root/tp.PINDER_VALIDATION) + ' (400 complexes; clusters are candidates again)'),
                    validation_chain_sequences=len(val_sequences), datasets={})
    for d, (rows, info) in candidates.items():
        outside = [r for r in rows if r['group'] not in val_groups[d]]
        if cdr is None or d != 'pinder': train = [r for r in outside if not any(s in near or s in val_sequences for s in r['_sequences'])]
        else: train = [r for r in outside if not row_leaks(r, near, val_sequences, cdr, notes)]
        kept = {r['id'] for r in train}
        strata = {}
        for r in sorted(train, key=lambda r: tp.rank(r['id'], version=V4)): strata.setdefault(r['group'], r['stratum'])
        entry = tp.min_fractions(strata, version=V4)
        for r in train: r['group_stratum'] = strata[r['group']]; r['min_fraction'] = round(entry[r['group']], 9)
        train.sort(key=lambda r: (r['min_fraction'], r['id']))
        tp._write_tsv(out/f'{d}.tsv', train); tp._write_tsv(out/f'{d}-val.tsv', validation[d])

        def counts(selected):
            groups = {r['group'] for r in selected}
            return dict(structures=len(selected), groups=len(groups), labelled_sites=sum(int(r['labelled_sites']) for r in selected),
                        **({'labelled_interface_sites': sum(int(r['labelled_interface_sites']) for r in selected)} if d == 'pinder' else {}),
                        by_group_stratum=dict(sorted(Counter(strata[g] for g in groups).items())))
        manifest['datasets'][d] = dict(
            candidates=dict(structures=len(rows), groups=len({r['group'] for r in rows}), **info),
            validation=dict(file=str(out/f'{d}-val.tsv'), sha256=digest(out/f'{d}-val.tsv'), structures=len(validation[d]),
                            labelled_sites=sum(int(r['labelled_sites']) for r in validation[d]),
                            **({'labelled_interface_sites': sum(int(r['labelled_interface_sites']) for r in validation[d])} if d == 'pinder' else {}),
                            cells=cells[d]),
            removed=dict(in_validation_groups=len(rows) - len(outside), chain_near_or_equal_validation=len(outside) - len(train),
                         chain_matches_by_validation_set={', '.join(k): n for k, n in sorted(Counter(
                             tuple(v for v in near_by if any(s in near_by[v] for s in r['_sequences'])) for r in outside if r['id'] not in kept).items())}),
            pool=dict(file=str(out/f'{d}.tsv'), sha256=digest(out/f'{d}.tsv')),
            pdb_ids_shared_with_benchmark_val_test=sorted({r['pdb_id'].lower() for r in train + validation[d]} & held),
            fractions={str(f): counts(tp.subset(train, f)) for f in tp.FRACTIONS})
    atomic_json(out/f'{V4}.json', manifest)
    return manifest


def finish(root, out, anarci_json):
    with open(out / 'state.pkl', 'rb') as f: state = pickle.load(f)
    numbered = load_anarci(anarci_json); cdr = CdrScreen(numbered, state['val_sequences']); notes = {}
    dest = out / 'pool-v5'
    manifest = write_pool(root, state, dest, cdr=cdr, notes=notes)
    os.remove(dest / 'pool-v4.json')  # replaced by pool-v5.json below
    for name in ('pkpdb.tsv', 'pkpdb-val.tsv'):  # pKPDB is unchanged: keep pool-v4's, assert identical, do not duplicate
        if digest(dest / name) != digest(root / tp.V4_DIR / name): raise AssertionError((name, 'pKPDB pool changed'))
        os.remove(dest / name)
    v4 = {r['id']: r for r in tp.read_pool(root / tp.V4_DIR / 'pinder.tsv')}
    v5 = {r['id']: r for r in tp.read_pool(dest / 'pinder.tsv')}
    if digest(dest / 'pinder-val.tsv') != digest(root / tp.V4_DIR / 'pinder-val.tsv'): raise AssertionError('validation changed')
    added = sorted(set(v5) - set(v4)); removed = sorted(set(v4) - set(v5))
    changed_fraction = Counter((v4[i]['group_stratum'], v5[i]['group_stratum'] != v4[i]['group_stratum'], v5[i]['min_fraction'] != v4[i]['min_fraction'])
                               for i in set(v4) & set(v5))
    # funnel: PINDER Ab/Ag candidates outside the validation groups
    pinder_rows = state['candidates']['pinder'][0]
    outside = [r for r in pinder_rows if r['group'] not in state['val_groups']['pinder'] and r['stratum'] == 'Ab/Ag']
    ab_v5 = [r for r in v5.values() if r['stratum'] == 'Ab/Ag']; ab_v4 = [r for r in v4.values() if r['stratum'] == 'Ab/Ag']
    rule_counts = Counter(notes[r['id']]['rule'] for r in outside)
    leak_kind = Counter()
    for r in outside:
        n = notes[r['id']]
        if not n['leak']: continue
        if n['rule'] != 'antibody CDRs': leak_kind[n['rule'] + ' leak'] += 1; continue
        parts = []
        if n['ag_leak']: parts.append('antigen whole-chain')
        if n['ab_exact_val']: parts.append('antibody exact')
        if n['cdr3'] >= CDR3_MIN - 1e-12: parts.append('CDR3>=0.90')
        if n['cdrs'] >= CDRS_MIN - 1e-12: parts.append('CDRs>=0.80')
        leak_kind[' + '.join(parts) or '?'] += 1
    # CDR3 hits split by chain type of the best CDR3 match
    funnel = dict(abag_candidates_outside_val_groups=len(outside), clusters=len({r['group'] for r in outside}),
                  pool_v4=dict(structures=len(ab_v4), groups=len({r['group'] for r in ab_v4})),
                  pool_v5=dict(structures=len(ab_v5), groups=len({r['group'] for r in ab_v5}),
                               pinder_fractions=manifest['datasets']['pinder']['fractions']),
                  rule_applied=dict(rule_counts), leaks_by_reason=dict(leak_kind),
                  added_vs_v4=len(added), removed_vs_v4=len(removed), removed_ids=removed,
                  added_by_stratum=dict(Counter(v5[i]['stratum'] for i in added)),
                  common_rows_by_group_stratum_changed_stratum_changed_min_fraction={'|'.join(map(str, k)): v for k, v in sorted(changed_fraction.items())},
                  val_ig_domains=len(cdr.val_domains), val_sequences_numbered=sum(tp.sequence_key(s) in numbered for s in state['val_sequences']),
                  numbered_input_sequences=len(numbered))
    # per admitted Ab/Ag row: nearest validation CDR identities of its antibody chain (memorisation split)
    with open(out / 'cdr-identity.tsv', 'w', newline='') as f:
        cols = ['id', 'group', 'in_pool_v4', 'in_pool_v5', 'min_fraction_v5', 'rule', 'types', 'cdr3', 'cdrs', 'cdr3_val', 'cdrs_val',
                'ab_whole_chain_hit', 'ag_leak', 'leak']
        w = csv.DictWriter(f, cols, delimiter='\t'); w.writeheader()
        for r in sorted(outside, key=lambda r: r['id']):
            n = notes[r['id']]
            w.writerow({'id': r['id'], 'group': r['group'], 'in_pool_v4': int(r['id'] in v4), 'in_pool_v5': int(r['id'] in v5),
                        'min_fraction_v5': v5[r['id']]['min_fraction'] if r['id'] in v5 else '', 'rule': n['rule'],
                        **{k: (round(n[k], 4) if isinstance(n.get(k), float) else n.get(k, '')) for k in ('types', 'cdr3', 'cdrs', 'cdr3_val', 'cdrs_val', 'ab_whole_chain_hit', 'ag_leak', 'leak')}})
    # distribution of nearest-val identity for admitted antibodies
    hist = defaultdict(Counter)
    for r in ab_v5:
        n = notes.get(r['id'], {})
        if 'cdr3' not in n: hist['not numbered'][r['id'] in v4] += 1; continue
        for k in ('cdr3', 'cdrs'):
            b = min(9, int(n[k] * 10)); hist[k][f'{b / 10:.1f}-{(b + 1) / 10:.1f}'] += 1
    funnel['admitted_nearest_val_identity'] = {k: dict(sorted(v.items())) for k, v in hist.items()}
    manifest.update(version='pool-v5', created_by='abdata-v5 pool_v5.py (pkabench.training_pools build_v4 body + antibody CDR screen)',
                    base=dict(pool_v4=str(root / tp.V4_DIR), pinder_sha256=digest(root / tp.V4_DIR / 'pinder.tsv'),
                              pinder_val_sha256=digest(root / tp.V4_DIR / 'pinder-val.tsv'), pkpdb='unchanged: use pool-v4 pkpdb.tsv / pkpdb-val.tsv'),
                    abag_rule=dict(antibody='PINDER Ab/Ag row with exactly one UNDEFINED-UniProt chain: that chain (score_masked2 ROLE)',
                                   numbering='ANARCI 2024.05.21 (bioconda), HMMER 3.4, scheme imgt, allow H/K/L; CDR1 27-38, CDR2 56-65, CDR3 105-117',
                                   leak=f'exact validation chain, or any domain vs any validation Ig domain of the same type (H; L = K or L): CDR3 >= {CDR3_MIN}, or CDR1+2+3 >= {CDRS_MIN}',
                                   identity='identical residues at the same IMGT position (with insertion codes) / longer CDR length; concatenated = summed matches / summed longer lengths',
                                   fallback='antibody chain not numbered -> pool-v4 whole-chain rule; antigen chain, rows without one UNDEFINED chain, non-Ab/Ag rows -> pool-v4 rule',
                                   validation_ig='every ANARCI-numbered domain of every chain of the PINDER and pKPDB validation sets',
                                   anarci_json=str(anarci_json), anarci_json_sha256=digest(anarci_json)),
                    funnel=funnel)
    manifest['datasets'] = {'pinder': manifest['datasets']['pinder']}
    atomic_json(dest / 'pool-v5.json', manifest)
    atomic_json(out / 'funnel.json', funnel)
    return funnel


if __name__ == '__main__':
    root = Path(os.environ['PKABENCH_RUNTIME']); action = sys.argv[1]; out = Path(sys.argv[2])
    if action == 'prepare':
        workers = int(os.environ.get('POOL_WORKERS') or os.environ['SLURM_CPUS_PER_TASK'])  # 24 = pool-v4 job 748141
        require_compute(threads=workers)
        print(json.dumps(prepare(root, workers, out), indent=1))
    elif action == 'finish':
        print(json.dumps(finish(root, out, Path(sys.argv[3])), indent=1, default=str))
    else: raise SystemExit(action)
