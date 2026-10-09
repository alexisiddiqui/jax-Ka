"""Training pools and nested subsets for the full PINDER and pKPDB pretraining sets (2026-10-09).

Pools are training-only. Validation sets: the benchmark validation split (structural-freeze-v1 'val', 142 complexes in
use; PypKa, pKAI and pKAI+ for AB/A/B in campaigns/production-full-v2), which both sets were leakage-screened against, and
the 400 PINDER validation complexes of training/ogqt-pinder-factorial-v1 (pKAI; current PypKa from the teacher audit).
  PINDER  pretraining/pinder-pkai-v1: accepted complexes not in pinder_heldout_exclusions_v3.tsv, outside the PINDER
          validation complexes' clusters, with >= 1 train-mask
          interface site carrying bound (AB) and own-free pKAI labels and both site weights. All PINDER splits and all
          prepared complexes per cluster. Group = PINDER cluster_id; stratum = hetero / homo / Ab/Ag.
  pKPDB   pretraining/pkpdb-full-v1: every pilot.json record. Group = sorted set of the entry's MMseqs2 chain clusters
          (30% identity, 80% coverage of both), as PINDER's cluster_id pairs its partners' clusters; stratum = monomer /
          homomer / heteromer (one chain / one cluster / several clusters). Entries with a chain >= 70% identical over
          >= 80% of both sequences to a PINDER validation chain are removed.
          A group's stratum is that of its first-ranked structure.
Both pools also drop any structure with a chain exactly identical to a held-out benchmark chain (pinder-prefilter-v1
ref_heldout_all.fasta) or a pKPDB reference sequence (benchmark or experimental); this catches peptides too short for
MMseqs2 to align (e.g. MHC epitopes, chaperone tail peptides). pKPDB also checks the PINDER validation chains.
Subsets: each group gets a seeded hash rank within its stratum and min_fraction = (position + 1) / groups in stratum.
Fraction f keeps groups with min_fraction <= f, so subsets are nested (10% in 50% in 75% in 100%), keep the stratum mix
and never split a group. Writes <dataset>/<VERSION>.tsv (one row per structure) and <VERSION>.json (pool-v1: v2 exclusions, no PINDER-validation removal; pool-v2: no exact-chain rule; both superseded).
Usage (compute node): python -m pkabench.training_pools {pinder,pkpdb}
"""
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from .runtime import atomic_json, digest, require_compute

VERSION = 'pool-v3'
SEED = 20261009
FRACTIONS = (0.1, 0.5, 0.75, 1.0)
PINDER = 'pretraining/pinder-pkai-v1'
PINDER_EXCLUSIONS = PINDER + '/pinder_heldout_exclusions_v3.tsv'
PINDER_VALIDATION = 'training/ogqt-pinder-factorial-v1/cohort.json'
PREFILTER = 'audits/pinder-prefilter-v1'
PKPDB = 'pretraining/pkpdb-full-v1'
MMSEQS = 'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
CLUSTER_ARGS = ('--min-seq-id', '0.3', '-c', '0.8', '--cov-mode', '0')
GROUP_ALIAS = {'NTR': 'NTERM', 'CTR': 'CTERM'}
SCREEN_ARGS = ('--min-seq-id', '0.7', '-c', '0.8', '--cov-mode', '0')


def rank(group, seed=SEED):
    return hashlib.sha256(f'{VERSION}|{seed}|{group}'.encode()).hexdigest()


def min_fractions(strata, seed=SEED):
    """strata: {group: stratum}. Returns {group: fraction at which the group enters the nested subsets}."""
    by = defaultdict(list)
    for group, stratum in strata.items(): by[stratum].append(group)
    out = {}
    for groups in by.values():
        groups.sort(key=lambda g: rank(g, seed))
        for i, g in enumerate(groups): out[g] = (i + 1) / len(groups)
    return out


def in_subset(min_fraction, fraction):
    return min_fraction <= fraction + 1e-12


def subset(rows, fraction):
    """Rows of a pool TSV (dicts) belonging to the nested subset at `fraction`."""
    return [r for r in rows if in_subset(float(r['min_fraction']), fraction)]


def read_pool(path):
    with open(path) as f: return list(csv.DictReader(f, delimiter='\t'))


def heldout_pdb_ids(root):
    """PDB IDs of benchmark validation and test complexes (the pools must not contain them)."""
    import pyarrow.parquet as pq
    split = pq.read_table(root/'universe/structural-freeze-v1/assignments.parquet', columns=['complex_id', 'split']).to_pydict()
    held = {c for c, s in zip(split['complex_id'], split['split']) if s in ('val', 'test')}
    return {c['pdb_id'][:4].lower() for c in json.loads((root/'universe/combined-split-v1/index.json').read_text())['candidates'] if c['complex_id'] in held}


def _fasta(path):
    out = {}; key = None
    for line in open(path):
        line = line.strip()
        if line.startswith('>'): key = line[1:]; out[key] = ''
        elif key: out[key] += line
    return out


def pinder_chains(root):
    """{PINDER complex id: (receptor sequence key, ligand sequence key)} and {key: sequence} from the prefilter."""
    import pyarrow.parquet as pq
    t = pq.read_table(root/PREFILTER/'prefiltered_v1.parquet', columns=['id', 'hR', 'hL']).to_pydict()
    return {i: (r, l) for i, r, l in zip(t['id'], t['hR'], t['hL'])}, _fasta(root/PREFILTER/'uniq.fasta')


def reference_sequences(root):
    """Held-out benchmark chains and every pKPDB reference sequence (benchmark and experimental), for exact matching."""
    refs = json.loads((root/PKPDB/'references.json').read_text())['references']
    return set(_fasta(root/PREFILTER/'ref_heldout_all.fasta').values()) | {v['sequence'] for v in refs.values()}


def pinder_validation(root):
    """The PINDER validation complexes (ids, clusters) of training/ogqt-pinder-factorial-v1."""
    records = [r for r in json.loads((root/PINDER_VALIDATION).read_text())['records'] if r['split'] == 'val']
    return {r['id'] for r in records}, {r['cluster_id'] for r in records}


def screen(root, queries, targets, threads):
    """Query keys with a hit >= 70% identity over >= 80% of both sequences (MMseqs2 easy-search)."""
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get('TMPDIR')))
    for name, seqs in (('q', queries), ('t', targets)): (tmp/f'{name}.fasta').write_text(''.join(f'>{k}\n{v}\n' for k, v in seqs.items()))
    subprocess.run([str(root/MMSEQS), 'easy-search', str(tmp/'q.fasta'), str(tmp/'t.fasta'), str(tmp/'hits.tsv'), str(tmp/'work'), *SCREEN_ARGS,
                    '--threads', str(threads), '--format-output', 'query,target,fident'], check=True, stdout=subprocess.DEVNULL)
    return {line.split('\t')[0] for line in (tmp/'hits.tsv').read_text().splitlines()}


def _pinder_complex(task):
    folder, row = task; folder = Path(folder)
    sites = json.loads((folder/'sites.json').read_text()); labels = json.loads((folder/'labels.json').read_text())['pkai']
    have = {state: {(str(c), int(n), str(i), GROUP_ALIAS.get(g, g)) for c, n, i, g, v in rows if v is not None} for state, rows in labels.items()}
    n = interface = 0
    for s in sites:
        key = (s['chain'], s['resnum'], s['icode'], s['group'])
        if s['train_mask'] and key in have.get('AB', ()) and key in have.get(s['partner'], ()) and s.get('w_burial') is not None and s.get('w_interface') is not None:
            n += 1; interface += bool(s['interface'])
    stratum = 'Ab/Ag' if row['kind'] == 'Ab/Ag' else ('hetero' if row['ctype'] == 'heteromer' else 'homo')
    return dict(id=row['id'], pdb_id=row['pdb'], group=row['cluster_id'], stratum=stratum, pinder_split=row['split'], n_res=row['n_res'],
                labelled_sites=n, labelled_interface_sites=interface)


def pinder_rows(root, workers):
    src = root/PINDER
    excluded = {r['id'] for r in csv.DictReader(open(root/PINDER_EXCLUSIONS), delimiter='\t')}
    tasks = []
    for path in sorted((src/'index').glob('label_*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['status'] == 'accepted' and row['id'] not in excluded: tasks.append((str(src/'entries'/row['id']), row))
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        rows = [r for r in pool.map(_pinder_complex, tasks, chunksize=64) if r['labelled_interface_sites']]
    val_ids, val_clusters = pinder_validation(root); before = len(rows)
    rows = [r for r in rows if r['group'] not in val_clusters]
    chains, seqs = pinder_chains(root); held = reference_sequences(root); after_clusters = len(rows)
    rows = [r for r in rows if not any(seqs[k] in held for k in chains[r['id']])]
    val_seqs = {k: seqs[k] for i in val_ids for k in chains[i]}
    near = screen(root, {k: seqs[k] for r in rows for k in chains[r['id']]}, val_seqs, workers)
    for r in rows: r['chain_70_to_validation'] = int(any(k in near for k in chains[r['id']]))
    inputs = dict(exclusions=str(root/PINDER_EXCLUSIONS), exclusions_sha256=digest(root/PINDER_EXCLUSIONS),
                  validation_cohort=str(root/PINDER_VALIDATION), validation_cohort_sha256=digest(root/PINDER_VALIDATION),
                  removed_in_validation_clusters=before - after_clusters, validation_clusters=len(val_clusters),
                  removed_exact_reference_chain=after_clusters - len(rows),
                  chain_70_to_validation=sum(r['chain_70_to_validation'] for r in rows),
                  criterion='accepted, not excluded, outside PINDER validation clusters, >= 1 train-mask interface site with AB and own-free pKAI labels and '
                            'both site weights; chain_70_to_validation flags (does not remove) complexes with a chain >= 70% / 80% both to a validation chain')
    return rows, inputs


def _pkpdb_chains(task):
    folder, record = task
    sequences = json.loads((Path(folder)/'defects.json').read_text())['sequences']
    return record['pdb_id'], [s['sequence'] for s in sequences]


def chain_clusters(root, sequences, threads):
    """MMseqs2 easy-cluster of unique chain sequences; returns {sequence: representative id}."""
    keys = {s: 's' + hashlib.sha256(s.encode()).hexdigest()[:20] for s in sequences}
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get('TMPDIR')))
    (tmp/'chains.fasta').write_text(''.join(f'>{k}\n{s}\n' for s, k in keys.items()))
    subprocess.run([str(root/MMSEQS), 'easy-cluster', str(tmp/'chains.fasta'), str(tmp/'clusters'), str(tmp/'work'), *CLUSTER_ARGS,
                    '--threads', str(threads)], check=True, stdout=subprocess.DEVNULL)
    member = {}
    for line in (tmp/'clusters_cluster.tsv').read_text().splitlines():
        representative, key = line.split('\t'); member[key] = representative
    return {s: member[k] for s, k in keys.items()}


def pkpdb_rows(root, workers):
    src = root/PKPDB
    assert json.loads((src/'verification.json').read_text())['passed']
    records = json.loads((src/'pilot.json').read_text())['records']
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        chains = dict(pool.map(_pkpdb_chains, [(str(src/'entries'/r['pdb_id']), r) for r in records], chunksize=64))
    unique = {s for seqs in chains.values() for s in seqs}
    cluster = chain_clusters(root, unique, workers)
    val_ids, _ = pinder_validation(root); pchains, pseqs = pinder_chains(root)
    keys = {'s' + hashlib.sha256(s.encode()).hexdigest()[:20]: s for s in unique}
    val_seqs = {k: pseqs[k] for i in val_ids for k in pchains[i]}
    near = {keys[k] for k in screen(root, keys, val_seqs, workers)}; exact = reference_sequences(root) | set(val_seqs.values())
    rows = []; removed = 0; removed_exact = 0
    for r in records:
        seqs = chains[r['pdb_id']]
        if any(s in exact for s in seqs): removed_exact += 1; continue
        if any(s in near for s in seqs): removed += 1; continue
        groups = sorted({cluster[s] for s in seqs})
        stratum = 'monomer' if len(seqs) == 1 else ('homomer' if len(groups) == 1 else 'heteromer')
        rows.append(dict(id=r['pdb_id'], pdb_id=r['pdb_id'], group='|'.join(groups), stratum=stratum, n_chains=len(seqs), n_res=r['n'],
                         labelled_sites=r['counts']['clean_sites'], raw_sites=r['counts']['raw_sites']))
    inputs = dict(pilot_sha256=digest(src/'pilot.json'), protocol_sha256=digest(src/'protocol.json'), clustering=dict(tool='mmseqs easy-cluster', args=list(CLUSTER_ARGS)),
                  unique_chain_sequences=len(cluster), chain_clusters=len(set(cluster.values())), validation_cohort=str(root/PINDER_VALIDATION),
                  validation_cohort_sha256=digest(root/PINDER_VALIDATION), removed_chain_70_to_pinder_validation=removed,
                  removed_exact_reference_or_validation_chain=removed_exact,
                  criterion='every pilot.json record without a chain >= 70% / 80% both to a PINDER validation chain')
    return rows, inputs


def build(root, dataset, workers):
    rows, inputs = (pinder_rows if dataset == 'pinder' else pkpdb_rows)(root, workers)
    # A group's structures can differ in class (PINDER clusters mixing classes; pKPDB monomer and homomer entries of one
    # cluster); the group's stratum is the class of its first-ranked structure.
    strata = {}
    for r in sorted(rows, key=lambda r: rank(r['id'])): strata.setdefault(r['group'], r['stratum'])
    entry = min_fractions(strata)
    for r in rows: r['group_stratum'] = strata[r['group']]; r['min_fraction'] = round(entry[r['group']], 9)
    # Leakage is sequence-based; a shared PDB ID (different chains) is reported, not excluded.
    held = heldout_pdb_ids(root); overlap = sorted({r['pdb_id'].lower() for r in rows} & held)
    out = root/(PINDER if dataset == 'pinder' else PKPDB)
    fields = list(rows[0].keys())
    tmp = out/f'{VERSION}.tsv.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fields, delimiter='\t'); w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r['min_fraction'], r['id'])))
    tmp.rename(out/f'{VERSION}.tsv')
    def counts(selected):
        groups = {r['group'] for r in selected}
        return dict(structures=len(selected), groups=len(groups), labelled_sites=sum(int(r['labelled_sites']) for r in selected),
                    **({'labelled_interface_sites': sum(int(r['labelled_interface_sites']) for r in selected)} if dataset == 'pinder' else {}),
                    by_group_stratum={k: v for k, v in sorted(Counter(strata[g] for g in groups).items())},
                    by_structure_stratum={k: v for k, v in sorted(Counter(r['stratum'] for r in selected).items())})
    manifest = dict(version=VERSION, dataset=dataset, seed=SEED, pool=str(out/f'{VERSION}.tsv'), pool_sha256=digest(out/f'{VERSION}.tsv'),
                    inputs=inputs, validation=dict(benchmark='universe/structural-freeze-v1 split == val; labels: campaigns/production-full-v2 pypka, pkai, pkai_plus for AB, A, B',
                                                   pinder=str(root/PINDER_VALIDATION) + ' split == val (400 complexes; pKAI, current PypKa audit)',
                                                   pdb_ids_shared_with_benchmark_val_test=overlap),
                    subset_rule='group min_fraction = (rank position + 1) / groups in stratum; fraction f keeps min_fraction <= f (nested, stratified)',
                    fractions={str(f): counts(subset(rows, f)) for f in FRACTIONS})
    atomic_json(out/f'{VERSION}.json', manifest)
    return manifest


if __name__ == '__main__':
    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    require_compute(threads=workers, allow_comp1400=os.environ.get('PKABENCH_ALLOW_COMP1400') == '1')  # explicit per-run opt-in
    print(json.dumps(build(Path(os.environ['PKABENCH_RUNTIME']), sys.argv[1], workers)['fractions'], indent=1))
