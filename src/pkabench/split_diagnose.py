"""Diagnose connected-component mergers using existing sequence hits."""
import json
from pathlib import Path
from collections import Counter, defaultdict
from .runtime import require_compute, atomic_json, config_hash, digest


def run(root):
    require_compute()
    import pyarrow.parquet as pq
    root = Path(root)
    index = json.loads((root/'index.json').read_text())
    rows = index['candidates']
    diversity = json.loads((root/'sequence/diversity.json').read_text())
    assignments = diversity['assignments']['cdr_partner']
    biggest = Counter(assignments.values()).most_common(1)[0][0]
    accepted, usable = set(), set()
    for source in index['sources']:
        masks = pq.read_table(Path(source['campaign'])/'site_masks.parquet').to_pylist()
        accepted.update(s['complex_id'] for s in masks)
        usable.update(s['complex_id'] for s in masks if s['train_mask'])
    seqs, nodes, labels = {}, {}, defaultdict(Counter)
    def key(seq):
        k = 's'+config_hash(seq)[:20]
        seqs[k] = seq
        return k
    for row in rows:
        cid = row['complex_id']
        if cid not in assignments:
            continue
        if row['stratum'] == 'antibody':
            fields = ['CDR-H1','CDR-H2','CDR-H3'] + (['CDR-L1','CDR-L2','CDR-L3'] if len(row['partner_A_chains']) == 2 else [])
            n = key(''.join(''.join(row['sabdab'][f].split()).upper() for f in fields))
            ns = [n]
            labels[n]['concatenated antibody CDR'] += 1
            chains = [c for c in row['chains'] if c['chain'] in row['partner_B_chains']]
        else:
            ns = []
            chains = row['chains']
        for c in chains:
            n = key(c['sequence']); ns.append(n)
            labels[n][c.get('description','unknown')] += 1
        nodes[cid] = ns
    links = []
    for line in (root/'sequence/hits.tsv').read_text().splitlines():
        a,b,identity,qcov,tcov = line.split('\t')
        if a in seqs and b in seqs and float(identity)>=.3 and min(float(qcov),float(tcov))>=.8:
            links.append((a,b))
    def components(ids, join_partners=True):
        allowed = {n for c in ids for n in nodes[c]}
        parent = {n:n for n in allowed}
        def find(n):
            while parent[n]!=n:
                parent[n]=parent[parent[n]]; n=parent[n]
            return n
        def union(a,b):
            parent[find(b)] = find(a)
        for a,b in links:
            if a in allowed and b in allowed: union(a,b)
        if join_partners:
            for c in ids:
                for n in nodes[c][1:]: union(nodes[c][0],n)
        return {n:find(n) for n in allowed}
    scenarios = {}
    for name, ids in [('whole',set(nodes)),('accepted_only',accepted & nodes.keys()),('usable_only',usable & nodes.keys()),('general_only',{r['complex_id'] for r in rows if r['stratum']=='general'} & nodes.keys())]:
        groups = components(ids)
        counts = Counter(groups[nodes[c][0]] for c in ids)
        scenarios[name] = {'pairs':len(ids),'components':len(counts),'largest':max(counts.values(),default=0),'largest_ten':sorted(counts.values(),reverse=True)[:10]}
    assert scenarios['whole']['largest'] == Counter(assignments.values()).most_common(1)[0][1]
    families = components(set(nodes),False)
    bigrows = [r for r in rows if assignments.get(r['complex_id'])==biggest]
    family_pairs = defaultdict(set)
    family_labels = defaultdict(Counter)
    visited = set()
    for r in bigrows:
        for n in set(nodes[r['complex_id']]):
            f = families[n]
            family_pairs[f].add(r['complex_id'])
            if n not in visited:
                family_labels[f].update(labels[n])
                visited.add(n)
    top = sorted(family_pairs,key=lambda f:len(family_pairs[f]),reverse=True)[:20]
    diagnostics = {'large_component': {'candidates':len(bigrows),'strata':dict(Counter(r['stratum'] for r in bigrows)),
                    'accepted':sum(r['complex_id'] in accepted for r in bigrows),'usable':sum(r['complex_id'] in usable for r in bigrows)},
                   'scenarios':scenarios,
                   'top_sequence_families':[{'family':f,'pairs':len(family_pairs[f]),'descriptions':family_labels[f].most_common(8)} for f in top],
                   'barnase_barstar_candidates':[{'complex_id':r['complex_id'],'pdb_id':r['pdb_id'],'component':assignments.get(r['complex_id']),'descriptions':[c.get('description','') for c in r['chains']]} for r in rows if any(any(t in c.get('description','').lower() for t in ('barnase','barstar')) for c in r['chains'])],
                   'limitations':'Removal scenarios are diagnostics only, not authorized split-policy changes. Description matches are reservation candidates, not sequence-complete experimental decontamination.'}
    atomic_json(root/'diagnosis.json',diagnostics)
    import csv
    import os
    source = Path(os.environ['PKABENCH_RUNTIME'])/'sources/KaMLs/KaML-CBTrees/train_test_split'
    proteins = {}
    receipts = []
    for path in sorted(source.glob('*.csv')):
        receipts.append({'path':str(path),'sha256':digest(path)})
        with path.open() as stream:
            for r in csv.DictReader(stream):
                pdb = r['PDB_ID'].strip().lower()
                chain = r['Chain'].strip()
                rec = proteins.setdefault((pdb,chain), {'pdb_id':pdb,'chain':chain,'uniprot_ids':set()})
                rec['uniprot_ids'].add(r['Uniprot_ID'].strip())
    if not receipts:
        raise ValueError('No local experimental tables found')
    records = [dict(r,uniprot_ids=sorted(r['uniprot_ids'])) for _,r in sorted(proteins.items())]
    pdb_ids = {r['pdb_id'] for r in records}
    overlaps = [{'complex_id':r['complex_id'],'pdb_id':r['pdb_id'],'component':assignments.get(r['complex_id'])} for r in rows if r['pdb_id'].split('-')[0].lower() in pdb_ids]
    atomic_json(root/'experimental-reservation-inventory.json', {
        'source':'Local KaML-CBtree released train/test tables; completeness against PKAD-3 not yet verified',
        'source_files':receipts,'proteins':records,'protein_chains':len(records),'pdb_entries':len(pdb_ids),
        'direct_pdb_overlaps':overlaps,'complete':False,
        'required_next':'Resolve reference chain sequences and search them against the universe. PDB-ID overlap alone is insufficient. Curate additional set-2 systems before freezing.',
        'set2_seed':{'system':'barnase-barstar','evidence':'https://pubmed.ncbi.nlm.nih.gov/8494892/','status':'Literature-supported candidate; constructs/conditions/structure sequence mapping remain to curate'}})
    print(json.dumps(diagnostics,indent=2),flush=True)
