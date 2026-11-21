"""Source-pinned biological partner units for the PINDER subset of v5.

PINDER A/B are partner maps, not PDB chain letters. In particular, gap fragments
of a partner must be surfaced together. ANARCI determines Ab/Ag roles; UniProt
UNDEFINED and alphabetical chain ordering are not role annotations.
"""
import csv
import hashlib
import json
from pathlib import Path


UNIT_POLICY = 'v5-ab-ag-partner-assembly-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def antibody_side(sequences, numbered):
    hits = []
    for sequence in sequences:
        key = 's' + hashlib.sha256(sequence.encode()).hexdigest()[:20]
        hits.append([d['chain_type'] for d in numbered.get(key, [])
                     if d['chain_type'] in ('H', 'K', 'L') and d.get('cdr3')])
    if sum(bool(h) for h in hits) != 1:
        raise ValueError('Ab/Ag assignment requires exactly one ANARCI-positive partner')
    return (0 if hits[0] else 1), hits


def inventory_v5(root, runtime, split='all'):
    from pkabench.training_pools import pinder_chains
    root, runtime = Path(root), Path(runtime)
    paths = dict(pool=root/'pool/pool-v5/pool-v5.json',
                 train=root/'pool/pool-v5/pinder.tsv',
                 val=root/'pool/pool-v5/pinder-val.tsv',
                 anarci=root/'pool/anarci.json',
                 chains=runtime/'audits/pinder-prefilter-v1/prefiltered_v1.parquet',
                 sequences=runtime/'audits/pinder-prefilter-v1/uniq.fasta')
    source_hashes = {k: sha(p) for k, p in paths.items()}
    pool = json.loads(paths['pool'].read_text())
    if pool['abag_rule']['anarci_json_sha256'] != source_hashes['anarci']:
        raise ValueError('ANARCI role evidence differs from v5 pool manifest')
    # The v5 membership contract pins both table byte streams.
    for key, field in [('train', 'pool'), ('val', 'validation')]:
        if pool['datasets']['pinder'][field]['sha256'] != source_hashes[key]:
            raise ValueError('v5 membership table differs from pool manifest')
    chains, sequences = pinder_chains(runtime)
    numbered = json.loads(paths['anarci'].read_text())['numbered']
    units = []; seen = set()
    for partition in ('train', 'val'):
        if split not in ('all', partition):
            continue
        with paths[partition].open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        for row in rows:
            if row['stratum'] != 'Ab/Ag':
                continue
            cid = row['id']
            if cid in seen:
                raise ValueError('v5 train/validation overlap or duplicate: ' + cid)
            seen.add(cid)
            seq = [sequences[k] for k in chains[cid]]
            side, types = antibody_side(seq, numbered)
            entry = runtime/'pretraining/pinder-pkai-v1/entries'/cid
            hashes = {name: sha(entry/name) for name in ('AB.cif.gz', 'map_A.json', 'map_B.json')}
            maps = {s: json.loads((entry/('map_' + s + '.json')).read_text()) for s in ('A', 'B')}
            keys = [{(str(c), int(n), str(i)) for _, _, c, n, i in maps[s]} for s in ('A', 'B')]
            if not all(keys) or keys[0] & keys[1]:
                raise ValueError('empty or overlapping partner maps: ' + cid)
            for index, state in enumerate(('A', 'B')):
                role = 'Ab' if index == side else 'Ag'
                units.append(dict(id='pinder-v5/' + cid + '/' + role, dataset='pinder',
                    cid=cid, state=state, role=role, split=partition, group=row['group'],
                    family='antibody_partner' if role == 'Ab' else 'antigen_partner',
                    n=len(keys[index]), unit_policy=UNIT_POLICY,
                    source_chain_labels=sorted({k[0] for k in keys[index]}),
                    fragment_chain_labels=sorted({str(r[0]) for r in maps[state]}),
                    antibody_domain_types=types[side], role_evidence='ANARCI IMGT H/K/L',
                    source_files_sha256=hashes,
                    assembly_scope='complete supplied PINDER partner; absent chains are not reconstructed'))
    metadata = dict(unit_policy=UNIT_POLICY, source_files_sha256=source_hashes,
                    source='native v5 antibody pool, split=' + split)
    metadata['inventory_sha256'] = hashlib.sha256(json.dumps(
        dict(metadata=metadata, units=units), sort_keys=True).encode()).hexdigest()
    return units, metadata
