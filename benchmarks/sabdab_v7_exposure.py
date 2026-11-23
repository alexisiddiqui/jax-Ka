"""Versioned, read-only full-pool exposure inventory for SAbDab integration.

The source policy is configurable; this program writes audit artifacts only.
It never edits current pool memberships, labels, structures or augmentation.
Scientific actions require a real Slurm allocation with --mem-per-cpu=2G.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

from pkabench.runtime import atomic_json, digest, require_compute
from pkabench.source_policy import SourcePolicy


def read(path):
    return json.loads(Path(path).read_text())


def skey(sequence):
    return 's'+hashlib.sha256(sequence.encode()).hexdigest()[:20]


def fasta(path):
    out = {}; name = None
    with Path(path).open() as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].strip(); out[name] = ''
            elif name:
                out[name] += line.strip()
    return out


def write_fasta(path, sequences):
    with Path(path).open('w') as f:
        for key, sequence in sorted(sequences.items()):
            f.write('>'+key+'\n'+sequence+'\n')


def tsv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_tsv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter='\t'); w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})


def native_fv_sequences(path):
    """Read the actual retained chains, not longer RCSB reference sequences."""
    from biotite.structure.io import pdbx
    import biotite.structure as struc
    from biotite.sequence import ProteinSequence
    with gzip.open(path, 'rt') as f:
        cif = pdbx.CIFFile.read(f)
    atoms = pdbx.get_structure(cif, model=1, altloc='first')
    out = {}
    for chain in ['H','L','G']:
        subset = atoms[atoms.chain_id == chain]
        _, names = struc.get_residues(subset)
        if not len(names):
            raise ValueError('Missing native Fv chain '+chain+': '+str(path))
        letters = []
        for name in names:
            try:
                letters.append(ProteinSequence.convert_letter_3to1(name))
            except KeyError:
                letters.append({'MSE':'M'}.get(name, 'X'))
        out[chain] = ''.join(letters)
    return out


def prepare(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    require_compute(threads=args.workers, allow_comp1400=True)
    started = time.monotonic(); out = args.output
    if out.exists():
        raise FileExistsError('Choose a new audit root: '+str(out))
    out.mkdir(parents=True)
    policy = SourcePolicy.load(args.policy)
    atomic_json(out/'source-policy.json', policy.to_dict())
    paths = {'policy': args.policy, 'state': args.v5/'pool/state.pkl',
             'sabdab_csv':args.sabdab/'inputs/metadata/splits_final/abag_split.csv',
             'v6_numbered':args.baseline/'work/numbered.json',
             'pkpdb_numbered':args.pkpdb_inventory/'numbered.json',
             'pkpdb_prepared':args.pkpdb_inventory/'prepared.json',
             'pkpdb_sequences':args.pkpdb_inventory/'sequences.fasta',
             'pkpdb_roles':args.pkpdb_inventory/'inventory.json',
             'reference_sequences':args.baseline/'work/screen-sequences.json',
             'reference_pairs':args.baseline/'work/reference-pairs.json',
             'prior_overlap':args.sabdab/'results/instances.tsv',
             'prior_context':args.sabdab/'results/context-diagnostics.json'}
    pools = {}
    for source in ['pkpdb','pinder']:
        for role, suffix in [('train',''),('validation','-val')]:
            name=source+suffix
            paths['pool_'+name] = args.baseline/'pool'/(name+'.tsv')
            pools[(source,role)] = tsv(paths['pool_'+name])
    # Trusted, locally generated builder snapshot, never an uploaded pickle.
    with paths['state'].open('rb') as f:
        state = pickle.load(f)
    legacy = {source:{r['id']:r for r in state['candidates'][source][0]} for source in ['pkpdb','pinder']}
    for source in legacy:
        legacy[source].update({r['id']:r for r in state['validation'][source]})
    prepared = read(paths['pkpdb_prepared'])
    for name, expected in prepared['membership_sha256'].items():
        if digest(args.baseline/'pool'/name) != expected:
            raise ValueError('pKPDB numbering membership differs from baseline: '+name)
    cached_sequences = fasta(paths['pkpdb_sequences'])
    if digest(paths['pkpdb_sequences']) != prepared['sequences_sha256']:
        raise ValueError('pKPDB sequence cache checksum differs')
    pk_numbered = read(paths['pkpdb_numbered'])
    if pk_numbered['sequences_sha256'] != prepared['sequences_sha256']:
        raise ValueError('pKPDB numbering source differs')
    raw = read(paths['v6_numbered'])['numbered']
    # Keep richer, already-frozen domains from v6c over old CDR3-only entries.
    for k,v in pk_numbered['numbered'].items():
        if k not in raw or (not raw[k] and v):
            raw[k] = v
    attempted = set(raw) | set(pk_numbered['failed']) | set(pk_numbered['numbered'])
    annotations = {r['id']:r for r in read(paths['pkpdb_roles'])['parents']}
    seqs = {}; memberships=[]; units=[]; official_domains={}; observed_pins={}

    def add_sequence(sequence):
        if not sequence or any(c.isspace() for c in sequence):
            raise ValueError('Empty/whitespace-containing protein sequence')
        k=skey(sequence)
        if k in seqs and seqs[k] != sequence:
            raise ValueError('Sequence digest collision')
        seqs[k]=sequence
        return k

    def member(uid, sequence, name, annotation='', kind='source_sequence_inventory'):
        k=add_sequence(sequence)
        memberships.append(dict(unit_id=uid,chain_id=name,sequence_key=k,
                                annotation_role=annotation,sequence_kind=kind))
        return k

    for (source,role), rr in pools.items():
        for r in rr:
            uid=source+':'+r['id']; keys=[]; annotation=''
            if source == 'pinder' and r.get('unit') == 'Fv':
                folder=args.runtime/'pretraining/pinder-pkai-fv-v1/entries'/r['id']
                meta=read(folder/'structure.json')
                if digest(folder/'AB.cif.gz') != meta['cif_sha256']:
                    raise ValueError('Native Fv structure checksum differs: '+r['id'])
                chains=native_fv_sequences(folder/'AB.cif.gz')
                for name,annotation in [('H','antibody_H'),('L','antibody_L'),('G','antigen')]:
                    keys.append(member(uid,chains[name],name,annotation,'native_observed_coordinates'))
                # References can contain unresolved residues: retain as separate
                # exposure evidence, never pretend they are the native inputs.
                for name in ['H','L']:
                    keys.append(member(uid,meta['assembly']['references'][name]['sequence'],name,
                                       'antibody_'+name,'numbering_reference_including_unresolved'))
                observed_pins[r['id']]=dict(structure_json=digest(folder/'structure.json'),
                                            cif=meta['cif_sha256'])
                annotation='paired_Fv'
            elif source == 'pinder':
                record=legacy[source][r['id']]
                for i,sequence in enumerate(record['_rl']):
                    keys.append(member(uid,sequence,'RL'[i],
                                       'AbAg_member_role_pending' if r['stratum']=='Ab/Ag' else 'protein'))
                annotation='AbAg' if r['stratum']=='Ab/Ag' else ''
            else:
                record=legacy[source][r['id']]
                sequences=record['_sequences']
                if {skey(s) for s in sequences} != set(prepared['parent_sequence_keys'][r['id']]):
                    raise ValueError('pKPDB parent sequence set differs: '+r['id'])
                if r['id'] in annotations:
                    # Only 121 antibody-bearing source parents need chain-level
                    # descriptions. Other parents remain honest sequence sets.
                    path=args.runtime/'pretraining/pkpdb-full-v1/entries'/r['id']/'defects.json'
                    chains=read(path)['sequences']
                    evidence={e['chain']:e['role'] for e in annotations[r['id']]['role_evidence']}
                    for c in chains:
                        keys.append(member(uid,c['sequence'],c['chain'],
                                           'antibody' if evidence.get(c['chain'])=='Ab' else 'antigen_context'))
                    observed_pins['pkpdb:'+r['id']]=dict(defects_json=digest(path))
                    annotation=annotations[r['id']]['status']
                else:
                    for i,sequence in enumerate(sequences):
                        keys.append(member(uid,sequence,'unique-sequence-'+str(i),'protein'))
            groups=json.loads(r['source_groups']) if r.get('source_groups') else [r['group']]
            units.append(dict(unit_id=uid,source=source,source_native_id=r['id'],
                              pdb_id=r['pdb_id'],original_role=role,stratum=r['stratum'],
                              sequence_keys=sorted(set(keys)),variable_keys=[],antigen_keys=[],
                              source_groups=groups,source_annotation=annotation,
                              sabdab_split='',sabdab_component='',protein_partner=True,
                              prepared_key=None,teacher_context_id=None,
                              preparation_status='not_established_by_sequence_inventory'))
    native_keys=set(seqs)
    with paths['sabdab_csv'].open() as f:
        sab=list(csv.DictReader(f))
    assert len(sab)==16415
    antigen_targets=[]
    for r in sab:
        if not r['agchains']:
            continue
        uid='sabdab:'+r['INSTANCE']; keys=[]; variable_keys=[]; agkeys=[]
        for role in ['H','L']:
            for col in [role+'seq',role+'seq_expected']:
                if r[col]:
                    keys.append(member(uid,r[col],r[role+'chain'],'antibody_'+role,col))
            col='V'+role+'_numerable_seq'
            if r[col]:
                k=member(uid,r[col],r[role+'chain'],'antibody_'+role,'published_variable_domain')
                keys.append(k);variable_keys.append(k)
                numbering=[[int(p),str(i).strip(),aa] for aa,(p,i) in json.loads(r['V'+role+'_numbering_list']) if aa!='-']
                if r['type']!='VNAR':
                    d=dict(chain_type=role,start=0,end=len(r[col])-1,numbering=numbering,
                           annotation_source='SAbDab2_ML_v0.3.0_IMGT')
                    # Identical sequences can have different published IMGT
                    # framework gap placements. Preserve alternatives; do not
                    # conflate position identity with sequence identity.
                    variants=official_domains.setdefault(k,[])
                    if d not in variants:
                        variants.append(d)
        chains=r['agchains'].split('/');types=r['agtypes'].split('/')
        resolved=r['agresolvedseqs'].split('/');expected=r['agexpectedseqs'].split('/')
        if not (len(chains)==len(types)==len(resolved)==len(expected)):
            raise ValueError('Antigen metadata does not align: '+r['INSTANCE'])
        for name,typ,obs,exp in zip(chains,types,resolved,expected):
            if typ not in {'PROTEIN','PEPTIDE'}:
                continue
            for kind,sequence in [('resolved',obs),('expected',exp)]:
                if not sequence:
                    continue
                k=member(uid,sequence,name,'antigen',kind+'_antigen')
                keys.append(k);agkeys.append(k)
                antigen_targets.append(dict(unit_id=uid,sequence_key=k,antigen_type=typ,
                                            sequence_kind=kind,component=r['ab_ag_cluster'],
                                            original_role=r['ab_ag_split']))
        units.append(dict(unit_id=uid,source='sabdab',source_native_id=r['INSTANCE'],
                          pdb_id=r['PDB_ID'][-4:].lower(),original_role=r['ab_ag_split'],
                          stratum='Ab/Ag',sequence_keys=sorted(set(keys)),
                          variable_keys=sorted(set(variable_keys)),antigen_keys=sorted(set(agkeys)),
                          source_groups=[r['ab_ag_cluster']],source_annotation=r['type'],
                          sabdab_split=r['ab_ag_split'],sabdab_component=r['ab_ag_cluster'],
                          protein_partner=bool(agkeys),prepared_key=None,teacher_context_id=None,
                          preparation_status='native_source_reconstruction_required_not_ML_crop'))
    if len({u['unit_id'] for u in units}) != len(units):
        raise ValueError('Source unit appears more than once across roles')
    # Number every unassessed existing-pool sequence, not just prefilter hits.
    # Cache negative results as attempted/no-HKL-domain, never proof of non-Ig.
    todo={k:seqs[k] for k in native_keys if k not in attempted or
          any('numbering' not in d for d in raw.get(k,[]))}
    refs=read(paths['reference_sequences'])
    for scope in ['validation','heldout']:
        for sequence in refs[scope]:
            k=add_sequence(sequence)
            if any('numbering' not in d for d in raw.get(k,[])) or k not in attempted:
                todo[k]=sequence
    ambiguous_official={k:v for k,v in official_domains.items() if len(v)>1}
    for u in units:
        u['official_numbering_variants']=any(k in ambiguous_official for k in u['variable_keys'])
    for k,domains in official_domains.items():
        if k not in raw or not raw[k]:
            raw[k]=domains
    pq.write_table(pa.Table.from_pylist(units),out/'units.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(memberships),out/'sequence-membership.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(antigen_targets),out/'sabdab-antigen-targets.parquet',compression='zstd')
    write_fasta(out/'sequences.fasta',seqs)
    atomic_json(out/'numbering-input.json',todo)
    atomic_json(out/'numbering-cached.json',dict(numbered=raw,attempted=sorted(attempted),
                                               official_variable_keys=sorted(official_domains)))
    atomic_json(out/'official-numbering-variants.json',dict(variants=ambiguous_official,
                note='Alternative published annotations, not distinct biological domains or prepared units.'))
    atomic_json(out/'native-structure-pins.json',observed_pins)
    pins={name:dict(path=str(p),sha256=digest(p)) for name,p in paths.items()}
    manifest=dict(status='prepared_inventory_not_a_release',policy=policy.to_dict(),policy_sha256=policy.sha256,
                  input_pins=pins,code_sha256=digest(__file__),slurm_job_id=os.environ['SLURM_JOB_ID'],
                  baseline=str(args.baseline),runtime=str(args.runtime),sabdab_audit=str(args.sabdab),
                  counts=dict(units=len(units),sequence_memberships=len(memberships),unique_sequences=len(seqs),
                              existing_pool_unique_sequences=len(native_keys),numbering_needed=len(todo),
                              official_sequences_with_numbering_variants=len(ambiguous_official),
                              source_roles=dict(Counter(u['source']+'/'+u['original_role'] for u in units))),
                  no_current_release_mutation=True,no_new_teacher_or_features=True,
                  limitations=['Inventory sequence sets are not prepared-structure identity proofs.',
                               'No-HKL-domain is an attempted numbering result, not proof of non-antibody identity.',
                               'Antigen homology and unmapped structural interfaces require later checks.'],
                  wall_s=time.monotonic()-started)
    atomic_json(out/'inventory-manifest.json',manifest)
    print(json.dumps(manifest['counts'],indent=2),flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare')
    for name in ['baseline','runtime','v5','sabdab','pkpdb-inventory','policy','output']:
        q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--workers',type=int,default=32)
    args=p.parse_args()
    if args.action=='prepare':prepare(args)


if __name__=='__main__':
    cli()
