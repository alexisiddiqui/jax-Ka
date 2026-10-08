"""Antibody role review, external sequence reservations and usable-size proposal."""
import csv
import json
import os
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from .runtime import require_compute, atomic_json, digest, config_hash
from .antigen_split import Groups


def prepare(root):
    require_compute()
    from biotite.structure.io import pdbx
    from .pool import entry_id
    root=Path(root); out=root/'usable-proposal-v2'; out.mkdir(exist_ok=True)
    runtime=Path(os.environ['PKABENCH_RUNTIME'])
    index=json.loads((root/'index.json').read_text())
    metadata=runtime/'universe/metadata-20261003/sabdab2-summary.csv'
    annotations=defaultdict(list)
    with metadata.open() as f:
        for r in csv.DictReader(f): annotations[entry_id(r['PDB'])].append(r)
    sources={}
    for c in index['sources']:
        manifest=json.loads((Path(c['campaign'])/'manifest.json').read_text())
        for r in manifest['candidates']:
            sources[r['pdb_id']]=Path(manifest['source_audit'])/'sources'/f"{r['pdb_id']}.cif"
    suspects={r['complex_id'] for r in json.loads((root/'antigen-proposal-v1/antibody_review.json').read_text())}
    maps={}; review=[]
    for r in index['candidates']:
        cid=r['complex_id']; pdb=r['pdb_id'].split('-')[0]
        rec={'complex_id':cid,'pdb_id':r['pdb_id'],'role':'general','antigen_chains':[], 'benchmark_eligible':True}
        if r['stratum']=='antibody':
            rec.update(role='antibody_antigen',antigen_chains=r['partner_B_chains'],evidence='existing SAbDab protein-antigen mapping')
        elif pdb in annotations:
            if r['pdb_id'] not in maps:
                cat=pdbx.CIFFile.read(sources[r['pdb_id']]).block['atom_site']
                mapping=defaultdict(set)
                for label,author in zip(cat['label_asym_id'].as_array(str),cat['auth_asym_id'].as_array(str)):
                    mapping[str(label)].add(str(author))
                maps[r['pdb_id']]=mapping
            mapping=maps[r['pdb_id']]
            selected=set(r['partner_A_chains']+r['partner_B_chains'])
            outcomes=set()
            for a in annotations[pdb]:
                ab={c for c in selected if mapping.get(c,set()) & {a['Hchain'],a['Lchain']}}
                antigen_auth={c.strip() for c,t in zip(a['antigen_chain'].split('|'),a['antigen_type'].split('|')) if t.strip().upper()=='PROTEIN'}
                ag={c for c in selected if mapping.get(c,set()) & antigen_auth}
                if selected<=ab: outcomes.add(('antibody_internal',()))
                elif ab and ag and selected<=ab|ag and not ab&ag:
                    # Never relabel a within-partner antibody/antigen mixture.
                    parts=[set(r['partner_A_chains']),set(r['partner_B_chains'])]
                    if any(p<=ab and selected-p<=ag for p in parts): outcomes.add(('antibody_antigen',tuple(sorted(ag))))
            if len(outcomes)==1:
                role,ags=outcomes.pop(); rec.update(role=role,antigen_chains=list(ags),benchmark_eligible=role!='antibody_internal',evidence='full SAbDab author-chain mapping')
            elif outcomes or cid in suspects:
                rec.update(role='unresolved_antibody',benchmark_eligible=False,evidence='ambiguous or incomplete SAbDab mapping; review required')
        elif cid in suspects:
            rec.update(role='unresolved_antibody',benchmark_eligible=False,evidence='description flag without SAbDab mapping')
        review.append(rec)
    atomic_json(out/'roles.json',{'rows':review,'histogram':dict(Counter(r['role'] for r in review)), 'metadata_sha256':digest(metadata),
        'limits':'Antibody-internal and unresolved pairs stay in graph/artifacts but are not counted toward benchmark quotas. Name-negative entries without SAbDab annotations are not proven antibody-free.'})
    inventory=json.loads((root/'experimental-reservation-inventory.json').read_text())
    refs=list(inventory['proteins'])
    # Literature-supported seed: include both partners from the canonical complex.
    refs.extend([{'pdb_id':'1brs','chain':c,'uniprot_ids':[],'source':'barnase-barstar seed'} for c in ('A','D')])
    atomic_json(out/'reference-inventory.json',{'rows':refs,'complete_set2':False,'source_sha256':digest(root/'experimental-reservation-inventory.json')})
    print(json.dumps({'roles':dict(Counter(r['role'] for r in review)),'reference_chains':len(refs)}),flush=True)


def references(root):
    require_compute()
    from .download import fetch
    from biotite.structure.io import pdbx
    root=Path(root); out=root/'usable-proposal-v2'
    refs=list({(r['pdb_id'],r['chain']):r for r in json.loads((out/'reference-inventory.json').read_text())['rows']}.values()); ids=sorted({r['pdb_id'] for r in refs})
    def download(pdb):
        import re
        if not re.fullmatch(r'[0-9][a-z0-9]{3}',pdb):
            return pdb,{'status':'not_pdb_identifier','error':'Released table uses a model/mutant identifier, not a PDB accession'}
        path=out/'reference-cifs'/f'{pdb}.cif'
        try:
            receipt=fetch(f'https://files.rcsb.org/download/{pdb}.cif',path) if not path.exists() else {'sha256':digest(path)}
            return pdb,{'status':'complete',**receipt}
        except Exception as exc: return pdb,{'status':'failed','error':str(exc)}
    with ThreadPoolExecutor(max_workers=8) as pool: receipts=dict(pool.map(download,ids))
    atomic_json(out/'reference-downloads.json',receipts)
    sequences={}; resolved=[]; failed=[]
    for r in refs:
        pdb=r['pdb_id']; chain=r['chain']
        try:
            if receipts[pdb]['status']!='complete': raise ValueError('download failed')
            if pdb not in sequences:
                cif=pdbx.CIFFile.read(out/'reference-cifs'/f'{pdb}.cif'); poly=cif.block['entity_poly']
                mapping=defaultdict(set)
                for names,seq in zip(poly['pdbx_strand_id'].as_array(str),poly['pdbx_seq_one_letter_code_can'].as_array(str)):
                    seq=''.join(str(seq).split())
                    if not seq or not set(seq)<=set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'): continue
                    for name in str(names).split(','): mapping[name.strip()].add(seq)
                sequences[pdb]=mapping
            choices=sequences[pdb].get(chain,set())
            if len(choices)!=1: raise ValueError(f'Expected one sequence for author chain, got {len(choices)}')
            seq=next(iter(choices)); resolved.append({**r,'sequence':seq,'id':'r'+config_hash((pdb,chain))[:20]})
        except Exception as exc: failed.append({**r,'error':str(exc)})
    proxies=[]
    for accession in sorted({u for r in failed for u in r['uniprot_ids'] if u and u.upper() not in ('NA','NAN','NONE')}):
        path=out/'reference-uniprot'/f'{accession}.fasta'
        try:
            receipt=fetch(f'https://rest.uniprot.org/uniprotkb/{accession}.fasta',path) if not path.exists() else {'sha256':digest(path)}
            lines=path.read_text().splitlines()
            if sum(x.startswith('>') for x in lines)!=1: raise ValueError('Expected a single UniProt FASTA sequence')
            seq=''.join(x.strip() for x in lines if not x.startswith('>'))
            if not seq or not set(seq)<=set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'): raise ValueError('Invalid UniProt sequence')
            proxies.append({'id':'u'+config_hash(accession)[:20],'sequence':seq,'uniprot_id':accession,'receipt':receipt,
                            'scope':'Parent-sequence family reservation, not the exact experimental mutant construct'})
        except Exception as exc: proxies.append({'uniprot_id':accession,'error':str(exc)})
    successful_proxies=[p for p in proxies if 'sequence' in p]
    atomic_json(out/'reference-sequences.json',{'resolved':resolved,'failed':failed,'parent_proxies':proxies,'complete_set2':False})
    fasta=out/'references.fasta'; fasta.write_text(''.join(f">{r['id']}\n{r['sequence']}\n" for r in resolved+successful_proxies))
    if not resolved: raise ValueError('No external reference sequences resolved')
    binary=Path(os.environ['PKABENCH_RUNTIME'])/'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
    command=[str(binary),'easy-search',str(fasta),str(root/'sequence/sequences.fasta'),str(out/'reference-hits.tsv'),str(Path(os.environ['TMPDIR'])/'reference-search'),
             '--threads','1','--min-seq-id','0.3','-c','0.8','--cov-mode','0','--alignment-mode','3','--max-seqs','10000','-s','7.5','--split-memory-limit','2G','--format-output','query,target,fident,qcov,tcov']
    with (out/'reference-search.log').open('w') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=5400)
    atomic_json(out/'reference-report.json',{'resolved':len(resolved),'failed':failed,'parent_proxies':[{k:v for k,v in p.items() if k!='sequence'} for p in proxies],'command':command,'fasta_sha256':digest(fasta),'hits_sha256':digest(out/'reference-hits.tsv'),'complete_set2':False})
    print(json.dumps({'resolved':len(resolved),'failed':len(failed)}),flush=True)


def allocate(root):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=Path(root); out=root/'usable-proposal-v2'
    index=json.loads((root/'index.json').read_text()); rows=index['candidates']
    roles={r['complex_id']:r for r in json.loads((out/'roles.json').read_text())['rows']}
    prior={r['complex_id']:r for r in pq.read_table(root/'antigen-proposal-v1/proposal.parquet').to_pylist()}
    verification=json.loads((root/'antigen-proposal-v1/verification.json').read_text())
    if digest(root/'antigen-proposal-v1/proposal.parquet')!=verification['proposal_sha256']: raise ValueError('Verified source proposal changed')
    reference_receipt=json.loads((out/'reference-report.json').read_text())
    if digest(out/'reference-hits.tsv')!=reference_receipt['hits_sha256']: raise ValueError('Reference hits changed')
    nodes={r['complex_id']:['s'+config_hash(c['sequence'])[:20] for c in r['chains'] if roles[r['complex_id']]['role']!='antibody_antigen' or c['chain'] in roles[r['complex_id']]['antigen_chains']] for r in rows}
    allowed={n for ns in nodes.values() for n in ns}; graph=Groups(allowed); edges=[]
    for line in (root/'sequence/hits.tsv').read_text().splitlines():
        a,b,i,q,t=line.split('\t')
        if a in allowed and b in allowed and float(i)>=.3 and min(float(q),float(t))>=.8:
            graph.join(a,b); edges.append((a,b))
    for ns in nodes.values():
        if not ns: raise ValueError('Empty split partner')
        for n in ns[1:]: graph.join(ns[0],n)
    components=defaultdict(list)
    for cid,ns in nodes.items(): components[graph.find(ns[0])].append(cid)
    grouped={'c'+config_hash(sorted(ids))[:20]:sorted(ids) for ids in components.values()}
    component={cid:g for g,ids in grouped.items() for cid in ids}
    reference_nodes=set()
    for line in (out/'reference-hits.tsv').read_text().splitlines():
        a,b,i,q,t=line.split('\t')
        if b in allowed and float(i)>=.3 and min(float(q),float(t))>=.8: reference_nodes.add(b)
    reserved_roots={graph.find(n) for n in reference_nodes}
    forced={component[c] for c,ns in nodes.items() if graph.find(ns[0]) in reserved_roots}
    weights={g:sum(roles[c]['benchmark_eligible'] and prior[c]['eval_interface_sites']>0 for c in ids) for g,ids in grouped.items()}
    trainweights={g:sum(roles[c]['benchmark_eligible'] and prior[c]['train_interface_sites']>0 for c in ids) for g,ids in grouped.items()}
    # Bounded subset sum: closest count, then lowest training opportunity cost.
    def choose(available,target):
        if target<=0: return set()
        dp={0:(0,())}; cap=target+max((weights[g] for g in available),default=0)
        for g in sorted(available):
            w=weights[g]
            if not w: continue
            for n,(cost,chosen) in list(dp.items()):
                new=n+w
                if new>cap: continue
                candidate=(cost+trainweights[g],chosen+(g,))
                if new not in dp or candidate[0]<dp[new][0]: dp[new]=candidate
        best=min(dp,key=lambda n:(abs(n-target),n<target,dp[n][0]))
        return set(dp[best][1])
    test=forced|choose(set(grouped)-forced,500-sum(weights[g] for g in forced))
    val=choose(set(grouped)-test,150)
    assignment={g:('test' if g in test else 'val' if g in val else 'train') for g in grouped}
    result=[]
    for cid in sorted(component):
        # Old novelty labels are invalid after reassignment and are deliberately omitted.
        result.append({'complex_id':cid,'component_id':component[cid],'split':assignment[component[cid]],
                       'role':roles[cid]['role'],'benchmark_eligible':roles[cid]['benchmark_eligible'],
                       'reference_reserved':component[cid] in forced,
                       **{k:prior[cid][k] for k in ('train_sites','eval_sites','train_interface_sites','eval_interface_sites')}})
    node_split={n:assignment[component[c]] for c,ns in nodes.items() for n in ns}
    assert all(node_split[a]==node_split[b] for a,b in edges)
    assert all(assignment[g]=='test' for g in forced)
    assert len(result)==len(rows)==len({r['complex_id'] for r in result})
    cdr_nodes={}
    for r in rows:
        if r['stratum']!='antibody': continue
        fields=['CDR-H1','CDR-H2','CDR-H3']+(['CDR-L1','CDR-L2','CDR-L3'] if len(r['partner_A_chains'])==2 else [])
        parts=[''.join(r['sabdab'].get(f,'').split()).upper() for f in fields]
        if all(p and p not in ('NA','NAN','NONE') and set(p)<=set('ACDEFGHIKLMNPQRSTVWY') for p in parts):
            cdr_nodes[r['complex_id']]='s'+config_hash(''.join(parts))[:20]
    cdr_set=set(cdr_nodes.values()); cdr_hits=[]
    for line in (root/'sequence/hits.tsv').read_text().splitlines():
        a,b,i,q,t=line.split('\t')
        if a in cdr_set and b in cdr_set and min(float(q),float(t))>=.8: cdr_hits.append((a,b,float(i)))
    novelty={}
    for cutoff in (30,50,70,90):
        cg=Groups(cdr_set)
        for a,b,i in cdr_hits:
            if i>=cutoff/100: cg.join(a,b)
        train={cg.find(n) for c,n in cdr_nodes.items() if assignment[component[c]]=='train'}
        count=Counter()
        for r in result:
            label=None
            if r['role']=='antibody_antigen' and r['split']=='test':
                n=cdr_nodes.get(r['complex_id'])
                label='unknown' if n is None else 'seen_antibody' if cg.find(n) in train else 'both_unseen'
                if r['benchmark_eligible'] and r['eval_interface_sites']>0: count[label]+=1
            r['novelty_cdr_'+str(cutoff)]=label
        novelty[str(cutoff)]=dict(count)
    pq.write_table(pa.Table.from_pylist(result),out/'proposal.parquet')
    assert pq.read_table(out/'proposal.parquet').to_pylist()==result
    summary={}
    for split in ('train','val','test'):
        rs=[r for r in result if r['split']==split]; mode='train' if split=='train' else 'eval'
        summary[split]={'candidate_pairs':len(rs),'eligible_pairs':sum(r['benchmark_eligible'] and r[mode+'_sites']>0 for r in rs),
                       'interface_eligible_pairs':sum(r['benchmark_eligible'] and r[mode+'_interface_sites']>0 for r in rs),
                       'components':len({r['component_id'] for r in rs}),
                       'antibody_interface_pairs':sum(r['benchmark_eligible'] and r['role']=='antibody_antigen' and r[mode+'_interface_sites']>0 for r in rs)}
    reference=json.loads((out/'reference-report.json').read_text())
    report={'frozen':False,'splits':summary,'antibody_test_interface_novelty':novelty,'largest_component':max(map(len,grouped.values())),'reference_reserved_components':len(forced),
            'resolved_reference_chains':reference['resolved'],'failed_reference_chains':reference['failed'],'parent_sequence_proxies':reference.get('parent_proxies',[]),
            'roles':dict(Counter(r['role'] for r in result)),'checks':'All candidates retained; protein similarity edges remain within splits; all reference-hit components assigned test.',
            'targets_met':summary['test']['interface_eligible_pairs']>=500 and summary['val']['interface_eligible_pairs']>=150 and summary['train']['interface_eligible_pairs']>=500,
            'limitations':['Full set-2 curation and PKAD-3 source completeness remain unresolved; no production freeze.',
                          'Unknown antibody roles excluded from quotas, retained in conservative graph.',
                          'Name-negative unannotated antibodies may remain; CDR novelty is provisional and uses annotated training candidates at four sensitivity thresholds.',
                          'Subset selection favors retaining training-eligible pairs; this is a feasibility proposal, not a random representative test sample.',
                          'External sequence matching uses 30% identity and 80% bidirectional coverage; fragments may evade this rule.'],
            'proposal_sha256':digest(out/'proposal.parquet'),'roles_sha256':digest(out/'roles.json'),
            'reference_report_sha256':digest(out/'reference-report.json'),'implementation_sha256':digest(Path(__file__))}
    atomic_json(out/'report.json',report); print(json.dumps(report,indent=2),flush=True)
