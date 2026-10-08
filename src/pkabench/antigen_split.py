"""Provisional antigen-held-out assignments; never freezes production splits."""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from .runtime import require_compute, atomic_json, config_hash, digest


class Groups:
    def __init__(self, nodes): self.parent = {n:n for n in nodes}
    def find(self,n):
        while self.parent[n]!=n:
            self.parent[n]=self.parent[self.parent[n]]; n=self.parent[n]
        return n
    def join(self,a,b): self.parent[self.find(b)] = self.find(a)


def run(root):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=Path(root); out=root/'antigen-proposal-v1'; out.mkdir(exist_ok=False)
    index=json.loads((root/'index.json').read_text()); rows=index['candidates']
    key=lambda s:'s'+config_hash(s)[:20]
    protein_nodes={}; antibody_nodes={}; roles={}; warnings=[]
    # Name screening is a triage flag, not antibody sequence annotation.
    pattern=re.compile(r'\b(antibody|immunoglobulin|nanobody|fab|scfv|vhh)\b|heavy chain|light chain',re.I)
    for r in rows:
        cid=r['complex_id']; chains=r['chains']
        if r['stratum']=='antibody':
            protein_nodes[cid]=[key(c['sequence']) for c in chains if c['chain'] in r['partner_B_chains']]
            fields=['CDR-H1','CDR-H2','CDR-H3']+(['CDR-L1','CDR-L2','CDR-L3'] if len(r['partner_A_chains'])==2 else [])
            parts=[''.join(r['sabdab'].get(f,'').split()).upper() for f in fields]
            valid=all(p and p not in ('NA','NAN','NONE') and set(p)<=set('ACDEFGHIKLMNPQRSTVWY') for p in parts)
            antibody_nodes[cid]=key(''.join(parts)) if valid else None
            roles[cid]='antibody_antigen'
        else:
            protein_nodes[cid]=[key(c['sequence']) for c in chains]
            suspects=[{'chain':c['chain'],'description':c.get('description','')} for c in chains if pattern.search(c.get('description',''))]
            roles[cid]='unresolved_antibody_candidate' if suspects else 'general'
            if suspects: warnings.append({'complex_id':cid,'pdb_id':r['pdb_id'],'suspect_chains':suspects})
        if not protein_nodes[cid]: raise ValueError(f'Missing protein nodes: {cid}')
    hits=[]
    for line in (root/'sequence/hits.tsv').read_text().splitlines():
        a,b,i,q,t=line.split('\t')
        if float(i)>=.3 and min(float(q),float(t))>=.8: hits.append((a,b,float(i)))
    allowed={n for ns in protein_nodes.values() for n in ns}; graph=Groups(allowed)
    for a,b,i in hits:
        if a in allowed and b in allowed: graph.join(a,b)
    for ns in protein_nodes.values():
        for n in ns[1:]: graph.join(ns[0],n)
    members=defaultdict(list)
    for cid,ns in protein_nodes.items(): members[graph.find(ns[0])].append(cid)
    # Stable IDs independent of union order.
    component={cid:'c'+config_hash(sorted(ids))[:20] for ids in members.values() for cid in ids}
    grouped=defaultdict(list)
    for cid,g in component.items(): grouped[g].append(cid)
    inventory=json.loads((root/'experimental-reservation-inventory.json').read_text())
    direct={x['complex_id'] for x in inventory['direct_pdb_overlaps']}
    forced={component[c] for c in direct}
    target={'train':.8*len(rows),'val':.1*len(rows),'test':.1*len(rows)}
    totals=Counter(); assignment={}
    for g in sorted(forced):
        assignment[g]='test'; totals['test']+=len(grouped[g])
    for g in sorted(set(grouped)-forced,key=lambda g:(-len(grouped[g]),g)):
        split=max(('train','val','test'),key=lambda s:target[s]-totals[s])
        assignment[g]=split; totals[split]+=len(grouped[g])
    byid={r['complex_id']:r for r in rows}; sitecounts=defaultdict(Counter)
    for source in index['sources']:
        campaign=Path(source['campaign'])
        if digest(campaign/'site_masks.parquet')!=source['mask_sha256']: raise ValueError('Masks changed')
        for s in pq.read_table(campaign/'site_masks.parquet').to_pylist():
            for mode in ('train','eval'):
                if s[mode+'_mask']:
                    sitecounts[s['complex_id']][mode+'_sites']+=1
                    sitecounts[s['complex_id']][mode+'_interface_sites']+=int(s['interface'])
    table=[]
    for cid in sorted(byid):
        table.append({'complex_id':cid,'component_id':component[cid],'split':assignment[component[cid]],'role':roles[cid],
                      'direct_reference_reservation':component[cid] in forced,
                      **{k:sitecounts[cid][k] for k in ('train_sites','eval_sites','train_interface_sites','eval_interface_sites')}})
    # Verify all protein similarity edges and general partner groups stay in one split.
    node_splits=defaultdict(set)
    for cid,ns in protein_nodes.items():
        for n in ns: node_splits[n].add(assignment[component[cid]])
    assert all(len(v)==1 for v in node_splits.values())
    for a,b,i in hits:
        if a in allowed and b in allowed: assert node_splits[a]==node_splits[b]
    cdrs={n for n in antibody_nodes.values() if n}; novelty={}
    for cutoff in (.3,.5,.7,.9):
        cg=Groups(cdrs)
        for a,b,i in hits:
            if a in cdrs and b in cdrs and i>=cutoff: cg.join(a,b)
        train={cg.find(n) for c,n in antibody_nodes.items() if n and assignment[component[c]]=='train'}
        ccounts=Counter(cg.find(n) for n in antibody_nodes.values() if n)
        stats=Counter()
        for r in table:
            if r['role']!='antibody_antigen' or r['split']!='test': continue
            n=antibody_nodes[r['complex_id']]
            label='unknown_cdr' if n is None else ('seen_antibody' if cg.find(n) in train else 'both_unseen')
            r['novelty_cdr_'+str(int(cutoff*100))]=label
            stats[label+'_pairs']+=1
            stats[label+'_eval_eligible_pairs']+=int(r['eval_sites']>0)
            stats[label+'_interface_eligible_pairs']+=int(r['eval_interface_sites']>0)
        novelty[str(cutoff)]={'cdr_families':len(ccounts),'largest_cdr_family_pairs':max(ccounts.values(),default=0),'test':dict(stats)}
    summary={}
    for split in ('train','val','test'):
        subset=[r for r in table if r['split']==split]; mode='train' if split=='train' else 'eval'
        summary[split]={'candidates':len(subset),'eligible_pairs':sum(r[mode+'_sites']>0 for r in subset),
                        'interface_eligible_pairs':sum(r[mode+'_interface_sites']>0 for r in subset),
                        'eligible_sites':sum(r[mode+'_sites'] for r in subset),'roles':dict(Counter(r['role'] for r in subset))}
    for r in table:
        for cutoff in (30,50,70,90):
            r.setdefault('novelty_cdr_'+str(cutoff),None)
    pq.write_table(pa.Table.from_pylist(table),out/'proposal.parquet')
    atomic_json(out/'antibody_review.json',warnings)
    result={'frozen':False,'components':len(grouped),'largest_component':max(map(len,grouped.values())),
            'splits':summary,'antibody_novelty_sensitivity':novelty,'general_pairs_flagged_for_antibody_review':len(warnings),
            'direct_reference_forced_components':len(forced),'checks':'No retained protein-node or protein-similarity edge crosses splits; no pairs dropped.',
            'policy':'80/10/10 by candidate count, antigen-only nodes for annotated antibody-antigen pairs; ordinary partner graph retained; CDRs used only for novelty reporting.',
            'limitations':['General-pool antibody name flags require sequence annotation; conservatively retain their full partner graph until resolved.',
                          'CDR novelty is relative to annotated antibody training candidates only; unannotated antibodies can hide overlap.',
                          '30/50/70/90% CDR thresholds are sensitivity analyses, not calibrated biological family definitions.',
                          'Forced-test reservations use direct PDB overlaps only; external reference sequence matching and full set-2 curation remain incomplete.',
                          'Existing heuristic MMseqs2 matches reused; no new all-against-all search.',
                          'Proposal only; 500-pair test target and giant-component policy must be assessed before freezing.'],
            'source_index_sha256':digest(root/'index.json'),'hits_sha256':digest(root/'sequence/hits.tsv')}
    atomic_json(out/'report.json',result); print(json.dumps(result,indent=2),flush=True)


def verify(root):
    require_compute()
    import pyarrow.parquet as pq
    root=Path(root); out=root/'antigen-proposal-v1'
    table=pq.read_table(out/'proposal.parquet')
    required={'novelty_cdr_'+str(c) for c in (30,50,70,90)}
    if not required.issubset(table.column_names):
        raise ValueError('Novelty columns missing from serialized proposal; regenerate with explicit nullable fields')
    rows=table.to_pylist(); report=json.loads((out/'report.json').read_text())
    assert len(rows)==len({r['complex_id'] for r in rows})==8683
    for cutoff in (30,50,70,90):
        counts=Counter(r['novelty_cdr_'+str(cutoff)]+'_pairs' for r in rows if r['role']=='antibody_antigen' and r['split']=='test')
        expected=report['antibody_novelty_sensitivity'][str(cutoff/100)]['test']
        assert all(expected[k]==v for k,v in counts.items())
    atomic_json(out/'verification.json',{'status':'passed','rows':len(rows),'novelty_columns':sorted(required),'proposal_sha256':digest(out/'proposal.parquet')})
    print('Serialized proposal checks passed',flush=True)
