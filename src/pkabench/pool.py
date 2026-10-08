"""Frozen, bounded coordinate acquisition; failures remain in the denominator."""
import gzip
import json
import random
import time
from pathlib import Path
from .download import fetch
from .runtime import atomic_json, digest, require_compute


def entry_id(value):
    value=value.lower()
    return value[8:] if value.startswith('pdb_0000') and len(value)==12 else value


def initialise(metadata, out, count=1000, antibody_count=250, seed=20261003, exclude_pool=None):
    require_compute(); metadata=Path(metadata); out=Path(out)
    assemblies=json.loads((metadata/'assemblies.json').read_text())['assembly_ids']
    excluded=set()
    if exclude_pool is not None:
        excluded={r['assembly_id'] for r in json.loads((Path(exclude_pool)/'download-manifest.json').read_text())['assemblies']}
        assemblies=[a for a in assemblies if a not in excluded]
    annotations=json.loads((metadata/'sabdab-candidates.json').read_text())['candidates']
    by_entry={}
    for row in annotations: by_entry.setdefault(entry_id(row['PDB']),[]).append(row)
    antibody=sorted(a for a in assemblies if entry_id(a.rsplit('-',1)[0]) in by_entry)
    general=sorted(set(assemblies)-set(antibody)); rng=random.Random(seed)
    selected=[(a,'antibody') for a in rng.sample(antibody,min(antibody_count,len(antibody)))]
    selected += [(a,'general') for a in rng.sample(general,min(count-len(selected),len(general)))]
    manifest={'seed':seed,'requested':count,'excluded_previous_assemblies':len(excluded),'exclude_pool':str(exclude_pool) if exclude_pool else None,'available':{'general':len(general),'antibody':len(antibody)},
        'source_hashes':{n:digest(metadata/n) for n in ('assemblies.json','sabdab-candidates.json')},
        'selection':'Seeded sampling without replacement, stratified by SAbDab protein-antigen annotation; failures are not replaced.',
        'production_allowed':False,'assemblies':[]}
    for identifier,stratum in selected:
        entry,assembly=identifier.rsplit('-',1); entry=entry_id(entry)
        manifest['assemblies'].append({'assembly_id':identifier,'pdb_id':f'{entry}-assembly{assembly}',
            'stratum':stratum,'url':f'https://files.rcsb.org/download/{entry}-assembly{assembly}.cif.gz',
            'sabdab':by_entry.get(entry,[])})
    out.mkdir(parents=True,exist_ok=False); atomic_json(out/'download-manifest.json',manifest)
    print(json.dumps({'selected':len(selected),'available':manifest['available']}))


def download(out, shard=0, shards=8):
    require_compute(); out=Path(out)
    manifest=json.loads((out/'download-manifest.json').read_text())
    for row in manifest['assemblies'][shard::shards]:
        name=row['pdb_id']; receipt=out/'receipts'/f'{name}.json'; target=out/'sources'/f'{name}.cif'
        if receipt.exists():
            prior=json.loads(receipt.read_text())
            if prior['status']=='complete' and target.exists() and digest(target)==prior['source_sha256']: continue
            if prior['status']=='failed': continue
            raise ValueError(f'cached coordinate integrity failure: {name}')
        try:
            compressed=out/'compressed'/f'{name}.cif.gz'; result=fetch(row['url'],compressed)
            target.parent.mkdir(parents=True,exist_ok=True); temp=target.with_suffix('.partial'); size=0
            with gzip.open(compressed,'rb') as source, temp.open('wb') as dest:
                while chunk:=source.read(1024*1024):
                    size+=len(chunk)
                    if size>100*1024*1024: raise ValueError('uncompressed coordinate limit exceeded')
                    dest.write(chunk)
            with temp.open('rb') as source:
                if not source.read(100).lstrip().startswith(b'data_'): raise ValueError('not mmCIF')
            temp.replace(target)
            result.update(status='complete',source_sha256=digest(target),uncompressed_bytes=size)
        except Exception as exc: result={'status':'failed','error':str(exc)}
        atomic_json(receipt,{'pdb_id':name,**result}); time.sleep(.25)
    atomic_json(out/'download-shards'/f'{shard}.json',{'status':'complete','shard':shard,'shards':shards})


def index(out):
    import itertools
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from scipy.spatial import cKDTree
    from .conformers import resolve
    from .dataset_audit import chain_metadata
    from .prep import CANONICAL, Rejection
    from .runtime import config_hash
    require_compute(); out=Path(out)
    manifest=json.loads((out/'download-manifest.json').read_text()); candidates=[]; records=[]
    for row in manifest['assemblies']:
        receipt=json.loads((out/'receipts'/f"{row['pdb_id']}.json").read_text())
        record={'pdb_id':row['pdb_id'],'stratum':row['stratum'],'download_status':receipt['status']}
        if receipt['status']!='complete': records.append(record); continue
        try:
            source=out/'sources'/f"{row['pdb_id']}.cif"
            if digest(source)!=receipt['source_sha256']: raise ValueError('source hash mismatch')
            cif=pdbx.CIFFile.read(source); cat=cif.block['atom_site']
            mapping={}
            for label,author in zip(cat['label_asym_id'].as_array(str),cat['auth_asym_id'].as_array(str)):
                mapping.setdefault(str(author),set()).add(str(label))
            cif,_=resolve(cif,sorted(set(cat['label_asym_id'].as_array(str))))
            atoms=pdbx.get_structure(cif,model=1,altloc='first',use_author_fields=False)
            protein=atoms[np.isin(atoms.res_name,list(CANONICAL)) & ~np.isin(np.char.upper(atoms.element),['H','D'])]
            if len(struc.get_residue_starts(protein))>1500:
                raise Rejection('size_cap','Assembly exceeds existing 1500-residue cap before contact enumeration')
            chains=sorted(set(protein.chain_id)); arrays={c:protein[protein.chain_id==c] for c in chains}
            trees={c:cKDTree(a.coord) for c,a in arrays.items()}; contacts={}
            for a,b in itertools.combinations(chains,2):
                aa,bb=arrays[a],arrays[b]
                def nres(at,mask): return len(set(zip(at.res_id[mask].tolist(),at.ins_code[mask].tolist())))
                contacts[a,b]=nres(aa,trees[b].query(aa.coord)[0]<=5)+nres(bb,trees[a].query(bb.coord)[0]<=5)
            def contact(a,b): return contacts.get(tuple(sorted((a,b))),0)
            choices=[]; mapping_failures=0
            if row['stratum']=='general':
                choices=[([a],[b],score,None) for (a,b),score in contacts.items() if score>=10]
            else:
                for annotation in row['sabdab']:
                    heavy=sorted(mapping.get(annotation['Hchain'],set()) & set(chains))
                    light_id=annotation['Lchain']; has_light=light_id.strip().lower() not in ('','na','nan','none','-')
                    light=sorted(mapping.get(light_id,set()) & set(chains)) if has_light else [None]
                    if not heavy or not light: mapping_failures+=1; continue
                    for antigen,kind in zip(annotation['antigen_chain'].split('|'),annotation['antigen_type'].split('|')):
                        if kind.upper()!='PROTEIN': continue
                        targets=sorted(mapping.get(antigen.strip(),set()) & set(chains))
                        if not targets: mapping_failures+=1
                        for h,l,b in itertools.product(heavy,light,targets):
                            a=[h]+([l] if l else [])
                            if len(set(a+[b]))!=len(a)+1 or (l and contact(h,l)<10): continue
                            score=sum(contact(c,b) for c in a)
                            if score>=10: choices.append((a,[b],score,annotation))
            choices.sort(key=lambda x:(-x[2],x[0],x[1])); unique={}
            for a,b,score,annotation in choices: unique.setdefault((tuple(a),tuple(b)),(a,b,score,annotation))
            cap=2 if row['stratum']=='antibody' else 3
            record.update(status='indexed',interfaces_before_cap=len(unique),mapping_failures=mapping_failures)
            for a,b,score,annotation in list(unique.values())[:cap]:
                item={'complex_id':config_hash((row['pdb_id'],a,b))[:16],'pdb_id':row['pdb_id'],
                    'partner_A_chains':a,'partner_B_chains':b,'source_sha256':receipt['source_sha256'],
                    'stratum':row['stratum'],'contact_residues_5A':score,'sabdab':annotation,
                    'chains':chain_metadata(cif,protein,a+b)}
                candidates.append(item)
            record['candidates']=min(cap,len(unique))
        except Rejection as exc: record.update(status='rejected',code=exc.code,detail=str(exc))
        except Exception as exc:
            import traceback
            record.update(status='pipeline_error',detail=str(exc),traceback=traceback.format_exc())
        records.append(record)
    atomic_json(out/'index.json',{'candidates':candidates,'assemblies':records,
        'download_manifest_sha256':digest(out/'download-manifest.json'),
        'selection':'First positive-occupancy conformer; >=10 contact residues at 5 A; rank by contacts, max 3 general or 2 antibody interfaces per assembly. SAbDab H+L versus one annotated protein antigen; chain copies require physical contacts. Chemistry and BSA gates follow in curation.',
        'production_allowed':False})
    print(json.dumps({'assemblies':len(records),'candidates':len(candidates),'pipeline_errors':[r for r in records if r.get('status')=='pipeline_error']}))


def sequence(out):
    import os
    import subprocess
    from collections import Counter
    from .runtime import config_hash
    require_compute(); out=Path(out); rows=json.loads((out/'index.json').read_text())['candidates']
    work=out/'sequence'; work.mkdir(exist_ok=True); seqs={}; pairs={}; missing=[]
    def key(seq):
        identifier='s'+config_hash(seq)[:20]; seqs[identifier]=seq; return identifier
    for row in rows:
        nodes=[key(c['sequence']) for c in row['chains']]
        cdrnodes=nodes
        if row['stratum']=='antibody':
            ann=row['sabdab']; fields=['CDR-H1','CDR-H2','CDR-H3']
            if len(row['partner_A_chains'])==2: fields+=['CDR-L1','CDR-L2','CDR-L3']
            parts=[''.join(ann.get(f,'').split()).upper() for f in fields]
            if all(p and p not in ('NA','NAN','NONE') and set(p)<=set('ACDEFGHIKLMNPQRSTVWY') for p in parts):
                cdrnodes=[key(''.join(parts))]+[key(c['sequence']) for c in row['chains'] if c['chain'] in row['partner_B_chains']]
            else: missing.append(row['complex_id']); cdrnodes=None
        pairs[row['complex_id']]={'full_chain':nodes,'cdr_partner':cdrnodes}
    fasta=work/'sequences.fasta'; fasta.write_text(''.join(f'>{k}\n{s}\n' for k,s in sorted(seqs.items())))
    binary=Path(os.environ['PKABENCH_RUNTIME'])/'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
    hits=work/'hits.tsv'; command=[str(binary),'easy-search',str(fasta),str(fasta),str(hits),str(Path(os.environ['TMPDIR'])/'pool-mmseqs'),
        '--threads','1','--min-seq-id','0.3','-c','0.8','--cov-mode','0','--alignment-mode','3','--max-seqs','10000','-s','7.5','--split-memory-limit','2G','--format-output','query,target,fident,qcov,tcov']
    with (work/'mmseqs.log').open('w') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=5400)
    links=[]
    for line in hits.read_text().splitlines():
        a,b,identity,qcov,tcov=line.split('\t')
        if float(identity)>=.3 and min(float(qcov),float(tcov))>=.8: links.append((a,b))
    summaries={}; assignments={}
    for mode in ('full_chain','cdr_partner'):
        parent={}
        def find(x):
            parent.setdefault(x,x)
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def union(a,b): parent[find(b)]=find(a)
        allowed={n for pair in pairs.values() for n in (pair[mode] or [])}
        for a,b in links:
            if a in allowed and b in allowed: union(a,b)
        for pair in pairs.values():
            nodes=pair[mode]
            if nodes:
                for node in nodes[1:]: union(nodes[0],node)
        assignment={cid:find(pair[mode][0]) for cid,pair in pairs.items() if pair[mode]}
        counts=Counter(assignment.values()); assignments[mode]=assignment
        summaries[mode]={'candidates_with_sequences':len(assignment),'components':len(counts),'largest_component':max(counts.values(),default=0)}
    atomic_json(work/'diversity.json',{'summaries':summaries,'assignments':assignments,'missing_cdr':missing,
        'sequence_sources':dict(Counter(c['sequence_source'] for r in rows for c in r['chains'])),
        'command':command,'binary_sha256':digest(binary),'fasta_sha256':digest(fasta),'hits_sha256':digest(hits),
        'note':'30% identity and 80% coverage on both sequences, complex partner edges. CDR mode concatenates annotated H/L CDRs; missing CDR cases excluded explicitly. Heuristic feasibility audit, not a frozen production split; no examples deleted.'})
    print(json.dumps({'summaries':summaries,'missing_cdr':len(missing)},indent=2))


def report(out,campaign):
    from collections import Counter
    from .runtime import config_hash
    require_compute(); out=Path(out); campaign=Path(campaign)
    index=json.loads((out/'index.json').read_text()); manifest=json.loads((out/'download-manifest.json').read_text())
    rows=[json.loads((campaign/'rows'/f"{r['complex_id']}.json").read_text()) for r in index['candidates']]
    diversity=json.loads((out/'sequence/diversity.json').read_text())
    if digest(out/'sequence/hits.tsv')!=diversity['hits_sha256']: raise ValueError('sequence alignment hash mismatch')
    accepted={r['complex_id'] for r in rows if r['status']=='accepted'}
    prepared={}
    for mode,assignment in diversity['assignments'].items():
        counts=Counter(component for cid,component in assignment.items() if cid in accepted)
        prepared[mode]={'covered_accepted':sum(counts.values()),'components':len(counts),'largest_component':max(counts.values(),default=0)}
    accepted_only={}
    for mode in ('full_chain','cdr_partner'):
        pairs={}
        for row in rows:
            if row['complex_id'] not in accepted: continue
            seqs=[c['sequence'] for c in row['chains']]
            if mode=='cdr_partner' and row['stratum']=='antibody':
                if row['complex_id'] in diversity['missing_cdr']: continue
                fields=['CDR-H1','CDR-H2','CDR-H3']+(['CDR-L1','CDR-L2','CDR-L3'] if len(row['partner_A_chains'])==2 else [])
                seqs=[''.join(''.join(row['sabdab'].get(f,'').split()).upper() for f in fields)]
                seqs += [c['sequence'] for c in row['chains'] if c['chain'] in row['partner_B_chains']]
            pairs[row['complex_id']]=['s'+config_hash(s)[:20] for s in seqs]
        allowed={node for nodes in pairs.values() for node in nodes}; parent={n:n for n in allowed}
        def find(n):
            while parent[n]!=n: parent[n]=parent[parent[n]]; n=parent[n]
            return n
        def union(a,b): parent[find(b)]=find(a)
        for line in (out/'sequence/hits.tsv').read_text().splitlines():
            a,b,identity,qcov,tcov=line.split('\t')
            if a in allowed and b in allowed and float(identity)>=.3 and min(float(qcov),float(tcov))>=.8: union(a,b)
        for nodes in pairs.values():
            for node in nodes[1:]: union(nodes[0],node)
        counts=Counter(find(nodes[0]) for nodes in pairs.values())
        accepted_only[mode]={'covered_accepted':len(pairs),'components':len(counts),'largest_component':max(counts.values(),default=0)}
    result={'selected_assemblies':len(manifest['assemblies']),
        'downloaded':sum(r['download_status']=='complete' for r in index['assemblies']),
        'assembly_discovery_statuses':dict(Counter(r.get('status',r['download_status']) for r in index['assemblies'])),
        'candidates':len(rows),'accepted':len(accepted),'accepted_assemblies':len({r['pdb_id'] for r in rows if r['status']=='accepted'}),
        'preparation_statuses':dict(Counter(r.get('code',r['status']) for r in rows)),
        'strata':{s:dict(Counter(r.get('code',r['status']) for r in rows if r['stratum']==s)) for s in ('general','antibody')},
        'diversity_before_preparation':diversity['summaries'],'diversity_after_preparation':prepared,
        'diversity_accepted_only_graph':accepted_only,
        'teacher_coverage':'Measured separately on expanded-radial references; structural eligibility is not teacher coverage.',
        'limitations':'Assembly-1 bounded stratified sample, contact-ranked interface caps, protein-only chemistry. Sequence clustering groups examples; it does not remove them. No production split or distance cutoff selected.'}
    atomic_json(out/'report.json',result); print(json.dumps(result,indent=2))
