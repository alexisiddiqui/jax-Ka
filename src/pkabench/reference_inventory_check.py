"""Inspect public reference releases and reconcile local experimental identifiers."""
import csv
import json
import os
from pathlib import Path
from collections import Counter
from .runtime import require_compute,atomic_json,digest


def run(out):
    require_compute()
    from .download import fetch
    out=Path(out); out.mkdir(exist_ok=True)
    runtime=Path(os.environ['PKABENCH_RUNTIME'])
    sources=runtime/'sources/KaMLs/KaML-CBTrees/train_test_split'
    rows={}; files=[]
    for p in sorted(sources.glob('*.csv')):
        files.append({'path':str(p),'sha256':digest(p)})
        with p.open() as f:
            for r in csv.DictReader(f):
                key=tuple(r.get(k,'').strip() for k in ('PDB_ID','Chain','Res_Name','Res_ID','Expt_pKa','Uniprot_ID'))
                rows[key]={k:r.get(k,'').strip() for k in ('PDB_ID','Chain','Res_Name','Res_ID','Expt_pKa','Uniprot_ID','info','infor')}
    records=list(rows.values())
    atomic_json(out/'local-observations.json',records)
    receipts=[]
    urls={'pkad-page.html':'https://database.computchem.org/pkad-3',
          'dash-layout.json':'https://database.computchem.org/_dash-layout',
          'pkad-dash-layout.json':'https://database.computchem.org/pkad-3/_dash-layout',
          'paper.xml':'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12323819/fullTextXML'}
    for name,url in urls.items():
        try:
            receipt=fetch(url,out/name)
            receipts.append({'file':name,'status':'downloaded',**receipt})
        except Exception as e: receipts.append({'file':name,'status':'failed','error':str(e),'url':url})
    tables=[]
    def walk(value,path):
        if isinstance(value,list):
            if value and isinstance(value[0],dict):
                keys=set(value[0])
                if any('pka' in k.lower() or 'pdb' in k.lower() for k in keys): tables.append({'path':path,'rows':len(value),'keys':sorted(keys),'preview':value[:2]})
            for i,v in enumerate(value): walk(v,path+f'/{i}')
        elif isinstance(value,dict):
            for k,v in value.items(): walk(v,path+'/'+k)
    for name in ('dash-layout.json','pkad-dash-layout.json'):
        try: walk(json.loads((out/name).read_text()),name)
        except (ValueError,FileNotFoundError): pass
    paper=[]
    if (out/'paper.xml').exists():
        import xml.etree.ElementTree as ET
        try:
            xml=ET.parse(out/'paper.xml')
            for p in xml.iter('p'):
                text=''.join(p.itertext())
                if 'PKAD-3' in text: paper.append(text)
            atomic_json(out/'paper-pkad-passages.json',paper)
        except ET.ParseError: pass
    report={'local_unique_structure_observations':len(records),'local_unique_uniprot_ids':len({r['Uniprot_ID'] for r in records}),
            'by_residue':dict(Counter(r['Res_Name'] for r in records)),
            'unique_uniprot_residue_values':len({(r['Uniprot_ID'],r['infor'],r['Res_Name'],r['Expt_pKa']) for r in records}),
            'files':files,'downloads':receipts,'public_tables':tables,'paper_passages':len(paper),
            'completeness_verified':False,'note':'Counts alone cannot certify coverage. Compare accession/construct inventory against the official database or released complete source table.'}
    atomic_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2),flush=True)


def download_all(out):
    """Use the public NiceGUI Download All control via its normal polling transport."""
    require_compute()
    import re
    import uuid
    import urllib.request
    import urllib.parse
    import http.cookiejar
    import time
    out=Path(out); base='https://database.computchem.org'
    client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def request(url,data=None):
        req=urllib.request.Request(url,data=None if data is None else data.encode(),headers={'User-Agent':'pkabench-research-metadata/1.0','Content-Type':'text/plain;charset=UTF-8'})
        with client.open(req,timeout=35) as response: return response.read()
    page=request(base+'/pkad-3').decode(); (out/'pkad-download-session.html').write_text(page)
    raw=re.search(r'parseElements\(String.raw`(.*?)`\)',page,re.S).group(1)
    try: elements=json.loads(raw)
    except ValueError: elements=json.loads(json.loads('"'+raw+'"'))
    cid=re.search(r"'client_id': '([^']+)'",page).group(1)
    controls=[(k,v) for k,v in elements.items() if v.get('props',{}).get('label')=='Download All' and 'hidden' not in v.get('class',[])]
    if len(controls)!=1: raise ValueError('Expected one public Download All control')
    button,control=controls[0]; listener=next(e['listener_id'] for e in control['events'] if e['type']=='click')
    url=base+'/_nicegui_ws/socket.io/?'+urllib.parse.urlencode({'EIO':4,'transport':'polling','client_id':cid})
    handshake=request(url).decode()
    if not handshake.startswith('0'): raise ValueError('Unexpected Engine.IO handshake')
    sid=json.loads(handshake[1:])['sid']; url+='&sid='+urllib.parse.quote(sid)
    def send(packet): request(url,packet)
    def poll():
        packets=request(url).decode().split('\x1e')
        for p in packets:
            if p=='2': send('3')
        return packets
    send('40'); poll()
    send('420'+json.dumps(['handshake',{'client_id':cid,'tab_id':str(uuid.uuid4())}]))
    packets=poll()
    if not any(p.startswith('430') and json.loads(p[3:])==[True] for p in packets): raise ValueError('Public page handshake was not accepted')
    send('42'+json.dumps(['event',{'id':int(button),'client_id':cid,'listener_id':listener,'args':[]}]))
    deadline=time.monotonic()+90
    binary_download=None
    while time.monotonic()<deadline:
        for p in poll():
            if p.startswith('51'):
                event=json.loads(p.split('-',1)[1])
                if event[0]=='download': binary_download=event[1]
                continue
            if p.startswith('b') and binary_download is not None:
                import base64
                event=['download',{**binary_download,'src':base64.b64decode(p[1:])}]
            elif p.startswith('42'): event=json.loads(p[2:])
            else: continue
            if event[0]!='download': continue
            msg=event[1]; src=msg['src']; target=out/'pkad-official-download.csv'
            if isinstance(src,str):
                download_url=urllib.parse.urljoin(base,src)
                if urllib.parse.urlparse(download_url).hostname!='database.computchem.org': raise ValueError('Unexpected download host')
                payload=request(download_url)
            elif isinstance(src,(list,bytes)): payload=bytes(src)
            else: raise ValueError('Unsupported download format')
            target.write_bytes(payload)
            atomic_json(out/'official-download-receipt.json',{'source':base+'/pkad-3','action':'public Download All control','filename':msg.get('filename'),'bytes':len(payload),'sha256':digest(target)})
            print(json.dumps({'downloaded':str(target),'filename':msg.get('filename'),'bytes':len(payload)}),flush=True)
            send('41'); return
    raise TimeoutError('No download event from public control')


def reconcile(out):
    require_compute()
    import re
    out=Path(out); runtime=Path(os.environ['PKABENCH_RUNTIME'])
    source=runtime/'universe/combined-split-v1/usable-proposal-v2'
    ref=json.loads((source/'reference-sequences.json').read_text())
    exact={(r['pdb_id'].lower(),r['chain']) for r in ref['resolved']}
    accessions={u for r in ref['resolved'] for u in r['uniprot_ids']} | {r['uniprot_id'] for r in ref['parent_proxies'] if 'sequence' in r}
    with (out/'pkad-official-download.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['id']!='id']
    if not rows or any(not r['id'].isdigit() for r in rows): raise ValueError('Unexpected official table records')
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate official record IDs')
    missing={}; categories=Counter(); official_accessions=set()
    for r in rows:
        pdb=r['pdb'].strip().lower(); chain=r['chain'].strip(); u=r['uniprot_id'].strip(); official_accessions.add(u)
        if (pdb,chain) in exact: kind='exact_structure_chain_covered'
        elif u in accessions: kind='known_accession_different_construct'
        else: kind='new_accession'
        categories[kind]+=1
        if kind!='exact_structure_chain_covered':
            k=(pdb,chain,u)
            missing.setdefault(k,{'pdb_id':pdb,'chain':chain,'uniprot_ids':[u],'category':kind,'official_ids':[],'valid_pdb_id':bool(re.fullmatch('[0-9][a-z0-9]{3}',pdb))})['official_ids'].append(r['id'])
    atomic_json(out/'official-reference-differences.json',list(missing.values()))
    report={'official_records':len(rows),'official_uniprot_ids':len(official_accessions),'official_pdb_chain_identifiers':len({(r['pdb'],r['chain']) for r in rows}),
            'coverage_by_record':dict(categories),'new_accessions':sorted(official_accessions-accessions),'reference_differences':len(missing),
            'official_sha256':digest(out/'pkad-official-download.csv'),'reserved_sequences_sha256':digest(source/'reference-sequences.json'),
            'completeness_verified':not missing,'note':'Known accession is family evidence, not proof of construct/sequence coverage. All non-exact mappings are retained for review.'}
    atomic_json(out/'reconciliation.json',report); print(json.dumps(report,indent=2),flush=True)


def check_delta(out):
    require_compute()
    import pyarrow.parquet as pq
    from .split_targets import references
    from .runtime import config_hash
    out=Path(out); runtime=Path(os.environ['PKABENCH_RUNTIME']); root=runtime/'universe/combined-split-v1'
    source=root/'usable-proposal-v2'; delta=out/'official-delta'; work=delta/'usable-proposal-v2'; work.mkdir(parents=True,exist_ok=True)
    (delta/'sequence').mkdir(exist_ok=True)
    link=delta/'sequence/sequences.fasta'
    if not link.exists(): link.symlink_to(root/'sequence/sequences.fasta')
    differences=json.loads((out/'official-reference-differences.json').read_text())
    needed=[r for r in differences if r['valid_pdb_id']]
    atomic_json(work/'reference-inventory.json',{'rows':needed})
    references(delta)
    seqs=json.loads((work/'reference-sequences.json').read_text())
    if seqs['failed']: raise ValueError('Official delta reference chain unresolved')
    hits={}
    for line in (work/'reference-hits.tsv').read_text().splitlines():
        a,b,i,q,t=line.split('\t')
        if float(i)>=.3 and min(float(q),float(t))>=.8: hits.setdefault(b,[]).append(a)
    report=json.loads((source/'report.json').read_text())
    if digest(source/'proposal.parquet')!=report['proposal_sha256']: raise ValueError('Proposal changed')
    proposal={r['complex_id']:r for r in pq.read_table(source/'proposal.parquet').to_pylist()}
    candidates=json.loads((root/'index.json').read_text())['candidates']; matches=[]
    for r in candidates:
        matched=[{'chain':c['chain'],'references':hits['s'+config_hash(c['sequence'])[:20]]} for c in r['chains'] if 's'+config_hash(c['sequence'])[:20] in hits]
        if matched: matches.append({'complex_id':r['complex_id'],'pdb_id':r['pdb_id'],'chains':matched,**{k:proposal[r['complex_id']][k] for k in ('split','component_id','benchmark_eligible','train_interface_sites','eval_interface_sites')}})
    conflicts=[r for r in matches if r['split']!='test']
    prior=json.loads((source/'reference-sequences.json').read_text())
    parents={p['uniprot_id'] for p in prior['parent_proxies'] if 'sequence' in p}
    parent_rows=[r for r in differences if not r['valid_pdb_id'] and set(r['uniprot_ids']) & parents]
    peptide_rows=[r for r in differences if r['pdb_id'].upper() in ('AADAA','AAEAA','AAHAA','AACAA','AAKAA','AAYAA')]
    accounted={x for r in needed+parent_rows+peptide_rows for x in r['official_ids']}
    all_missing={x for r in differences for x in r['official_ids']}
    summary={'official_sha256':digest(out/'pkad-official-download.csv'),'proposal_sha256':report['proposal_sha256'],
             'new_structure_chains':[{k:r[k] for k in ('pdb_id','chain')} for r in needed],
             'matched_pairs':len(matches),'matched_by_split':dict(Counter(r['split'] for r in matches)),'non_test_conflicts':conflicts,
             'parent_proxy_records':sum(len(r['official_ids']) for r in parent_rows),'model_pentapeptide_records':sum(len(r['official_ids']) for r in peptide_rows),
             'unaccounted_official_ids':sorted(all_missing-accounted),'official_inventory_accounted_for':all_missing==accounted,
             'reservation_check_pass':not conflicts and all_missing==accounted,'split_unchanged':True,
             'limits':'Official downloaded version only. Six alanine-flanked model pentapeptide entries are outside the protein-complex benchmark scope. Parent sequences reserve mutant families rather than exact constructs. Set-2 literature curation remains separate.'}
    atomic_json(out/'official-delta-matches.json',matches); atomic_json(out/'official-delta-report.json',summary)
    base_audit=json.loads((source/'leakage-audit-scoped.json').read_text())
    scope=json.loads((source/'independent-experimental-scope.json').read_text())
    if base_audit['proposal_sha256']!=report['proposal_sha256']: raise ValueError('Base audit uses another proposal')
    if digest(source/'independent-experimental-scope.json')!=base_audit['experimental_scope_sha256']: raise ValueError('Base experimental scope changed')
    scope.update(version='independent-experimental-v2',
                 previous_scope_sha256=digest(source/'independent-experimental-scope.json'),
                 official_pkad_sha256=summary['official_sha256'],
                 supplemental_sequences_sha256=digest(work/'reference-sequences.json'),
                 pkad_release_reservation_check_pass=summary['reservation_check_pass'] and base_audit['checks_available_reference_inventory_pass'],
                 complete_experimental_inventory=False,
                 pkad_coverage_note='Downloaded official PKAD-3 release accounted for using exact chains, parent-family proxies, six out-of-scope model pentapeptides, and approved 1AXT H exception. Set-2 curation remains unfinished.')
    known={r['reference_id'] for r in scope['reference_eligibility']}
    scope['reference_eligibility'].extend({'reference_id':r['id'],'independent_evaluation_eligible':True} for r in seqs['resolved'] if r['id'] not in known)
    atomic_json(source/'independent-experimental-scope-v2.json',scope)
    print(json.dumps(summary,indent=2),flush=True)
