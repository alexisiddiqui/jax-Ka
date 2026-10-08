"""Resumable official metadata downloads, confined to compute allocations."""
import datetime
import csv
import gzip
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
from .runtime import atomic_json, digest, require_compute


def fetch(url, path, limit=50*1024*1024):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent':'pkabench-research-metadata/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response:
                temp = path.with_suffix(path.suffix+'.partial'); total = 0
                with temp.open('wb') as stream:
                    while chunk := response.read(1024*1024):
                        total += len(chunk)
                        if total>limit: raise ValueError('download size limit exceeded')
                        stream.write(chunk)
                temp.replace(path)
                return {'requested_url':url,'resolved_url':response.url,'bytes':total,'sha256':digest(path),
                    'content_type':response.headers.get('Content-Type'),'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        except Exception:
            if attempt == 2: raise
            time.sleep(2**attempt)


def metadata(out):
    require_compute(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    terminal = lambda attribute,operator,value: {'type':'terminal','service':'text','parameters':{'attribute':attribute,'operator':operator,'value':value}}
    query = {'type':'group','logical_operator':'and','nodes':[
        terminal('rcsb_assembly_container_identifiers.assembly_id','exact_match','1'),
        terminal('rcsb_assembly_info.polymer_entity_instance_count_protein','greater_or_equal',2),
        terminal('rcsb_assembly_info.polymer_monomer_count','less_or_equal',1500),
        terminal('rcsb_assembly_info.polymer_entity_instance_count_nucleic_acid','equals',0),
        terminal('exptl.method','in',['X-RAY DIFFRACTION','ELECTRON MICROSCOPY'])]}
    atomic_json(out/'selection-query.json',query)
    results=[]; receipts=[]; errors=[]; total=None
    try:
        start=0
        while total is None or start<total:
            payload={'query':query,'return_type':'assembly','request_options':{'paginate':{'start':start,'rows':10000},'results_content_type':['experimental']}}
            url='https://search.rcsb.org/rcsbsearch/v2/query?json='+urllib.parse.quote(json.dumps(payload,separators=(',',':')))
            path=out/f'rcsb-{start:07d}.json'
            receipt=fetch(url,path) if not path.exists() else {'cached_file':str(path),'sha256':digest(path)}
            data=json.loads(path.read_text()); receipts.append(receipt)
            total=data['total_count']; batch=data.get('result_set',[])
            if not batch and start<total: raise ValueError('empty RCSB pagination before total_count')
            results.extend(r['identifier'] for r in batch); start+=len(batch)
        if len(set(results))!=total: raise ValueError('duplicate/missing assembly IDs across pages')
        atomic_json(out/'assemblies.json',{'assembly_ids':results,'count':len(results),'query_sha256':digest(out/'selection-query.json')})
    except Exception as exc: errors.append({'source':'RCSB','error':str(exc)})
    for name,url in (
        ('sabdab2-summary.csv','https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary'),
        ('sabdab2-about.html','https://sabdab.opig.stats.ox.ac.uk/about')):
        try:
            path=out/name
            receipt=fetch(url,path) if not path.exists() else {'cached_file':str(path),'sha256':digest(path)}
            receipt['file']=name; receipts.append(receipt)
            if name.endswith('.csv'):
                with path.open(newline='') as stream:
                    reader=csv.DictReader(stream)
                    if not {'PDB','Hchain','Lchain','antigen_chain','antigen_type'} <= set(reader.fieldnames or []):
                        raise ValueError('SAbDab response is not the expected summary table')
                    rows=list(reader)
                protein=[r for r in rows if 'PROTEIN' in r['antigen_type'].upper().split('|')]
                atomic_json(out/'sabdab-candidates.json',{'instances':len(rows),'entries':len({r['PDB'] for r in rows}),
                    'protein_antigen_instances':len(protein),'protein_antigen_entries':len({r['PDB'] for r in protein}),
                    'source_sha256':digest(path),'candidates':protein,
                    'note':'Source author chain annotations; assembly-copy/label mapping and interface verification still required.'})
        except Exception as exc: errors.append({'source':name,'error':str(exc)})
    result={'status':'complete' if not errors else 'partial','rcsb_assemblies':len(results),'receipts':receipts,'errors':errors,
        'scope':'Metadata only. Assembly 1, >=2 protein chains, <=1500 deposited polymer residues, no nucleic acid chains, X-ray/EM. Ligand/glycan counts are not prefiltered. No production split or coordinates downloaded.',
        'job':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURMD_NODENAME']}
    atomic_json(out/'download-report.json',result); print(json.dumps({k:v for k,v in result.items() if k!='receipts'},indent=2))


def coordinate_sample(out, count=24):
    """Bounded sample to check official coordinate access and estimate storage."""
    import random
    require_compute(); out=Path(out)
    ids=json.loads((out/'assemblies.json').read_text())['assembly_ids']
    chosen=random.Random(20261003).sample(sorted(ids),min(count,len(ids)))
    receipts=[]; errors=[]
    for identifier in chosen:
        entry,assembly=identifier.rsplit('-',1)
        path=out/'coordinate-sample'/f'{entry.lower()}-assembly{assembly}.cif.gz'
        try:
            receipt=fetch(f'https://files.rcsb.org/download/{entry.lower()}-assembly{assembly}.cif.gz',path)
            with gzip.open(path,'rt') as stream:
                if not stream.read(100).lstrip().startswith('data_'): raise ValueError('response is not gzipped mmCIF')
            receipts.append({'assembly_id':identifier,**receipt})
        except Exception as exc: errors.append({'assembly_id':identifier,'error':str(exc)})
    sizes=[r['bytes'] for r in receipts]
    report={'attempted':len(chosen),'downloaded':len(sizes),'bytes':sum(sizes),'errors':errors,'receipts':receipts,
        'estimated_compressed_universe_gib':sum(sizes)/len(sizes)*len(ids)/1024**3 if sizes else None,
        'note':'Seeded 24-assembly storage pilot, before quality filtering; rough estimate, not a bulk-download commitment.',
        'job':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURMD_NODENAME']}
    atomic_json(out/'coordinate-sample-report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='receipts'},indent=2))
