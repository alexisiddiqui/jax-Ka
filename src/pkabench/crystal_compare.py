"""Stored pKPDB versus fresh methods on original crystal coordinates."""
import csv
import json
import os
from pathlib import Path
import sys
from .runtime import atomic_json, digest, require_compute


def initialise(audit,out):
    require_compute(); audit=Path(audit); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    report=json.loads((audit/'report.json').read_text())
    entries=[r['pdb_id'] for r in report['entries'] if r.get('precomputed_pkpdb')]
    atomic_json(out/'manifest.json',{'entries':entries,'source_audit':str(audit),
        'scope':'All eight confirmed precomputed entries in the preceding gap-enriched audit; not the full pKPDB.',
        'comparison':'Absolute site pKa, whole crystallographic entry first model; no AB/free shift subtraction.',
        'policy':'First positive-occupancy alternate per residue. Canonical protein heavy atoms; HETATM context recorded. Shared PDB2PQR atom completion, observed atoms preserved. Physical gaps exported as separate segments. No whole-residue reconstruction.',
        'limitations':'Historical preparation/version/settings may differ; current PypKa pH -2..16 does not cover all deposited values. Agreement is not proof of historical whole-residue reconstruction.',
        'implementation_sha256':digest(Path(__file__))})
    print(json.dumps({'entries':entries}))


def run(out,index):
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from .conformers import resolve
    from .prep import CANONICAL,complete,export_pdb
    from .supervision import inventory,clearance
    from .annotate import SITE_ATOMS
    from .adapters.base import execute,TEACHER
    require_compute(); out=Path(out); manifest=json.loads((out/'manifest.json').read_text()); entry=manifest['entries'][index]
    work=out/entry; work.mkdir(exist_ok=True); audit=Path(manifest['source_audit']); runtime=Path(os.environ['PKABENCH_RUNTIME'])
    receipt=work/'result.json'
    if receipt.exists(): raise ValueError('Entry already attempted; preserve result and use a new output campaign')
    result={'entry':entry,'methods':{},'job':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURMD_NODENAME']}
    try:
        source=audit/'downloads'/f'{entry}-original.cif'; cif=pdbx.CIFFile.read(source)
        experimental=list(cif.block['exptl']['method'].as_array(str)); result['experimental_methods']=experimental
        if 'X-RAY DIFFRACTION' not in experimental: raise ValueError('Not an X-ray entry')
        cat=cif.block['atom_site']; selected=sorted(set(cat['label_asym_id'].as_array(str)[np.isin(cat['label_comp_id'].as_array(str),list(CANONICAL))]))
        cif,alternates=resolve(cif,selected); atomic_json(work/'conformers.json',alternates)
        atoms=pdbx.get_structure(cif,model=1,altloc='first',use_author_fields=False)
        author=pdbx.get_structure(cif,model=1,altloc='first',use_author_fields=True)
        evidence=inventory(cif,{'A':selected,'B':[]}); labels_to_author={str(a):str(b) for a,b in zip(atoms.chain_id,author.chain_id)}
        for d in evidence['defects']:
            if 'chain' in d: d['chain']=labels_to_author[d['chain']]
            if 'key' in d: d['key'][0]=labels_to_author[d['key'][0]]
        keep=np.isin(author.res_name,list(CANONICAL)) & ~np.isin(np.char.upper(author.element),['H','D'])
        result['omitted_components']=sorted(set(map(str,author.res_name[~np.isin(author.res_name,list(CANONICAL))])))
        atoms=author[keep]; atoms.coord=np.round(atoms.coord,3)
        result['source_sha256']=digest(source); result['observed_residues']=len(struc.get_residue_starts(atoms))
        atoms=complete(atoms,str(runtime/'envs/pypka/bin/pdb2pqr30'),work)
        mapping=export_pdb(atoms,work/'input.pdb'); atomic_json(work/'mapping.json',[{'exported':list(k),'original':list(v)} for k,v in mapping.items()])
        atomic_json(work/'input_atom_mask.json',evidence); result['input_sha256']=digest(work/'input.pdb')
        stored=json.loads((audit/'downloads'/f'{entry}-pkpdb.json').read_text())
        atomic_json(work/'stored-pkpdb.json',stored); result['stored_params']=stored.get('params')
        for method in ('propka','pypka'):
            methodwork=work/method; methodwork.mkdir(exist_ok=True)
            config=TEACHER if method=='pypka' else {'model':'propka'}
            atomic_json(methodwork/'request.json',{'method':method,'config':config,'pdb':str((work/'input.pdb').resolve())})
            env=os.environ.copy()
            if method=='pypka': env['LD_LIBRARY_PATH']=str(runtime/'fortran/lib')
            try:
                seconds=execute([str(runtime/'envs'/('pypka' if method=='pypka' else 'runner')/'bin/python'),'-m','pkabench.adapters.worker',str((methodwork/'request.json').resolve()),str((methodwork/'native.json').resolve())],methodwork,5400,env)
                native=json.loads((methodwork/'native.json').read_text()); normalized=[]
                for r in native['rows']:
                    chain,number,icode=mapping[r['chain'],r['resnum']]
                    normalized.append({'chain':chain,'resnum':number,'icode':icode,'group':r['group'],'pka':r['pka']})
                result['methods'][method]={'status':'complete','seconds':seconds,'version':native['version'],'rows':normalized}
            except Exception as exc: result['methods'][method]={'status':'failed','error':str(exc)}
        starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
        residues={ (str(atoms.chain_id[s]),int(atoms.res_id[s]),str(atoms.ins_code[s]).strip()):atoms[s:e] for s,e in zip(starts[:-1],starts[1:])}
        rows=[]
        for chain,group,number,value in stored['pKas']:
            group={'NTR':'NTERM','CTR':'CTERM'}.get(group,group)
            keys=[k for k in residues if k[:2]==(str(chain),int(number))]
            row={'entry':entry,'chain':str(chain),'resnum':int(number),'group':group,'pkpdb_pka':value,
                'mapping_status':'unique' if len(keys)==1 else 'absent' if not keys else 'ambiguous_insertion_code'}
            row['icode']=keys[0][2] if len(keys)==1 else None
            if len(keys)==1 and group in SITE_ATOMS:
                residue=residues[keys[0]]; points=residue.coord[np.isin(residue.atom_name,SITE_ATOMS[group])]
                gaps=[d for d in evidence['defects'] if d['kind'] in ('terminal_gap','internal_gap')]
                row['gap_envelope_clearance_A']=min((clearance(points,d) for d in gaps),default=None)
                row['missing_coordinate_regions']=len(gaps)
            for method,data in result['methods'].items():
                matches=[r for r in data.get('rows',[]) if len(keys)==1 and (r['chain'],r['resnum'],r['icode'],r['group'])==(*keys[0],group)]
                predicted=matches[0]['pka'] if len(matches)==1 else None
                ok=predicted is not None and np.isfinite(predicted)
                row[method+'_pka']=float(predicted) if ok else None
                row[method+'_status']='ok' if ok else data['status'] if data['status']=='failed' else 'not_reported'
                row[method+'_minus_pkpdb']=float(predicted)-float(value) if ok and value is not None else None
            rows.append(row)
        result['comparisons']=rows; result['status']='complete'
    except Exception as exc: result.update(status='preparation_failed',error=str(exc))
    atomic_json(receipt,result); print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','methods')},indent=2))


def collect(out):
    import numpy as np
    from collections import Counter
    require_compute(); out=Path(out); manifest=json.loads((out/'manifest.json').read_text())
    results=[json.loads((out/entry/'result.json').read_text()) for entry in manifest['entries']]
    rows=[row for result in results for row in result.get('comparisons',[])]
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'site_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    def stats(values):
        a=np.asarray(values,float)
        return {'n':len(a),'bias':float(a.mean()),'mae':float(np.abs(a).mean()),'rmse':float(np.sqrt(np.mean(a*a))),'p95_abs':float(np.quantile(np.abs(a),.95))} if len(a) else {'n':0}
    report={'entries':len(results),'entry_statuses':dict(Counter(r['status'] for r in results)),
        'failures':[{'entry':r['entry'],'error':r.get('error')} for r in results if r['status']!='complete'],
        'stored_sites_in_prepared_entries':len(rows),'comparisons':{},'by_group':{},'by_gap_clearance':{},
        'limits':manifest['limitations'],'production_allowed':False}
    for method in ('pypka','propka'):
        field=method+'_minus_pkpdb'
        report['comparisons'][method+'_vs_pkpdb']={**stats([r[field] for r in rows if r.get(field) is not None]),'coverage':dict(Counter(r.get(method+'_status','absent') for r in rows))}
        report['by_group'][method]={g:stats([r[field] for r in rows if r['group']==g and r.get(field) is not None]) for g in sorted({r['group'] for r in rows})}
        report['by_gap_clearance'][method]={str(radius):stats([r[field] for r in rows if r.get(field) is not None and (r.get('gap_envelope_clearance_A') is None or r['gap_envelope_clearance_A']>=radius)]) for radius in (0,10,20)}
    report['comparisons']['propka_vs_fresh_pypka']=stats([r['propka_pka']-r['pypka_pka'] for r in rows if r.get('propka_pka') is not None and r.get('pypka_pka') is not None])
    atomic_json(out/'report.json',report); print(json.dumps(report,indent=2))
