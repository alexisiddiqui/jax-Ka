"""Component removal sensitivity, preserving protein geometry and native typing."""
import csv
import json
import os
import contextlib
from pathlib import Path
from collections import Counter
from .runtime import require_compute,atomic_json,digest

IONS={'MG':2,'CA':2,'ZN':2,'NA':1,'CL':-1,'MN':2,'K':1,'CD':2,'FE':3,'SR':2,'CU':2,'IOD':-1,'HG':2,'BR':-1,'CO':2,'NI':2,'FE2':2}


def initialise(out):
    require_compute(); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    runtime=Path(os.environ['PKABENCH_RUNTIME']); campaign=runtime/'campaigns/coordinate-pool-inclusive-v1'; tasks=[]
    for p in sorted((runtime/'audits/ligand-review-v1').glob('*/result.json')):
        r=json.loads(p.read_text())
        if r['status']=='prepared': tasks.append({'complex_id':r['complex_id'],'kind':'ligand','review':str(p),'protein_root':str(p.parent/'prepared'),'components':r['removed']})
    inventory=[r for p in sorted((runtime/'audits/component-support-v1').glob('inventory-*.json')) for r in json.loads(p.read_text())]
    metal=[]
    for r in inventory:
        if r['status']!='complete': continue
        cc=r['components']; metals=[c for c in cc if c['code']=='metal']
        if metals and all(c['name'] in IONS and c['end']-c['start']==1 for c in metals) and all(c['code']=='metal' or c['name'] in ('GOL','EDO','PEG') for c in cc):
            metal.append({'complex_id':r['complex_id'],'kind':'metal','components':metals})
    # Bounded metal pilot, distinct assemblies where possible.
    seen=set()
    for t in metal:
        r=json.loads((campaign/'rows'/f"{t['complex_id']}.json").read_text())
        if r['pdb_id'] in seen: continue
        seen.add(r['pdb_id']); tasks.append(t)
        if len(seen)==12: break
    atomic_json(out/'manifest.json',{'campaign':str(campaign),'tasks':tasks,'code_sha256':digest(Path(__file__)),
        'scope':'39 prepared ligand-review pairs and up to 12 distinct-assembly supported-metal pairs. Up to four component instances deleted individually per pair, all other selected components retained. No coordinate relaxation. Free-state ownership assigned to nearest partner and explicitly recorded; bridging cases are conditional diagnostics.',
        'cutoffs_A':{'coulomb':10,'burial':15,'desolvation':20},'production_allowed':False})
    print(json.dumps({'tasks':len(tasks),'kinds':dict(Counter(t['kind'] for t in tasks))}))


def run(out,shard,shards):
    require_compute()
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from .prep import read_cif,complete,export_pdb,CANONICAL
    from .annotate import SITE_ATOMS
    from .audit import component_inventory
    from propka.run import single
    out=Path(out).resolve(); manifest=json.loads((out/'manifest.json').read_text()); campaign=Path(manifest['campaign']); runtime=Path(os.environ['PKABENCH_RUNTIME'])
    for task in manifest['tasks'][shard::shards]:
        cid=task['complex_id']; work=out/cid; work.mkdir(exist_ok=False)
        result={'complex_id':cid,'kind':task['kind'],'components':[],'errors':[]}
        try:
            row=json.loads((campaign/'rows'/f'{cid}.json').read_text()); result['pdb_id']=row['pdb_id']
            source=campaign/'structures'/cid/'resolved-source.cif'; cif=pdbx.CIFFile.read(source)
            raw=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
            auth=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=True)
            selected=row['partner_A_chains']+row['partner_B_chains']
            if task['kind']=='ligand': atoms=read_cif(Path(task['protein_root'])/'AB.cif')
            else:
                raw.res_id=auth.res_id; raw.ins_code=auth.ins_code
                atoms=raw[np.isin(raw.chain_id,selected)&np.isin(raw.res_name,list(CANONICAL))&~np.isin(np.char.upper(raw.element),['H','D'])]
                atoms.coord=np.round(atoms.coord,3); atoms=complete(atoms,str(runtime/'envs/pypka/bin/pdb2pqr30'),work)
                raw=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
            wanted={(c['chain'],c['name']) for c in task['components']}; components=[]
            for c in component_inventory(raw,cif,selected):
                if (c['chain'],c['name']) not in wanted: continue
                a=raw[c['start']:c['end']]; a=a[~np.isin(np.char.upper(a.element),['H','D'])]
                if len(c['name'])>3: raise ValueError('Component identifier exceeds PDB width; explicit alias mapping needed')
                dist={p:float(np.linalg.norm(a.coord[:,None]-atoms.coord[np.isin(atoms.chain_id,row[f'partner_{p}_chains'])][None,:],axis=-1).min()) for p in ('A','B')}
                owner=min(dist,key=dist.get)
                iso=float(np.sum(struc.sasa(a,ignore_ions=False,vdw_radii='Single')))
                bound=float(np.sum(struc.sasa(atoms+a,ignore_ions=False,vdw_radii='Single')[-len(a):]))
                components.append({'atoms':a,'name':c['name'],'chain':c['chain'],'resnum':c['resnum'],'owner':owner,'partner_distances':dist,'bridges':max(dist.values())<=4,'sasa_A2':bound,'isolated_sasa_A2':iso,'exposed_fraction':bound/iso})
            if not components: raise ValueError('No matching components')
            starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
            residues={(str(atoms.chain_id[s]),int(atoms.res_id[s]),str(atoms.ins_code[s]).strip()):atoms[s:e] for s,e in zip(starts[:-1],starts[1:])}
            mappings={}; texts={}
            for state in ('AB','A','B'):
                a=atoms if state=='AB' else atoms[np.isin(atoms.chain_id,row[f'partner_{state}_chains'])]
                mappings[state]=export_pdb(a,work/f'{state}.pdb'); texts[state]='\n'.join(l for l in (work/f'{state}.pdb').read_text().splitlines() if l.startswith(('ATOM','TER')))+'\n'
                if any(k[0]=='Z' for k in mappings[state]): raise ValueError('Reserved component chain collision')
            def predict(state,removed=None):
                tag=f'{state}-'+('full' if removed is None else f'delete{removed}'); text=texts[state]; serial=80000
                included=[]
                for i,c in enumerate(components):
                    if i==removed or (state!='AB' and c['owner']!=state): continue
                    included.append(i)
                    for atom in c['atoms']:
                        serial+=1; x,y,z=atom.coord
                        text+=f'HETATM{serial:5d} {str(atom.atom_name):>4s} {c["name"]:>3s} Z{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{0.:6.2f}          {str(atom.element):>2s}\n'
                pdb=work/f'{tag}.pdb'; pdb.write_text(text+'END\n')
                with (work/f'{tag}.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log): model=single(str(pdb),write_pka=False)
                conf=model.conformations[model.conformation_names[0]]; predictions={}; typing=[]
                for g in conf.groups:
                    atom=g.atom
                    if atom.chain_id=='Z': typing.append({'component_index':int(atom.res_num)-1,'atom':atom.name,'group_type':g.type,'charge':g.charge,'residue_type':g.residue_type})
                    else:
                        key=mappings[state].get((atom.chain_id,atom.res_num)); group={'N+':'NTERM','C-':'CTERM'}.get(g.residue_type.strip(),g.residue_type.strip())
                        if key is not None and group in SITE_ATOMS and np.isfinite(g.pka_value): predictions[(*key,group)]=float(g.pka_value)
                counts={i:sum(a.chain_id=='Z' and a.res_num==i+1 and a.element!='H' for a in conf.atoms) for i in included}
                for i in included:
                    if counts[i]!=len(components[i]['atoms']): raise ValueError('Component atom retention mismatch')
                    if components[i]['name'] in IONS and not any(t['component_index']==i and t['charge']==IONS[components[i]['name']] for t in typing): raise ValueError('Ion charge recognition failed')
                atomic_json(work/f'{tag}-typing.json',{'groups':typing,'retained_atom_counts':counts})
                return predictions,typing
            baseline={}; baseline_types={}
            for state in ('AB','A','B'): baseline[state],baseline_types[state]=predict(state)
            comparisons=[]
            for i,c in enumerate(components[:4]):
                ab,unused=predict('AB',i); free,unused=predict(c['owner'],i)
                metadata={k:v for k,v in c.items() if k!='atoms'}; metadata['component_index']=i
                metadata['propka_groups']=[g for g in baseline_types['AB'] if g['component_index']==i]
                result['components'].append(metadata)
                for key,value in baseline['AB'].items():
                    group=key[3]; a=residues[key[:3]]; points=a.coord[np.isin(a.atom_name,SITE_ATOMS[group])]
                    if not len(points): continue
                    partner='A' if key[0] in row['partner_A_chains'] else 'B'
                    ref_free=baseline[partner].get(key); new_free=free.get(key) if partner==c['owner'] else ref_free
                    if key not in ab or ref_free is None or new_free is None: continue
                    distance=float(np.linalg.norm(points[:,None]-c['atoms'].coord[None,:],axis=-1).min())
                    comparisons.append({'complex_id':cid,'kind':task['kind'],'component_index':i,'component_name':c['name'],'group':group,'chain':key[0],'resnum':key[1],'icode':key[2],
                        'distance_A':distance,'sasa_A2':c['sasa_A2'],'exposed_fraction':c['exposed_fraction'],'ab_pka_change':ab[key]-value,
                        'delta_pka_change':(ab[key]-new_free)-(value-ref_free),'flag_train15':distance<15,'flag_eval25':distance<25})
            if comparisons:
                with (work/'site_changes.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(comparisons[0])); w.writeheader(); w.writerows(comparisons)
            result.update(status='complete',site_component_observations=len(comparisons),components_in_context=len(components),source_sha256=digest(source))
        except Exception as exc: result.update(status='failed',error=str(exc))
        atomic_json(work/'result.json',result)


def collect(out):
    require_compute()
    import numpy as np
    out=Path(out); tasks=json.loads((out/'manifest.json').read_text())['tasks']; results=[json.loads((out/t['complex_id']/'result.json').read_text()) for t in tasks]; rows=[]
    for r in results:
        p=out/r['complex_id']/'site_changes.csv'
        if p.exists():
            with p.open() as f: rows.extend(csv.DictReader(f))
    summaries=[]
    for kind in ('ligand','metal'):
        for exposure,lo,hi in [('all',0,1.01),('buried',0,.3),('partial',.3,.7),('exposed',.7,1.01)]:
            for radius in (0,10,15,20,25):
                rr=[r for r in rows if r['kind']==kind and lo<=float(r['exposed_fraction'])<hi and float(r['distance_A'])>=radius]
                for field in ('ab_pka_change','delta_pka_change'):
                    x=np.abs([float(r[field]) for r in rr]); summaries.append({'kind':kind,'exposure':exposure,'radius_A':radius,'metric':field,'n':len(x),'complexes':len({r['complex_id'] for r in rr}),'mae':float(x.mean()) if len(x) else None,'p95':float(np.quantile(x,.95)) if len(x) else None,'max':float(x.max()) if len(x) else None})
    if rows:
        with (out/'site_changes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    atomic_json(out/'report.json',{'statuses':dict(Counter(r['status'] for r in results)),'failures':[r for r in results if r['status']!='complete'],'summaries':summaries,
        'limits':'PROPKA sensitivity, not physical error bounds. Built-in cutoffs 10/15/20 A. Per-component typing recorded; generic ligand typing does not establish correct charge/protonation. Fixed geometry; free ownership assigned to nearest partner. Exposure groups are descriptive, not causal; repeated sites/components and small support. No production policy changes.'})
    print(json.dumps({'statuses':dict(Counter(r['status'] for r in results)),'observations':len(rows)}))
