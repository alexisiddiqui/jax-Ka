"""Corrected neutral-buffer PROPKA sensitivity; isolated from frozen campaigns."""
import csv
import json
import os
import contextlib
from pathlib import Path
from collections import Counter, defaultdict
from .runtime import require_compute, atomic_json, digest

# Bounded chemistry scope: complete neutral C/O alcohols/ethers, standard PROPKA.
SUPPORTED={'GOL','EDO','MPD','PGE','PG4','1PE'}
IONS={}

def initialise(out):
    require_compute()
    import pyarrow.parquet as pq
    from .glycan_buffer_policy import BUFFERS
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    runtime=Path(os.environ['PKABENCH_RUNTIME'])
    old={r['complex_id']:r for r in pq.read_table(runtime/'universe/combined-split-v1/usable-proposal-v2/proposal.parquet').to_pylist()}
    new={r['complex_id']:r for r in pq.read_table(runtime/'universe/glycan-buffer-v2-recount/eligibility-with-existing-assignments.parquet').to_pylist()}
    candidates=[]; inventory=[]
    for name in ('glycan-buffer-v2-pool1','glycan-buffer-v2-pool5000'):
        campaign=runtime/'campaigns'/name
        for path in sorted((campaign/'structures').glob('*/removed_components.json')):
            cid=path.parent.name; cc=json.loads(path.read_text())['components']
            buffers=[c for c in cc if c['name'] in BUFFERS]
            if not buffers: continue
            lost=bool(old[cid]['split']=='train' and old[cid]['benchmark_eligible'] and old[cid]['train_interface_sites']>0 and new[cid]['train_interface_sites']==0)
            for c in buffers:
                inventory.append({'complex_id':cid,'name':c['name'],'chain':c['chain'],'supported_chemistry':c['name'] in SUPPORTED,'lost_training_pair':lost,
                    'reason':'neutral C/O scope; per-structure typing gate still required' if c['name'] in SUPPORTED else 'ignored by default PROPKA or charge/chemistry outside verified neutral C/O scope'})
            supported=[c for c in buffers if c['name'] in SUPPORTED]
            if not supported: continue
            row=json.loads((campaign/'rows'/f'{cid}.json').read_text())
            candidates.append({'complex_id':cid,'pdb_id':row['pdb_id'],'campaign':str(campaign),'protein_root':str(path.parent),'kind':'ligand','components':supported,'lost_training_pair':lost,
                'other_removed_components':[c['name'] for c in cc if c['name'] not in SUPPORTED]})
    # Deliberately oversample affected pairs; round robin chemical identities,
    # distinct PDBs, then remaining distinct PDBs. This is a diagnostic sample.
    chosen=[]; seen=set()
    buckets=defaultdict(list)
    for c in candidates:
        for name in sorted({r['name'] for r in c['components']}): buckets[name].append(c)
    for b in buckets.values(): b.sort(key=lambda c:(not c['lost_training_pair'],c['complex_id']))
    while len(chosen)<30:
        added=False
        for name in sorted(buckets):
            b=buckets[name]
            while b and b[0]['pdb_id'] in seen: b.pop(0)
            if b and len(chosen)<30:
                c=b.pop(0); chosen.append(c); seen.add(c['pdb_id']); added=True
        if not added: break
    atomic_json(out/'inventory.json',inventory)
    atomic_json(out/'manifest.json',{'tasks':chosen,'code_sha256':digest(Path(__file__)),
        'scope':'Corrected PDB element alignment. Native PROPKA parameters; complete neutral C/O buffers only, explicit element/bond/neutral-group gates. Other removed species stay absent. Individual and all supported buffers removed; fixed protein geometry. Diagnostic convenience sample, oversampling lost training pairs. No policy changes.',
        'inventory_instances':dict(Counter(r['name'] for r in inventory)), 'supported_candidates':len(candidates),'production_allowed':False})
    print(json.dumps({'tasks':len(chosen),'lost_training_pairs':sum(c['lost_training_pair'] for c in chosen),'inventory':dict(Counter(r['name'] for r in inventory))}),flush=True)


def validate_components(conf, components, included):
    from biotite.structure.info import get_from_ccd, bonds_in_residue
    for i in included:
        c=components[i]; name=c['name']
        if name not in SUPPORTED: raise ValueError('Unsupported buffer chemistry')
        cat=get_from_ccd('chem_comp_atom',name)
        ids=cat['atom_id'].as_array(str); els=dict(zip(ids,cat['type_symbol'].as_array(str))); charges=dict(zip(ids,cat['charge'].as_array(int)))
        expected={n for n in ids if els[n]!='H'}
        atoms={a.name:a for a in conf.atoms if a.chain_id=='Z' and a.res_num==i+1 and a.element!='H'}
        if set(atoms)!=expected: raise ValueError(f'{name}: absent/extra heavy atoms relative to CCD')
        bonds=bonds_in_residue(name)
        for n,a in atoms.items():
            if a.element!=els[n] or charges[n]!=0 or a.element not in ('C','O'): raise ValueError(f'{name}: element/charge mismatch')
            neighbors={y if x==n else x for (x,y),order in bonds.items() if n in (x,y) and (y if x==n else x) in atoms}
            actual={b.name for b in a.get_bonded_heavy_atoms() if b.chain_id=='Z' and b.res_num==i+1}
            if actual!=neighbors or any(b.chain_id!='Z' or b.res_num!=i+1 for b in a.get_bonded_heavy_atoms()): raise ValueError(f'{name}:{n}: geometry/CCD connectivity mismatch')
            if a.sybyl_type != ('C.3' if a.element=='C' else 'O.3'): raise ValueError(f'{name}:{n}: unexpected native atom type {a.sybyl_type}')
        groups=[g for g in conf.groups if g.atom.chain_id=='Z' and g.atom.res_num==i+1]
        if not groups or any(g.charge!=0 for g in groups): raise ValueError(f'{name}: no neutral interacting groups')

def run(out,shard,shards):
    require_compute()
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from .prep import read_cif,complete,export_pdb,CANONICAL
    from .annotate import SITE_ATOMS
    from .audit import component_inventory
    from propka.run import single
    out=Path(out).resolve(); manifest=json.loads((out/'manifest.json').read_text()); runtime=Path(os.environ['PKABENCH_RUNTIME'])
    for task in manifest['tasks'][shard::shards]:
        campaign=Path(task['campaign']); cid=task['complex_id']; work=out/cid; work.mkdir(exist_ok=False)
        result={'complex_id':cid,'kind':task['kind'],'components':[],'errors':[]}
        try:
            row=json.loads((campaign/'rows'/f'{cid}.json').read_text()); result['pdb_id']=row['pdb_id']
            source=campaign/'structures'/cid/'original-resolved.cif'; cif=pdbx.CIFFile.read(source)
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
                    if removed == 'all' or i==removed or (state!='AB' and c['owner']!=state): continue
                    included.append(i)
                    for atom in c['atoms']:
                        serial+=1; x,y,z=atom.coord
                        name=str(atom.atom_name); element=str(atom.element)
                        atom_field=f' {name:<3s}' if len(element)==1 and len(name)<4 else f'{name:<4s}'
                        text+=f'HETATM{serial:5d} {atom_field} {c["name"]:>3s} Z{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{0.:6.2f}          {str(atom.element):>2s}\n'
                pdb=work/f'{tag}.pdb'; pdb.write_text(text+'END\n')
                with (work/f'{tag}.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log): model=single(str(pdb),write_pka=False)
                conf=model.conformations[model.conformation_names[0]]; predictions={}; typing=[]
                for g in conf.groups:
                    atom=g.atom
                    if atom.chain_id=='Z': typing.append({'component_index':int(atom.res_num)-1,'atom':atom.name,'group_type':g.type,'charge':g.charge,'residue_type':g.residue_type})
                    else:
                        key=mappings[state].get((atom.chain_id,atom.res_num)); group={'N+':'NTERM','C-':'CTERM'}.get(g.residue_type.strip(),g.residue_type.strip())
                        if key is not None and group in SITE_ATOMS and np.isfinite(g.pka_value): predictions[(*key,group)]=float(g.pka_value)
                validate_components(conf, components, included)
                counts={i:sum(a.chain_id=='Z' and a.res_num==i+1 and a.element!='H' for a in conf.atoms) for i in included}
                for i in included:
                    if counts[i]!=len(components[i]['atoms']): raise ValueError('Component atom retention mismatch')
                    if components[i]['name'] in IONS and not any(t['component_index']==i and t['charge']==IONS[components[i]['name']] for t in typing): raise ValueError('Ion charge recognition failed')
                atomic_json(work/f'{tag}-typing.json',{'groups':typing,'retained_atom_counts':counts})
                return predictions,typing
            baseline={}; baseline_types={}
            from .schema import read_table
            site_metadata={(r['chain'],r['resnum'],r['icode'],r['group']):r for r in read_table(Path(task['protein_root'])/'sites.parquet')}
            for state in ('AB','A','B'): baseline[state],baseline_types[state]=predict(state)
            repeat,_=predict('AB')
            if repeat != baseline['AB']: raise ValueError('Unmodified repeat differs')
            comparisons=[]
            all_ab,_=predict('AB','all')
            all_free={p:predict(p,'all')[0] for p in ('A','B')}
            variants=list(enumerate(components))+[('all',{'atoms':sum((c['atoms'] for c in components[1:]),components[0]['atoms'].copy()),'name':'ALL','chain':'','resnum':0,'owner':'AB','sasa_A2':sum(c['sasa_A2'] for c in components),'exposed_fraction':sum(c['sasa_A2'] for c in components)/sum(c['isolated_sasa_A2'] for c in components)})]
            for i,c in variants:
                if i=='all': ab=all_ab; free={}
                else:
                    ab,unused=predict('AB',i); free,unused=predict(c['owner'],i)
                metadata={k:v for k,v in c.items() if k!='atoms'}; metadata['component_index']=i
                metadata['propka_groups']=[g for g in baseline_types['AB'] if g['component_index']==i]
                result['components'].append(metadata)
                for key,value in baseline['AB'].items():
                    meta=site_metadata.get(key,{})
                    if not meta.get('functional_atoms_complete') or meta.get('is_break_terminus'): continue
                    group=key[3]; a=residues[key[:3]]; points=a.coord[np.isin(a.atom_name,SITE_ATOMS[group])]
                    if not len(points): continue
                    partner='A' if key[0] in row['partner_A_chains'] else 'B'
                    ref_free=baseline[partner].get(key); new_free=all_free[partner].get(key) if i=='all' else free.get(key) if partner==c['owner'] else ref_free
                    if key not in ab or ref_free is None or new_free is None: continue
                    distance=float(np.linalg.norm(points[:,None]-c['atoms'].coord[None,:],axis=-1).min())
                    comparisons.append({'complex_id':cid,'kind':task['kind'],'component_index':i,'component_name':c['name'],'group':group,'chain':key[0],'resnum':key[1],'icode':key[2],
                        'interface':bool(meta['residue_delta_sasa']>10), 'lost_training_pair':task['lost_training_pair'],'distance_A':distance,'sasa_A2':c['sasa_A2'],'exposed_fraction':c['exposed_fraction'],'ab_pka_change':ab[key]-value,
                        'delta_pka_change':(ab[key]-new_free)-(value-ref_free),'flag_train15':distance<15,'flag_eval25':distance<25})
            if comparisons:
                with (work/'site_changes.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(comparisons[0])); w.writeheader(); w.writerows(comparisons)
            result.update(status='complete',site_component_observations=len(comparisons),components_in_context=len(components),source_sha256=digest(source))
        except Exception as exc:
            import traceback
            result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        atomic_json(work/'result.json',result)


def collect(out):
    require_compute()
    import numpy as np
    out=Path(out); manifest=json.loads((out/'manifest.json').read_text()); results=[]; rows=[]
    for t in manifest['tasks']:
        p=out/t['complex_id']; r=json.loads((p/'result.json').read_text()); results.append(r)
        if r['status']=='complete':
            with (p/'site_changes.csv').open() as f: rows.extend(csv.DictReader(f))
    # Interface annotation is the shared buried-area definition, not a sites-table field.
    from .schema import read_table
    metadata={}
    for t in manifest['tasks']:
        for site in read_table(Path(t['protein_root'])/'sites.parquet'):
            metadata[(t['complex_id'],site['chain'],site['resnum'],site['icode'],site['group'])]=site['residue_delta_sasa']>10
    for r in rows:
        r['interface']=str(metadata[(r['complex_id'],r['chain'],int(r['resnum']),r['icode'],r['group'])])
    if rows:
        with (out/'site_changes.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries=[]; retention=[]
    for variant in ('individual','all'):
        vv=[r for r in rows if (r['component_index']=='all')==(variant=='all')]
        for radius in (0,5,10,15,20,25,30):
            rr=[r for r in vv if float(r['distance_A'])>=radius]
            for exposure,lo,hi in [('all',0,1.01),('buried',0,.3),('partial',.3,.7),('exposed',.7,1.01)]:
                ee=[r for r in rr if lo<=float(r['exposed_fraction'])<hi]
                for metric in ('ab_pka_change','delta_pka_change'):
                    x=np.abs([float(r[metric]) for r in ee])
                    summaries.append({'variant':variant,'radius_A':radius,'exposure':exposure,'metric':metric,'observations':len(x),'complexes':len({r['complex_id'] for r in ee}),'p95':float(np.quantile(x,.95)) if len(x) else None,'max':float(x.max()) if len(x) else None,'above_0.1':int(sum(x>.1))})
            if variant=='all':
                interface=[r for r in rr if r['interface']=='True']
                retention.append({'radius_A':radius,'sites':len(rr),'interface_sites':len(interface),'pairs':len({r['complex_id'] for r in rr}),'interface_pairs':len({r['complex_id'] for r in interface})})
    report={'statuses':dict(Counter(r['status'] for r in results)),'failures':[r for r in results if r['status']!='complete'],'observations':len(rows),'summaries':summaries,'retention':retention,
        'limits':'PROPKA sensitivity, not physical label error. Native finite cutoffs: Coulomb 10 A, burial 15 A, desolvation 20 A. Complete neutral C/O buffers only; ignored/unsupported chemistry cannot establish safety. Other stripped species absent. Nearest-partner free ownership is conditional for bridging species. Diagnostic sample oversamples lost training pairs. Retention is buffer-only among matched native eligible sites; natural-gap and other-component masks not applied. Repeated observations correlated; no statistical safety guarantee.',
        'collector_code_sha256':digest(Path(__file__)),'manifest_sha256':digest(out/'manifest.json'),'successful_complexes':[r['complex_id'] for r in results if r['status']=='complete']}
    atomic_json(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('statuses','observations','retention')}),flush=True)


def plots(out):
    require_compute()
    import subprocess
    subprocess.run([str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/radial-plots/bin/python'),'-m','pkabench.buffer_plots',str(out)],check=True)
