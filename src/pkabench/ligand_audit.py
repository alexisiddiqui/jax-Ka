"""Inventory ligand-rejected pairs without altering chemistry policy."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from .runtime import require_compute,atomic_json,digest


def audit(campaign,out):
    require_compute()
    import numpy as np
    from scipy.spatial import cKDTree
    from biotite.structure.io import pdbx
    from .audit import component_inventory
    from .prep import CANONICAL
    campaign=Path(campaign); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((campaign/'manifest.json').read_text()); policy=manifest['policy']
    selected=[r for p in sorted((campaign/'rows').glob('*.json')) if (r:=json.loads(p.read_text())).get('code')=='ligand']
    candidates={'GOL','EDO','PEG','PGE','PG4','1PE','MPD','DMS','SO4','PO4','ACT','FMT','TRS','MES','HEP','BME'}
    records=[]; pairs=[]; errors=[]; sources={}; identities=defaultdict(set)
    for row in selected:
        cid=row['complex_id']; path=campaign/'structures'/cid/'resolved-source.cif'
        try:
            cif=pdbx.CIFFile.read(path); atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
            partners={p:row[f'partner_{p}_chains'] for p in ('A','B')}
            trees={p:cKDTree(atoms.coord[np.isin(atoms.chain_id,chains)&np.isin(atoms.res_name,list(CANONICAL))&~np.isin(np.char.upper(atoms.element),['H','D'])]) for p,chains in partners.items()}
            components=component_inventory(atoms,cif,partners['A']+partners['B']); declared=defaultdict(set)
            conn=cif.block.get('struct_conn')
            if conn is not None:
                for i,kind in enumerate(conn['conn_type_id'].as_array(str)):
                    if not str(kind).lower().startswith(('covale','metalc')): continue
                    for p in ('ptnr1','ptnr2'):
                        c,n=f'{p}_label_asym_id',f'{p}_label_comp_id'
                        if c in conn and n in conn: declared[str(conn[c].as_array(str)[i]),str(conn[n].as_array(str)[i])].add(str(kind))
            protection=set(declared); local=[]
            for component in components:
                a=atoms[component['start']:component['end']]; xyz=a.coord[~np.isin(np.char.upper(a.element),['H','D'])]
                distances={p:float(t.query(xyz)[0].min()) for p,t in trees.items()}
                bridge=max(distances.values())<=4.; close=min(distances.values())<1.9
                if bridge or close: protection.add((component['chain'],component['name']))
                local.append((component,distances,bridge,close))
            remaining=[]
            for component,d,bridge,close in local:
                key=component['chain'],component['name']; protected=key in protection
                removed=component['name'] in policy['neutral_additives'] and component['code']=='ligand' and not protected
                category='already_removable' if removed else 'other_chemistry' if component['code']!='ligand' else 'connected_or_bridging' if protected else 'remote_gt10A' if min(d.values())>10 else 'additive_candidate_near' if component['name'] in candidates else 'other_ligand_near'
                rec={k:component[k] for k in ('chain','resnum','name','code')}
                rec.update(complex_id=cid,pdb_id=row['pdb_id'],distance_A=d['A'],distance_B=d['B'],bridges_partners=bridge,within_1_9A=close,
                    declared_connections='|'.join(sorted(declared[key])),protected=protected,already_removable=removed,
                    additive_candidate=component['name'] in candidates,category=category)
                records.append(rec)
                if not removed: remaining.append(rec)
                identities[component['name']].add(cid)
            remote_only=bool(remaining) and all(r['category']=='remote_gt10A' for r in remaining)
            additive_only=bool(remaining) and all(r['code']=='ligand' and r['additive_candidate'] and not r['protected'] for r in remaining)
            mixed_possible=bool(remaining) and all(r['category'] in ('remote_gt10A','additive_candidate_near') for r in remaining)
            pairs.append({'complex_id':cid,'pdb_id':row['pdb_id'],'original_rejection':row['detail'],
                'remaining_components':len(remaining),'remaining_names':sorted({r['name'] for r in remaining}),
                'remote_only':remote_only,'unprotected_additives_only':additive_only,'remote_or_additives_only':mixed_possible,
                'has_other_chemistry':any(r['code']!='ligand' for r in remaining),'has_protected_component':any(r['protected'] for r in remaining)})
            sources[cid]=digest(path)
        except Exception as exc: errors.append({'complex_id':cid,'error':str(exc)})
        atomic_json(out/'progress.json',{'done':len(pairs),'errors':len(errors),'total':len(selected)})
    if records:
        with (out/'components.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    atomic_json(out/'pairs.json',pairs)
    report={'selected_pairs':len(selected),'audited_pairs':len(pairs),'unique_assemblies':len({p['pdb_id'] for p in pairs}),
        'errors':errors,'component_categories':dict(Counter(r['category'] for r in records)),
        'top_components_by_pairs':sorted(({'name':n,'pairs':len(ids)} for n,ids in identities.items()),key=lambda r:(-r['pairs'],r['name']))[:30],
        'pair_counterfactuals':{k:sum(p[k] for p in pairs) for k in ('remote_only','unprotected_additives_only','remote_or_additives_only','has_other_chemistry','has_protected_component')},
        'limits':'Inventory only. Counterfactual categories overlap and do not establish recoverable pairs: no preparation rerun. First rejection can hide additional chemistry. Distance >10 A is descriptive, not proof of negligible pKa effect. Additive names are review candidates, not approved removals; buffers/ions can be charged. Component instances repeat across candidate pairs; pair-frequency counts are deduplicated by identity. Connection protection is conservative at chain/component-name level. Source conformer resolution retained.',
        'policy_changed':False}
    atomic_json(out/'report.json',report)
    atomic_json(out/'manifest.json',{'campaign':str(campaign),'source_manifest_sha256':digest(campaign/'manifest.json'),'sources':sources,'code_sha256':digest(Path(__file__)),'existing_policy':policy})
    print(json.dumps(report,indent=2))
