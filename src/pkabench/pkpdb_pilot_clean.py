"""Map deposited pKPDB midpoints without renumbering; apply existing cleaning rules."""
import json
import re
import sqlite3
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx
from scipy.spatial import cKDTree
from .runtime import atomic_json,digest,config_hash
from .pkpdb_pilot_refs import cif,sequences
from .prep import CANONICAL,Rejection
from .annotate import SITE_ATOMS
from .anchor_tiers import terminal_radius,classify
from .supervision import inventory,clearance


def metadata(task):
    pdb,path=task
    try:
        rows,other=sequences(cif(path))
        if other:return dict(pdb_id=pdb,status='rejected',reason='nonprotein_polymer')
        n=sum(len(r['sequence']) for r in rows)
        if not 30<=n<=1500:return dict(pdb_id=pdb,status='rejected',reason='size_outside_30_1500',n=n)
        return dict(pdb_id=pdb,status='candidate',chains=rows,n=n)
    except Exception as exc:return dict(pdb_id=pdb,status='rejected',reason='sequence_metadata_unavailable',detail=repr(exc))


def gap_context(evidence,residues):
    bykey={tuple(r['key']):r for r in evidence['atoms']}
    positions={(r['key'][0],r['label_seq_id']):tuple(r['key']) for r in evidence['atoms']}
    gaps=[]
    for d in evidence['defects']:
        if d['kind'] not in ('terminal_gap','internal_gap'):continue
        internal=d['kind']=='internal_gap'
        radius=(15. if d['length']<=3 else None) if internal else terminal_radius(d['length'])
        keys=[positions[d['chain'],p] for p in (d['start']-1,d['end']+1) if (d['chain'],p) in positions]
        anchor=None
        if radius is not None and len(keys)==(2 if internal else 1) and all(k in residues for k in keys):
            atoms=[residues[k].coord[np.isin(residues[k].atom_name,list(set(('N','CA','C','O'))-set(bykey[k]['missing_atoms'])))] for k in keys]
            anchor=np.concatenate(atoms)
            if any(len(a)!=4 for a in atoms):radius=None
        else:radius=None
        gaps.append((d,radius,anchor))
    known=bool(evidence['sequences']) and all(r['sequence_source']=='entity_poly_canonical' for r in evidence['sequences'])
    return gaps,known


def gap_tier(points,eligible,gaps):
    if not eligible:return 'ineligible'
    near=False;uncalibrated=False;distances=[]
    for d,radius,anchor in gaps:
        distance=clearance(points,d);distances.append(distance)
        if radius is not None:near |= float(np.linalg.norm(points[:,None]-anchor[None,:],axis=-1).min())<radius
        else:uncalibrated |= distance<20
    return classify(True,all(x>=20 for x in distances),near,uncalibrated)


def clean(task):
    root,out,row=task;root=Path(root);out=Path(out);pdb=row['pdb_id'];dest=out/'entries'/pdb
    dest.mkdir(parents=True,exist_ok=True);resultfile=dest/'receipt.json'
    if resultfile.exists():return json.loads(resultfile.read_text())
    try:
        path=root/'pretraining/pkpdb-v1/structures'/pdb[1:3]/f'{pdb}.cif.gz'
        from .conformers import resolve
        from .glycan_buffer_policy import classify_components,BUFFERS
        from pkanet.graph import geometry
        from jaxpropka.parameters import THREE_TO_INDEX,GROUPS
        source_receipt=json.loads((path.parent/f'{pdb}.json').read_text());assert digest(path)==source_receipt['sha256']
        selected=[r['chain'] for r in row['chains']];partners={'A':selected,'B':[]}
        file,conformers=resolve(cif(path),selected)
        evidence=inventory(file,partners)
        label=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=False)
        author=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True)
        assert len(label)==len(author) and np.array_equal(label.coord,author.coord)
        working=label.copy();working.res_id=author.res_id.copy();working.ins_code=author.ins_code.copy()
        cat=file.block['chem_comp'];types=dict(zip(cat['id'].as_array(str),cat['type'].as_array(str)))
        # Noncanonical peptide residues are absent from the backbone model and become sequence gaps.
        modified={name for name,kind in types.items() if 'PEPTIDE' in str(kind).upper() and name not in CANONICAL}
        component_atoms=working[~np.isin(working.res_name,list(modified))]
        removals,_=classify_components(component_atoms,file,partners)
        for c in removals:
            if c['name'] in BUFFERS:c['train_radius_A']=15.;c['eval_radius_A']=20.
        trees=[(cKDTree(c['coordinates']),c) for c in removals]
        keep=np.isin(working.res_name,list(CANONICAL))&np.isin(working.chain_id,selected)&~np.isin(np.char.upper(working.element),['H','D'])
        starts=struc.get_residue_starts(working,add_exclusive_stop=True)
        residues={};nodes=[];backbone=[];chainindex=[];lookup=defaultdict(list);original=[];positions=[]
        polylen={r['chain']:len(r['sequence']) for r in row['chains']}
        for s,e in zip(starts[:-1],starts[1:]):
            if not keep[s]:continue
            a=working[s:e];k=(str(a.chain_id[0]),int(a.res_id[0]),str(a.ins_code[0]).strip())
            if k in residues:raise Rejection('ambiguous_residue_key',str(k))
            residues[k]=a
            names={str(n):i for i,n in enumerate(a.atom_name)}
            if not {'N','CA','C'}<=names.keys():raise Rejection('incomplete_backbone','Backbone input requires N, CA and C; no coordinates invented')
            seqpos=int(label.res_id[s]);authchain=str(author.chain_id[s]);n=len(nodes)
            bb=np.stack([a.coord[names[name]] if name in names else a.coord[names['C']] for name in ('N','CA','C','O')])
            backbone.append(bb);chainindex.append(selected.index(k[0]));positions.append(seqpos)
            aa=np.eye(20,dtype=np.float32)[THREE_TO_INDEX[str(a.res_name[0])]]
            nodes.append(np.concatenate((aa,[seqpos==1,seqpos==polylen[k[0]],False,True])))
            lookup[authchain,k[1]].append((n,k,str(a.res_name[0])));original.append(dict(chain=authchain,resnum=k[1],icode=k[2],label_chain=k[0],label_seq_id=seqpos))
        if not nodes:raise Rejection('no_backbone','No canonical protein backbone')
        graphs,valid=geometry(np.asarray(backbone),np.asarray(chainindex));nodes=np.asarray(nodes,np.float32);nodes[:,23]=valid
        graphs.update(nodes=nodes,node_mask=np.ones(len(nodes),bool))
        gaps,known=gap_context(evidence,residues);breaks={tuple(k) for k in evidence['artificial_terminal_keys']}
        db=sqlite3.connect(f'file:{out}/labels.sqlite?mode=ro',uri=True)
        labels=db.execute('select chain,kind,number,pka from labels where pdb=?',(pdb,)).fetchall();db.close()
        counts=Counter();mapped=[];queries=[];values=[];seen=Counter((chain,kind,number) for chain,kind,number,_ in labels)
        for chain,kind,number,value in labels:
            counts['deposited_labels']+=1
            if seen[chain,kind,number]>1:counts['duplicate_label_key']+=1;continue
            match=re.fullmatch(r'(-?\d+)([A-Za-z]?)',number)
            if not match:counts['unparseable_residue_number']+=1;continue
            num=int(match[1]);ins=match[2];choices=lookup.get((chain,num),[])
            if not ins and any(k[2] for _,k,_ in choices):counts['ambiguous_insertion_code']+=1;continue
            choices=[r for r in choices if r[1][2]==ins]
            if len(choices)!=1:counts['unmapped_or_ambiguous_site']+=1;continue
            n,k,resname=choices[0];group={'NTR':'NTERM','CTR':'CTERM'}.get(kind,kind)
            if group not in GROUPS or (group not in ('NTERM','CTERM') and group!=resname):counts['residue_type_mismatch']+=1;continue
            if value is None or not np.isfinite(value):counts['nonfinite_label']+=1;continue
            a=residues[k];points=a.coord[np.isin(a.atom_name,SITE_ATOMS[group])]
            functional=set(SITE_ATOMS[group])<=set(a.atom_name)
            terminal=(group=='NTERM' and positions[n]!=1) or (group=='CTERM' and positions[n]!=polylen[k[0]])
            tier=gap_tier(points,known and functional and not terminal and not (group in ('NTERM','CTERM') and k in breaks),gaps)
            distances=[float(tree.query(points)[0].min()) if len(points) else 0. for tree,_ in trees]
            train=tier in ('clean','uncertain') and all(d>=c['train_radius_A'] for d,(_,c) in zip(distances,trees))
            evaluation=tier in ('clean','uncertain') and all(d>=c['eval_radius_A'] for d,(_,c) in zip(distances,trees))
            mapped.append(dict(complex_id=pdb,**original[n],group=group,pka=float(value),train_mask=bool(train),eval_mask=bool(evaluation),
                natural_gap_tier=tier,functional_atoms_complete=functional,nearest_component_A=min(distances,default=None)))
            queries.append((n,GROUPS.index(group)));values.append(value);counts['raw_sites']+=1;counts['clean_sites']+=bool(train);counts['eval_sites']+=bool(evaluation)
        if not values:raise Rejection('no_mapped_labels',json.dumps(counts))
        if not counts['clean_sites']:raise Rejection('no_clean_sites',json.dumps(counts))
        q=np.asarray(queries,np.int32);graphs.update(query_residue=q[:,0],query_group=q[:,1])
        np.savez_compressed(dest/'graph.npz',**graphs,labels=np.asarray(values,np.float32))
        atomic_json(dest/'sites.json',mapped);atomic_json(dest/'conformers.json',conformers)
        atomic_json(dest/'defects.json',evidence);atomic_json(dest/'removed_components.json',removals)
        result=dict(pdb_id=pdb,complex_id=pdb,status='accepted',counts=dict(counts),n=len(nodes),k=graphs['neighbors'].shape[1],q=len(values),
            component_id='exact-'+config_hash(sorted(r['sequence'] for r in row['chains']))[:20],
            sha256=digest(dest/'graph.npz'),sites_sha256=digest(dest/'sites.json'),source_sha256=digest(path),
            noncanonical_residue_names=sorted(modified),keys=[[pdb,r['chain'],r['resnum'],r['icode'],r['group']] for r in mapped])
    except Rejection as exc:result=dict(pdb_id=pdb,status='rejected',reason=exc.code,detail=str(exc))
    except Exception as exc:
        import traceback
        result=dict(pdb_id=pdb,status='pipeline_error',reason=type(exc).__name__,detail=repr(exc),traceback=traceback.format_exc())
    atomic_json(resultfile,result);return result
