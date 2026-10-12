import os,json,csv,gzip,io,time,hashlib,traceback,concurrent.futures,multiprocessing,sys
from pathlib import Path
R=Path(os.environ['PKABENCH_RUNTIME']); S=R/'pretraining/atlas-high-confidence-v1'; O=R/'pretraining/atlas-prepared-v1'; D=O/'pilot'; D.mkdir(exist_ok=True)

def fa(p):
 d={};k=None
 for l in p.read_text().splitlines():
  if l.startswith('>'):k=l[1:];d[k]=''
  elif k:d[k]+=l.strip()
 return d

def initprep():
 global INDEX,SEQ
 INDEX={r['id']:r for r in csv.DictReader((S/'structure-index.tsv').open(),delimiter='\t')};SEQ=fa(S/'metadata/chains.fasta')

def prepare(r):
 start=time.monotonic(); dest=D/r['id'];dest.mkdir(exist_ok=True)
 try:
  import numpy as np
  import biotite.structure as st
  from biotite.structure.io.pdb import PDBFile
  from biotite.structure.io import pdbx
  from pkabench import curation,prep
  from pkabench.dataset_audit import AA
  from pkabench.pkpdb_pilot_clean import gap_context
  from pkabench.long_gap_policy import site_usable
  from pkabench.annotate import SITE_ATOMS
  ix=INDEX[r['id']]
  with (S/ix['shard']).open('rb') as f:f.seek(int(ix['offset']));raw=f.read(int(ix['compressed_bytes']))
  assert hashlib.sha256(raw).hexdigest()==ix['sha256']
  atoms=PDBFile.read(io.StringIO(gzip.decompress(raw).decode())).get_structure(model=1,extra_fields=['b_factor','occupancy'])
  chains=list(dict.fromkeys(map(str,atoms.chain_id)));assert len(chains)==2
  declared=[SEQ[k] for k in r['chains']];observed=[]
  for c in chains:
   sub=atoms[atoms.chain_id==c];observed.append(''.join(AA.get(str(sub.res_name[i]),'X') for i in st.get_residue_starts(sub)))
  assert observed==declared or observed==declared[::-1], 'source sequence mismatch'
  cf=pdbx.CIFFile();pdbx.set_structure(cf,atoms)
  cf.block['entity_poly']=pdbx.CIFCategory({'entity_id':['1','2'],'type':['polypeptide(L)']*2,'pdbx_seq_one_letter_code_can':observed})
  cf.block['struct_asym']=pdbx.CIFCategory({'id':chains,'entity_id':['1','2']})
  # Label positions must be canonical ordinal positions, independent of author numbering.
  labels=[]
  for c in chains:
   sub=atoms[atoms.chain_id==c]; starts=st.get_residue_starts(sub,add_exclusive_stop=True)
   labels.extend(str(i+1) for i,(a,b) in enumerate(zip(starts[:-1],starts[1:])) for _ in range(b-a))
  cf.block['atom_site']['label_seq_id']=pdbx.CIFColumn(labels)
  src=dest/'source.cif';cf.write(src)
  original=prep.prepare_pair
  def with_exe(cif,row,out,exe,**kw):return original(cif,row,out,'/scratch/u6xq/hussain.u6xq/_install/pdb2pqr/bin/pdb2pqr30',**kw)
  curation.prepare_pair=with_exe
  row=dict(complex_id=r['id'],pdb_id=r['id'],source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),partner_A_chains=[chains[0]],partner_B_chains=[chains[1]])
  info,sites,_=curation.prepare_revised(src,row,dest,diagnostic_perturbation=True)
  ab=prep.read_cif(dest/'AB.cif');rs=st.get_residue_starts(ab,add_exclusive_stop=True)
  residues={(str(ab.chain_id[a]),int(ab.res_id[a]),str(ab.ins_code[a]).strip()):ab[a:b] for a,b in zip(rs[:-1],rs[1:])}
  ev=json.loads((dest/'input_atom_mask.json').read_text());gaps,known=gap_context(ev,residues)
  # ATLAS source contains only canonical protein; component-bearing input is rejected by source checks/prep.
  for s in sites:
   a=residues[(s['chain'],s['resnum'],s['icode'])];pts=a.coord[np.isin(a.atom_name,SITE_ATOMS[s['group']])]
   eligible=known and s['functional_atoms_complete'] and not s['was_completed'] and not s['is_break_terminus']
   usable,reason=site_usable(pts,eligible,gaps) if len(pts) else (False,'no_atoms')
   s.update(train_mask=bool(usable),gap_rule=reason,interface=bool(s['in_interface_zone']))
  (dest/'sites.json').write_text(json.dumps(sites))
  for state in ('AB','A','B'):
   mapping=prep.export_pdb(prep.read_cif(dest/f'{state}.cif'),dest/f'{state}.pdb')
   (dest/f'map_{state}.json').write_text(json.dumps([[c,n,*v] for (c,n),v in mapping.items()]))
  (dest/'meta.json').write_text(json.dumps(dict(id=r['id'],source_metadata=r,training_enabled=False,geometry_gate='source HC selection; same diagnostic setting as PINDER',confidence_policy='pLDDT preserved; additional site confidence policy pending',content_sha256=info['content_sha256'])))
  result=dict(id=r['id'],status='prepared',seconds=time.monotonic()-start,residues=sum(r['lengths']),sites=len(sites))
 except Exception as e:result=dict(id=r['id'],status='rejected' if hasattr(e,'code') else 'error',reason=getattr(e,'code',str(e)),traceback=traceback.format_exc(),seconds=time.monotonic()-start,residues=sum(r['lengths']))
 return result

def initlabel():
 global Protein,PK_MODS,MODELS,torch
 import torch
 from importlib.util import find_spec
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 sys.path.insert(0,str(R/'envs/pkai/lib/python3.11/site-packages/pkai'))
 from protein import Protein,PK_MODS
 from pKAI import load_model
 MODELS={n:load_model(m,'cpu').eval() for n,m in [('pkai','pKAI'),('pkai_plus','pKAI+')]}

def label(r):
 t=time.monotonic();d=D/r['id']
 try:
  from pkai_termini import add_termini
  sites=json.loads((d/'sites.json').read_text());out={k:{} for k in MODELS}
  for state in ('AB','A','B'):
   mapping={(c,int(n)):(oc,int(on),oi) for c,n,oc,on,oi in json.loads((d/f'map_{state}.json').read_text())};reverse={v:k for k,v in mapping.items()}
   terms={g:[reverse[(s['chain'],s['resnum'],s['icode'])] for s in sites if s['group']==g and not s['is_break_terminus'] and (s['chain'],s['resnum'],s['icode']) in reverse] for g in ('NTERM','CTERM')}
   text,extra=add_termini((d/f'{state}.pdb').read_text(),terms['NTERM'],terms['CTERM'])
   for key,(_,c,n) in extra.items():mapping[key]=mapping[c,n]
   p=d/f'{state}-termini.pdb';p.write_text(text)
   protein=Protein(str(p));protein.apply_cutoff();rs=list(protein.iter_residues(titrable_only=True));assert rs,'no supported sites'
   x=torch.stack([r.input_layer for r in rs]);assert torch.isfinite(x).all()
   for name,model in MODELS.items():
    with torch.inference_mode():pred=torch.cat([model(b).reshape(-1) for b in x.split(256)]).tolist()
    out[name][state]=[[*mapping[(s.chain,int(s.resnumb))],s.resname,float(v)+PK_MODS[s.resname]] for s,v in zip(rs,pred)]
  out.update(termini=True,errors=[],precision='unrounded native model output',batch_size=256)
  (d/'labels.json').write_text(json.dumps(out));return dict(id=r['id'],status='labelled',seconds=time.monotonic()-t,sites=sum(len(x) for x in out['pkai'].values()))
 except Exception as e:return dict(id=r['id'],status='error',reason=str(e),traceback=traceback.format_exc(),seconds=time.monotonic()-t)

if __name__=='__main__':
 assert os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_CPUS_PER_TASK'])==72
 action=sys.argv[1];limit=int(sys.argv[2]) if len(sys.argv)>2 else 288
 rows=json.loads((O/'benchmark-sample.json').read_text())[:limit]
 if action=='label':rows=[r for r in json.loads((O/f'prep-{limit}.json').read_text())['results'] if r['status']=='prepared']
 workers=min(72,len(rows));beg=time.monotonic();results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initprep if action=='prep' else initlabel,mp_context=multiprocessing.get_context('spawn')) as ex:
  for result in ex.map(prepare if action=='prep' else label,rows,chunksize=1):
   results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='traceback'}),flush=True)
 report=dict(action=action,job=os.environ['SLURM_JOB_ID'],workers=workers,wall_seconds=time.monotonic()-beg,results=results)
 (O/f'{action}-{limit}.json').write_text(json.dumps(report,indent=2));print('FINISHED',action,report['wall_seconds'],flush=True)
