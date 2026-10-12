"""Full ATLAS: stream canonical zstd graph records into 72 resumable shards."""
import os,json,time,io,gzip,tarfile,hashlib,traceback,collections,concurrent.futures,multiprocessing,shutil
from pathlib import Path
R=Path(os.environ['PKABENCH_RUNTIME']);O=R/'pretraining/atlas-prepared-v1/graph-v1'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def init(q):os.sched_setaffinity(0,{q.get()})
def one(row):
 cid=row['id']
 for folder in [O/'graphs'/cid,R/'pretraining/atlas-human-v1/graphs'/cid]:
  if not (folder/'receipt.json').exists():continue
  rec=read(folder/'receipt.json')
  if rec['status']=='accepted' and rec['source_sha256']==row['archive']['sha256'] and rec['source_split']==row['split'] and rec['component_id']==row['sequence_component']:
   blob=(folder/'record.zst').read_bytes()
   if hashlib.sha256(blob).hexdigest()==rec['blob_sha256']:
    audit=read(folder/'site-audit.json')
    return blob,dict(rec,reused_verified_record=True),audit
  elif rec['status']=='excluded':return None,dict(rec,source_split=row['split']),read(folder/'site-audit.json') if (folder/'site-audit.json').exists() else []
 try:
  import numpy as np
  import biotite.structure as st
  from biotite.structure.io import pdbx
  from jaxpropka.topology import load_topology
  from jaxpropka.parameters import GROUPS,GROUP_AA
  from pkanet.graph import geometry
  from pkatrain.site_graph_data import build_site_graph
  from pkatrain.production_graphs import FIELDS,DIMS,shape,encode,arrays_digest
  assert row['source_metadata']['type'] in ('human','mouse','arabidopsis','additional_bacteria_or_archaea','36_pathogens','virulence-factor','human-virus')
  ix=row['archive']
  with open(ix['archive'],'rb') as f:f.seek(ix['offset']);b=f.read(ix['bytes'])
  assert hashlib.sha256(b).hexdigest()==ix['sha256']
  with tarfile.open(fileobj=io.BytesIO(gzip.decompress(b))) as tf:
   def j(n):return json.load(tf.extractfile(n))
   meta=j('meta.json');labels=j('labels.json');sites=j('sites.json');evidence=j('input_atom_mask.json')
   assert meta['source_metadata']==row['source_metadata'] and not labels['errors']
   ab=pdbx.CIFFile.read(io.StringIO(tf.extractfile('AB.cif').read().decode()))
   source=pdbx.CIFFile.read(io.StringIO(tf.extractfile('source.cif').read().decode()))
  atoms=pdbx.get_structure(source,model=1,extra_fields=['b_factor'])
  ca=atoms[atoms.atom_name=='CA'];conf={(str(a.chain_id),int(a.res_id),str(a.ins_code).strip()):float(a.b_factor) for a in ca}
  assert len(conf)==len(ca) and all(np.isfinite(v) and 0<=v<=100 for v in conf.values())
  ab_atoms=pdbx.get_structure(ab,model=1,altloc='occupancy',use_author_fields=True,include_bonds=True)
  topo=load_topology(ab_atoms,gap_policy='cap',freeze_disulfides=True)
  assert np.isfinite(topo.backbone).all()
  graph,valid=geometry(topo.backbone,topo.chain_index,radius=20.)
  nodes=np.concatenate((np.eye(20,dtype=np.float32)[topo.native_index],np.stack((topo.nterm,topo.cterm,np.zeros(topo.n_residues,bool),valid),axis=-1)),axis=-1).astype(np.float32)
  graph.update(nodes=nodes,node_mask=np.ones(topo.n_residues,bool))
  lookup={(k.chain,k.number,k.insertion):i for i,k in enumerate(topo.keys)}
  assert len(lookup)==topo.n_residues
  ev={tuple(x['key']):x for x in evidence['atoms']}
  maps={};alias={'NTR':'NTERM','CTR':'CTERM'}
  for model in ['pkai','pkai_plus']:
   maps[model]={}
   for state in ['AB','A','B']:
    mm={}
    for c,n,ins,g,v in labels[model][state]:
     key=(c,int(n),ins,alias.get(g,g));assert key not in mm and np.isfinite(v);mm[key]=v
    maps[model][state]=mm
  queries=[];targets=[];flags=[];distances=[];audit=[];reasons=collections.Counter()
  seen=set()
  for s in sites:
   key=tuple(s[k] for k in ['chain','resnum','icode','group']);assert key not in seen;seen.add(key)
   rk=key[:3];reason=None;confidence=conf.get(rk)
   if not s['train_mask']:reason='source_mask'
   elif not s['functional_atoms_complete'] or s['was_completed'] or s['is_break_terminus']:reason='invalid_source_mask'
   elif rk not in ev or rk not in lookup:reason='unmapped'
   elif confidence is None or confidence<70:reason='site_pLDDT_below_70_or_missing'
   elif key not in maps['pkai']['AB'] or key not in maps['pkai'][s['partner']]:reason='teacher_unsupported_or_missing'
   elif s['group']=='ARG':reason='unsupported_ARG'
   if reason:reasons[reason]+=1;audit.append(dict(key=key,eligible=False,reason=reason,pLDDT=confidence));continue
   i=lookup[rk];g=GROUPS.index(s['group'])
   assert g>=7 or topo.native_index[i]==GROUP_AA[g]
   assert g!=7 or topo.nterm[i];assert g!=8 or topo.cterm[i]
   assert valid[i]
   distance=float(s['min_partner_distance']);assert np.isfinite(distance) and distance>=0
   queries.append((i,g));targets.append((maps['pkai']['AB'][key],maps['pkai'][s['partner']][key]));flags.append(bool(s['interface']));distances.append(distance)
   audit.append(dict(key=key,eligible=True,pLDDT=confidence,weight=1.,partner=s['partner']))
  if not queries or not any(flags):
   rec=dict(id=cid,status='excluded',source_split=row['split'],reason='no_eligible_paired_interface_sites',masked=dict(reasons),sites=len(sites));return None,rec,audit
  q=np.asarray(queries,np.int32);graph.update(query_residue=q[:,0],query_group=q[:,1]);site=build_site_graph(topo.backbone,topo.chain_index,nodes,q[:,0],q[:,1]);graph.update(site)
  graph.update(branch_edge_mask=np.stack((graph['edge_mask'],graph['edge_mask']&(graph['edge'][...,-1]>.5))),branch_site_edge_mask=np.stack((site['site_edge_mask'],site['site_edge_mask']&(site['site_edge'][...,31]>.5))),targets=np.asarray(targets,np.float32).T,w_burial=np.ones(len(q),np.float32),w_interface=np.ones(len(q),np.float32),interface=np.asarray(flags,bool),partner_distance_A=np.asarray(distances,np.float32),rsa_free=np.full(len(q),np.nan,np.float32))
  dims=dict(n=len(nodes),k=graph['neighbors'].shape[1],q=len(q),s=len(site['site_type']),sk=site['site_neighbors'].shape[1]);fields=FIELDS['pinder']
  for f in fields:
   assert graph[f].shape==shape(f,*(dims[d] for d in DIMS)),f
   if f!='rsa_free':assert np.isfinite(graph[f]).all(),f
  for f,limit in [('neighbors',dims['n']),('query_residue',dims['n']),('site_residue',dims['n']),('site_neighbors',dims['s']),('query_site',dims['s'])]:assert ((graph[f]>=0)&(graph[f]<limit)).all(),f
  assert np.array_equal(site['site_residue'][site['query_site']],q[:,0]);assert np.array_equal(site['site_type'][site['query_site']],q[:,1])
  blob=encode(graph,fields)
  rec=dict(id=cid,status='accepted',source_split=row['split'],component_id=row['sequence_component'],stratum=row['stratum'],**{k:int(v) for k,v in dims.items()},q_interface=sum(flags),masked=dict(reasons),sites=len(sites),arrays_sha256=arrays_digest(graph,fields),blob_sha256=hashlib.sha256(blob).hexdigest(),source_sha256=ix['sha256'],dtypes={f:graph[f].dtype.str for f in fields},w_burial_mean=1.,w_interface_mean=1.)
  return blob,rec,audit

 except Exception as e:
  return None,dict(id=cid,status='error',source_split=row['split'],error=str(e),traceback=traceback.format_exc()),[]
def shard(task):
 number,rows=task;folder=O/'shards'/f'{number:04d}-of-0072';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'receipt.json').exists():
  old=read(folder/'receipt.json');assert old['ids']==[r['id'] for r in rows];return old
 results=[];pending=folder/'records.bin.partial';auditpath=folder/'site-audits.jsonl.gz.partial';cleanup=[];started=time.time()
 with pending.open('wb') as f,gzip.open(auditpath,'wt',compresslevel=3) as auditstream:
  for i,row in enumerate(rows,1):
   blob,record,audit=one(row)
   if blob is not None:
    record.update(shard_path=str(folder/'records.bin'),blob=[f.tell(),len(blob)])
    f.write(blob)
   auditstream.write(json.dumps(dict(id=row['id'],sites=audit),separators=(',',':'))+'\n')
   results.append(record);cleanup.append(O/'graphs'/row['id'])
   if i%32==0 or i==len(rows):
    f.flush();os.fsync(f.fileno());auditstream.flush()
    # These are only disposable graph intermediates generated by this task;
    # original labelled archives remain the reproducible source of every record.
    for path in cleanup:
     if path.exists():shutil.rmtree(path)
    cleanup=[]
   if i%100==0 or i==len(rows):save(folder/'progress.json',dict(task=number,done=i,total=len(rows),counts=dict(collections.Counter(x['status'] for x in results)),seconds=time.time()-started))
  f.flush();os.fsync(f.fileno())
 if any(r['status']=='error' for r in results):
  save(folder/'failures.json',[r for r in results if r['status']=='error']);raise RuntimeError(f'Errors in shard {number}; source archives retained')
 pending.replace(folder/'records.bin');auditpath.replace(folder/'site-audits.jsonl.gz')
 receipt=dict(task=number,ids=[r['id'] for r in rows],records=results,seconds=time.time()-started)
 save(folder/'receipt.json',receipt);return receipt
if __name__=='__main__':
 assert os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_CPUS_PER_TASK'])==72 and int(os.environ['SLURM_JOB_NUM_NODES'])==1
 rows=read(O/'source-inventory.json');ctx=multiprocessing.get_context('spawn');q=ctx.Queue();cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=72
 for c in cpus[:72]:q.put(c)
 (O/'shards').mkdir(exist_ok=True);started=time.time();results=[]
 with concurrent.futures.ProcessPoolExecutor(72,mp_context=ctx,initializer=init,initargs=(q,)) as pool:
  futures=[pool.submit(shard,(i,rows[i::72])) for i in range(72)]
  for fut in concurrent.futures.as_completed(futures):
   rec=fut.result();results.extend(rec['records']);print(json.dumps(dict(stage='shard_committed',task=rec['task'],records=len(rec['records']),seconds=rec['seconds'])),flush=True)
 byid={r['id']:r for r in results};assert len(byid)==len(rows)
 results=[byid[r['id']] for r in rows];save(O/'build-results.json',results)
 save(O/'build-progress.json',dict(done=len(rows),total=len(rows),seconds=time.time()-started,counts=dict(collections.Counter(r['status'] for r in results)),format='compressed shards'))
 print(json.dumps(dict(stage='all_shards_complete',count=len(rows),seconds=time.time()-started)),flush=True)
