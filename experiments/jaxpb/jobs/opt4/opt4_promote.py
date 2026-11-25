"""Promote independently selected measured CPU arms; local source-only operation."""
import argparse, ast, re, subprocess, tempfile
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser()
p.add_argument('--maps',type=int,choices=(0,1,2,3,4,5),required=True)
p.add_argument('--geometry',type=int,choices=(0,1,2),required=True)
p.add_argument('--slab',type=int,choices=(32,128),default=128)
p.add_argument('--modeltiles',action='store_true')
p.add_argument('--site-chunk',type=int,choices=(4096,16384,65536),default=4096)
p.add_argument('--small-geometry',action='store_true')
p.add_argument('--post',action='store_true')
a=p.parse_args()
factory=Path('jobs/opt4_arms.py').read_text(); tree=ast.parse(factory)
kt=Path('jobs/opt4_kernels.py').read_text(); funcs={n.name:ast.get_source_segment(kt,n) for n in ast.parse(kt).body if isinstance(n,ast.FunctionDef)}
factory_funcs={n.name:ast.get_source_segment(factory,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
with tempfile.TemporaryDirectory(prefix='opt4-baseline-') as tmp:
 root=Path(tmp)
 for name in ('maps','geometry','pb'):
  root.joinpath(name+'.py').write_bytes(subprocess.check_output(['git','show','bdffc91:src/jaxpb/'+name+'.py']))
 ns=dict(Path=Path,M=SimpleNamespace(__file__=str(root/'maps.py')),G=SimpleNamespace(__file__=str(root/'geometry.py')),P=SimpleNamespace(__file__=str(root/'pb.py')),module=lambda name,source,extra:source)
 for name in ('compact_query','reach_need','span_hash','span_cand','compact_burial','site_burial','boundary_laplacian'):ns[name]=None
 # Select global geometry, site tiling, and chunk sizing independently.
 gf=factory_funcs['geometry_arm'].replace('if step>=3:', 'if MODEL_TILES:').replace('if step>=4:', 'if SMALL_GEOMETRY:')
 pf=factory_funcs['pb_arm'].replace(' from jaxpb import pb as P\n','')
 ns.update(MODEL_TILES=a.modeltiles,SMALL_GEOMETRY=a.small_geometry)
 exec(compile(factory_funcs['maps_arm']+'\n'+gf+'\n'+pf,'opt4-source-factories','exec'),ns)
 maps=ns['maps_arm'](a.maps); geo=ns['geometry_arm'](a.geometry)
 def helper(name,new,prefix):
  source=funcs.get(name,factory_funcs.get(name))
  source=re.sub(r'\b'+prefix+r'\.', '',source)
  return re.sub(r'\b'+name+r'\b',new,source)+'\n'
 common=funcs['compact_indices'].replace('compact_indices(start,count,rank)','_compact_slots(count,rank)')
 common='"""Occupied-slot rank inversion shared by CPU spatial queries."""\nimport jax.numpy as jnp\n\n'+common+'\n'
 def compact_call(s):return s.replace('compact_indices(st,count,rank)','_compact_slots(count,rank)')
 if a.maps in (1,2,3,5):maps+='\n\n'+compact_call(helper('compact_query','_compact_query','M')).replace('offs,slab=128)',f'offs,slab={a.slab})')
 if a.maps in (2,3,4):
  maps+='\n\n'+compact_call(helper('reach_query','_reach_query','M'))
  maps+='\n\n'+re.sub(r'\breach_query\b','_reach_query',helper('reach_need','_reach_need','M'))
 shared=a.maps!=0 or a.geometry>=2
 if a.maps:maps=maps.replace('from . import records as RC','from . import records as RC\nfrom ._cpu_cells import _compact_slots')
 if a.geometry:
  geo+='\n\nGEO_NCELL = 128\n'
  for name,new in (('span_hash','_span_hash'),('candidate_spans','_candidate_spans'),('span_cand','_span_cand')):
   geo+='\n\n'+re.sub(r'\bcandidate_spans\b','_candidate_spans',helper(name,new,'G'))
 if a.geometry>=2:
  geo=geo.replace('from . import records as RC','from . import records as RC\nfrom ._cpu_cells import _compact_slots')
  geo+='\n\n'+re.sub(r'\bcandidate_spans\b','_candidate_spans',compact_call(helper('compact_burial','_compact_burial','G')))
 if a.modeltiles:geo+='\n\n'+helper('site_burial','_site_burial','G').replace('ch=4096',f'ch={a.site_chunk}')
 def note(source,lines):
  end=source.index('"""',3)
  return source[:end]+'\n'+lines+'\n'+source[end:]
 if a.maps:
  notes=[]
  if a.maps in (1,2,3,5):notes.append('OPT4 CPU queries use occupied CSR slots in the original offset/slot order.\nThe running minimum and float32 norms preserve first-min ties.')
  if a.maps in (2,3,4):notes.append('CPU reach queries visit nearby probes at the needed midpoints, retaining the original float32 norm and strict radius test.')
  notes.append('GPU paths retain their original programs.')
  maps=note(maps,'\n'.join(notes))
 if a.geometry or a.modeltiles:
  notes=[]
  if a.geometry:notes.append('OPT4 CPU geometry adds direct cell spans, with the original hash lookup as an out-of-range fallback.')
  if a.geometry>=2:notes.append('Burial compacts candidates in their original order and keeps float64 arithmetic.')
  if a.modeltiles:notes.append(f'CPU model-site burial uses {a.site_chunk}-point tiles and sequential site mapping to bound temporary storage.')
  geo=note(geo,'\n'.join(notes))
 sources={'maps':maps,'geometry':geo,'pb':(root/'pb.py').read_text()}
 if shared:sources['_cpu_cells']=common
 if a.post:sources['pb']=ns['pb_arm']()+'\n\n'+helper('boundary_laplacian','_boundary_laplacian','P')
 # Parse every result before writing any production module.
 for name,source in sources.items():ast.parse(source,name)
 for name,source in sources.items():
  path=Path('src/jaxpb/'+name+'.py')
  if not path.exists() or path.read_text()!=source:path.write_text(source)
 print('Promoted:',vars(a))
