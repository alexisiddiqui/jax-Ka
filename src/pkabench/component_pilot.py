"""Nonprotein support audit and explicit-ion removal diagnostics."""
import json
import os
from pathlib import Path
from collections import Counter,defaultdict
from .runtime import require_compute,atomic_json,digest


def inventory(campaign,out,shard,shards):
    require_compute()
    import numpy as np
    from biotite.structure.io import pdbx
    from .audit import component_inventory
    campaign=Path(campaign); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    candidates=[r for p in sorted((campaign/'rows').glob('*.json')) if (r:=json.loads(p.read_text())).get('code') in ('metal','nonstandard_residue')]
    rows=[]
    for row in candidates[shard::shards]:
        cid=row['complex_id']; source=campaign/'structures'/cid/'resolved-source.cif'
        try:
            cif=pdbx.CIFFile.read(source); atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
            components=component_inventory(atoms,cif,row['partner_A_chains']+row['partner_B_chains'])
            conn=cif.block.get('struct_conn'); connections=[]
            if conn is not None:
                for i,kind in enumerate(conn['conn_type_id'].as_array(str)):
                    if str(kind).lower().startswith(('covale','metalc')):
                        connections.append({k:str(conn[k].as_array(str)[i]) for k in conn if k=='conn_type_id' or k.startswith(('ptnr1_','ptnr2_'))})
            chem=cif.block.get('chem_comp'); metadata={}
            if chem is not None:
                for i,n in enumerate(chem['id'].as_array(str)):
                    metadata[str(n)]={k:str(chem[k].as_array(str)[i]) for k in ('name','type','mon_nstd_parent_comp_id','pdbx_formal_charge') if k in chem}
            for c in components:
                c['ccd']=metadata.get(c['name'],{}); c['elements']=sorted(set(map(str,atoms.element[c['start']:c['end']])))
                c['declared_connections']=[x for x in connections if any(x.get(p+'_label_asym_id')==c['chain'] and x.get(p+'_label_comp_id')==c['name'] for p in ('ptnr1','ptnr2'))]
            result={'complex_id':cid,'pdb_id':row['pdb_id'],'original_code':row['code'],'source_sha256':digest(source),'components':components,'status':'complete'}
        except Exception as exc: result={'complex_id':cid,'status':'error','error':str(exc)}
        rows.append(result)
    atomic_json(out/f'inventory-{shard}.json',rows); print(json.dumps({'shard':shard,'pairs':len(rows),'errors':sum(r['status']=='error' for r in rows)}))


def collect(campaign,out,shards):
    require_compute(); campaign=Path(campaign); out=Path(out)
    rows=[r for shard in range(shards) for r in json.loads((out/f'inventory-{shard}.json').read_text())]
    bykind=defaultdict(lambda:defaultdict(set)); connected=defaultdict(set); eligible=[]
    for r in rows:
        if r['status']!='complete': continue
        for c in r['components']:
            bykind[c['code']][c['name']].add(r['complex_id'])
            if c['declared_connections']: connected[c['code']].add(r['complex_id'])
        unsupported=[c for c in r['components'] if c['name'] not in ('NA','CL','GOL','EDO','PEG')]
        sodium=[c for c in r['components'] if c['name']=='NA' and c['end']-c['start']==1]
        if sodium and not unsupported: eligible.append(r)
    report={'audited_pairs':len(rows),'errors':[r for r in rows if r['status']=='error'],
        'identities':{kind:sorted(({'name':n,'pairs':len(ids)} for n,ids in names.items()),key=lambda x:-x['pairs']) for kind,names in bykind.items()},
        'pairs_with_declared_connections':{k:len(v) for k,v in connected.items()},'sodium_only_context_candidates':[r['complex_id'] for r in eligible],
        'support':'Installed PypKa keep_ions whitelist is NA+/CL- only; G54A7 charges +1/-1 and radii 1.097/1.820 A. Generic ligands and other metals lack a verified retention/parameterization path. Nonstandard residues require identity-specific review; no automatic deletion or conversion.',
        'ligand_sasa_error_status':'Not measurable with current generic-ligand reference support. Exposure must be computed with the retained component represented, and removal error stratified at 10/20 A once validated parameters exist.'}
    atomic_json(out/'report.json',report); atomic_json(out/'sodium-candidates.json',eligible)
    print(json.dumps(report,indent=2))


def gate(out):
    """Synthetic sodium is a retention/parameter plumbing test, not calibration."""
    require_compute()
    import subprocess
    from .adapters.base import TEACHER
    out=Path(out); work=out/'sodium-retention-gate-v2'; work.mkdir(exist_ok=False)
    runtime=Path(os.environ['PKABENCH_RUNTIME'])
    source=runtime/'audits/pkpdb-crystal-compare-v1/1ciq/input.pdb'
    lines=[line for line in source.read_text().splitlines() if line.startswith(('ATOM','TER'))]
    first=next(line for line in lines if line.startswith('ATOM'))
    xyz=[float(first[a:b]) for a,b in ((30,38),(38,46),(46,54))]; xyz[0]+=25
    lines.append(f'ATOM  {90000:5d} {"NA+":>4s} {"NA+":>3s} {first[21]}{900:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.:6.2f}{0.:6.2f}          NA')
    (work/'input.pdb').write_text('\n'.join(lines)+'\nEND\n')
    atomic_json(work/'request.json',{'pdb':str(work/'input.pdb'),'config':TEACHER,'ion_xyz':xyz,'source_sha256':digest(source),'scope':'Synthetic ion backend gate; no biological removal-error claim'})
    env=os.environ.copy(); env['LD_LIBRARY_PATH']=str(runtime/'fortran/lib')
    with (work/'stdout.log').open('w') as stdout, (work/'stderr.log').open('w') as stderr:
        p=subprocess.run([str(runtime/'envs/pypka/bin/python'),'-m','pkabench.component_gate_worker',str(work/'request.json')],cwd=work,env=env,stdout=stdout,stderr=stderr,timeout=300)
    if p.returncode: raise RuntimeError(f'Ion gate failed; inspect {work}')
    print((work/'result.json').read_text())
