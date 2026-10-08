"""Method representability and rebuild probes; outputs are evidence, not scores."""
import json
import os
from pathlib import Path
import numpy as np
from .adapters.base import Adapter, execute
from .prep import read_cif, write_cif, export_pdb, complete, topology, missing_atoms
from .runtime import atomic_json, digest, require_compute
from .schema import key, read_table


def preflight(campaign, methods):
    require_compute(); campaign=Path(campaign).resolve(); runtime=Path(os.environ["PKABENCH_RUNTIME"])
    target=min(read_table(campaign/"structures.parquet"),key=lambda s:s["n_residues"])
    cid=target["complex_id"]; source=campaign/"structures"/cid
    sites=read_table(source/"sites.parquet"); atoms=read_cif(source/"AB.cif")
    root=campaign/"preflight"/os.environ["SLURM_JOB_ID"]; root.mkdir(parents=True,exist_ok=True)
    controls={}; chains=list(dict.fromkeys(map(str,atoms.chain_id)))
    permutation={c: f"control_{i}" for i,c in enumerate(reversed(chains))}
    # Reorder partners as well as relabel them, so PDB remapping cannot turn this
    # into a byte-identical input and accidentally make the control vacuous.
    order=np.concatenate([np.flatnonzero(atoms.chain_id==c) for c in reversed(chains)])
    relabel=atoms[order].copy(); relabel.chain_id=np.asarray([permutation[str(c)] for c in relabel.chain_id])
    write_cif(root/"relabel.cif",relabel)
    far=atoms.copy(); far.coord[np.isin(far.chain_id,target["partner_B_chains"]),0]+=1000
    write_cif(root/"far.cif",far)
    for method in methods:
        if method == "kaml":
            controls[method]={"status":"excluded","reason":"Released predictor explicitly requires a single chain; joint-system G3 unsupported",
                "source":str(runtime/"sources/KaMLs.json")}; continue
        results={}; errors={}; free_results={}
        for label,path,expected in (("native",source/"AB.cif",sites),("relabel",root/"relabel.cif",[{**s,"chain":permutation[s["chain"]]} for s in sites]),("far",root/"far.cif",sites)):
            adapter=Adapter(method,expected,runtime,timeout=1800)
            files={s:source/f"{s}.cif" for s in ("AB","A","B")} if label=="native" else {"AB":path}
            rows=adapter.run(files,root/method/label)
            if label=="native": free_results={key(r):r["pka"] for r in rows if r["state"]!="AB" and r["status"]=="ok"}
            if label=="relabel":
                inverse={v:k for k,v in permutation.items()}
                rows=[{**r,"chain":inverse[r["chain"]]} for r in rows]
            results[label]={key(r):r["pka"] for r in rows if r["state"]=="AB" and r["status"]=="ok" and r["group"] not in {"NTERM","CTERM"}}
            errors[label]=adapter.errors
        common=set(results["native"])&set(results["relabel"])&set(free_results)
        # Input chain labels must not change chemistry. Float/Monte Carlo methods
        # receive a looser tolerance; response to the partner is reported separately.
        tolerance=.1 if method=="pypka" else .03
        difference=max((abs(results["native"][k]-results["relabel"][k]) for k in common),default=None)
        response=max((abs(results["native"][k]-free_results[k]) for k in common),default=None)
        controls[method]={"status":"pass" if common and difference<=tolerance and response>0 and not errors["native"] and not errors["relabel"] else "unresolved",
            "n":len(common),"max_relabel_difference":difference,"max_partner_response":response,"errors":errors,
            "detail":"Chain-label invariance and AB versus isolated partner response; far-box control failures reported separately."}
        if method=="propka" and common and not errors['native']:
            work=root/method/'graph'; work.mkdir(parents=True,exist_ok=True)
            try:
                execute([str(runtime/'envs/runner/bin/python'),'-m','pkabench.propka_probe',
                    str(root/method/'native/AB/input.pdb'),str(work/'result.json')],work,300)
                graph=json.loads((work/'result.json').read_text())
                controls[method]['interaction_graph']=graph
                if graph['cross_chain_term_count']>0:
                    controls[method]['status']='pass'
                    controls[method]['order_sensitive']=difference>tolerance
                    controls[method]['detail']='Native PROPKA determinants explicitly cross chains. Partner-order sensitivity is reported independently of joint-system eligibility.'
            except Exception as exc: controls[method]['graph_error']=str(exc)
    # Explicitly test PDB2PQR with a distal-style deletion of a titratable sidechain.
    rebuild={"status":"fallback"}
    candidates=[i for i,a in enumerate(atoms) if str(a.atom_name) in {"OE1","OD1","NZ"}]
    if candidates:
        work=root/"completion"; work.mkdir()
        damaged=atoms[np.arange(len(atoms))!=candidates[0]]
        try:
            fixed=complete(damaged,str(runtime/"envs/pypka/bin/pdb2pqr30"),work)
            if missing_atoms(topology(fixed)): raise ValueError("missing atoms remain after rebuilding")
            rebuild={"status":"pass","evidence":str(work/"completion.log")}
        except Exception as exc:
            rebuild={"status":"fallback","reason":str(exc),"evidence":str(work/"completion.log")}
    training={"status":"unresolved"}
    if (runtime/"envs/pkai/bin/python").exists():
        work=root/"training"; work.mkdir()
        export_pdb(atoms,work/"input.pdb")
        atomic_json(work/"request.json",{"method":"pkai_training_probe","pdb":str(work/"input.pdb")})
        try:
            execute([str(runtime/"envs/pkai/bin/python"),"-m","pkabench.adapters.worker",str(work/"request.json"),str(work/"result.json")],work,300)
            training=json.loads((work/"result.json").read_text())
        except Exception as exc: training={"status":"fail","detail":str(exc)}
    report={"G3":controls,"G4":training,"G5":rebuild,"complex_id":cid,"workdir":str(root),
        "model_hashes":{p.name:digest(p) for p in (runtime/"envs/pkai/lib/python3.11/site-packages/pkai/models").glob("*.pt")}}
    atomic_json(campaign/"preflight.json",report); print(json.dumps(report,indent=2))
    if training["status"]=="pass":
        atomic_json(runtime/"receipts/pkai.validated.json",{"status":"pass","gate":"G4","evidence":str(root/"training/result.json"),"model_hashes":report["model_hashes"]})
