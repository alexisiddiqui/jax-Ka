"""Relate frozen-GQT cross-chain causal use to paired PypKa accuracy."""
from __future__ import annotations
import argparse,json,os
from collections import defaultdict
from pathlib import Path
import numpy as np

from .runtime import atomic_json,digest,require_compute
from .gqt_approach_attention import build_clusters,edge_categories,read,write_parquet,_checkpoint,_partner_chains

KEY=("complex_id","chain","resnum","icode","group")
PRIMARY=.05


def state_graph(path,query_keys,strict=True):
    from jaxpropka.topology import load_topology
    from jaxpropka.parameters import GROUPS,GROUP_AA
    from pkanet.graph import geometry
    from .prep import read_cif
    top=load_topology(read_cif(path),gap_policy="cap",freeze_disulfides=True)
    graph,valid=geometry(top.backbone,top.chain_index)
    graph["nodes"]=np.concatenate((np.eye(20,dtype=np.float32)[top.native_index],
        np.stack((top.nterm,top.cterm,top.disulfide,valid),axis=-1)),axis=-1).astype(np.float32)
    if strict:graph["nodes"][:,22]=0
    graph["node_mask"]=np.ones(top.n_residues,bool)
    lookup={(k.chain,k.number,k.insertion):i for i,k in enumerate(top.keys)};kept=[];queries=[]
    for key in query_keys:
        residue=tuple(key[:3]);i=lookup.get(residue)
        if i is None:continue
        g=GROUPS.index(key[3])
        if g<7 and top.native_index[i]!=GROUP_AA[g]:continue
        if g==7 and not top.nterm[i]:continue
        if g==8 and not top.cterm[i]:continue
        kept.append(tuple(key));queries.append((i,g))
    q=np.asarray(queries,np.int32).reshape((-1,2));graph.update(query_residue=q[:,0],query_group=q[:,1])
    chains=np.array([k.chain for k in top.keys])
    return graph,kept,chains


def ablate_edges(graph,cut):
    out={k:np.array(v,copy=True) for k,v in graph.items()}
    out["edge_mask"][cut]=False;out["switch"][cut]=0
    return out


def hl_cut(chains,neighbors,mask,hchain,lchain):
    receiver=np.arange(len(chains))[:,None];a=chains[receiver];b=chains[neighbors]
    return mask&(((a==hchain)&(b==lchain))|((a==lchain)&(b==hchain)))


def pad_one(graph,capacity):
    from pkatrain.graph_data import pad
    labels=np.zeros(len(graph["query_residue"]),np.float32)
    return pad(graph,labels,capacity)[0]


def _prediction_map(path,methods,cids):
    import pyarrow.parquet as pq
    table=pq.read_table(path,columns=list(KEY)+["state","method","pka","status"],
        filters=[("method","in",list(methods)),("complex_id","in",list(cids))]).to_pylist()
    return {(tuple(r[k] for k in KEY),r["state"],r["method"]):r for r in table}


def run(root,out):
    require_compute(threads=8,gpu_benchmark=True,allow_comp1400=True)
    import jax
    import jax.numpy as jnp
    import pyarrow.parquet as pq
    from pkanet.model import predict_pkpdb
    from pkatrain.graph_data import bucket,mask_features
    from .gqt_alignment_audit import structure_nodes
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=root/"pretraining/gqt-backbone-5k-pkmod-v1/unweighted";manifest=read(source/"manifest.json")
    graphroot=Path(manifest["validation_source"]);graphparent=read(graphroot/"manifest.json");recordroot=Path(graphparent["source"])
    approach=root/"audits/gqt-approach-attention-v1";audit=root/"audits/gqt-data-alignment-v1"
    params,checkpoint,metadata=_checkpoint(source,manifest);predict=jax.jit(jax.vmap(predict_pkpdb,in_axes=(None,0)))
    records=[r for r in manifest["records"] if r["split"]=="val"]
    cids=[r["complex_id"] for r in records];campaign=root/"campaigns/production-nojax-v1"
    pred=_prediction_map(campaign/"predictions.parquet",("pypka","pkai"),cids)
    sites=pq.read_table(campaign/"sites.parquet",filters=[("complex_id","in",cids)]).to_pylist();site_info={tuple(r[k] for k in KEY):r for r in sites}
    masks=pq.read_table(campaign/"site_masks.parquet",filters=[("complex_id","in",cids)]).to_pylist();mask_info={tuple(r[k] for k in KEY):r for r in masks}
    val_sites=pq.read_table(audit/"validation_titratable_sites.parquet").to_pylist();pairs=pq.read_table(audit/"teacher_pairs.parquet").to_pylist()
    bysites=defaultdict(list);bypairs=defaultdict(list)
    for r in val_sites:bysites[r["complex_id"]].append(r)
    for r in pairs:bypairs[r["complex_id"]].append(r)
    universe=read(root/"universe/combined-split-v1/index.json");candidate={r["complex_id"]:r for r in universe["candidates"]}
    rows=[];complexes=[]
    for number,row in enumerate(records,1):
        cid=row["complex_id"];original=read(recordroot/"records"/f"{cid}.json");achains,bchains=_partner_chains(original)
        pchain={c:0 for c in achains}|{c:1 for c in bchains};keys=[tuple(x[1:]) for x in row["keys"]];capacity=manifest["capacities"][bucket(row)]
        path=graphroot/"data"/cid/"graph.npz";assert digest(path)==row["sha256"]
        with np.load(path,allow_pickle=False) as f:ab={k:f[k] for k in f.files if k!="labels"}
        ab=mask_features(pad_one(ab,capacity),manifest["config"]);node_keys,_=structure_nodes(root,manifest,row)
        node_chains=np.array([x[0] for x in node_keys]);node_partners=np.array([pchain[x] for x in node_chains],np.int8)
        cat=edge_categories(np.pad(node_chains,(0,capacity[0]-len(node_chains)),constant_values=""),
            np.pad(node_partners,(0,capacity[0]-len(node_partners)),constant_values=-2),ab["neighbors"],ab["edge_mask"])
        variants=[ab,ablate_edges(ab,cat==2)];names=["full","no_partner"]
        hchain=lchain=None;cand=candidate.get(cid)
        if row["role"]=="antibody_antigen" and cand:
            hchain=cand.get("sabdab",{}).get("Hchain");lchain=cand.get("sabdab",{}).get("Lchain")
        if hchain and lchain:
            variants.append(ablate_edges(ab,hl_cut(np.pad(node_chains,(0,capacity[0]-len(node_chains)),constant_values=""),ab["neighbors"],ab["edge_mask"],hchain,lchain)));names.append("no_hl")
        abvalues=np.asarray(predict(params,{k:jnp.asarray(np.stack([g[k] for g in variants])) for k in ab}))[:,:row["q"]]
        values={name:{key:float(v) for key,v in zip(keys,abvalues[i])} for i,name in enumerate(names)}
        state_values={};state_nohl={}
        for state in ("A","B"):
            raw,kept,chains=state_graph(original["structures"][state],keys)
            full=pad_one(raw,capacity);state_variants=[full]
            if hchain and lchain and hchain in chains and lchain in chains:
                padded_chains=np.pad(chains,(0,capacity[0]-len(chains)),constant_values="")
                state_variants.append(ablate_edges(full,hl_cut(padded_chains,full["neighbors"],full["edge_mask"],hchain,lchain)))
            pv=np.asarray(predict(params,{k:jnp.asarray(np.stack([g[k] for g in state_variants])) for k in full}))
            state_values[state]={key:float(v) for key,v in zip(kept,pv[0,:len(kept)])}
            state_nohl[state]={key:float(v) for key,v in zip(kept,pv[-1,:len(kept)])}
        clusters,membership=build_clusters(bysites[cid],bypairs[cid],pchain)
        matched=0
        for key in keys:
            fullkey=(cid,*key);info=site_info.get(fullkey);mask=mask_info.get(fullkey)
            if info is None or mask is None or not mask["evaluation_eligible"]:continue
            free=info["partner"]
            if key not in values["full"] or key not in state_values[free]:continue
            teacher=[];pkai=[]
            for state in ("AB",free):
                for method,target in (("pypka",teacher),("pkai",pkai)):
                    r=pred.get((fullkey,state,method));target.append(None if r is None or r["status"]!="ok" or r["pka"] is None else float(r["pka"]))
            if any(x is None for x in teacher):continue
            model_ab=values["full"][key];model_free=state_values[free][key];teacher_shift=teacher[0]-teacher[1]
            outrow=dict(complex_id=cid,component_id=row["component_id"],role=row["role"],chain=key[0],resnum=key[1],icode=key[2],group=key[3],free_state=free,
                teacher_ab=teacher[0],teacher_free=teacher[1],teacher_shift=teacher_shift,model_ab=model_ab,model_free=model_free,model_shift=model_ab-model_free,
                shift_error=(model_ab-model_free)-teacher_shift,ab_error=model_ab-teacher[0],free_error=model_free-teacher[1],
                causal_partner_signed=model_ab-values["no_partner"][key],causal_partner_abs=abs(model_ab-values["no_partner"][key]),
                causal_use="strong" if abs(model_ab-values["no_partner"][key])>=PRIMARY else "weak",
                geometric_cluster=membership["geometric"].get(key),coupling_cluster=membership["coupling"].get(key),interface=bool(mask["interface"]),
                pkai_shift=None if any(x is None for x in pkai) else pkai[0]-pkai[1],hchain=hchain,lchain=lchain)
            if "no_hl" in values and key in state_nohl[free]:
                ab_hl=model_ab-values["no_hl"][key];free_hl=model_free-state_nohl[free][key]
                outrow.update(causal_hl_ab_signed=ab_hl,causal_hl_free_signed=free_hl,causal_hl_shift_signed=ab_hl-free_hl)
            else:outrow.update(causal_hl_ab_signed=None,causal_hl_free_signed=None,causal_hl_shift_signed=None)
            rows.append(outrow);matched+=1
        complexes.append(dict(complex_id=cid,component_id=row["component_id"],role=row["role"],matched_sites=matched,hchain=hchain,lchain=lchain,
            geometric_clusters=sum(x["cluster_definition"]=="geometric" for x in clusters),coupling_clusters=sum(x["cluster_definition"]=="coupling" for x in clusters)))
        atomic_json(out/"progress.json",dict(completed=number,total=len(records),last=cid))
    # Exact AB parity against the previous audit on its available sites.
    previous=pq.read_table(approach/"site_trajectory.parquet",filters=[("separation_A","=",0)]).to_pylist()
    old={(r["complex_id"],r["chain"],r["resnum"],r["icode"],r["group"]):r["predicted_shift"] for r in previous}
    from pkanet.model import PKPDB_PK_MOD
    from jaxpropka.parameters import GROUPS
    differences=[]
    for r in rows:
        k=tuple(r[x] for x in KEY)
        if k in old:differences.append(abs(r["model_ab"]-(float(PKPDB_PK_MOD[GROUPS.index(r["group"])])+old[k])))
    parity=max(differences,default=0.);assert parity<=1e-5,parity
    write_parquet(out/"site_results.parquet",rows);write_parquet(out/"complexes.parquet",complexes)
    atomic_json(out/"verification.json",dict(complete=True,complexes=len(records),paired_sites=len(rows),ab_parity_sites=len(differences),ab_parity_max_abs_pka=parity,
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint/"state.npz"),checkpoint_metadata=metadata,
        source_manifest_sha256=digest(source/"manifest.json"),prediction_source_sha256=digest(campaign/"predictions.parquet"),
        outputs={p.name:digest(p) for p in (out/"site_results.parquet",out/"complexes.parquet")}))


def metrics(rows,field):
    y=np.array([r["teacher_shift"] for r in rows]);p=np.array([r[field] for r in rows]);e=p-y
    slope=float(np.cov(y,p,ddof=0)[0,1]/np.var(y)) if len(y)>1 and np.var(y)>0 else None
    return dict(n=len(rows),mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e*e))),
        sign_accuracy=float(np.mean(np.sign(y[abs(y)>=.5])==np.sign(p[abs(y)>=.5]))) if np.any(abs(y)>=.5) else None,
        slope=slope,prediction_std=float(p.std()),teacher_std=float(y.std()),error_cancellation=float(np.corrcoef([r["ab_error"] for r in rows],[r["free_error"] for r in rows])[0,1]) if field=="model_shift" and len(rows)>1 else None)


def aggregate(rows,field,seed=20261007):
    bycid=defaultdict(list);component={}
    for r in rows:bycid[r["complex_id"]].append(r);component[r["complex_id"]]=r["component_id"]
    bygroup=defaultdict(list)
    for cid,rr in bycid.items():bygroup[component[cid]].append(metrics(rr,field))
    gm={g:{m:float(np.mean([x[m] for x in v if x[m] is not None and np.isfinite(x[m])])) for m in ("mae","rmse","sign_accuracy","slope","prediction_std","teacher_std","error_cancellation") if any(x[m] is not None and np.isfinite(x[m]) for x in v)} for g,v in bygroup.items()}
    result=metrics(rows,field)|dict(complexes=len(bycid),components=len(gm));rng=np.random.default_rng(seed);names=sorted(gm)
    for metric in ("mae","rmse"):
        vals={g:gm[g][metric] for g in names};result[metric]=float(np.mean(list(vals.values())))
        if len(names)>=5:
            boot=[np.mean([vals[x] for x in rng.choice(names,len(names),replace=True)]) for _ in range(2000)];result[metric+"_ci95"]=np.quantile(boot,[.025,.975]).tolist()
        else:result[metric+"_ci95"]=None
    return result


def report(out):
    require_compute(threads=4,allow_comp1400=True)
    import pyarrow.parquet as pq
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(out);rows=pq.read_table(out/"site_results.parquet").to_pylist();clustered=[r for r in rows if r["geometric_cluster"] or r["coupling_cluster"]]
    summary=[]
    strata=(("either",lambda r:r["geometric_cluster"] or r["coupling_cluster"]),("geometric",lambda r:r["geometric_cluster"]),("coupling",lambda r:r["coupling_cluster"]))
    for name,select in strata:
        base=[r for r in clustered if select(r)]
        for use in ("all","weak","strong"):
            rr=[r for r in base if use=="all" or r["causal_use"]==use]
            if rr:
                for model,field in (("gqt","model_shift"),("zero","zero_shift"),("pkai","pkai_shift")):
                    selected=[]
                    for r in rr:
                        if model=="zero":r=dict(r,zero_shift=0.)
                        if r.get(field) is not None:selected.append(r)
                    if selected:summary.append(dict(cluster_definition=name,causal_use=use,model=model,**aggregate(selected,field)))
    write_parquet(out/"summary.parquet",summary)
    plots=out/"plots";plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True)
    for ax,use in zip(axes,("weak","strong")):
        rr=[r for r in clustered if r["causal_use"]==use];ax.hexbin([r["teacher_shift"] for r in rr],[r["model_shift"] for r in rr],gridsize=35,mincnt=1,bins="log");ax.axline((0,0),slope=1,color="crimson",ls="--");ax.set(title=f"{use.capitalize()} partner use (n={len(rr)})",xlabel="PypKa binding shift",ylabel="GQT binding shift")
    fig.tight_layout();fig.savefig(plots/"binding_shift_strong_weak.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));labels=[];x=[];y=[];lo=[];hi=[]
    for definition in ("either","geometric","coupling"):
        for use in ("weak","strong"):
            r=next(z for z in summary if z["cluster_definition"]==definition and z["causal_use"]==use and z["model"]=="gqt");labels.append(f"{definition}\n{use}");x.append(len(x));y.append(r["mae"]);ci=r["mae_ci95"] or [r["mae"],r["mae"]];lo.append(r["mae"]-ci[0]);hi.append(ci[1]-r["mae"])
    ax.bar(x,y,color=["#6baed6","#2171b5"]*3);ax.errorbar(x,y,yerr=[lo,hi],fmt="none",color="black",capsize=3);ax.set_xticks(x,labels);ax.set(ylabel="Binding-shift MAE (pKa)",title="Accuracy by causal partner use");fig.tight_layout();fig.savefig(plots/"mae_by_partner_use.png",dpi=180);plt.close(fig)
    bins=(0,.01,.05,.1,.25,float("inf"));curve=[]
    for a,b in zip(bins[:-1],bins[1:]):
        rr=[r for r in clustered if a<=r["causal_partner_abs"]<b];curve.append((a,b,len(rr),float(np.mean([abs(r["shift_error"]) for r in rr])) if rr else np.nan))
    fig,ax=plt.subplots(figsize=(6,4));ax.plot(range(len(curve)),[r[3] for r in curve],marker="o");ax.set_xticks(range(len(curve)),[f"{a:g}–{'∞' if not np.isfinite(b) else f'{b:g}'}" for a,b,_,_ in curve]);ax.set(xlabel="Bound edge-removal effect (pKa)",ylabel="Mean absolute binding-shift error",title="Error versus causal partner use");fig.tight_layout();fig.savefig(plots/"error_vs_causal_use.png",dpi=180);plt.close(fig)
    antibody=[r for r in rows if r["role"]=="antibody_antigen" and r["causal_hl_shift_signed"] is not None]
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter([r["causal_partner_signed"] for r in antibody],[r["causal_hl_shift_signed"] for r in antibody],s=9,alpha=.35);ax.axhline(0,color="grey",lw=1);ax.axvline(0,color="grey",lw=1);ax.set(xlabel="Antibody–antigen causal contribution (pKa)",ylabel="Differential heavy–light contribution (pKa)",title=f"Antibody contexts (n={len(antibody)})");fig.tight_layout();fig.savefig(plots/"antibody_antigen_vs_heavy_light.png",dpi=180);plt.close(fig)
    key={f"{r['cluster_definition']}:{r['causal_use']}:{r['model']}":r for r in summary if r["causal_use"] in ("weak","strong")}
    gweak=key["either:weak:gqt"];gstrong=key["either:strong:gqt"]
    zweak=key["either:weak:zero"];zstrong=key["either:strong:zero"]
    pweak=key.get("either:weak:pkai");pstrong=key.get("either:strong:pkai")
    strong_rows=[r for r in clustered if r["causal_use"]=="strong"]
    aligned=[r for r in strong_rows if abs(r["teacher_shift"])>=.1]
    causal_alignment=dict(n=len(aligned),correlation=float(np.corrcoef([r["teacher_shift"] for r in aligned],[r["causal_partner_signed"] for r in aligned])[0,1]) if len(aligned)>1 else None,
        sign_accuracy=float(np.mean([np.sign(r["teacher_shift"])==np.sign(r["causal_partner_signed"]) for r in aligned])) if aligned else None)
    antibody_summary=dict(n=len(antibody),antigen_mean_abs=float(np.mean([abs(r["causal_partner_signed"]) for r in antibody])) if antibody else None,
        antigen_fraction_005=float(np.mean([abs(r["causal_partner_signed"])>=.05 for r in antibody])) if antibody else None,
        heavy_light_mean_abs=float(np.mean([abs(r["causal_hl_shift_signed"]) for r in antibody])) if antibody else None,
        heavy_light_fraction_005=float(np.mean([abs(r["causal_hl_shift_signed"])>=.05 for r in antibody])) if antibody else None)
    result=dict(paired_sites=len(rows),clustered_sites=len(clustered),strong_sites=sum(r["causal_use"]=="strong" for r in clustered),weak_sites=sum(r["causal_use"]=="weak" for r in clustered),
        antibody_hl_sites=len(antibody),gqt_weak=gweak,gqt_strong=gstrong,zero_weak=zweak,zero_strong=zstrong,pkai_weak=pweak,pkai_strong=pstrong,
        strong_skill_vs_zero_mae=1-gstrong["mae"]/zstrong["mae"],weak_skill_vs_zero_mae=1-gweak["mae"]/zweak["mae"],
        causal_alignment=causal_alignment,antibody=antibody_summary,mae_difference_strong_minus_weak=gstrong["mae"]-gweak["mae"])
    atomic_json(out/"key_results.json",result)
    text=f"""# Partner-use accuracy audit

This frozen-checkpoint validation analysis contains {len(rows)} matched PypKa/GQT paired sites; {len(clustered)} belong to a geometric or native-coupling cross-partner cluster. No test data or retraining is used.

| Causal-use group | Sites | GQT MAE | Zero-shift MAE | Frozen-pKAI MAE | GQT slope |
|---|---:|---:|---:|---:|---:|
| Weak (<0.05 pKa) | {result['weak_sites']} | {gweak['mae']:.4f} | {zweak['mae']:.4f} | {pweak['mae']:.4f} | {gweak['slope']:.3f} |
| Strong (≥0.05 pKa) | {result['strong_sites']} | {gstrong['mae']:.4f} | {zstrong['mae']:.4f} | {pstrong['mae']:.4f} | {gstrong['slope']:.3f} |

Strong-use sites have larger teacher shifts, so strong-versus-weak raw MAE is not a measure of benefit. Within the strong stratum, GQT's MAE skill relative to predicting zero shift is {result['strong_skill_vs_zero_mae']:+.1%}. For strong sites with |PypKa shift| ≥0.1, the signed causal contribution has correlation {causal_alignment['correlation']:.3f} with the teacher shift and sign agreement {causal_alignment['sign_accuracy']:.1%}.

For {antibody_summary['n']} antibody-site observations with an H/L pair, the mean absolute antibody–antigen causal contribution is {antibody_summary['antigen_mean_abs']:.4f} pKa ({antibody_summary['antigen_fraction_005']:.1%} ≥0.05), while the differential heavy–light contribution is {antibody_summary['heavy_light_mean_abs']:.4f} pKa ({antibody_summary['heavy_light_fraction_005']:.1%} ≥0.05).

![Shift calibration](plots/binding_shift_strong_weak.png)

![MAE](plots/mae_by_partner_use.png)

![Continuous causal use](plots/error_vs_causal_use.png)

![Antibody contexts](plots/antibody_antigen_vs_heavy_light.png)

PypKa is the teacher, not experimental truth. Geometric and ≥1 kBT coupling memberships overlap. Antibody–antigen and heavy–light interventions are separate nonlinear counterfactuals and are not summed.
"""
    (out/"report.md").write_text(text);v=read(out/"verification.json");v.update(report_complete=True,summary_sha256=digest(out/"summary.parquet"),key_results_sha256=digest(out/"key_results.json"));atomic_json(out/"verification.json",v)


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest="command",required=True)
    for name in ("run","report"):
        q=s.add_parser(name);q.add_argument("root",type=Path) if name=="run" else None;q.add_argument("out",type=Path)
    a=p.parse_args();run(a.root,a.out) if a.command=="run" else report(a.out)


if __name__=="__main__":main()
