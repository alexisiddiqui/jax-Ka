"""Attention and causal partner-use audit along a rigid separation trajectory."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .runtime import atomic_json, digest, require_compute

DISTANCES = tuple(range(0, 41, 2))
SHUFFLES = 20
EFFECT_THRESHOLDS = (0.01, 0.05, 0.10)


def read(path):
    return json.loads(Path(path).read_text())


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(".pending-" + path.name)
    pq.write_table(pa.Table.from_pylist(rows), pending)
    os.replace(pending, path)


def components(items, edges):
    """Connected component labels for hashable items; isolated items included."""
    parent = {x: x for x in items}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        if a not in parent or b not in parent: continue
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    groups = defaultdict(list)
    for x in items: groups[find(x)].append(x)
    return [sorted(v, key=str) for v in groups.values()]


def edge_categories(chains, partners, neighbors, mask):
    """0=same chain, 1=other chain same partner, 2=opposite partner, -1=padding."""
    chains = np.asarray(chains); partners = np.asarray(partners)
    out = np.full(neighbors.shape, -1, np.int8)
    receiver = np.arange(len(chains))[:, None]
    same_chain = chains[receiver] == chains[neighbors]
    same_partner = partners[receiver] == partners[neighbors]
    out[mask & same_chain] = 0
    out[mask & ~same_chain & same_partner] = 1
    out[mask & ~same_partner] = 2
    return out


def choose_separation_direction(backbone, partners):
    """Choose a contact-interface normal that monotonically separates C-alpha sets."""
    ca = np.asarray(backbone)[:, 1]
    p = np.asarray(partners)
    a, b = ca[p == 0], ca[p == 1]
    if not len(a) or not len(b): raise ValueError("both biological partners are required")
    delta=b[None]-a[:,None];distance=np.linalg.norm(delta,axis=-1)
    candidates=[b.mean(0)-a.mean(0)]
    contact=distance<=10
    if contact.any(): candidates.insert(0,delta[contact].mean(0))
    i,j=np.unravel_index(np.argmin(distance),distance.shape);candidates.append(b[j]-a[i])
    def minimum(axis, d):
        shifted = b + d * axis
        return float(np.sqrt(((a[:, None] - shifted[None]) ** 2).sum(-1)).min())
    trials=[]
    for raw in candidates:
        norm=np.linalg.norm(raw)
        if norm<1e-8:continue
        for axis in (raw/norm,-raw/norm):
            values=np.array([minimum(axis,d) for d in DISTANCES]);trials.append((np.min(np.diff(values)),values[-1]-values[0],axis,values))
    for mindiff,_,axis,values in sorted(trials,key=lambda x:(x[0],x[1]),reverse=True):
        if mindiff>=-1e-5:return axis,values
    raise ValueError("no contact, centroid or closest-pair axis separates monotonically")


def perturb_identities(nodes, partners, query_partners, seed, shuffles=SHUFFLES):
    """Return query-specific partner masks and deterministic within-partner shuffles."""
    nodes = np.asarray(nodes); partners = np.asarray(partners); qp = np.asarray(query_partners)
    variants=[]; names=[]
    for side in (0, 1):
        x=nodes.copy(); x[partners != side, :20]=0
        variants.append(x); names.append(f"mask_opposite_for_{side}")
    rng=np.random.default_rng(seed)
    for rep in range(shuffles):
        perms={}
        for side in (0,1):
            idx=np.flatnonzero(partners==side); perms[side]=rng.permutation(idx)
        for side in (0,1):
            x=nodes.copy(); other=1-side; idx=np.flatnonzero(partners==other)
            x[idx,:20]=nodes[perms[other],:20]
            variants.append(x); names.append(f"shuffle_{rep:02d}_for_{side}")
    return np.stack(variants), names


def same_partner_chain_masks(nodes, chains, partners, query_chains):
    """One graph per query chain with other chains of that biological partner masked."""
    nodes=np.asarray(nodes);chains=np.asarray(chains);partners=np.asarray(partners)
    variants=[];names=[]
    for chain in sorted(set(map(str,query_chains))):
        side=int(partners[np.flatnonzero(chains==chain)[0]])
        x=nodes.copy();x[(partners==side)&(chains!=chain),:20]=0
        variants.append(x);names.append(f"mask_same_partner_other_chain_for_{chain}")
    return np.stack(variants),names


def _site_key(row, prefix=""):
    if prefix: return (row[f"chain_{prefix}"], int(row[f"resnum_{prefix}"]), row[f"icode_{prefix}"], row[f"group_{prefix}"])
    return (row["chain"], int(row["resnum"]), row["icode"], row["group"])


def build_clusters(site_rows, pair_rows, partner_by_chain):
    """Build cross-partner geometric and >=1 kBT native-coupling components."""
    site = {_site_key(r): r for r in site_rows}; keys=list(site)
    geo=[]
    from .gqt_alignment_audit import site_distance
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            d=site_distance(np.asarray(site[a]["coords"]),np.asarray(site[b]["coords"]))
            if d is not None and d <= 10: geo.append((a,b))
    coupling=[]
    for r in pair_rows:
        if float(r["strength_kbt"]) < 1: continue
        def parse(x):
            c,n,ic,g=x.split("|"); return c,int(n),ic,g
        a,b=parse(r["site_i"]),parse(r["site_j"])
        if a in site and b in site: coupling.append((a,b))
    rows=[]; membership={"geometric":{},"coupling":{}}
    for kind,edges in (("geometric",geo),("coupling",coupling)):
        for number,group in enumerate(components(keys,edges)):
            ps={partner_by_chain[x[0]] for x in group}
            internal=[e for e in edges if e[0] in group and e[1] in group]
            if len(group)<2 or len(ps)<2 or not any(partner_by_chain[a[0]] != partner_by_chain[b[0]] for a,b in internal): continue
            cid=f"{kind[:3]}-{number:04d}"
            for x in group: membership[kind][x]=cid
            rows.append(dict(cluster_definition=kind,cluster_id=cid,n_sites=len(group),
                             n_edges=len(internal),n_chains=len({x[0] for x in group}),
                             sites=["|".join(map(str,x)) for x in group]))
    return rows,membership


def _checkpoint(source, manifest):
    import jax
    from pkanet.model import initialize
    from pkatrain.graph_pkmod_compare import ExplicitShiftEngine
    from pkatrain.trainer import load_checkpoint
    cfg=manifest["config"]; params=initialize(jax.random.PRNGKey(cfg["seed"]),**cfg["architecture"])
    engine=ExplicitShiftEngine(cfg["shift_bin_weights"],cfg["learning_rate"])
    state=engine.optimizer.init(params)
    run=source/f"seed-{cfg['seed']}"; latest=read(run/"checkpoints/latest.json")["checkpoint"]
    params,_,meta=load_checkpoint(run/"checkpoints"/latest,(params,state))
    if int(meta["epoch"]) != 20: raise ValueError("analysis is registered to fixed epoch 20")
    return params,run/"checkpoints"/latest,meta


def _partner_chains(record):
    from .prep import read_cif
    answer=[]
    for state in ("A","B"):
        path=Path(record["structures"][state]); assert digest(path)==record["structure_sha256"][state]
        arr=read_cif(path); answer.append(sorted(set(map(str,arr.chain_id))))
    if set(answer[0]) & set(answer[1]): raise ValueError("A/B chain sets overlap")
    return answer


def run(root, out, limit=None):
    require_compute(threads=8,gpu_benchmark=True,allow_comp1400=True)
    os.environ.setdefault("JAX_ENABLE_X64","false")
    import jax
    import jax.numpy as jnp
    import pyarrow.parquet as pq
    from pkanet.graph import geometry
    from pkanet.model import predict_pkpdb_with_trace, predict_shift
    from pkatrain.graph_data import bucket,mask_features,pad
    from .gqt_alignment_audit import structure_nodes

    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=root/"pretraining/gqt-backbone-5k-pkmod-v1/unweighted"
    audit=root/"audits/gqt-data-alignment-v1"
    manifest=read(source/"manifest.json"); graphroot=Path(manifest["validation_source"])
    graphparent=read(graphroot/"manifest.json"); recordroot=Path(graphparent["source"])
    params,checkpoint,metadata=_checkpoint(source,manifest)
    site_all=pq.read_table(audit/"validation_titratable_sites.parquet").to_pylist()
    pair_all=pq.read_table(audit/"teacher_pairs.parquet").to_pylist()
    bysites=defaultdict(list);bypairs=defaultdict(list)
    for r in site_all:bysites[r["complex_id"]].append(r)
    for r in pair_all:bypairs[r["complex_id"]].append(r)
    records=[r for r in manifest["records"] if r["split"]=="val"]
    if limit: records=records[:limit]
    trace_fn=jax.jit(predict_pkpdb_with_trace)
    shift_batch=jax.jit(jax.vmap(predict_shift,in_axes=(None,0)))
    trajectory=[];attention=[];cluster_rows=[];complex_rows=[]
    for number,row in enumerate(records,1):
        cid=row["complex_id"]
        original=read(recordroot/"records"/f"{cid}.json")
        achains,bchains=_partner_chains(original); pchain={c:0 for c in achains}|{c:1 for c in bchains}
        keys,atoms=structure_nodes(root,manifest,row)
        chains=np.array([k[0] for k in keys]); partners=np.array([pchain[c] for c in chains],np.int8)
        backbone=np.stack([[a[n] for n in ("N","CA","C","O")] for a in atoms]).astype(np.float32)
        try: direction,minimum=choose_separation_direction(backbone,partners)
        except ValueError as error:
            complex_rows.append(dict(complex_id=cid,component_id=row["component_id"],role=row["role"],
                chains_A=achains,chains_B=bchains,n_residues=row["n"],n_queries=row["q"],excluded=True,exclusion_reason=str(error),
                geometric_clusters=0,coupling_clusters=0))
            atomic_json(out/"progress.json",dict(completed=number,total=len(records),last=cid,excluded=cid));continue
        path=graphroot/"data"/cid/"graph.npz";assert digest(path)==row["sha256"]
        with np.load(path,allow_pickle=False) as f:
            static={k:f[k] for k in f.files if k not in ("labels","neighbors","edge","edge_mask","switch")}; labels=f["labels"]
        clusters,membership=build_clusters(bysites[cid],bypairs[cid],pchain)
        for c in clusters:c.update(complex_id=cid,component_id=row["component_id"],role=row["role"])
        cluster_rows.extend(clusters)
        query_keys=[tuple(k[1:]) for k in row["keys"]]; qpartner=partners[static["query_residue"]]
        seed=int.from_bytes(cid.encode()[:8],"little",signed=False)
        identity_variants,variant_names=perturb_identities(static["nodes"],partners,qpartner,seed)
        chain_variants,chain_variant_names=same_partner_chain_masks(
            static["nodes"],chains,partners,[x[0] for x in query_keys])
        complex_rows.append(dict(complex_id=cid,component_id=row["component_id"],role=row["role"],
            chains_A=achains,chains_B=bchains,n_residues=row["n"],n_queries=row["q"],
            excluded=False,exclusion_reason=None,
            geometric_clusters=sum(c["cluster_definition"]=="geometric" for c in clusters),
            coupling_clusters=sum(c["cluster_definition"]=="coupling" for c in clusters)))
        capacity=manifest["capacities"][bucket(row)]
        for frame,separation in enumerate(DISTANCES):
            moved=backbone.copy();moved[partners==1]+=np.float32(separation)*direction.astype(np.float32)
            dynamic,_=geometry(moved,chains,radius=float(manifest["config"]["radius_A"]))
            raw={**static,**dynamic}; padded,_,eligible=pad(raw,labels,capacity);padded=mask_features(padded,manifest["config"])
            cat=edge_categories(np.pad(chains,(0,capacity[0]-len(chains)),constant_values=""),
                np.pad(partners,(0,capacity[0]-len(partners)),constant_values=-2),padded["neighbors"],padded["edge_mask"])
            full=trace_fn(params,{k:jnp.asarray(v) for k,v in padded.items()})
            base=np.asarray(full["predicted_shift"])[:row["q"]]
            variants=[];vnames=[]
            for category,name in ((2,"remove_opposite_edges"),(1,"remove_same_partner_other_chain_edges")):
                g={k:np.array(v,copy=True) for k,v in padded.items()}; cut=cat==category
                g["edge_mask"][cut]=False;g["switch"][cut]=0
                variants.append(g);vnames.append(name)
            for ni,name in enumerate(variant_names):
                g={k:np.array(v,copy=True) for k,v in padded.items()};g["nodes"][:row["n"],:20]=identity_variants[ni,:,:20]
                variants.append(g);vnames.append(name)
            for ni,name in enumerate(chain_variant_names):
                g={k:np.array(v,copy=True) for k,v in padded.items()};g["nodes"][:row["n"],:20]=chain_variants[ni,:,:20]
                variants.append(g);vnames.append(name)
            stacked={k:jnp.asarray(np.stack([g[k] for g in variants])) for k in padded}
            pred=np.asarray(shift_batch(params,stacked))[:,:row["q"]]
            values={name:pred[i] for i,name in enumerate(vnames)}
            for qi,key in enumerate(query_keys):
                side=int(qpartner[qi]);sh=np.array([values[f"shuffle_{r:02d}_for_{side}"][qi]-base[qi] for r in range(SHUFFLES)])
                rowout=dict(complex_id=cid,component_id=row["component_id"],role=row["role"],frame=frame,
                    separation_A=separation,min_partner_ca_A=float(minimum[frame]),chain=key[0],resnum=key[1],icode=key[2],group=key[3],
                    query_index=qi,query_partner=side,predicted_shift=float(base[qi]),
                    delta_remove_opposite_edges=float(values["remove_opposite_edges"][qi]-base[qi]),
                    delta_mask_opposite_identity=float(values[f"mask_opposite_for_{side}"][qi]-base[qi]),
                    shuffle_opposite_mean=float(sh.mean()),shuffle_opposite_mean_abs=float(np.mean(abs(sh))),shuffle_opposite_sd=float(sh.std()),
                    delta_remove_same_partner_other_chain_edges=float(values["remove_same_partner_other_chain_edges"][qi]-base[qi]),
                    delta_mask_same_partner_other_chain_identity=float(values[f"mask_same_partner_other_chain_for_{key[0]}"][qi]-base[qi]),
                    geometric_cluster=membership["geometric"].get(key),coupling_cluster=membership["coupling"].get(key))
                trajectory.append(rowout)
            layer_traces=list(full["encoder"])+[full["query"]]
            for li,t in enumerate(layer_traces):
                weights=np.asarray(t["weights"]); rows=static["query_residue"] if li<2 else np.arange(row["q"])
                if li<2: weights=weights[static["query_residue"]]
                localcat=cat[static["query_residue"]]
                localswitch=np.asarray(padded["switch"])[static["query_residue"]]
                for qi,key in enumerate(query_keys):
                    if not (membership["geometric"].get(key) or membership["coupling"].get(key)): continue
                    for head in range(weights.shape[-1]):
                        denom=float(localswitch[qi].sum())
                        for category,name in enumerate(("same_chain","same_partner_other_chain","opposite_partner")):
                            select=localcat[qi]==category;mass=float(weights[qi,select,head].sum())
                            opportunity=float(localswitch[qi,select].sum()/denom) if denom else 0.
                            attention.append(dict(complex_id=cid,component_id=row["component_id"],role=row["role"],frame=frame,
                                separation_A=separation,chain=key[0],resnum=key[1],icode=key[2],group=key[3],query_index=qi,
                                layer=(f"encoder_{li}" if li<2 else "query"),head=head,edge_class=name,mass=mass,
                                opportunity=opportunity,enrichment=(mass/opportunity if opportunity>0 else None),
                                geometric_cluster=membership["geometric"].get(key),coupling_cluster=membership["coupling"].get(key)))
        atomic_json(out/"progress.json",dict(completed=number,total=len(records),last=cid))
    write_parquet(out/"site_trajectory.parquet",trajectory);write_parquet(out/"attention_summary.parquet",attention)
    write_parquet(out/"clusters.parquet",cluster_rows);write_parquet(out/"complexes.parquet",complex_rows)
    provenance=dict(complete=True,complexes=len(records),frames=list(DISTANCES),shuffles=SHUFFLES,
        source=str(source),source_manifest_sha256=digest(source/"manifest.json"),checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint/"state.npz"),checkpoint_metadata=metadata,
        audit=str(audit),audit_hashes={p.name:digest(p) for p in (audit/"validation_titratable_sites.parquet",audit/"teacher_pairs.parquet")},
        outputs={p.name:digest(p) for p in (out/"site_trajectory.parquet",out/"attention_summary.parquet",out/"clusters.parquet",out/"complexes.parquet")})
    atomic_json(out/"verification.json",provenance)


def report(out):
    require_compute(threads=4,allow_comp1400=True)
    import pyarrow.parquet as pq
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(out);rows=pq.read_table(out/"site_trajectory.parquet").to_pylist();att=pq.read_table(out/"attention_summary.parquet").to_pylist()
    complexes=pq.read_table(out/"complexes.parquet").to_pylist();clusters=pq.read_table(out/"clusters.parquet").to_pylist()
    clustered=[r for r in rows if r["geometric_cluster"] or r["coupling_cluster"]]
    summary=[];rng=np.random.default_rng(20261007)
    metrics=("delta_remove_opposite_edges","delta_mask_opposite_identity","shuffle_opposite_mean_abs")
    def component_estimate(rr,metric):
        bycomplex=defaultdict(list);component={}
        for r in rr:bycomplex[r["complex_id"]].append(abs(r[metric]));component[r["complex_id"]]=r["component_id"]
        bycomponent=defaultdict(list)
        for cid,v in bycomplex.items():bycomponent[component[cid]].append(float(np.mean(v)))
        values={c:float(np.mean(v)) for c,v in bycomponent.items()};keys=sorted(values)
        estimate=float(np.mean(list(values.values()))) if values else float("nan")
        boot=[]
        if keys:
            for _ in range(2000):boot.append(float(np.mean([values[x] for x in rng.choice(keys,len(keys),replace=True)])))
        return estimate,(float(np.quantile(boot,.025)) if boot else None),(float(np.quantile(boot,.975)) if boot else None),len(keys)
    strata=(("either",lambda r:r["geometric_cluster"] or r["coupling_cluster"]),
            ("geometric",lambda r:r["geometric_cluster"] is not None),
            ("coupling",lambda r:r["coupling_cluster"] is not None))
    roles=sorted({r["role"] for r in clustered})
    for definition,select in strata:
      for role in ("all",*roles):
       for d in DISTANCES:
        rr=[r for r in clustered if r["separation_A"]==d and select(r) and (role=="all" or r["role"]==role)]
        for metric in metrics:
            if not rr:continue
            v=np.abs([r[metric] for r in rr]);estimate,lo,hi,ncomponents=component_estimate(rr,metric);summary.append(dict(cluster_definition=definition,role=role,separation_A=d,metric=metric,n=len(v),n_components=ncomponents,mean_abs=estimate,ci_low=lo,ci_high=hi,
                median_abs=float(np.median(v)),fraction_001=float(np.mean(v>=.01)),fraction_005=float(np.mean(v>=.05)),fraction_010=float(np.mean(v>=.1))))
    write_parquet(out/"summary.parquet",summary)
    onset=[]
    sitegroups=defaultdict(list)
    for r in clustered:sitegroups[(r["complex_id"],r["chain"],r["resnum"],r["icode"],r["group"])].append(r)
    for key,rr in sitegroups.items():
        first=rr[0]
        for metric in metrics:
            for threshold in EFFECT_THRESHOLDS:
                passing=[r["separation_A"] for r in rr if abs(r[metric])>=threshold]
                onset.append(dict(complex_id=key[0],chain=key[1],resnum=key[2],icode=key[3],group=key[4],component_id=first["component_id"],role=first["role"],metric=metric,threshold=threshold,
                    onset_separation_A=max(passing) if passing else None,geometric_cluster=first["geometric_cluster"],coupling_cluster=first["coupling_cluster"]))
    write_parquet(out/"onsets.parquet",onset)
    plots=out/"plots";plots.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for metric,label in zip(metrics,("Remove cross-partner edges","Mask partner identity","Shuffle partner identity")):
        s=[r for r in summary if r["metric"]==metric and r["cluster_definition"]=="either" and r["role"]=="all"];x=[r["separation_A"] for r in s];y=[r["mean_abs"] for r in s]
        ax.plot(x,y,marker="o",label=label);ax.fill_between(x,[r["ci_low"] for r in s],[r["ci_high"] for r in s],alpha=.12)
    ax.axhline(.05,color="grey",ls="--",lw=1);ax.set(xlabel="Rigid separation from bound structure (Å)",ylabel="Mean absolute prediction change (pKa)",title="Causal partner use in titratable clusters");ax.legend();fig.tight_layout();fig.savefig(plots/"causal_effect_vs_separation.png",dpi=180);plt.close(fig)
    cross=[r for r in att if r["edge_class"]=="opposite_partner" and (r["geometric_cluster"] or r["coupling_cluster"])]
    fig,ax=plt.subplots(figsize=(7,4.5))
    for layer in ("encoder_0","encoder_1","query"):
        y=[]
        for d in DISTANCES:
            v=[r["mass"] for r in cross if r["layer"]==layer and r["separation_A"]==d];y.append(float(np.mean(v)) if v else np.nan)
        ax.plot(DISTANCES,y,marker="o",label=layer)
    ax.set(xlabel="Rigid separation from bound structure (Å)",ylabel="Mean cross-partner attention mass",title="Cross-partner attention in titratable clusters");ax.legend();fig.tight_layout();fig.savefig(plots/"attention_vs_separation.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for metric,label in zip(metrics,("Remove edges","Mask identity","Shuffle identity")):
        s=[r for r in summary if r["metric"]==metric and r["cluster_definition"]=="either" and r["role"]=="all"]
        ax.plot([r["separation_A"] for r in s],[r["fraction_005"] for r in s],marker="o",label=label)
    ax.set(xlabel="Rigid separation from bound structure (Å)",ylabel="Fraction with |effect| ≥ 0.05 pKa",ylim=(-.02,1.02),title="Sites with practically detectable partner use");ax.legend();fig.tight_layout();fig.savefig(plots/"partner_use_fraction.png",dpi=180);plt.close(fig)
    bound=[r for r in clustered if r["separation_A"]==0]
    ranked=[]
    for definition,field in (("geometric","geometric_cluster"),("coupling","coupling_cluster")):
        groups=defaultdict(list)
        for r in bound:
            if r[field]:groups[(r["complex_id"],r[field])].append(abs(r["delta_remove_opposite_edges"]))
        ranked.extend((float(np.mean(v)),definition,cid,cluster) for (cid,cluster),v in groups.items())
    chosen=sorted(ranked,reverse=True)[:12]
    if chosen:
        matrix=[];labels=[]
        for _,definition,cid,cluster in chosen:
            field=definition+"_cluster";line=[]
            for d in DISTANCES:
                v=[abs(r["delta_remove_opposite_edges"]) for r in clustered if r["complex_id"]==cid and r[field]==cluster and r["separation_A"]==d]
                line.append(float(np.mean(v)) if v else np.nan)
            matrix.append(line);labels.append(f"{cid[:8]} {definition[:3]}:{cluster.split('-')[-1]}")
        fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(matrix,aspect="auto",cmap="magma",extent=(-1,41,len(matrix)-.5,-.5))
        ax.set_yticks(range(len(labels)),labels);ax.set(xlabel="Rigid separation (Å)",title="Most sensitive cross-partner titratable clusters");fig.colorbar(im,ax=ax,label="Mean |edge-removal effect| (pKa)");fig.tight_layout();fig.savefig(plots/"selected_cluster_heatmap.png",dpi=180);plt.close(fig)
    key_rows=[r for r in summary if r["cluster_definition"]=="either" and r["role"]=="all" and r["separation_A"] in (0,10,20,40)]
    isolated=[r for r in rows if r["separation_A"]==40 and r["min_partner_ca_A"]>20]
    isolated_max={metric:max((abs(r[metric]) for r in isolated),default=0.) for metric in metrics}
    if max(isolated_max.values(),default=0.)>1e-4:
        raise AssertionError(f"disconnected-partner intervention invariant failed: {isolated_max}")
    key_results=dict(validation_complexes=len(complexes),analysed_complexes=sum(not r["excluded"] for r in complexes),
        trajectory_exclusions=sum(r["excluded"] for r in complexes),sites_with_cluster_trajectory=len(sitegroups),
        geometric_clusters=sum(r["cluster_definition"]=="geometric" for r in clusters),
        coupling_clusters=sum(r["cluster_definition"]=="coupling" for r in clusters),
        disconnected_sites_checked=len(isolated),disconnected_invariant_max_abs_pka=isolated_max,results=key_rows)
    atomic_json(out/"key_results.json",key_results)
    labels={"delta_remove_opposite_edges":"remove edges","delta_mask_opposite_identity":"mask identity","shuffle_opposite_mean_abs":"shuffle identity"}
    table=["| Separation | Probe | Mean | 95% component CI | Fraction ≥0.05 |","|---:|---|---:|---:|---:|"]
    for r in sorted(key_rows,key=lambda x:(x["separation_A"],x["metric"])):
        table.append(f"| {r['separation_A']} Å | {labels[r['metric']]} | {r['mean_abs']:.4f} | {r['ci_low']:.4f}–{r['ci_high']:.4f} | {r['fraction_005']:.3f} |")
    table_text="\n".join(table)
    verification=read(out/"verification.json")
    report_text=f"""# GQT approach attention and causal partner-use audit

This diagnostic uses the fixed epoch-20 unweighted explicit-pKPDB-shift backbone GQT on all {verification['complexes']} frozen validation complexes. It does not use the test set. {key_results['analysed_complexes']} complexes yielded a monotonic trajectory; {key_results['trajectory_exclusions']} are explicit trajectory-geometry exclusions. The analysed sites span {key_results['geometric_clusters']} geometric and {key_results['coupling_clusters']} native-coupling cross-partner clusters.

The B partner was translated 0–40 Å along a monotonic interface-normal path in 2 Å increments, with centroid and closest-pair axes as registered fallbacks. This is a model sensitivity trajectory, not a physical association pathway. Titratable clusters are defined both by functional-atom proximity (≤10 Å connected components) and native PypKa coupling (≥1 kBT connected components); only components spanning both partners are included in the headline curves.

Causal probes remove cross-partner graph edges, mask opposite-partner residue identities while retaining geometry, and shuffle opposite-partner identities 20 times within each partner. A 0.05 pKa change is the registered primary practical threshold; 0.01 and 0.10 pKa are reported in `summary.parquet`.

{table_text}

![Causal effects](plots/causal_effect_vs_separation.png)

![Attention](plots/attention_vs_separation.png)

![Partner-use fraction](plots/partner_use_fraction.png)

![Selected clusters](plots/selected_cluster_heatmap.png)

Attention is descriptive; the edge-removal and identity interventions are the causal evidence for whether information from another chain affects a prediction. Biological partner membership is kept separate from chain membership, so antibody heavy/light communication is reported as same-partner, other-chain context rather than antigen context.

The disconnected-graph invariant passed on {key_results['disconnected_sites_checked']} site trajectories at 40 Å: the largest intervention effect was {max(isolated_max.values()):.2g} pKa.
"""
    (out/"report.md").write_text(report_text)
    verification["report_complete"]=True;verification["summary_sha256"]=digest(out/"summary.parquet");verification["key_results_sha256"]=digest(out/"key_results.json");atomic_json(out/"verification.json",verification)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run");r.add_argument("root",type=Path);r.add_argument("out",type=Path);r.add_argument("--limit",type=int)
    q=sub.add_parser("report");q.add_argument("out",type=Path)
    a=p.parse_args();run(a.root,a.out,a.limit) if a.command=="run" else report(a.out)


if __name__ == "__main__": main()
