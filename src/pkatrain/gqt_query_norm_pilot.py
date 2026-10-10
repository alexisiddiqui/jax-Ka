"""Versioned matched shared/separate local-query normalization experiment."""
from __future__ import annotations
import csv
import json
import os
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from pkanet.ogqt import initialize_auxiliary
from pkanet.site_query_norm import predict_multi
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.gqt_auxiliary_pilot import AuxiliaryEngine,_bootstrap,_rate,evaluate,read,MIN_DELTA
from pkatrain.gqt_paired_pinder import Loader,_plans,_prefetched
from pkatrain.trainer import load_checkpoint,save_checkpoint

VERSION="ogqt-query-norm-v1"
ARMS=("shared","separate")
SEED=int(os.environ.get("PKATRAIN_NORM_SEED","17"))
if SEED not in (17,29,43):raise ValueError(SEED)


def roots(root):
    base=Path(root)/"training"
    if SEED!=17:
        output=base/VERSION/f"seed-{SEED}"
        return output/"data",output
    return base/"ogqt-auxiliary-pilot-v1",base/VERSION


def prepare_seed(root):
    """Fresh initialization and common one-epoch pKa-only warmup for a repeat seed."""
    if SEED==17:raise ValueError("seed 17 uses the preserved original warmup")
    source=Path(root)/"training/ogqt-auxiliary-pilot-v1"; base,out=roots(root)
    base.mkdir(parents=True,exist_ok=True)
    if (base/"common-warmup").exists():raise FileExistsError(base/"common-warmup")
    for name in ("manifest.json","graphs","mmap-v1","calibration.json"):
        destination=base/name
        if not destination.exists():destination.symlink_to(source/name,target_is_directory=(source/name).is_dir())
    manifest=read(base/"manifest.json"); train=[r for r in manifest["records"] if r["split"]=="train"]
    val=[r for r in manifest["records"] if r["split"]=="val"]
    if {r["cluster_id"] for r in train}&{r["cluster_id"] for r in val}:raise AssertionError("cluster overlap")
    params=initialize_auxiliary(jax.random.PRNGKey(SEED),**manifest["architecture"])
    engine=AuxiliaryEngine(params,0,{"burial":1,"interface":1}); state=engine.optimizer.init(params)
    import hashlib
    initial_hash=hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree.leaves(params))).hexdigest()
    plans=_plans(train,np.random.default_rng(SEED),manifest); loader=Loader(base,manifest)
    started=time.monotonic(); burial_sum=0.; burial_n=0
    atomic_json(out/"warmup-registration.json",{"seed":SEED,"initial_parameters_sha256":initial_hash,
        "code_hashes":hashes(),"manifest_sha256":digest(base/"manifest.json"),"fixed_calibration_sha256":digest(base/"calibration.json"),
        "initialization":"fresh seed-specific parameters; same pKa-only first epoch as seed 17","test_data_included":False})
    for number,(_,batch) in enumerate(_prefetched(loader,plans),1):
        _,mask,burial,_=engine.targets(batch,manifest["normalization"])
        burial_sum+=float(burial[mask].sum()); burial_n+=int(mask.sum())
        params,state,_=engine.update(params,state,batch,manifest["normalization"],1e-3)
        if number%50==0 or number==len(plans):
            progress={"phase":"warmup","seed":SEED,"batch":number,"batches":len(plans),"seconds":time.monotonic()-started}
            atomic_json(out/"warmup-progress.json",progress); print(json.dumps(progress),flush=True)
    loader.close(); mean=burial_sum/burial_n
    metrics,_=evaluate(base,manifest,val,engine,params,mean)
    folder=base/"common-warmup"
    save_checkpoint(folder/"checkpoint",params,state,{"seed":SEED,"epoch":1,"initial_parameters_sha256":initial_hash})
    atomic_json(folder/"verification.json",{"passed":True,"seed":SEED,"epoch":1,"validation":metrics,
        "train_burial_mean":mean,"checkpoint_sha256":digest(folder/"checkpoint/state.npz"),
        "wall_seconds":time.monotonic()-started,"test_data_included":False})


def hashes():
    src=Path(__file__).parents[1]
    return {str(p):digest(p) for p in (Path(__file__),src/"pkanet/site_query_norm.py",src/"pkanet/model.py",
        src/"pkanet/site_model.py",src/"pkanet/triton_attention.py",src/"pkatrain/gqt_auxiliary_pilot.py",
        src/"pkatrain/gqt_paired_pinder.py")}


def setup(root,arm):
    if arm not in ARMS:raise ValueError(arm)
    base,out=roots(root); manifest=read(base/"manifest.json")
    p=initialize_auxiliary(jax.random.PRNGKey(SEED),**manifest["architecture"])
    coeff=read(base/"calibration.json")["coefficients"]
    template=AuxiliaryEngine(p,1,coeff)
    p,_,_=load_checkpoint(base/"common-warmup/checkpoint",(p,template.optimizer.init(p)))
    if arm=="separate":
        p={**p,"query_context_norm":jnp.array(p["query"]["norm1"])}
        engine=AuxiliaryEngine(p,1,coeff,predict_fn=predict_multi)
    else:engine=AuxiliaryEngine(p,1,coeff)
    # Shared-moment history has no unique split into query/context moments.
    # Reset the entire optimizer identically in both arms, preserving all model weights.
    return base,out,manifest,p,engine.optimizer.init(p),engine


def separation(p):
    if "query_context_norm" not in p:return {"scale_rms":0.,"offset_rms":0.}
    difference=np.asarray(p["query_context_norm"]-p["query"]["norm1"])
    return {"scale_rms":float(np.sqrt(np.mean(difference[0]**2))),"offset_rms":float(np.sqrt(np.mean(difference[1]**2)))}


def register(root):
    base,out=roots(root); out.mkdir(parents=True,exist_ok=True)
    if (out/"protocol.json").exists():raise FileExistsError(out/"protocol.json")
    manifest=read(base/"manifest.json"); train=[r for r in manifest["records"] if r["split"]=="train"]
    val=[r for r in manifest["records"] if r["split"]=="val"]
    if {r["cluster_id"] for r in train}&{r["cluster_id"] for r in val}:raise AssertionError("cluster overlap")
    rng=np.random.default_rng(SEED)
    atomic_json(out/"plans.json",{str(e):_plans(train,rng,manifest) for e in range(2,11)})
    atomic_json(out/"protocol.json",{"version":VERSION,"arms":ARMS,"seed":SEED,"epochs_total":10,
        "train_complexes":len(train),"validation_complexes":len(val),"initialization":"same saved epoch-1 warmup; extra context scale/offset copied from local query norm1",
        "optimizer_initialization":"all Adam moments and counter reset in BOTH arms; no ambiguous shared-to-split moment transplant",
        "change":"local query norm1 remains query-only; a separate 2 x width array normalizes residue context for K/V; no changes to encoder/site attention or FF norms",
        "extra_parameters":2*manifest["architecture"]["width"],"objective":"state MSE + paired MSE + standard calibrated auxiliary losses",
        "coefficients":read(base/"calibration.json")["coefficients"],"optimizer":"AdamW 1e-4, global norm clip 1; inherited epoch-2-to-10 cosine schedule 1e-3 to 1e-5",
        "selection":"unweighted state MAE + interface paired MAE; improvement threshold 0.001; common warmup eligible",
        "augmentation":"none; no dropout","test_data_included":False,"code_hashes":hashes(),
        "manifest_sha256":digest(base/"manifest.json"),"checkpoint_sha256":digest(base/"common-warmup/checkpoint/state.npz"),"plans_sha256":digest(out/"plans.json")})


def check(base,out):
    p=read(out/"protocol.json")
    if p["code_hashes"]!=hashes():raise AssertionError("code changed since registration")
    for path,key in ((base/"manifest.json","manifest_sha256"),(base/"common-warmup/checkpoint/state.npz","checkpoint_sha256"),(out/"plans.json","plans_sha256")):
        if digest(path)!=p[key]:raise AssertionError((key,"mismatch"))
    return p


def smoke(root):
    base,out,m,p,s,e=setup(root,"separate"); _,_,_,q,qs,qe=setup(root,"shared")
    check(base,out); loader=Loader(base,m); batch=loader.batch(read(out/"plans.json")["2"][0])
    a=qe.predictions(q,batch[0]); b=e.predictions(p,batch[0])
    max_error=max(float(jnp.max(jnp.abs(a[k]-b[k]))) for k in a)
    for name in a:np.testing.assert_allclose(a[name],b[name],rtol=1e-6,atol=1e-6)
    target,mask,burial,interface=e.targets(batch,m["normalization"]); args=(batch[0],target,mask,burial,interface,jnp.ones(len(target),bool))
    def objective(engine,params):
        parts=engine.losses(params,*args)
        return parts[0]+parts[1]+engine.lambdas["burial"]*parts[2]+engine.lambdas["interface"]*parts[3]
    grad=jax.jit(jax.grad(lambda x:objective(e,x)))(p)
    reference=jax.jit(jax.grad(lambda x:objective(qe,x)))(q)
    # At identical affine parameters, gradients on the two paths must SUM to the tied gradient.
    collapsed={k:v for k,v in grad.items() if k!="query_context_norm"}
    collapsed={**collapsed,"query":{**collapsed["query"],"norm1":grad["query"]["norm1"]+grad["query_context_norm"]}}
    error=np.concatenate([np.asarray(a-b).ravel() for a,b in zip(jax.tree.leaves(collapsed),jax.tree.leaves(reference))])
    vector=np.concatenate([np.asarray(a).ravel() for a in jax.tree.leaves(reference)])
    relative=float(np.linalg.norm(error)/max(np.linalg.norm(vector),1e-12))
    if not np.isfinite(relative) or relative>2e-4:raise AssertionError(("collapsed gradient",relative))
    norms={"query":float(jnp.linalg.norm(grad["query"]["norm1"])),"context":float(jnp.linalg.norm(grad["query_context_norm"]))}
    if min(norms.values())<1e-8:raise AssertionError("norm path has no gradient")
    finite_checks=[]
    for which in ("query","context"):
        gradient=grad["query"]["norm1"] if which=="query" else grad["query_context_norm"]
        direction=gradient/jnp.linalg.norm(gradient)
        def altered(step):
            return ({**p,"query":{**p["query"],"norm1":p["query"]["norm1"]+step*direction}}
                if which=="query" else {**p,"query_context_norm":p["query_context_norm"]+step*direction})
        epsilon=.001
        fd=float((objective(e,altered(epsilon))-objective(e,altered(-epsilon)))/(2*epsilon))
        expected=float(jnp.sum(gradient*direction))
        np.testing.assert_allclose(fd,expected,rtol=.03,atol=5e-4)
        finite_checks.append({"path":which,"analytical":expected,"finite_difference":fd})
    print(json.dumps({"smoke":"predictions, collapsed gradients and finite differences passed"}),flush=True)
    updated,state,loss=e.update(p,s,batch,m["normalization"],1e-3)
    distance=separation(updated)
    if max(distance.values())<1e-8:raise AssertionError("norms did not separate")
    _,_,control_loss=qe.update(q,qs,batch,m["normalization"],1e-3)
    graph=batch[0]; padded={**graph,"nodes":np.where(graph["node_mask"][...,None],graph["nodes"],7.)}
    normal=e.predictions(updated,graph); changed=e.predictions(updated,padded)
    for name in normal:np.testing.assert_allclose(np.asarray(normal[name])*mask[:,None,:],
        np.asarray(changed[name])*mask[:,None,:],rtol=1e-5,atol=1e-6)
    save_checkpoint(out/"smoke-checkpoint",updated,state,{"version":VERSION,"arm":"separate","smoke":True})
    restored,rs,_=load_checkpoint(out/"smoke-checkpoint",(updated,state))
    for a,b in zip(jax.tree.leaves((updated,state)),jax.tree.leaves((restored,rs))):np.testing.assert_array_equal(a,b)
    loader.close(); atomic_json(out/"tests.json",{"passed":True,"max_prediction_difference":max_error,"collapsed_gradient_relative_difference":relative,
        "norm_gradient_norms":norms,"finite_differences":finite_checks,"norm_separation_after_step":distance,"loss":loss,"control_loss":control_loss,
        "extra_parameters":sum(x.size for x in jax.tree.leaves(p))-sum(x.size for x in jax.tree.leaves(q)),
        "padding_invariance":True,"checkpoint_roundtrip":True,"finite_jit_updates_both_arms":True,"code_hashes":hashes()})


def train(root,arm):
    base,out,m,p,state,engine=setup(root,arm); protocol=check(base,out)
    if not read(out/"tests.json")["passed"]:raise AssertionError("smoke incomplete")
    run=out/arm; run.mkdir(exist_ok=False); plans=read(out/"plans.json"); common=read(base/"common-warmup/verification.json")
    val=[r for r in m["records"] if r["split"]=="val"]; history=[]; loader=Loader(base,m); started=time.monotonic()
    best={"epoch":1,"selection":common["validation"]["primary"]["selection"],"norm_separation":separation(p)}
    metadata={"version":VERSION,"arm":arm,"predictor":"pkanet.site_query_norm.predict_multi" if arm=="separate" else "pkanet.ogqt.predict_multi"}
    atomic_json(run/"run.json",{**protocol,**metadata,"parameter_count":sum(x.size for x in jax.tree.leaves(p))})
    save_checkpoint(run/"checkpoints/epoch-001",p,state,{**metadata,"epoch":1})
    for epoch in range(2,11):
        began=time.monotonic(); plan=plans[str(epoch)]; logs=[]
        for number,(_,batch) in enumerate(_prefetched(loader,plan),1):
            p,state,loss=engine.update(p,state,batch,m["normalization"],_rate(epoch,number,len(plan))); logs.append(loss)
            if number%50==0 or number==len(plan):
                progress={"arm":arm,"epoch":epoch,"batch":number,"batches":len(plan),"seconds":time.monotonic()-started}
                atomic_json(run/"progress.json",progress); print(json.dumps(progress),flush=True)
        metrics,_=evaluate(base,m,val,engine,p,common["train_burial_mean"]); score=metrics["primary"]["selection"]
        if score<best["selection"]-MIN_DELTA:best={"epoch":epoch,"selection":score,"norm_separation":separation(p)}
        import hashlib
        row={"epoch":epoch,"train":{k:float(np.mean([x[k] for x in logs])) for k in logs[0]},"validation":metrics,
            "norm_separation":separation(p),"seconds":time.monotonic()-began,"batch_plan_digest":hashlib.sha256(json.dumps(plan).encode()).hexdigest()}
        history.append(row); atomic_json(run/"history.json",history); atomic_json(run/"best.json",best)
        save_checkpoint(run/f"checkpoints/epoch-{epoch:03d}",p,state,{**metadata,"epoch":epoch})
        print(json.dumps({"arm":arm,"epoch":epoch,"selection":score,"epoch_seconds":row["seconds"],"norm_separation":row["norm_separation"]}),flush=True)
    loader.close(); p,_,_=load_checkpoint(run/f"checkpoints/epoch-{best['epoch']:03d}",(p,state))
    final,_=evaluate(base,m,val,engine,p,common["train_burial_mean"],run/"validation_predictions.csv")
    atomic_json(run/"final.json",final); atomic_json(run/"verification.json",{"passed":True,"best":best,"epochs_completed":10,
        "wall_seconds":time.monotonic()-started,"peak_memory":jax.local_devices()[0].memory_stats(),
        "predictions_sha256":digest(run/"validation_predictions.csv"),"protocol_sha256":digest(out/"protocol.json")})


def report(root):
    _,out=roots(root); result={}; rows={}; plans={}
    for arm in ARMS:
        run=out/arm; receipt=read(run/"verification.json")
        if not receipt["passed"] or digest(run/"validation_predictions.csv")!=receipt["predictions_sha256"]:raise AssertionError(arm)
        result[arm]={**receipt,"metrics":read(run/"final.json")}
        with (run/"validation_predictions.csv").open() as stream:
            rows[arm]=[{**r,**{k:float(r[k]) for k in ("state_error","paired_error","distance","rsa_free")},"interface":r["interface"].lower()=="true"} for r in csv.DictReader(stream)]
        plans[arm]=[r["batch_plan_digest"] for r in read(run/"history.json")]
    if plans["shared"]!=plans["separate"]:raise AssertionError("unmatched plans")
    key=lambda r:tuple(r[k] for k in ("complex_id","chain","resnum","icode","group"))
    if [key(r) for r in rows["shared"]]!=[key(r) for r in rows["separate"]]:raise AssertionError("unmatched sites")
    bootstrap=_bootstrap(rows["shared"],rows["separate"])
    observed=result["separate"]["metrics"]["primary"]["selection"]-result["shared"]["metrics"]["primary"]["selection"]
    atomic_json(out/"summary.json",{"arms":result,"bootstrap":bootstrap,"observed_selection_delta":observed,"matched_batch_plans":True,"test_data_included":False})
    lines=["# oGQT local-query normalization pilot","",f"Shared versus separate query/context LayerNorm affine parameters (+88). Same epoch-1 weights, freshly reset Adam state in BOTH arms, same nine additional training epochs and standard auxiliary objective. Training seed {SEED}; 2,968 training and 400 validation complexes. No skip added and no test data used.","",
        "| Arm | State MAE | Paired MAE | Interface paired MAE | Selection | Epoch | Minutes | VRAM GiB |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm,item in result.items():
        p=item["metrics"]["primary"]
        lines.append(f"| {arm} | {p['state_mae']:.5f} | {p['paired_mae']:.5f} | {p['interface_paired_mae']:.5f} | {p['selection']:.5f} | {item['best']['epoch']} | {item['wall_seconds']/60:.1f} | {item['peak_memory']['peak_bytes_in_use']/2**30:.2f} |")
    lo,hi=bootstrap["ci95"]
    lines += ["",f"Selection difference (separate minus shared): {observed:.5f}; paired complex bootstrap 95% CI [{lo:.5f}, {hi:.5f}], 2,000 resamples, seed 17. Negative favours separate normalization.","",
        "This one-seed screen does not estimate seed variability. Intervals do not correct for checkpoint selection. Compare these two matched arms; the previous skip pilot preserved optimizer history and is not the control for this test. Auxiliary metrics, distance/RSA errors, norm separation, per-site predictions and timing are retained in JSON/CSV outputs.",""]
    (out/"report.md").write_text("\n".join(lines))


def main():
    import sys
    action=sys.argv[1]; root=os.environ["PKABENCH_RUNTIME"]
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK","1")),gpu_benchmark=action in ("prepare-seed","smoke","train"),allow_comp1400=True)
    if action=="prepare-seed":prepare_seed(root)
    elif action=="register":register(root)
    elif action=="smoke":smoke(root)
    elif action=="train":train(root,sys.argv[2])
    elif action=="report":report(root)
    else:raise ValueError(action)


if __name__=="__main__":main()
