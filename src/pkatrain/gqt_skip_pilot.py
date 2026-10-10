"""Matched one-seed control/learnable-skip screen using the auxiliary cohort."""
from __future__ import annotations
import csv
import functools
import hashlib
import json
import os
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from pkanet.ogqt import initialize_auxiliary
from pkanet.site_skip import predict_multi
from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain.gqt_auxiliary_pilot import (AuxiliaryEngine, _bootstrap, _rate, evaluate, read, MIN_DELTA)
from pkatrain.gqt_paired_pinder import Loader, _plans, _prefetched
from pkatrain.trainer import load_checkpoint, save_checkpoint

VERSION="ogqt-direct-skip-v1"
SEED=17
EPOCHS=10


def roots(root):
    training=Path(root)/"training"
    return training/"ogqt-auxiliary-pilot-v1", training/VERSION


def hashes():
    src=Path(__file__).parents[1]
    return {str(p):digest(p) for p in (Path(__file__),src/"pkanet/site_skip.py",src/"pkanet/site_model.py",
        src/"pkanet/model.py",src/"pkanet/triton_attention.py",src/"pkatrain/gqt_auxiliary_pilot.py")}


def setup(root, arm):
    if arm not in ("control","skip"): raise ValueError(arm)
    base,out=roots(root); manifest=read(base/"manifest.json")
    p=initialize_auxiliary(jax.random.PRNGKey(SEED),**manifest["architecture"])
    coeff=read(base/"calibration.json")["coefficients"]
    old_engine=AuxiliaryEngine(p,1,coeff)
    old_p,old_state,_=load_checkpoint(base/"common-warmup/checkpoint",(p,old_engine.optimizer.init(p)))
    if arm=="control":
        return base,out,manifest,old_p,old_state,old_engine,old_p,old_state,old_engine
    p={**old_p,"direct_skip":jnp.asarray(0.,jnp.float32)}
    engine=AuxiliaryEngine(p,1,coeff,predict_fn=functools.partial(predict_multi,enabled=arm=="skip"))
    # Preserve all existing Adam moments and its step counter; add zero moments only for the new scalar.
    leaves={jax.tree_util.keystr(path):value for path,value in jax.tree_util.tree_flatten_with_path(old_state)[0]}
    def transplant(path,value):
        key=jax.tree_util.keystr(path)
        if key in leaves:
            if value.shape!=leaves[key].shape: raise AssertionError((key,"shape"))
            return leaves[key]
        if "direct_skip" not in key: raise AssertionError((key,"unexpected new optimizer leaf"))
        return value
    state=jax.tree_util.tree_map_with_path(transplant,engine.optimizer.init(p))
    copied={jax.tree_util.keystr(path):value for path,value in jax.tree_util.tree_flatten_with_path(state)[0]}
    for key,value in leaves.items(): np.testing.assert_array_equal(np.asarray(value),np.asarray(copied[key]))
    return base,out,manifest,p,state,engine,old_p,old_state,old_engine


def register(root):
    base,out=roots(root); out.mkdir(parents=True,exist_ok=True)
    if (out/"protocol.json").exists(): raise FileExistsError(out/"protocol.json")
    manifest=read(base/"manifest.json"); train=[r for r in manifest["records"] if r["split"]=="train"]
    val=[r for r in manifest["records"] if r["split"]=="val"]
    if {r["cluster_id"] for r in train}&{r["cluster_id"] for r in val}: raise AssertionError("cluster overlap")
    rng=np.random.default_rng(SEED); plans={str(e):_plans(train,rng,manifest) for e in range(2,EPOCHS+1)}
    atomic_json(out/"plans.json",plans)
    atomic_json(out/"protocol.json",{"version":VERSION,"arms":["control","skip"],"seed":SEED,
        "epochs_total":EPOCHS,"common_warmup_epochs":1,"train_complexes":len(train),"validation_complexes":len(val),
        "initialization":"identical common epoch-1 checkpoint; existing optimizer state preserved; gate moments zero",
        "gate":"one unconstrained learnable scalar shared across both branches and all heads; zero initialized; no weight decay",
        "skip":"initial site tokens / sqrt(mean(tokens^2)+1e-5), added after final site block",
        "objective":"state MSE + paired MSE + fixed calibrated burial/interface losses (standard auxiliary arm)",
        "alpha":1,"coefficients":read(base/"calibration.json")["coefficients"],"dropout":0,"context_masking":0,
        "optimizer":"AdamW 1e-4; global-norm clip 1; inherited cosine schedule to 1e-5",
        "selection":"unweighted state MAE + interface paired MAE; common warmup eligible; minimum improvement 0.001",
        "budget":"fixed ten total epochs for both arms","test_data_included":False,
        "plans_sha256":digest(out/"plans.json"),"manifest_sha256":digest(base/"manifest.json"),
        "common_checkpoint_sha256":digest(base/"common-warmup/checkpoint/state.npz"),"code_hashes":hashes()})


def check_protocol(base,out):
    protocol=read(out/"protocol.json")
    if protocol["code_hashes"]!=hashes(): raise AssertionError("registered code changed")
    for path,key in ((base/"manifest.json","manifest_sha256"),(out/"plans.json","plans_sha256"),
        (base/"common-warmup/checkpoint/state.npz","common_checkpoint_sha256")):
        if digest(path)!=protocol[key]: raise AssertionError((str(path),"hash"))
    return protocol


def smoke(root):
    base,out,manifest,p,state,engine,old_p,old_state,old_engine=setup(root,"skip")
    check_protocol(base,out); loader=Loader(base,manifest)
    ids=read(out/"plans.json")["2"][0]; batch=loader.batch(ids)
    original=old_engine.predictions(old_p,batch[0]); changed=engine.predictions(p,batch[0])
    for name in original: np.testing.assert_allclose(np.asarray(original[name]),np.asarray(changed[name]),atol=1e-6,rtol=1e-6)
    target,mask,b,i=engine.targets(batch,manifest["normalization"]); valid=jnp.ones(len(ids),bool)
    args=(batch[0],target,mask,b,i,valid)
    def objective(params):
        parts=engine.losses(params,*args)
        return parts[0]+parts[1]+engine.lambdas["burial"]*parts[2]+engine.lambdas["interface"]*parts[3]
    def old_objective(params):
        parts=old_engine.losses(params,*args)
        return parts[0]+parts[1]+engine.lambdas["burial"]*parts[2]+engine.lambdas["interface"]*parts[3]
    gradient=jax.jit(jax.grad(objective))(p); reference=jax.jit(jax.grad(old_objective))(old_p)
    common={k:v for k,v in gradient.items() if k!="direct_skip"}
    differences=[np.asarray(a-b).ravel() for a,b in zip(jax.tree.leaves(common),jax.tree.leaves(reference))]
    scale=np.concatenate([np.asarray(a).ravel() for a in jax.tree.leaves(reference)])
    relative=float(np.linalg.norm(np.concatenate(differences))/max(np.linalg.norm(scale),1e-12))
    if not np.isfinite(relative) or relative>2e-4: raise AssertionError(("gradient parity",relative))
    print(json.dumps({"smoke":"prediction and gradient parity passed","relative_gradient_difference":relative}),flush=True)
    analytical=float(gradient["direct_skip"]); eps=.003
    finite=float((objective({**p,"direct_skip":p["direct_skip"]+eps})-objective({**p,"direct_skip":p["direct_skip"]-eps}))/(2*eps))
    np.testing.assert_allclose(finite,analytical,rtol=.03,atol=5e-4)
    updated,new_state,loss=engine.update(p,state,batch,manifest["normalization"],1e-3)
    if abs(float(updated["direct_skip"]))<1e-10: raise AssertionError("gate did not learn")
    _,_,_,cp,cs,ce,_,_,_=setup(root,"control")
    cp,cs,_=ce.update(cp,cs,batch,manifest["normalization"],1e-3)
    op,_,_=old_engine.update(old_p,old_state,batch,manifest["normalization"],1e-3)
    if float(cp.get("direct_skip",0.))!=0: raise AssertionError("control gate changed")
    update_errors=[]; reference_updates=[]
    for name in old_p:
        for a,z,start in zip(jax.tree.leaves(cp[name]),jax.tree.leaves(op[name]),jax.tree.leaves(old_p[name])):
            update_errors.append(np.asarray(a-z).ravel()); reference_updates.append(np.asarray(z-start).ravel())
    error=np.concatenate(update_errors); reference_update=np.concatenate(reference_updates)
    max_error=float(np.max(np.abs(error))); update_relative=float(np.linalg.norm(error)/max(np.linalg.norm(reference_update),1e-12))
    # Triton scatter reductions need not be bitwise reproducible. Adam can amplify tiny
    # gradient differences in near-zero bias channels; bound both absolute and global error.
    if not np.isfinite(update_relative) or max_error>1e-5 or update_relative>2e-3:
        raise AssertionError(("control update parity",max_error,update_relative))
    save_checkpoint(out/"smoke-checkpoint",updated,new_state,{"smoke":True})
    restored,rs,_=load_checkpoint(out/"smoke-checkpoint",(updated,new_state))
    for a,z in zip(jax.tree.leaves((updated,new_state)),jax.tree.leaves((restored,rs))):
        np.testing.assert_array_equal(np.asarray(a),np.asarray(z))
    loader.close(); atomic_json(out/"tests.json",{"passed":True,"zero_gate_prediction_parity":True,
        "shared_gradient_relative_difference":relative,"gate_gradient":analytical,"finite_difference":finite,
        "gate_after_one_step":float(updated["direct_skip"]),"control_update_parity":True,
        "control_update_max_abs_difference":max_error,"control_update_relative_difference":update_relative,
        "optimizer_moments_preserved":True,"checkpoint_roundtrip":True,"finite_jit_update":True,"loss":loss,"code_hashes":hashes()})


def train(root,arm):
    base,out,manifest,p,state,engine,*_=setup(root,arm); protocol=check_protocol(base,out)
    if not read(out/"tests.json")["passed"]: raise AssertionError("smoke not passed")
    run=out/arm; run.mkdir(exist_ok=False); plans=read(out/"plans.json")
    common=read(base/"common-warmup/verification.json"); val=[r for r in manifest["records"] if r["split"]=="val"]
    best={"epoch":1,"selection":common["validation"]["primary"]["selection"],"gate":0.}
    save_checkpoint(run/"checkpoints/epoch-001",p,state,{"epoch":1,"arm":arm})
    history=[]; loader=Loader(base,manifest); began=time.monotonic()
    atomic_json(run/"run.json",{**protocol,"arm":arm,"started_unix":time.time()})
    for epoch in range(2,EPOCHS+1):
        epoch_start=time.monotonic(); epoch_plan=plans[str(epoch)]; logs=[]
        for number,(_,batch) in enumerate(_prefetched(loader,epoch_plan),1):
            p,state,loss=engine.update(p,state,batch,manifest["normalization"],_rate(epoch,number,len(epoch_plan))); logs.append(loss)
            if number%50==0 or number==len(epoch_plan):
                progress={"arm":arm,"epoch":epoch,"batch":number,"batches":len(epoch_plan),"gate":float(p.get("direct_skip",0.)),"seconds":time.monotonic()-began}
                atomic_json(run/"progress.json",progress); print(json.dumps(progress),flush=True)
        metrics,_=evaluate(base,manifest,val,engine,p,common["train_burial_mean"])
        score=metrics["primary"]["selection"]
        if score<best["selection"]-MIN_DELTA: best={"epoch":epoch,"selection":score,"gate":float(p.get("direct_skip",0.))}
        row={"epoch":epoch,"train":{key:float(np.mean([x[key] for x in logs])) for key in logs[0]},"validation":metrics,
            "gate":float(p.get("direct_skip",0.)),"seconds":time.monotonic()-epoch_start,"best_epoch":best["epoch"],
            "batch_plan_digest":hashlib.sha256(json.dumps(epoch_plan).encode()).hexdigest()}
        history.append(row); atomic_json(run/"history.json",history); atomic_json(run/"best.json",best)
        save_checkpoint(run/f"checkpoints/epoch-{epoch:03d}",p,state,{"epoch":epoch,"arm":arm,"batch_plan_digest":row["batch_plan_digest"]})
        print(json.dumps({"arm":arm,"epoch":epoch,"selection":score,"gate":row["gate"],"epoch_seconds":row["seconds"]}),flush=True)
    loader.close(); p,_,_=load_checkpoint(run/f"checkpoints/epoch-{best['epoch']:03d}",(p,state))
    final,_=evaluate(base,manifest,val,engine,p,common["train_burial_mean"],run/"validation_predictions.csv")
    atomic_json(run/"final.json",final)
    atomic_json(run/"verification.json",{"passed":True,"best":best,"epochs_completed":EPOCHS,"wall_seconds":time.monotonic()-began,
        "peak_memory":jax.local_devices()[0].memory_stats(),"predictions_sha256":digest(run/"validation_predictions.csv"),"protocol_sha256":digest(out/"protocol.json")})


def report(root):
    _,out=roots(root); summaries={}; rows={}; plans={}
    for arm in ("control","skip"):
        run=out/arm; check=read(run/"verification.json")
        if not check["passed"] or check["predictions_sha256"]!=digest(run/"validation_predictions.csv"): raise AssertionError(arm)
        summaries[arm]={"metrics":read(run/"final.json"),**check}
        with (run/"validation_predictions.csv").open() as stream:
            rows[arm]=[{**r,"state_error":float(r["state_error"]),"paired_error":float(r["paired_error"]),
                "interface":r["interface"].lower()=="true","distance":float(r["distance"]),"rsa_free":float(r["rsa_free"])} for r in csv.DictReader(stream)]
        plans[arm]=[r["batch_plan_digest"] for r in read(run/"history.json")]
    if plans["control"]!=plans["skip"]: raise AssertionError("batch plans differ")
    if [(r["complex_id"],r["chain"],r["resnum"],r["icode"],r["group"]) for r in rows["control"]]!=[(r["complex_id"],r["chain"],r["resnum"],r["icode"],r["group"]) for r in rows["skip"]]: raise AssertionError("unmatched sites")
    bootstrap=_bootstrap(rows["control"],rows["skip"])
    atomic_json(out/"summary.json",{"arms":summaries,"bootstrap":bootstrap,"matched_batch_plans":True,"test_data_included":False})
    lines=["# oGQT direct-skip pilot","","Matched standard-auxiliary objective, seed 17, ten total epochs including the shared one-epoch warm-up. Full 400-complex validation each epoch; no test data. Only the scalar-gated normalized initial-site bypass differs.","",
        "| Arm | State MAE | Paired MAE | Interface paired MAE | Selection | Selected epoch | Gate | Minutes | Peak VRAM GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm,item in summaries.items():
        p=item["metrics"]["primary"]; b=item["best"]
        lines.append(f"| {arm} | {p['state_mae']:.5f} | {p['paired_mae']:.5f} | {p['interface_paired_mae']:.5f} | {p['selection']:.5f} | {b['epoch']} | {b['gate']:.5f} | {item['wall_seconds']/60:.1f} | {item['peak_memory']['peak_bytes_in_use']/2**30:.2f} |")
    lo,hi=bootstrap["ci95"]
    observed=summaries["skip"]["metrics"]["primary"]["selection"]-summaries["control"]["metrics"]["primary"]["selection"]
    lines += ["",f"Observed selection difference (skip minus control): {observed:.5f}. Paired complex bootstrap 95% interval: [{lo:.5f}, {hi:.5f}] (2,000 resamples, seed 17). Negative favours skip.","",
        "This is a one-seed screen; bootstrap intervals do not measure seed variability or correct for checkpoint selection. Auxiliary scores, distance/RSA strata, gate trajectories, predictions, runtime and checksums are retained in the JSON/CSV files.",""]
    (out/"report.md").write_text("\n".join(lines))


def main():
    import sys
    action=sys.argv[1]; root=os.environ["PKABENCH_RUNTIME"]
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK","1")),gpu_benchmark=action in ("smoke","train"),allow_comp1400=True)
    if action=="register":register(root)
    elif action=="smoke":smoke(root)
    elif action=="train":train(root,sys.argv[2])
    elif action=="report":report(root)
    else:raise ValueError(action)


if __name__=="__main__":main()
