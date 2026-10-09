"""Per-output cross-chain sensitivity accompanying the frozen flow diagnostic."""
from __future__ import annotations
import json
import hashlib
import os
import subprocess
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxpropka.topology import load_topology
from pkanet.model import linear
from pkanet.ogqt import initialize_auxiliary
from pkabench.ogqt_gradient_flow import base_gates, predicted, read, stats, zeros
from pkabench.ogqt_rotation_invariance import _read_cif
from pkabench.runtime import atomic_json, require_compute
from pkatrain.gqt_auxiliary_pilot import AuxiliaryEngine
from pkatrain.gqt_paired_pinder import Loader, _prefetched
from pkatrain.trainer import load_checkpoint

HEADS=("bound_shift","free_burial","bound_interface","free_interface")


def optional_stats(values):
    return {"eligible_sites":len(values),"distribution":stats(values) if values else None}


@jax.jit
def sensitivity(p,g,query):
    taps=zeros(g,p["embed"]["w"].shape[1])
    def objective(embedding):
        out,_=predicted(p,g,(embedding,*taps[1:]),base_gates())
        return jnp.stack((out["shift"][0,query],out["burial"][1,query],
                          out["interface"][0,query],out["interface"][1,query]))
    gradient=jnp.stack([jax.grad(lambda x:objective(x)[i])(taps[0]) for i in range(4)])
    # Branch-specific, per-residue squared norms. No pooling across target outputs.
    energy=jnp.sum(gradient**2,axis=(1,3))
    embedding=jax.vmap(lambda nodes:linear(p["embed"],nodes))(g["nodes"])
    scaled=jnp.sum((gradient*embedding[None])**2,axis=(1,3))
    return energy,scaled


def run(runtime):
    require_compute(threads=8,gpu_benchmark=True,allow_comp1400=True)
    started=time.monotonic(); runtime=Path(runtime); base=runtime/"training/ogqt-auxiliary-pilot-v1"
    output=base/"gradient-flow-v1"; protocol=read(output/"protocol.json"); manifest=read(base/"manifest.json")
    if not read(output/"verification.json")["passed"]: raise AssertionError("forward not verified")
    train_clusters={r["cluster_id"] for r in manifest["records"] if r["split"]=="train"}
    val_clusters={r["cluster_id"] for r in manifest["records"] if r["split"]=="val"}
    if train_clusters & val_clusters: raise AssertionError("training/validation cluster overlap")
    source=Path(__file__).parents[1]
    paths=[Path(__file__),Path(__file__).with_name("ogqt_gradient_flow.py"),Path(__file__).with_name("ogqt_gradient_flow_report.py"),
        source/"pkanet/model.py",source/"pkanet/site_model.py",source/"pkanet/triton_attention.py",
        source/"pkatrain/gqt_paired_pinder.py",base/"manifest.json",Path(protocol["checkpoint"])/"metadata.json"]
    atomic_json(output/"provenance.json",{"sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cluster_separation_passed":True,"test_data_included":False})
    p=initialize_auxiliary(jax.random.PRNGKey(17),**manifest["architecture"])
    engine=AuxiliaryEngine(p,0,{"burial":1,"interface":1})
    p,_,_=load_checkpoint(protocol["checkpoint"],(p,engine.optimizer.init(p)))
    records={r["id"]:r for r in manifest["records"]}; loader=Loader(base,manifest); results=[]
    for number,(ids,batch) in enumerate(_prefetched(loader,protocol["plans"]["val"]),1):
        for index,cid in enumerate(ids):
            folder=runtime/"pretraining/pinder-pkai-v1/entries"/cid
            topology=load_topology(_read_cif(folder/"AB.cif.gz"),gap_policy="cap",freeze_disulfides=True)
            chain=np.asarray([str(key.chain) for key in topology.keys]); n=len(chain)
            mapping={str(r["chain"]):str(r["partner"]) for r in read(folder/"sites.json")}
            partners=np.asarray([mapping.get(x,"unknown") for x in chain])
            g=jax.tree.map(lambda x:jnp.asarray(x[index]),batch[0]); count=int(batch[2][index].sum())
            if n!=records[cid]["n"]: raise AssertionError((cid,"residue count"))
            for q,key in enumerate(records[cid]["keys"]):
                residue=topology.keys[int(g["query_residue"][0,q])]
                if (str(residue.chain),residue.number,residue.insertion)!=tuple(key[:3]):
                    raise AssertionError((cid,"query topology mapping",q))
            distances=batch[5]["partner_distance_A"][index,:count]
            bins={"<=4":np.flatnonzero(distances<=4),"4-10":np.flatnonzero((distances>4)&(distances<=10)),">10":np.flatnonzero(distances>10)}
            for label,indices in bins.items():
                if len(indices)==0: continue
                # Stable median-index site in each distance stratum, never chosen by gradients.
                query=int(indices[len(indices)//2]); residue=int(g["query_residue"][0,query])
                energy,scaled=jax.device_get(sensitivity(p,g,jnp.asarray(query,jnp.int32)))
                same_chain=chain==chain[residue]; known=partners!="unknown"; target=partners[residue]
                regions={"own_chain":same_chain,"other_chain":~same_chain,"unknown_partner":~known}
                if target!="unknown":
                    regions.update(own_partner=known&(partners==target),other_partner=known&(partners!=target))
                for h,head in enumerate(HEADS):
                    total=float(energy[h,:n].sum()); total_scaled=float(scaled[h,:n].sum())
                    row={"id":cid,"query_index":query,"head":head,"distance_bin":label,"distance_A":float(distances[query]),"total_gradient_l2":float(np.sqrt(total)),"regions":{}}
                    for name,selected in regions.items():
                        amount=float(energy[h,:n][selected].sum()); weighted=float(scaled[h,:n][selected].sum())
                        row["regions"][name]={"residues":int(selected.sum()),"gradient_l2":float(np.sqrt(amount)),"energy_fraction":amount/total if total else None,"gradient_times_input_fraction":weighted/total_scaled if total_scaled else None}
                    if head.startswith("free_") and row["regions"]["other_chain"]["gradient_l2"]>1e-7:
                        raise AssertionError((cid,head,"free cross-chain leakage"))
                    results.append(row)
        print(json.dumps({"partner_batches":number,"total":len(protocol["plans"]["val"]),"seconds":time.monotonic()-started}),flush=True)
    loader.close(); atomic_json(output/"partner-per-site.json",results)
    summary={}
    for head in HEADS:
        summary[head]={}
        for label in ("<=4","4-10",">10"):
            selected=[r for r in results if r["head"]==head and r["distance_bin"]==label]
            if not selected: continue
            summary[head][label]={"sites":len(selected),"other_chain_energy_fraction":optional_stats([r["regions"]["other_chain"]["energy_fraction"] for r in selected if r["regions"]["other_chain"]["energy_fraction"] is not None]),"other_partner_energy_fraction":optional_stats([r["regions"]["other_partner"]["energy_fraction"] for r in selected if "other_partner" in r["regions"] and r["regions"]["other_partner"]["energy_fraction"] is not None])}
    atomic_json(output/"partner-summary.json",{"results":summary,"seconds":time.monotonic()-started,"free_cross_chain_leakage_check_passed":True,"method":"individual output Jacobian wrt initial residue embeddings; squared-norm shares, not causal attribution","test_data_included":False})
    subprocess.run([str(runtime/"envs/radial-plots/bin/python"),
        str(Path(__file__).with_name("ogqt_gradient_flow_report.py")),str(output)],check=True)


if __name__=="__main__": run(os.environ["PKABENCH_RUNTIME"])
