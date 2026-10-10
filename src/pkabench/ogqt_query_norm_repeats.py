"""Aggregate matched normalization runs without treating seeds as new structures."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute

SEEDS=(17,29,43)
ARMS=("shared","separate")


def read(path):return json.loads(Path(path).read_text())


def run(root):
    require_compute(threads=2,allow_comp1400=True)
    base=Path(root)/"training/ogqt-query-norm-v1"; entries={}; data={}; protocols=[]; site_sets=[]
    for seed in SEEDS:
        folder=base if seed==17 else base/f"seed-{seed}"
        protocol=read(folder/"protocol.json"); protocols.append(protocol)
        if protocol["seed"]!=seed:raise AssertionError("seed mismatch")
        histories=[]; entries[str(seed)]={}
        for arm in ARMS:
            run=folder/arm; receipt=read(run/"verification.json")
            if not receipt["passed"] or digest(run/"validation_predictions.csv")!=receipt["predictions_sha256"]:raise AssertionError((seed,arm))
            if receipt["protocol_sha256"]!=digest(folder/"protocol.json"):raise AssertionError("protocol mismatch")
            final=read(run/"final.json"); primary=final["primary"]
            entries[str(seed)][arm]={**{k:primary[k] for k in ("state_mae","paired_mae","interface_paired_mae","selection")},
                "burial_mae":final["auxiliary"]["burial"]["pooled_mae"],
                "interface_ap":final["auxiliary"]["interface"]["predicted_ranking"]["average_precision"],
                "epoch":receipt["best"]["epoch"],"minutes":receipt["wall_seconds"]/60}
            histories.append([r["batch_plan_digest"] for r in read(run/"history.json")])
            with (run/"validation_predictions.csv").open() as stream:rows=list(csv.DictReader(stream))
            site_sets.append([(r["complex_id"],r["chain"],r["resnum"],r["icode"],r["group"]) for r in rows])
            grouped={}
            for row in rows:
                value=grouped.setdefault(row["complex_id"],np.zeros(5))
                interface=row["interface"].lower()=="true"
                value+=np.asarray([float(row["state_error"]),1.,abs(float(row["paired_error"]))*interface,float(interface),abs(float(row["paired_error"]))])
            data[seed,arm]=grouped
        if histories[0]!=histories[1]:raise AssertionError((seed,"unmatched batch plans"))
    if any(keys!=site_sets[0] for keys in site_sets):raise AssertionError("different validation sites")
    for p in protocols[1:]:
        for key in ("manifest_sha256","coefficients","epochs_total","optimizer","selection"):
            if p[key]!=protocols[0][key]:raise AssertionError((key,"different experiment settings"))
    if len({p["checkpoint_sha256"] for p in protocols})!=3:raise AssertionError("reused common checkpoint across seeds")
    initial=[read(base/f"seed-{seed}/warmup-registration.json")["initial_parameters_sha256"] for seed in (29,43)]
    if initial[0]==initial[1]:raise AssertionError("repeat initializations are identical")
    metrics=("state_mae","paired_mae","interface_paired_mae","selection","burial_mae","interface_ap")
    averages={arm:{metric:{"mean":float(np.mean([entries[str(seed)][arm][metric] for seed in SEEDS])),
        "std":float(np.std([entries[str(seed)][arm][metric] for seed in SEEDS],ddof=1))} for metric in metrics} for arm in ARMS}
    deltas={metric:[entries[str(seed)]["separate"][metric]-entries[str(seed)]["shared"][metric] for seed in SEEDS] for metric in metrics}
    # Use the SAME bootstrap sample of complexes across arms AND seeds.
    # The 3 x 400 repeated predictions are not 1,200 independent structures.
    ids=sorted(data[17,"shared"])
    values=np.asarray([[[data[seed,arm][cid] for cid in ids] for arm in ARMS] for seed in SEEDS])
    draws=np.random.default_rng(17).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    totals=np.einsum("rc,sacf->rsaf",draws,values)
    scores=totals[...,0]/totals[...,1]+totals[...,2]/totals[...,3]
    differences=(scores[:,:,1]-scores[:,:,0]).mean(1)
    bootstrap={"replicates":2000,"seed":17,"resampling_unit":"complex, jointly across arms and seeds",
        "selection_mean_delta_ci95":np.quantile(differences,[.025,.975]).tolist(),
        "interpretation":"conditional on these three seeds; does not capture unobserved seed variability or selection bias"}
    output=base/"three-seed-summary.json"
    atomic_json(output,{"seeds":SEEDS,"per_seed":entries,"mean_std":averages,"paired_seed_deltas":deltas,"bootstrap":bootstrap,
        "matched_batch_plans":True,"matched_validation_sites":True,"fixed_auxiliary_coefficients":True,"test_data_included":False})
    lines=["# oGQT query/context normalization: three-seed confirmation","",
        "Seeds 17, 29 and 43; 2,968 training and 400 validation complexes. Each seed has its own fresh initialization and one-epoch pKa-only warmup, shared by its two arms. Both arms reset Adam after warmup and train through epoch 10 with identical batch plans. Auxiliary coefficients are fixed at the seed-17 calibration; no skip connection was added.","",
        "| Seed | Shared selection | Separate selection | Separate − shared |","|---|---:|---:|---:|"]
    for seed,delta in zip(SEEDS,deltas["selection"]):
        lines.append(f"| {seed} | {entries[str(seed)]['shared']['selection']:.5f} | {entries[str(seed)]['separate']['selection']:.5f} | {delta:+.5f} |")
    lines += ["","| Metric | Shared mean ± SD | Separate mean ± SD |","|---|---:|---:|"]
    for metric in metrics:
        a=averages["shared"][metric]; b=averages["separate"][metric]
        lines.append(f"| {metric} | {a['mean']:.5f} ± {a['std']:.5f} | {b['mean']:.5f} ± {b['std']:.5f} |")
    lo,hi=bootstrap["selection_mean_delta_ci95"]
    lines += ["",f"Paired selection difference across seeds: {np.mean(deltas['selection']):+.5f} ± {np.std(deltas['selection'],ddof=1):.5f} (sample SD). "
        f"Complex-bootstrap interval for the mean difference: [{lo:+.5f}, {hi:+.5f}]. Negative favours separate normalization.","",
        "The bootstrap resamples the same complexes jointly across all six predictions; it does not treat seed repetitions as independent structures. Three seeds still give limited evidence about initialization variability. Validation was used for checkpoint selection, and these intervals do not correct for that selection.",""]
    (base/"three-seed-report.md").write_text("\n".join(lines))


if __name__=="__main__":run(os.environ["PKABENCH_RUNTIME"])
