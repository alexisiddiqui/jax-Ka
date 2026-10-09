"""Human-readable tables for the frozen oGQT gradient diagnostic."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np


def report(output):
    output=Path(output)
    read=lambda name:json.loads((output/name).read_text())
    summary=read("summary.json"); check=read("verification.json"); protocol=read("protocol.json")
    rows=read("per-complex.json"); partner=read("partner-summary.json")
    lines=["# oGQT gradient-flow and recurrence diagnostic", "",
        "Frozen standard auxiliary checkpoint, seed 17, epoch 9; no training updates or test-set access. "
        "The sample contains 148 training complexes from 32 fixed batches and 24 validation complexes from eight batches. "
        "The auxiliary-trained checkpoint is used to inspect all three heads; it is not a declaration that this arm won the pKa pilot.", "",
        f"Prediction parity: max difference {check['max_prediction_difference']:.3g}. Primary parameter-gradient relative difference: "
        f"{check['parameter_gradient_relative_difference']:.3g}. Finite differences at two step sizes and padding checks passed.", "",
        "## Gradient flow", "", "RMS gradients are with respect to hidden representations, using separate per-complex losses before clipping. "
        "Values below are medians over validation complexes and exclude padding. Different losses have different scales; these are not percentages of importance.", "",
        "| Stage | Activation RMS | State gradient RMS | Paired gradient RMS | Burial gradient RMS | Interface gradient RMS |",
        "|---|---:|---:|---:|---:|---:|"]
    for stage in protocol["stages"]:
        values=summary["results"]["val"]["flow"][stage]
        line=[stage,f"{values['state']['activation_rms']['median']:.4g}"]
        line += [f"{values[loss]['gradient_rms']['median']:.4g}" for loss in protocol["losses"]]
        lines.append("| "+" | ".join(line)+" |")
    lines += ["", "## Residual branch sizes", "",
        "Each entry is the median across validation complexes of the bound/free mean update-RMS divided by incoming-RMS. "
        "These measure branch activity; they do not establish useful information or generalization.", "",
        "| Block | Attention update / input | FF update / input |", "|---|---:|---:|"]
    ratios=summary["results"]["val"]["residual_ratios"]
    for block in ("encoder_1","encoder_2","local_query","site"):
        lines.append(f"| {block} | {ratios[block+'_attention']['median']:.4f} | {ratios[block+'_ff']['median']:.4f} |")
    lines += ["", "## Extra recurrence and direct skip", "",
        "Gate zero reproduces the checkpoint. Negative derivatives favour a small positive gate locally. "
        "Recurrence adds the residual update of one extra application of the existing shared site block. "
        "The direct skip adds the per-token RMS-normalized initial site embedding. "
        "Their units differ, so derivative magnitudes must not be compared as an architecture ranking. "
        "Primary means unweighted state MSE + paired MSE. Intervals bootstrap complexes, 2,000 resamples, seed 17. "
        "They describe this frozen-checkpoint screen, not uncertainty across training seeds.", "",
        "| Split | Probe | Mean primary derivative | 95% bootstrap CI | Complexes favouring positive gate |", "|---|---|---:|---|---:|"]
    bootstrap={}
    for split in ("train","val"):
        data=np.asarray([r["gate_derivatives"] for r in rows if r["split"]==split]); n=len(data)
        indices=np.random.default_rng(17).integers(0,n,size=(2000,n))
        for gate_index in (8,9):
            gate=protocol["gates"][gate_index]; values=data[:,0,gate_index]+data[:,1,gate_index]
            interval=np.quantile(values[indices].mean(1),[.025,.975]); fraction=float(np.mean(values<0))
            bootstrap[f"{split}/{gate}"]={"mean":float(values.mean()),"ci95":interval.tolist(),"negative_fraction":fraction,"complexes":n}
            lines.append(f"| {split} | {gate} | {values.mean():.5f} | [{interval[0]:.5f}, {interval[1]:.5f}] | {fraction:.1%} |")
    lines += ["", "| Validation probe | State derivative | Paired derivative | Burial derivative | Interface derivative |", "|---|---:|---:|---:|---:|"]
    for gate in protocol["gates"][-2:]:
        values=summary["results"]["val"]["gate_derivatives"][gate]
        lines.append("| "+gate+" | "+" | ".join(f"{values[loss]['mean']:.5f}" for loss in protocol["losses"])+" |")
    lines += ["", "## Partner sensitivity", "",
        "One deterministic site per available distance stratum per validation complex. "
        "The table reports median fractions of the individual output's squared embedding-gradient norm on the other partner. "
        "These local sensitivity shares depend on representation and partner size; they are not causal importance. "
        "The free branches have zero cross-chain gradient within the 1e-7 absolute-norm tolerance.", "",
        "| Output | Distance to partner | Eligible sites | Median partner gradient-energy share |", "|---|---|---:|---:|"]
    for head,groups in partner["results"].items():
        for label,item in groups.items():
            value=item["other_partner_energy_fraction"]; distribution=value["distribution"]
            shown=f"{distribution['median']:.2%}" if distribution else "undefined"
            lines.append(f"| {head} | {label} Å | {value['eligible_sites']} | {shown} |")
    lines += ["", "## Limits", "",
        "Existing attention and FF residual connections remain active throughout. A nonzero gradient does not establish adequate optimization; "
        "a low gradient can reflect either saturation or a well-fitted target. The diagnostic cannot establish that extra depth or skips improve held-out prediction. "
        "Any promising change still requires a matched training comparison. Validation has only 24 complexes, and this screen uses one checkpoint.", "",
        f"Main diagnostic elapsed time: {summary['elapsed_s']/60:.1f} min. Partner check: {partner['seconds']/60:.1f} min. "
        "These include startup/compilation and are not training-step timings.", ""]
    (output/"report.md").write_text("\n".join(lines))
    (output/"bootstrap.json").write_text(json.dumps(bootstrap,indent=2)+"\n")
    with (output/"gate-derivatives.csv").open("w",newline="") as stream:
        writer=csv.writer(stream); writer.writerow(("complex_id","split","gate","state","paired","burial","interface"))
        for row in rows:
            data=np.asarray(row["gate_derivatives"])
            for index,gate in enumerate(protocol["gates"]):writer.writerow((row["id"],row["split"],gate,*data[:,index]))


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output); summary=json.loads((output/"summary.json").read_text())
    protocol=json.loads((output/"protocol.json").read_text()); bootstrap=json.loads((output/"bootstrap.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout="constrained")
    x=np.arange(len(protocol["stages"]))
    for loss in protocol["losses"]:
        values=[summary["results"]["val"]["flow"][stage][loss]["gradient_rms"] for stage in protocol["stages"]]
        axes[0].plot(x,[v["median"] for v in values],marker="o",label=loss)
        axes[0].fill_between(x,np.maximum([v["p10"] for v in values],1e-12),np.maximum([v["p90"] for v in values],1e-12),alpha=.12)
    axes[0].set_xticks(x,[s.replace("_"," ") for s in protocol["stages"]],rotation=25,ha="right")
    axes[0].set_yscale("log"); axes[0].set_ylabel("Hidden-state gradient RMS")
    axes[0].set_title("Validation gradient flow\nMedian and 10–90% across complexes")
    axes[0].legend(); axes[0].grid(alpha=.2)
    for offset,split in ((-.12,"train"),(.12,"val")):
        values=[bootstrap[f"{split}/{gate}"] for gate in protocol["gates"][-2:]]
        means=np.array([v["mean"] for v in values]); lo=np.array([v["ci95"][0] for v in values]); hi=np.array([v["ci95"][1] for v in values])
        axes[1].errorbar(np.arange(2)+offset,means,yerr=np.stack((means-lo,hi-means)),fmt="o",capsize=5,label=split)
    axes[1].axhline(0,color="grey",linestyle="--")
    axes[1].set_xticks([0,1],["Extra shared site block","Extra initial-site skip"])
    axes[1].set_ylabel("Derivative of state MSE + paired MSE")
    axes[1].set_title("Frozen-checkpoint probes\nNegative favours a small positive gate")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.suptitle("oGQT gradient diagnostic • 148 training / 24 validation complexes\nOne checkpoint; gates have different units; no retraining comparison",fontsize=12)
    fig.savefig(output/"gradient-flow.png",dpi=180); fig.savefig(output/"gradient-flow.pdf"); plt.close(fig)


if __name__=="__main__":
    import sys
    report(sys.argv[1]); plot(sys.argv[1])
