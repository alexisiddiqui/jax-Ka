"""Seed spread and decision-rule comparisons for production runs (2026-10-11; scaling plan, 07 note).
  python scripts/seed_compare.py RUNS_DIR BASELINE_LABEL LABEL=run1,run2,... [LABEL=...]
RUNS_DIR: $PKABENCH_RUNTIME/training/gqt-production-v2/runs. Pass every finished 3-seed group to pool sigma over
more degrees of freedom (52 df at 2026-10-12 gives sigma 0.0040).
sigma: pooled sample SD of selection over groups with >= 2 finished seeds. A contrast vs BASELINE is 'real' if
|mean difference in selection| > 2 sigma and the four confirmation metrics agree in direction on >= 3 of 4."""
import json, sys
from pathlib import Path
import numpy as np
B = Path(sys.argv[1]); base = sys.argv[2]
METRICS = ("selection", "state_mae", "interface_paired_mae", "pkpdb_mae", "benchmark_mae")
def row(run):
    d = B / run
    if not (d / "selection.json").exists(): return None
    s = json.loads((d / "selection.json").read_text()); v = s["validation"]
    h = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines()]
    return {"selection": v["selection"], "state_mae": v["pinder"]["state_mae"], "interface_paired_mae": v["pinder"]["interface_paired_mae"],
            "pkpdb_mae": v["pkpdb"]["site_mae"], "benchmark_mae": v["benchmark"]["overall"]["mae"], "epoch": s["selected_epoch"],
            "epochs_run": s["epochs_run"], "params": s.get("parameter_count"), "train_s": float(np.median([x["train_seconds"] for x in h]))}
groups = {}
for arg in sys.argv[3:]:
    label, runs = arg.split("="); rows = [(r, row(r)) for r in runs.split(",")]
    groups[label] = [x for _, x in rows if x]; missing = [r for r, x in rows if not x]
    if missing: print(f"# {label}: not finished {missing}")
var = [(np.var([r["selection"] for r in g], ddof=1), len(g) - 1) for g in groups.values() if len(g) >= 2]
sigma = float(np.sqrt(sum(v * d for v, d in var) / sum(d for _, d in var))) if var else float("nan")
print(f"pooled seed sigma (selection) = {sigma:.4f}  ({sum(d for _, d in var)} df)")
for label, g in groups.items():
    if not g: continue
    m = {k: np.mean([r[k] for r in g]) for k in METRICS}; sd = np.std([r["selection"] for r in g], ddof=1) if len(g) > 1 else float("nan")
    print(json.dumps({"group": label, "n": len(g), "params": g[0]["params"], **{k: round(float(m[k]), 4) for k in METRICS},
                      "selection_sd": round(float(sd), 4), "selection_seeds": [round(r["selection"], 4) for r in g],
                      "epochs": [r["epoch"] for r in g], "train_s": round(float(np.mean([r["train_s"] for r in g])), 1)}))
if base in groups and groups[base]:
    b = {k: np.mean([r[k] for r in groups[base]]) for k in METRICS}
    for label, g in groups.items():
        if label == base or not g: continue
        m = {k: np.mean([r[k] for r in g]) for k in METRICS}; d = {k: m[k] - b[k] for k in METRICS}
        agree = sum(np.sign(d[k]) == np.sign(d["selection"]) for k in METRICS[1:])
        print(json.dumps({"contrast": f"{label} - {base}", **{k: round(float(d[k]), 4) for k in METRICS}, "over_2sigma": bool(abs(d["selection"]) > 2 * sigma),
                          "agree_of_4": int(agree), "real": bool(abs(d["selection"]) > 2 * sigma and agree >= 3)}))
