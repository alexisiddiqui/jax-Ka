"""quartet-rank-v2 (2026-11-19): pool-v5 control / residue-quartet q03 / chain-quartet arms qc, conly (PINDER val, free branch,
unmasked, one seed). Markdown on stdout; reads existing reports and run folders only.

Sources (under $PKABENCH_RUNTIME/audits unless overridden):
  control  pool5-v1-50-asym50-control-w064-sS   masked2 / residue / quartet: abdata-v5/score/control[-sS] (or quartet-rank-v2/score/...)
  q03      pool5-v1-50-asym50-q03-w064-sS       same, abdata-v5/score/q03[-sS]
  qc       pool5-v2-50-asym50-qc-w064-sS        quartet-rank-v2/score/qc[-sS]/{masked2,residue,quartet}
  conly    pool5-v2-50-asym50-conly-w064-sS     quartet-rank-v2/score/conly[-sS]/{masked2,residue,quartet}
  chain scorer for all four: quartet-rank-v2/score/<arm>[-sS]/chain/report.json
Table 1: pKa clean / state / interface (score_residue), the six masked2 gates; delta vs control (and vs q03 for qc / conly).
Table 2: Ab/Ag detail (masked2 swap varying antibody / antigen, wrong antibody), residue quartet accuracy per stratum
(score_quartet), chain quartet accuracy per class and the length-shortcut Spearman (score_chain).
Decision rule (brief): success = Ab/Ag swap >= control, wrong-partner >= control + 0.04, swap all >= control + 0.02, pKa clean
<= control + 0.012. Table 3: training terms from history.jsonl, regroup / pair counts and gradient-norm records.
Missing inputs are listed (every path) and the script exits 1.

  python compile.py [--seed 17] [--arms qc,conly]
"""
import argparse
import json
import os
import sys
from pathlib import Path

RUNTIME = Path(os.environ.get("PKABENCH_RUNTIME", "/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench"))
AUD = RUNTIME / "audits"
GATES = ["Ab/Ag swap", "swap all", "wrong-partner", "within AUROC", "edge AUROC", "edge AP"]
PKA = ["pKa clean", "state MAE", "iface paired MAE"]
ABAG = ["Ab/Ag vary antibody", "Ab/Ag vary antigen", "Ab/Ag wrong antibody"]
RESQ = ["res-quartet all", "res-quartet hetero", "res-quartet homo", "res-quartet Ab/Ag"]
CHAINQ = ["chain-quartet all", "chain-quartet hetero", "chain-quartet homo", "chain-quartet Ab/Ag", "length rho all", "length rho Ab/Ag"]
LOWER = set(PKA)
BASE_ARMS = ("control", "q03")
TERMS = ["pinder_chain_softplus", "pinder_chain_accuracy", "pinder_chain_accuracy_hetero", "pinder_chain_accuracy_homo", "pinder_chain_accuracy_abag",
         "pinder_chain_quartets_per_batch", "pinder_chain_quartets_abag_per_batch", "pinder_quartet_softplus", "pinder_within_listwise",
         "pinder_mask_free_edge_bce", "pinder_state", "pinder_paired"]


class Missing(Exception):
    pass


def run_name(arm, seed): return f"pool5-v1-50-asym50-{arm}-w064-s{seed}" if arm in BASE_ARMS else f"pool5-v2-50-asym50-{arm}-w064-s{seed}"


def sub(arm, seed): return arm if seed == 17 else f"{arm}-s{seed}"


def load(p):
    p = Path(p)
    if not p.is_file(): raise Missing(str(p))
    return json.loads(p.read_text())


def entry(runs, run, key, where):
    hits = [v for v in runs.values() if isinstance(v, dict) and v.get("run") == run]
    if len(hits) == 1: return hits[0]
    if key in runs: return runs[key]
    raise Missing(f"{where}: no entry run={run} / key={key!r} (keys {sorted(runs)})")


def score_dir(aud, arm, seed, step):
    """quartet-rank-v2/score/<arm>/<step> when it holds a report, else (control / q03) abdata-v5/score/<arm>/<step>."""
    mine = aud / "quartet-rank-v2" / "score" / sub(arm, seed) / step
    if (mine / "report.json").is_file() or arm not in BASE_ARMS or step == "chain": return mine
    return aud / "abdata-v5" / "score" / sub(arm, seed) / step


def metrics(aud, arm, seed):
    run = run_name(arm, seed); errors = []; out = {}
    d = lambda step: score_dir(aud, arm, seed, step)
    try:
        m = entry(load(d("masked2") / "report.json")["runs"], run, arm, str(d("masked2")))["modes"]["unmasked"]; sw = m["swap_same_stratum"]
        out.update({"Ab/Ag swap": sw["Ab/Ag"]["accuracy"], "swap all": sw["all"]["accuracy"], "wrong-partner": m["wrong_partner"]["different_protein"]["accuracy"],
                    "within AUROC": m["within_complex"]["all"]["auroc"], "edge AUROC": m["edges_10A_free"]["auroc"], "edge AP": m["edges_10A_free"]["ap"],
                    "Ab/Ag vary antibody": sw["Ab/Ag varying antibody"]["accuracy"], "Ab/Ag vary antigen": sw["Ab/Ag varying antigen"]["accuracy"],
                    "Ab/Ag wrong antibody": m["wrong_partner"]["abag_wrong_antibody"]["accuracy"]})
    except (Missing, KeyError) as e: errors.append(f"masked2 {run}: {e!r}")
    try:
        p = entry(load(d("residue") / "report.json"), run, arm, str(d("residue")))["pka"]
        out.update({"pKa clean": p["selection_no_bonded"], "state MAE": p.get("state_no_bonded"), "iface paired MAE": p.get("interface_no_bonded")})
    except (Missing, KeyError) as e: errors.append(f"residue {run}: {e!r}")
    try:
        q = entry(load(d("quartet") / "report.json")["runs"], run, arm, str(d("quartet")))["metrics"]
        out.update({"res-quartet all": q["overall"]["all"]["accuracy"], **{f"res-quartet {s}": q[s]["all"]["accuracy"] for s in ("hetero", "homo", "Ab/Ag")}})
    except (Missing, KeyError) as e: errors.append(f"quartet {run}: {e!r}")
    try:
        c = entry(load(d("chain") / "report.json")["runs"], run, arm, str(d("chain")))["metrics"]
        out.update({"chain-quartet all": c["quartets"]["all"]["accuracy"], **{f"chain-quartet {s}": c["quartets"][s]["accuracy"] for s in ("hetero", "homo", "Ab/Ag")},
                    "length rho all": c["length_spearman"]["all"], "length rho Ab/Ag": c["length_spearman"]["Ab/Ag"]})
    except (Missing, KeyError) as e: errors.append(f"chain {run}: {e!r}")
    if errors: raise Missing("; ".join(errors))
    return run, out


def history_row(runs_dir, run):
    out = Path(runs_dir) / run; sel = load(out / "selection.json"); prot = load(out / "protocol.json")
    hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    row = next(h for h in hist if h["epoch"] == sel["selected_epoch"])
    grads = [g for h in hist for g in h.get("debug_quartet_grad_norms", []) if "error" not in g]
    return {"epoch": f"{sel['selected_epoch']}/{sel['epochs_run']}", "train_s": sum(h["train_seconds"] for h in hist) / len(hist),
            "selected": {k: row["train_loss"].get(k) for k in TERMS}, "first": {k: hist[0]["train_loss"].get(k) for k in TERMS},
            "last": {k: hist[-1]["train_loss"].get(k) for k in TERMS}, "grads": grads, "parameters": sel.get("parameter_count"),
            "chain": prot.get("chain_quartet"), "quartet": (prot.get("quartet_rank") or {}).get("weight", 0.0),
            "regroup": hist[-1].get("chain_regroup"), "counts": hist[-1].get("chain_quartet_counts")}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="compile.py"); ap.add_argument("--arms", default="qc,conly"); ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--aud", default=str(AUD)); ap.add_argument("--runs-dir", default=str(RUNTIME / "training/gqt-production-v2/runs"))
    a = ap.parse_args(argv); aud = Path(a.aud); arms = [x for x in a.arms.split(",") if x]
    rows = {}; errors = []; hist = {}
    for arm in (*BASE_ARMS, *arms):
        try: rows[arm] = metrics(aud, arm, a.seed)
        except Missing as e: errors.append(f"{arm}: {e}")
    for arm in arms:
        try: hist[arm] = history_row(a.runs_dir, run_name(arm, a.seed))
        except (Missing, FileNotFoundError, StopIteration) as e: errors.append(f"{arm} history: {e!r}")
    if errors: print("missing inputs:\n  " + "\n  ".join(errors), file=sys.stderr); return 1
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"
    C = rows["control"][1]; Q = rows["q03"][1]

    def cell(label, col):
        x = rows[label][1][col]
        if x is None: return "-"
        if label == "control": return f"{x:.4f}"
        s = f"{x:.4f} ({x - C[col]:+.4f}" if C[col] is not None else f"{x:.4f} (-"
        if label in arms and Q[col] is not None: s += f"; q03 {x - Q[col]:+.4f}"
        return s + ")"

    def table(title, cols):
        print(f"### {title}\n"); print("| row | run | " + " | ".join(cols) + " |"); print("|---|---|" + "---:|" * len(cols))
        for label, (run, _) in rows.items(): print(f"| {label} | {run} | " + " | ".join(cell(label, c) for c in cols) + " |")
        print()
    weights = {arm: f"quartet {hist[arm]['quartet']}, chain {hist[arm]['chain'].get('weight') if hist[arm]['chain'] else '-'}" for arm in arms}
    print(f"# quartet-rank-v2: chain-quartet arms on pool-v5, seed {a.seed}\n")
    print("Weights: " + "; ".join(f"{k}: {v}" for k, v in weights.items()) + ". Deltas in brackets: vs control, and for the v2 arms vs q03.\n")
    table(f"Table 1. pKa and pairing gates (PINDER val 800, free branch, unmasked)", PKA + GATES)
    table("Table 2. Ab/Ag detail, residue quartet accuracy (score_quartet) and chain quartet accuracy / length-shortcut Spearman (score_chain)", ABAG + RESQ + CHAINQ)
    print("pKa: MAE (pH units), lower is better; gates / accuracies higher is better; length rho = Spearman of native chain score with log(n_A n_B) surface pairs "
          "(0 = no length shortcut). Single-seed bands: 0.02 (gates), 0.012 (pKa clean).\n")
    print("Decision rule (brief): success = Ab/Ag swap >= control, wrong-partner >= control + 0.04, swap all >= control + 0.02, pKa clean <= control + 0.012.\n")
    for arm in arms:
        m = rows[arm][1]
        checks = {"Ab/Ag swap >= control": m["Ab/Ag swap"] - C["Ab/Ag swap"] >= 0, "wrong-partner >= +0.04": m["wrong-partner"] - C["wrong-partner"] >= 0.04,
                  "swap all >= +0.02": m["swap all"] - C["swap all"] >= 0.02, "pKa clean <= +0.012": m["pKa clean"] - C["pKa clean"] <= 0.012}
        print(f"- {arm}: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in checks.items())
              + f" -> {'SUCCESS' if all(checks.values()) else 'not met'} (Ab/Ag swap {m['Ab/Ag swap'] - C['Ab/Ag swap']:+.4f} vs control, {m['Ab/Ag swap'] - Q['Ab/Ag swap']:+.4f} vs q03)")
    print("\n### Table 3. Training terms (history.jsonl train_loss, unweighted; first / selected / last epoch)\n")
    print("| arm | epoch | " + " | ".join(TERMS) + " |"); print("|---|---|" + "---:|" * len(TERMS))
    for arm in arms:
        for which in ("first", "selected", "last"):
            print(f"| {arm} | {which} | " + " | ".join(f(hist[arm][which].get(k)) for k in TERMS) + " |")
    for arm in arms:
        h = hist[arm]; print(f"\n- {arm}: selected epoch {h['epoch']}, train {f(h['train_s'], 0)} s/epoch, parameters {h['parameters']}")
        print(f"  - last-epoch regroup {json.dumps({k: v for k, v in (h['regroup'] or {}).items() if k != 'buckets'})}; pair counts {json.dumps(h['counts'])}")
        for g in h["grads"]:
            print(f"  - grad norms epoch {g.get('epoch')} batch {g.get('batch')}: chain/within (encoder, unweighted) {f(g.get('ratio_chain_over_within_encoder'), 3)}, "
                  f"weighted {f(g.get('ratio_weighted_chain_over_within_encoder'), 3)}; quartet/within {f(g.get('ratio_quartet_over_within_encoder'), 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
