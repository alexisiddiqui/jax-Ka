"""quartet-rank-v1 Part B (2026-11-15): Table 1 (control vs the encoder-trained quartet arms, PINDER val, free branch, unmasked,
seed 17) and Table 2 (quartet training terms from history.jsonl), markdown on stdout. Reads existing reports only.

Sources (under $PKABENCH_RUNTIME/audits unless overridden):
  control  drsa-v1-50-within03-asym50-w064-s17    drsa-within-v1/{masked2,residue}/report.json (key "drsa")
                                                   + quartet-rank-v1/control/quartet/report.json (optional)
  arm A    quartet-v1-50-asym50-qAA-w064-s17       quartet-rank-v1/A/{masked2,residue,quartet}/report.json (key A; quartet optional)
  extra    --extra LABEL=DIR:KEY:RUN (optional rows, e.g. another recipe scored by the same masked2 / residue scorers)
Metrics: masked2 runs[*].modes.unmasked (Ab/Ag swap, swap all, wrong-partner = different protein, within AUROC, edge AUROC / AP),
score_residue pka (pKa clean = selection_no_bonded; state / interface no-bonded), score_quartet overall all / chemical accuracy.
Decision rule (brief): advance if any Table-1 gate >= +0.01 vs control with pKa clean within +0.012; flagged per arm, not decided here.
Missing required inputs are listed (every path) and the script exits 1.

  python compile.py [--arms q03,q10] [--seed 17] [--extra LABEL=DIR:KEY:RUN ...]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

RUNTIME = Path(os.environ.get("PKABENCH_RUNTIME", "/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench"))
AUD = RUNTIME / "audits"
GATES = ["Ab/Ag swap", "swap all", "wrong-partner", "within AUROC", "edge AUROC", "edge AP"]
PKA = ["pKa clean", "state MAE", "iface paired MAE"]
QUART = ["quartet acc", "quartet acc chem"]
COLS = PKA + GATES + QUART
LOWER = set(PKA)
WEIGHTS = {"q03": 0.3, "q10": 1.0}
TERMS = ["pinder_quartet_softplus", "pinder_quartet_accuracy", "pinder_quartet_covered", "pinder_within_listwise", "pinder_mask_free_edge_bce", "pinder_state", "pinder_paired"]


class Missing(Exception):
    pass


def arm_run(arm, seed): return f"quartet-v1-50-asym50-{arm}-w064-s{seed}"


def control_run(seed): return f"drsa-v1-50-within03-asym50-w064-s{seed}"


def load(p):
    p = Path(p)
    if not p.is_file(): raise Missing(str(p))
    return json.loads(p.read_text())


def entry(runs, run, key, where):
    hits = [v for v in runs.values() if isinstance(v, dict) and v.get("run") == run]
    if len(hits) == 1: return hits[0]
    if key in runs: return runs[key]
    raise Missing(f"{where}: no entry run={run} / key={key!r} (keys {sorted(runs)})")


def metrics(masked2_dir, residue_dir, quartet_dir, key, run):
    errors = []; out = {}
    try:
        m = entry(load(Path(masked2_dir) / "report.json")["runs"], run, key, str(Path(masked2_dir) / "report.json"))["modes"]["unmasked"]
        out.update({"Ab/Ag swap": m["swap_same_stratum"]["Ab/Ag"]["accuracy"], "swap all": m["swap_same_stratum"]["all"]["accuracy"],
                    "wrong-partner": m["wrong_partner"]["different_protein"]["accuracy"], "within AUROC": m["within_complex"]["all"]["auroc"],
                    "edge AUROC": m["edges_10A_free"]["auroc"], "edge AP": m["edges_10A_free"]["ap"]})
    except (Missing, KeyError) as e: errors.append(f"masked2 {run}: {e}")
    try:
        p = entry(load(Path(residue_dir) / "report.json"), run, key, str(Path(residue_dir) / "report.json"))["pka"]
        out.update({"pKa clean": p["selection_no_bonded"], "state MAE": p.get("state_no_bonded"), "iface paired MAE": p.get("interface_no_bonded")})
    except (Missing, KeyError) as e: errors.append(f"residue {run}: {e}")
    out["quartet acc"] = out["quartet acc chem"] = None
    if quartet_dir is not None and (Path(quartet_dir) / "report.json").is_file():
        q = entry(load(Path(quartet_dir) / "report.json")["runs"], run, key, str(quartet_dir))["metrics"]["overall"]
        out["quartet acc"], out["quartet acc chem"] = q["all"]["accuracy"], q["chemical"]["accuracy"]
    if errors: raise Missing("; ".join(errors))
    return out


def history_row(runs_dir, run):
    out = Path(runs_dir) / run; sel = load(out / "selection.json"); hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    row = next(h for h in hist if h["epoch"] == sel["selected_epoch"])
    grads = [g for h in hist for g in h.get("debug_quartet_grad_norms", []) if "error" not in g]
    return {"epoch": f"{sel['selected_epoch']}/{sel['epochs_run']}", "train_s": sum(h["train_seconds"] for h in hist) / len(hist),
            "selected": {k: row["train_loss"].get(k) for k in TERMS}, "first": {k: hist[0]["train_loss"].get(k) for k in TERMS},
            "last": {k: hist[-1]["train_loss"].get(k) for k in TERMS}, "quartet_counts": hist[-1].get("quartet_counts"), "grads": grads,
            "parameters": sel.get("parameter_count")}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="compile.py"); ap.add_argument("--arms", default="q03,q10"); ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--aud", default=str(AUD)); ap.add_argument("--runs-dir", default=str(RUNTIME / "training/gqt-production-v2/runs"))
    ap.add_argument("--control-dir", default=None); ap.add_argument("--extra", action="append", default=[])
    a = ap.parse_args(argv); aud = Path(a.aud); arms = [x for x in a.arms.split(",") if x]; Q = aud / "quartet-rank-v1"
    cdir = Path(a.control_dir) if a.control_dir else aud / ("drsa-within-v1" if a.seed == 17 else f"drsa-within-v1/seeds/s{a.seed}")
    rows = {}; errors = []; hist = {}
    try: rows["control"] = (control_run(a.seed), metrics(cdir / "masked2", cdir / "residue", Q / "control" / "quartet", "drsa", control_run(a.seed)))
    except Missing as e: errors.append(f"control: {e}")
    for spec in a.extra:
        label, rest = spec.split("=", 1); d, key, run = rest.split(":", 2)
        try: rows[label] = (run, metrics(Path(d) / "masked2", Path(d) / "residue", Path(d) / "quartet", key, run))
        except Missing as e: errors.append(f"{label}: {e}")
    for arm in arms:
        run = arm_run(arm, a.seed); d = Q / (arm if a.seed == 17 else f"{arm}-s{a.seed}")
        try: rows[arm] = (run, metrics(d / "masked2", d / "residue", d / "quartet", arm, run))
        except Missing as e: errors.append(f"{arm}: {e}")
        try: hist[arm] = history_row(a.runs_dir, run)
        except (Missing, FileNotFoundError, StopIteration) as e: errors.append(f"{arm} history: {e}")
    if errors: print("missing inputs:\n  " + "\n  ".join(errors), file=sys.stderr); return 1
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"
    C = rows["control"][1]
    def cell(x, c, col):
        if x is None: return "-"
        return f"{x:.4f}" if c is None or c is x else f"{x:.4f} ({x - c:+.4f})"
    print(f"### Table 1. Control vs quartet arms, seed {a.seed} (PINDER val 800, free branch, unmasked; delta vs control in brackets)\n")
    print("| row | run | quartet weight | " + " | ".join(COLS) + " | sel. epoch | train s/epoch |"); print("|---|---|---:|" + "---:|" * (len(COLS) + 2))
    for label, (run, m) in rows.items():
        h = hist.get(label, {}); base = None if label == "control" else C
        print(f"| {label} | {run} | {WEIGHTS.get(label, '-') if label != 'control' else 0} | " + " | ".join(cell(m[c], None if base is None else base[c], c) for c in COLS)
              + f" | {h.get('epoch', '-')} | {f(h.get('train_s'), 0)} |")
    print("\npKa clean / state / interface: MAE (pH units), lower is better; gates and quartet accuracy: higher is better. Quartet acc = score_quartet.py "
          "(exhaustive deterministic pools, general + chemically matched, residue-type charge classes). Single-seed bands: 0.02 (gates), 0.012 (pKa clean).\n")
    print("Decision-rule flags (brief: advance if any gate >= +0.01 with pKa clean within +0.012; watch Ab/Ag swap separately):\n")
    for arm in arms:
        m = rows[arm][1]; up = [g for g in GATES if m[g] - C[g] >= 0.01]; dp = m["pKa clean"] - C["pKa clean"]
        print(f"- {arm}: gates >= +0.01: {up or 'none'}; pKa clean {dp:+.4f} ({'clean' if dp <= 0.012 else 'NOT clean'}); Ab/Ag swap {m['Ab/Ag swap'] - C['Ab/Ag swap']:+.4f}"
              f" -> {'meets the advance rule' if up and dp <= 0.012 else 'does not meet the advance rule'}")
    print("\n### Table 2. Training terms (history.jsonl train_loss, unweighted; first / selected / last epoch)\n")
    print("| arm | epoch | " + " | ".join(TERMS) + " |"); print("|---|---|" + "---:|" * len(TERMS))
    for arm in arms:
        for which in ("first", "selected", "last"):
            print(f"| {arm} | {which} | " + " | ".join(f(hist[arm][which].get(k)) for k in TERMS) + " |")
    for arm in arms:
        h = hist[arm]; print(f"\n- {arm}: parameters {h['parameters']}; quartet loader counts {json.dumps(h['quartet_counts'])}")
        for g in h["grads"]:
            print(f"  - grad norms epoch {g.get('epoch')} batch {g.get('batch')}: ratio quartet/within (encoder, unweighted) {f(g.get('ratio_quartet_over_within_encoder'), 3)}, "
                  f"weighted {f(g.get('ratio_weighted_quartet_over_within_encoder'), 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
