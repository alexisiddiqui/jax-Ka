"""quartet-rank-v1 Part A (2026-11-15): pairing-rank-v1 heads grafted onto the control, scored by our masked2 scorer.

Reads (no re-scoring):
  graft masked2   GRAFT_DIR/masked2/report.json   (score_graft.py: parity + 6 heads, --modes unmasked) and GRAFT_DIR/masked2/graft.json
  graft quartet   GRAFT_DIR/quartet/report.json   (score_quartet.py on the same 7 runs; optional -> "-")
  control         CONTROL_DIR/masked2/report.json (drsa-within-v1, key "drsa", run drsa-v1-50-within03-asym50-w064-s17)
Prints markdown: parity check (every Table-1 gate of the parity graft vs the stored control, |diff| <= TOL), the per-head
table (Ab/Ag swap, swap all, wrong-partner, within AUROC, edge AUROC, edge AP, quartet accuracy all / chemical), lambda 0.3 -
lambda 0 per head seed and the mean over seeds. Exit 1 when an input is missing; the parity verdict is printed, not enforced.

  python compile_graft.py --graft-dir DIR [--control-dir DIR] [--tol 1e-6]
"""
import argparse
import json
import os
import sys
from pathlib import Path

RUNTIME = Path(os.environ.get("PKABENCH_RUNTIME", "/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench"))
CONTROL_RUN = "drsa-v1-50-within03-asym50-w064-s17"
SEEDS = (17, 29, 43)
LAMBDAS = ("w0p0", "w0p3")
GATES = ["Ab/Ag swap", "swap all", "wrong-partner", "within AUROC", "edge AUROC", "edge AP"]
QCOLS = ["quartet acc", "quartet acc chem"]


def name_of(tag, seed): return f"graft-{tag}-h{seed}"


def gates(m):
    return {"Ab/Ag swap": m["swap_same_stratum"]["Ab/Ag"]["accuracy"], "swap all": m["swap_same_stratum"]["all"]["accuracy"],
            "wrong-partner": m["wrong_partner"]["different_protein"]["accuracy"], "within AUROC": m["within_complex"]["all"]["auroc"],
            "edge AUROC": m["edges_10A_free"]["auroc"], "edge AP": m["edges_10A_free"]["ap"],
            "_n": (m["swap_same_stratum"]["Ab/Ag"]["n"], m["swap_same_stratum"]["all"]["n"], m["wrong_partner"]["different_protein"]["n"], m["edges_10A_free"]["pairs"])}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="compile_graft.py"); ap.add_argument("--graft-dir", required=True)
    ap.add_argument("--control-dir", default=str(RUNTIME / "audits/drsa-within-v1")); ap.add_argument("--control-key", default="drsa"); ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args(argv); gd = Path(a.graft_dir); errors = []
    def load(p):
        try: return json.loads(Path(p).read_text())
        except FileNotFoundError: errors.append(f"missing {p}"); return None
    rep = load(gd / "masked2/report.json"); ctl = load(Path(a.control_dir) / "masked2/report.json"); grafts = load(gd / "masked2/graft.json")
    qrep = json.loads((gd / "quartet/report.json").read_text()) if (gd / "quartet/report.json").is_file() else None
    if errors: print("\n".join(errors), file=sys.stderr); return 1
    names = ["parity"] + [name_of(t, s) for s in SEEDS for t in LAMBDAS]
    runs = rep["runs"]; missing = [n for n in names if n not in runs or "unmasked" not in runs[n].get("modes", {})]
    if missing: print(f"masked2 report lacks unmasked results for {missing}", file=sys.stderr); return 1
    c = ctl["runs"][a.control_key]
    if c["run"] != CONTROL_RUN: print(f"control key {a.control_key} is {c['run']}, expected {CONTROL_RUN}", file=sys.stderr); return 1
    C = gates(c["modes"]["unmasked"]); R = {n: gates(runs[n]["modes"]["unmasked"]) for n in names}
    for n in names:
        q = (qrep or {}).get("runs", {}).get(n)
        R[n]["quartet acc"] = q["metrics"]["overall"]["all"]["accuracy"] if q else None
        R[n]["quartet acc chem"] = q["metrics"]["overall"]["chemical"]["accuracy"] if q else None
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"
    sd = lambda x: "-" if x is None else f"{x:+.4f}"
    diffs = {g: abs(R["parity"][g] - C[g]) for g in GATES}; same_n = R["parity"]["_n"] == C["_n"]
    parity_ok = same_n and all(d <= a.tol for d in diffs.values())
    pg = grafts.get(runs["parity"]["run"], {}); head_diff = pg.get("max_abs_diff_vs_base_head")
    print("### Part A. pairing-rank-v1 free edge heads grafted onto the control checkpoint, scored by score_masked2.py --modes unmasked "
          "(PINDER val 800, free branch; encoder, pair norm, pKa / burial / gate heads = control)\n")
    print(f"Parity graft (edge-contrib-v1 baseline-edge.npz onto {CONTROL_RUN}): max |head - checkpoint head| per array {json.dumps(head_diff)}; "
          f"gates vs the stored control report (drsa-within-v1 masked2, key {a.control_key}): max |diff| {max(diffs.values()):.2e} "
          f"({', '.join(f'{g} {d:.1e}' for g, d in diffs.items())}); comparison counts equal: {same_n} -> **parity {'PASS' if parity_ok else 'FAIL'}** (tol {a.tol:g})\n")
    cols = GATES + QCOLS
    print("| row | head | sel. epoch (tune AP) | " + " | ".join(cols) + " |"); print("|---|---|---:|" + "---:|" * len(cols))
    def sel(n):
        h = (grafts.get(runs[n]["run"]) or {}).get("head_selection") or {}
        return f"{h.get('epoch', '-')} ({f(h.get('tune_anchor_ap'))})" if h else "-"
    print(f"| control (stored) | checkpoint | - | " + " | ".join(f(C[g]) for g in GATES) + " | - | - |")
    print(f"| parity | baseline-edge.npz | - | " + " | ".join(f(R['parity'][g]) for g in cols) + " |")
    for s in SEEDS:
        for t in LAMBDAS:
            n = name_of(t, s); print(f"| {n} | seed-{s}/{t} | {sel(n)} | " + " | ".join(f(R[n][g]) for g in cols) + " |")
    print("\n### lambda 0.3 - lambda 0 (paired by head seed) and vs the control\n")
    print("| contrast | " + " | ".join(cols) + " |"); print("|---|" + "---:|" * len(cols))
    dl = {s: {g: (None if R[name_of('w0p3', s)][g] is None or R[name_of('w0p0', s)][g] is None else R[name_of('w0p3', s)][g] - R[name_of('w0p0', s)][g]) for g in cols} for s in SEEDS}
    for s in SEEDS: print(f"| seed {s}: w0p3 - w0p0 | " + " | ".join(sd(dl[s][g]) for g in cols) + " |")
    mean = lambda xs: None if any(x is None for x in xs) else sum(xs) / len(xs)
    print("| mean over seeds: w0p3 - w0p0 | " + " | ".join(sd(mean([dl[s][g] for s in SEEDS])) for g in cols) + " |")
    for t in LAMBDAS:
        print(f"| mean {t} - control | " + " | ".join(sd(mean([R[name_of(t, s)][g] - C[g] for s in SEEDS])) if g in GATES else "-" for g in cols) + " |")
    print("\nGates: pairwise accuracy / AUROC / AP, higher is better (single-seed band ~0.02). Quartet acc: score_quartet.py, exhaustive deterministic "
          "pools on PINDER val (general + chemically matched; residue-type charge classes), not pairing-rank-v1's own evaluation. pKa is the control's "
          "by construction (the encoder and pKa head are not grafted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
