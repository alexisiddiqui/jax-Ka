#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# quartet-rank-v2 smoke (2026-11-19): the pool-v5 recipe + --quartet-weight 0.3 + --chain-quartet-weight SMOKE_CW (default 1.0) under the
# trainer's --smoke (2 epochs of SMOKE_BATCHES = 150 batches of the regrouped epoch plans -> 300 updates, then validation / selection).
# debug_quartet_grad_norms on the LAST batch of each smoke epoch (after 149 and 299 updates, not at init): encoder L2 norms of the
# UNWEIGHTED chain term (slot 13), the UNWEIGHTED residue quartet term (slot 10) and the UNWEIGHTED within listwise term (slot 2);
# ratios printed (lambda_c for parity with within x 0.3 = 0.3 / ratio_chain_over_within_encoder).
# Pass = 2 history rows, finite terms, chain softplus > 0, chain accuracy in [0, 1], chain quartets per batch > 0, chain_regroup with
# Ab/Ag batches and an unchanged visited set, Ab/Ag quartets drawn > 0, residue quartet softplus > 0, 2 finite grad-norm records with
# chain and within norms, protocol chain_quartet {SMOKE_CW, 16, 1, 32, ABAGMIN}, parameter count 180000. Printed: chain accuracy by class,
# Ab/Ag quartets per batch, step time (train s / update) vs the pool-v5 q03 run (epoch 1 and the median epoch).
# Then scorer code paths on the smoke run (CPU jax): score_chain.py --limit 60 (exit 0, quartets scored) and score_masked2.py --modes
# unmasked under timeout 900 (exit 0 or 124, checkpoint loaded, >= 1 batch, no Traceback).
# Outputs: run smoke-quartet-v2-w064-<job>; $A = audits/quartet-rank-v2/smoke/{chain, smoke-masked2.log} (overwritten per job).
#   sbatch -J q2_smoke smoke.sbatch
set -uo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/quartet-rank-v2/jobs/common.sh
export PYTHONPATH=$CODE/src XLA_PYTHON_CLIENT_PREALLOCATE=false
ulimit -c 0; cd "$CODE"; echo "[job] $(date -Is) ${SLURMD_NODENAME:-}"
sha256sum "$CODE"/src/pkatrain/production_{train,loading,residue,extra}.py "$CODE"/jobs/*.py "$0" 2>/dev/null || true
SMOKE_CW=${SMOKE_CW:-1.0}; SMOKE_BATCHES=${SMOKE_BATCHES:-150}
A=$R/audits/quartet-rank-v2/smoke; mkdir -p "$A"
for f in "$STORE_V2/verification.json" "$P/store-hotspot-v5abag/verification.json" "$R/training/pool-v5/pinder-manifest.json"; do
  [ -f "$f" ] || { echo "[job] missing $f" >&2; exit 3; }; done
RUN=smoke-quartet-v2-w064-${SLURM_JOB_ID:-local}
gpu "$PY" -m pkatrain.production_train train "$RUN" $(recipe "$STORE_V2") --seed 17 --quartet-weight 0.3 --quartet-count 128 $(chain_flags "$SMOKE_CW") \
    "${POOL5[@]}" --smoke --smoke-batches "$SMOKE_BATCHES"
rc=$?; echo "[job] train exit=$rc"; [ $rc -eq 0 ] || exit $rc
"$PY" - "$R/training/gqt-production-v2/runs" "$RUN" "$SMOKE_CW" "$SMOKE_BATCHES" "${ABAGMIN:-2}" <<'PYEOF'
import json, math, statistics, sys
from pathlib import Path
runs = Path(sys.argv[1]); out = runs / sys.argv[2]; cw = float(sys.argv[3]); nb = int(sys.argv[4]); amin = int(sys.argv[5])
history = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
protocol = json.loads((out / "protocol.json").read_text()); selection = json.loads((out / "selection.json").read_text())
tl = [h["train_loss"] for h in history]; cq = protocol.get("chain_quartet") or {}
grads = [g for h in history for g in h.get("debug_quartet_grad_norms", [])]
num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
finite = all(math.isfinite(v) for l in tl for v in l.values() if num(v))
counts = [h.get("chain_quartet_counts") or {} for h in history]; regroup = [h.get("chain_regroup") or {} for h in history]
parts = {
    "history_rows_2": len(history) == 2, "updates": all(h["updates"] == nb for h in history), "finite_terms": finite,
    "chain_softplus_positive": all((l.get("pinder_chain_softplus") or 0) > 0 for l in tl),
    "chain_accuracy_0_1": all(0.0 <= (l.get("pinder_chain_accuracy") if l.get("pinder_chain_accuracy") is not None else -1) <= 1.0 for l in tl),
    "chain_quartets_per_batch": all((l.get("pinder_chain_quartets_per_batch") or 0) > 0 for l in tl),
    "quartet_softplus_positive": all((l.get("pinder_quartet_softplus") or 0) > 0 for l in tl),
    "regroup": all(r.get("abag", 0) > 0 and r.get("abag_batches", 0) > 0 and r.get("minimum") == amin for r in regroup),
    "abag_drawn": sum(c.get("drawn", [0, 0, 0])[2] for c in counts) > 0,
    "grad_norms": len(grads) == 2 and all("error" not in g and g.get("batch") == nb and "chain_encoder" in g["norms"] and "within_encoder" in g["norms"]
                                          and all(num(v) and math.isfinite(v) and v > 0 for v in g["norms"].values()) for g in grads),
    "protocol_chain": (cq.get("weight"), cq.get("count"), cq.get("tau"), cq.get("k"), cq.get("abag_min")) == (cw, 16, 1.0, 32, amin),
    "protocol_quartet": (protocol.get("quartet_rank") or {}).get("weight") == 0.3 and "pinder_pool" in protocol,
    "parameter_count": selection["parameter_count"] == 180000,
}
ok = all(parts.values())
ref = runs / "pool5-v1-50-asym50-q03-w064-s17" / "history.jsonl"
ref_rows = [json.loads(l) for l in ref.read_text().splitlines()] if ref.is_file() else []
per = [h["train_seconds"] / h["updates"] for h in history]
ref_per = [h["train_seconds"] / h["updates"] for h in ref_rows]
print(json.dumps({"smoke_passed": ok, "failed": [k for k, v in parts.items() if not v], "chain_quartet_counts": counts, "chain_regroup": regroup,
                  "terms": [{k: l.get(k) for k in ("total", "pinder_state", "pinder_paired", "pkpdb", "pinder_within_listwise", "pinder_mask_free_edge_bce",
                                                   "pinder_quartet_softplus", "pinder_chain_softplus", "pinder_chain_accuracy", "pinder_chain_accuracy_hetero",
                                                   "pinder_chain_accuracy_homo", "pinder_chain_accuracy_abag", "pinder_chain_quartets_per_batch",
                                                   "pinder_chain_quartets_abag_per_batch")} for l in tl],
                  "train_seconds": [h["train_seconds"] for h in history], "validation_seconds": [h["validation_seconds"] for h in history],
                  "loader_wait_fraction": [h.get("loader_wait_fraction") for h in history], "protocol_chain_quartet": cq}, indent=1, default=str))
for h, l in zip(history, tl):
    print(f"[job] epoch {h['epoch']}: chain softplus {l.get('pinder_chain_softplus')}, chain accuracy {l.get('pinder_chain_accuracy')} "
          f"(hetero {l.get('pinder_chain_accuracy_hetero')}, homo {l.get('pinder_chain_accuracy_homo')}, Ab/Ag {l.get('pinder_chain_accuracy_abag')}), "
          f"quartets/batch {l.get('pinder_chain_quartets_per_batch')}, Ab/Ag quartets/batch {l.get('pinder_chain_quartets_abag_per_batch')}, "
          f"within {l.get('pinder_within_listwise')}, residue quartet {l.get('pinder_quartet_softplus')}, train {h['train_seconds']} s")
print(f"[job] step time: smoke {['%.3f' % x for x in per]} s/update (epoch 1 includes compilation); pool5 q03 (v1) epoch 1 "
      f"{('%.3f' % ref_per[0]) if ref_per else '-'} s/update, median epoch {('%.3f' % statistics.median(ref_per)) if ref_per else '-'} s/update")
for g in grads:
    if "error" in g: print(f"[job] grad norms ERROR {g}"); continue
    n = g["norms"]; rc = g["ratio_chain_over_within_encoder"]
    print(f"[job] grad norms (epoch {g['epoch']} batch {g['batch']}, after {g['updates_before']} updates, {g['seconds']} s): encoder |grad chain| {n['chain_encoder']:.4g}, "
          f"|grad quartet| {n.get('quartet_encoder', float('nan')):.4g}, |grad within| {n['within_encoder']:.4g}; chain/within {rc:.3g} (unweighted), "
          f"{g['ratio_weighted_chain_over_within_encoder']:.3g} (x{g['weights']['chain']} / x{g['weights']['within']}); quartet/within {g.get('ratio_quartet_over_within_encoder', float('nan')):.3g}; "
          f"lambda_c for parity with within x0.3 = {0.3 / rc:.3g}; aux_redge chain {n['chain_aux_redge']:.4g} vs within {n['within_aux_redge']:.4g}")
sys.exit(0 if ok else 1)
PYEOF
ok=$?; echo "[job] smoke checks exit=$ok"; [ $ok -eq 0 ] || exit $ok
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=8; sc=0
rm -rf "$A/chain" "$A/smoke-masked2" "$A"/smoke-*.log
PYTHONPATH=$CODE/jobs:$CODE/src "$PY" "$CODE/jobs/score_chain.py" "$A/chain" "smoke=$RUN" --limit 60 || sc=1
"$PY" - "$A/chain/report.json" <<'PYEOF' || sc=1
import json, sys
rep = json.load(open(sys.argv[1])); r = rep["runs"]["smoke"]; q = r["metrics"]["quartets"]
print(f"[job] score_chain: pairs {rep['pairs_per_class']}, scored {r['metrics']['complexes_scored']}, accuracy {json.dumps({k: q[k]['accuracy'] for k in q})}, "
      f"length rho {json.dumps(r['metrics']['length_spearman'])}, {r['seconds']} s")
sys.exit(0 if q["all"]["n"] > 0 and q["all"]["accuracy"] is not None else 1)
PYEOF
cd "$CODE1"; log=$A/smoke-masked2.log; trc=0
PYTHONPATH=$CODE1:$CODE1/src timeout -k 30 900 "$PY" - score_masked2.py "$A/smoke-masked2" "$R/training/gqt-production-v2/pinder/store-hotspot-v1" "smoke=$RUN" --modes unmasked > "$log" 2>&1 <<'PYEOF' || trc=$?
import runpy, sys
import pkatrain.loading as loading, pkatrain.production_train as pt
_sel = pt.selected_model
def selected_model(root, run):
    out = _sel(root, run); print(f"[smoke-trace] checkpoint loaded {run}", flush=True); return out
class Prefetcher(loading.Prefetcher):
    def __iter__(self):
        for i, item in enumerate(super().__iter__()):
            if i: print(f"[smoke-trace] processed batch {i}", flush=True)
            yield item
pt.selected_model = selected_model; loading.Prefetcher = Prefetcher
sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")
PYEOF
good=1; { [ $trc -eq 0 ] || [ $trc -eq 124 ]; } || good=0
grep -q "^\[smoke-trace\] checkpoint loaded" "$log" || good=0; grep -q "^\[smoke-trace\] processed batch" "$log" || good=0; grep -q "Traceback" "$log" && good=0
echo "[job] masked2 trace: exit=$trc checkpoint=$(grep -c '^\[smoke-trace\] checkpoint loaded' "$log") batches=$(grep -c '^\[smoke-trace\] processed batch' "$log") ok=$good"
[ $good -eq 1 ] || { tail -40 "$log" >&2; sc=1; }
echo "[job] scorer check exit=$sc"; exit $sc
