#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# quartet-rank-v1 smoke (2026-11-15): the train.sbatch recipe (asym50 dRSA, fraction 0.5) with --quartet-weight SMOKE_QW (default 1.0)
# under the trainer's --smoke (2 epochs, then validation / selection / auxiliary metrics) with --smoke-batches SMOKE_BATCHES (default
# 150 -> 300 updates). debug_quartet_grad_norms is recorded on the LAST batch of each smoke epoch (before its update: after 149 and
# 299 updates, not at init): L2 norm of the gradient of the UNWEIGHTED quartet term and of the UNWEIGHTED within listwise term over the
# encoder (every key not starting with aux_), the trunk (encoder minus head / groups) and aux_redge; ratios printed.
# Pass = 2 history rows, finite terms, pinder_quartet_softplus > 0, accuracy in [0, 1], covered > 0.5, within listwise and masked
# free edge BCE > 0, quartet_counts with general and chemical pools, 2 finite positive grad-norm records, protocol quartet_rank
# {weight SMOKE_QW, count 128}, within_listwise {0.3, drsa}, hotspot_mask {pesto, 0.5, one, 0.5}, parameter count 180000.
# Then scorer code paths on the smoke run (CPU jax): score_quartet.py --limit 40 (exit 0, report with > 0 complexes) and
# score_masked2.py --modes unmasked / score_residue.py under timeout 900 (exit 0 or 124, checkpoint loaded, >= 1 batch, no Traceback).
# Outputs: run smoke-quartet-v1-w064-<job> under training runs; $A = audits/quartet-rank-v1/smoke/{quartet, *.log} (overwritten per job).
#   sbatch -J qr_smoke smoke.sbatch
set -uo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
CODE=$S/_submission/quartet-rank-v1
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$CODE/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PKATRAIN_LOADER_WORKERS=15 PKATRAIN_PREFETCH_DEPTH=4 PYTHONFAULTHANDLER=1 GQT_PARAMETER_SCALING=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=$JAXKA_REPO/.venv/bin/python
ulimit -c 0; cd "$CODE"; echo "[job] $(date -Is) ${SLURMD_NODENAME:-}"
sha256sum "$CODE"/src/pkatrain/production_{train,loading,residue}.py "$CODE"/*.py "$0" 2>/dev/null || true
STORE=${MASK_STORE:-$PKABENCH_RUNTIME/training/gqt-production-v2/pinder/store-hotspot-v1}
SMOKE_QW=${SMOKE_QW:-1.0}; SMOKE_BATCHES=${SMOKE_BATCHES:-150}
A=$PKABENCH_RUNTIME/audits/quartet-rank-v1/smoke; mkdir -p "$A"
BASE=(--fraction 0.5 --seed 17 --shift-weight none --batch 16 --width 64 --dropout 0.1 --aux-weight 1 --aux-heads burial --orientation --dihedrals
      --variant relations --relation-maps 4,1 --all-residue-tokens --token-cutoff 15 --drop-bonded-cys --gate-weight 0.1 --gate-cys-weight 10
      --residue-edge-weight 0.1 --residue-burial-siamese --decoy-retype-weight 0 --within-weight 0.3
      --mask-oracle pesto --mask-store "$STORE" --mask-sides one --mask-prob 0.5 --mask-keep 0.5 --within-target drsa
      --quartet-weight "$SMOKE_QW" --quartet-count 128)
gpu() {
  for attempt in 1 2 3; do
    t0=$SECONDS; rc=0; "$@" || rc=$?
    [ $rc -eq 135 ] && [ $((SECONDS - t0)) -lt 120 ] || return $rc
    echo "[job] bus error at startup, attempt $attempt" >&2; sleep 20
  done
  return 135
}
if [ ! -f "$STORE/verification.json" ]; then echo "[job] store $STORE not built" >&2; exit 3; fi
RUN=smoke-quartet-v1-w064-${SLURM_JOB_ID:-local}
gpu "$PY" -m pkatrain.production_train train "$RUN" "${BASE[@]}" --smoke --smoke-batches "$SMOKE_BATCHES"
rc=$?; echo "[job] train exit=$rc"; [ $rc -eq 0 ] || exit $rc
"$PY" - "$PKABENCH_RUNTIME/training/gqt-production-v2/runs/$RUN" "$SMOKE_QW" "$SMOKE_BATCHES" <<'PYEOF'
import json, math, sys
from pathlib import Path
out = Path(sys.argv[1]); qw = float(sys.argv[2]); nb = int(sys.argv[3])
history = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
protocol = json.loads((out / "protocol.json").read_text()); selection = json.loads((out / "selection.json").read_text())
tl = [h["train_loss"] for h in history]; qr = protocol.get("quartet_rank") or {}; hm = protocol.get("hotspot_mask") or {}; wl = protocol.get("within_listwise") or {}
grads = [g for h in history for g in h.get("debug_quartet_grad_norms", [])]; counts = history[-1].get("quartet_counts") or {}
finite = all(isinstance(v, (int, float)) and math.isfinite(v) for l in tl for v in l.values() if not isinstance(v, dict))
parts = {
    "history_rows_2": len(history) == 2, "updates": all(h["updates"] == nb for h in history), "finite_terms": finite,
    "quartet_softplus_positive": all(l.get("pinder_quartet_softplus", 0) > 0 for l in tl),
    "quartet_accuracy_0_1": all(0.0 <= l.get("pinder_quartet_accuracy", -1) <= 1.0 for l in tl),
    "quartet_covered_gt_0.5": all(l.get("pinder_quartet_covered", 0) > 0.5 for l in tl),
    "within_and_masked_bce_positive": all(l.get("pinder_within_listwise", 0) > 0 and l.get("pinder_mask_free_edge_bce", 0) > 0 for l in tl),
    "quartet_counts": counts.get("complexes", 0) > 0 and counts.get("with_general", 0) > 0 and counts.get("with_chemical", 0) > 0,
    "grad_norms": len(grads) == 2 and all("error" not in g and g.get("batch") == nb and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in g["norms"].values()) for g in grads),
    "protocol_quartet": qr.get("weight") == qw and qr.get("count") == 128 and protocol.get("smoke_batches", 5) == nb,
    "within_listwise": wl.get("weight") == 0.3 and wl.get("target") == "drsa",
    "hotspot_mask": hm.get("oracle") == "pesto" and hm.get("keep") == 0.5 and hm.get("sides") == "one" and hm.get("prob") == 0.5,
    "parameter_count": selection["parameter_count"] == 180000, "width_64": protocol["architecture"]["width"] == 64,
}
ok = all(parts.values())
print(json.dumps({"smoke_passed": ok, "failed": [k for k, v in parts.items() if not v], "quartet_counts": counts,
                  "terms": [{k: l.get(k) for k in ("total", "pinder_state", "pinder_paired", "pkpdb", "pinder_within_listwise", "pinder_mask_free_edge_bce",
                                                   "pinder_quartet_softplus", "pinder_quartet_accuracy", "pinder_quartet_covered")} for l in tl],
                  "weighted": [{"within x0.3": 0.3 * l.get("pinder_within_listwise", 0), f"quartet x{qw}": qw * l.get("pinder_quartet_softplus", 0)} for l in tl],
                  "train_seconds": [h["train_seconds"] for h in history], "validation_seconds": [h["validation_seconds"] for h in history],
                  "loader_wait_fraction": [h.get("loader_wait_fraction") for h in history], "protocol_quartet_rank": qr}, indent=1))
for h, l in zip(history, tl):
    print(f"[job] epoch {h['epoch']}: quartet softplus {l.get('pinder_quartet_softplus'):.4f}, quartet accuracy {l.get('pinder_quartet_accuracy'):.4f}, "
          f"covered {l.get('pinder_quartet_covered'):.3f}, within listwise {l.get('pinder_within_listwise'):.4f}, train {h['train_seconds']} s")
for g in grads:
    if "error" in g: print(f"[job] grad norms ERROR {g}"); continue
    n = g["norms"]
    print(f"[job] grad norms (epoch {g['epoch']} batch {g['batch']}, after {g['updates_before']} updates, {g['seconds']} s): encoder |grad quartet| {n['quartet_encoder']:.4g} "
          f"vs |grad within| {n['within_encoder']:.4g} -> ratio {g['ratio_quartet_over_within_encoder']:.3g} (unweighted), "
          f"{g['ratio_weighted_quartet_over_within_encoder']:.3g} (x{g['weights']['quartet']} / x{g['weights']['within']}); trunk ratio {g['ratio_quartet_over_within_trunk']:.3g}; "
          f"aux_redge {n['quartet_aux_redge']:.4g} vs {n['within_aux_redge']:.4g}")
sys.exit(0 if ok else 1)
PYEOF
ok=$?; echo "[job] smoke checks exit=$ok"; [ $ok -eq 0 ] || exit $ok
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=8 PYTHONPATH=$CODE:$CODE/src; sc=0
rm -rf "$A/quartet" "$A/smoke-masked2" "$A/smoke-residue" "$A"/smoke-*.log
"$PY" score_quartet.py "$A/quartet" "smoke=$RUN" --limit 40 || sc=1
"$PY" - "$A/quartet/report.json" <<'PYEOF' || sc=1
import json, sys
r = json.load(open(sys.argv[1]))["runs"]["smoke"]; o = r["metrics"]["overall"]
print(f"[job] score_quartet: {r['complexes_iterated']} iterated, {o['all']['complexes']} with quartets, {r['complexes_without_quartets']} without, "
      f"accuracy all {o['all']['accuracy']}, chemical {o['chemical']['accuracy']}, {r['seconds_base']} s")
sys.exit(0 if o["all"]["complexes"] > 0 and o["all"]["accuracy"] is not None else 1)
PYEOF
trace() {  # trace LOG SCRIPT ARGS...: the scorer runs unchanged through a tracer (selected_model / Prefetcher), runpy as __main__
  local log=$1; shift; local rc=0
  timeout -k 30 900 "$PY" - "$@" > "$log" 2>&1 <<'PYEOF' || rc=$?
import runpy, sys
import pkatrain.loading as loading, pkatrain.production_train as pt
_sel = pt.selected_model
def selected_model(root, run):
    out = _sel(root, run); print(f"[smoke-trace] checkpoint loaded {run}", flush=True); return out
class Prefetcher(loading.Prefetcher):
    def __iter__(self):
        for i, item in enumerate(super().__iter__()):
            if i: print(f"[smoke-trace] processed batch {i} ({len(item[0])} complexes in the next)", flush=True)
            yield item
pt.selected_model = selected_model; loading.Prefetcher = Prefetcher
sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")
PYEOF
  local ok=1
  { [ $rc -eq 0 ] || [ $rc -eq 124 ]; } || ok=0
  grep -q "^\[smoke-trace\] checkpoint loaded" "$log" || ok=0
  grep -q "^\[smoke-trace\] processed batch" "$log" || ok=0
  grep -q "Traceback" "$log" && ok=0
  echo "[job] trace $(basename "$log"): exit=$rc checkpoint=$(grep -c '^\[smoke-trace\] checkpoint loaded' "$log") batches=$(grep -c '^\[smoke-trace\] processed batch' "$log") ok=$ok"
  [ $ok -eq 1 ] || { tail -40 "$log" >&2; return 1; }
}
trace "$A/smoke-masked2.log" score_masked2.py "$A/smoke-masked2" "$STORE" "smoke=$RUN" --modes unmasked || sc=1
trace "$A/smoke-residue.log" score_residue.py "$A/smoke-residue" "smoke=$RUN" || sc=1
echo "[job] scorer check exit=$sc"; exit $sc
