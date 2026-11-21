#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# quartet-rank-v1b checks (2026-11-16), run AFTER the pool-v5 data and store-hotspot-v5abag exist.
# 1 loader (CPU jax, no model step; loader_check.py): the q03 recipe on pool-v4 under quartet-rank-v1 and under v1b (flags off) ->
#   40 training batches hashed, must be bitwise equal; then v1b on pool-v5 -> 40 batches + every batch holding a newcomer: newcomers
#   load with residue + hotspot fields and no hotspot fallback.
# 2 smoke (GPU): q03 on pool-v5, --smoke --smoke-batches 40: 2 epochs + validation, finite terms.
# Outputs: $R/audits/abdata-v5/check/*.json; runs abdata-v5-{lc-*,smoke}-<job> (protocol only for lc-*).
#   sbatch -J v5_check /scratch/u6xq/hussain.u6xq/_submission/quartet-rank-v1b/jobs/check.sbatch
set -uo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/quartet-rank-v1b/jobs/common.sh
A=$R/audits/abdata-v5/check; mkdir -p "$A"; J=${SLURM_JOB_ID:-local}; cd "$CODE/jobs"
echo "[job] $(date -Is) ${SLURMD_NODENAME:-}"; sha256sum "$CODE"/src/pkatrain/production_{train,extra,loading,residue}.py "$CODE"/jobs/* 2>/dev/null
V1=$S/_submission/quartet-rank-v1
for f in "$R/training/pool-v5/pinder-manifest.json" "$P/abag-v5/training/gqt-production-v2/pinder/store-v2/verification.json" "$P/store-hotspot-v5abag/verification.json"; do
  [ -f "$f" ] || { echo "[job] missing $f" >&2; exit 3; }; done
Q=(--quartet-weight 0.3 --quartet-count 128 --seed 17)
JAX_PLATFORMS=cpu PYTHONPATH=$V1/src "$PY" loader_check.py v4 abdata-v5-lc-v4-q1-$J "$A/lc-v4-q1.json" $(recipe "$P/store-hotspot-v1") "${Q[@]}"; r1=$?
JAX_PLATFORMS=cpu PYTHONPATH=$CODE/src "$PY" loader_check.py v4 abdata-v5-lc-v4-q1b-$J "$A/lc-v4-q1b.json" $(recipe "$P/store-hotspot-v1") "${Q[@]}"; r2=$?
JAX_PLATFORMS=cpu PYTHONPATH=$CODE/src "$PY" loader_check.py v5 abdata-v5-lc-v5-q1b-$J "$A/lc-v5-q1b.json" $(recipe "$STORE_V2") "${Q[@]}" "${POOL5[@]}"; r3=$?
"$PY" - "$A" <<'PYEOF'
import json, sys
a = sys.argv[1]; x = json.load(open(f"{a}/lc-v4-q1.json")); y = json.load(open(f"{a}/lc-v4-q1b.json")); z = json.load(open(f"{a}/lc-v5-q1b.json"))
same = x["batches"] == y["batches"] and len(x["batches"]) == 40
out = {"v4_v1_vs_v1b_bitwise_equal_40_batches": same, "v5_extra_records_in_epoch": z["extra_records_in_epoch"], "v5_extra_checked": z["extra_checked"],
       "v5_extra_fallbacks": z["extra_fallbacks"], "v5_extra_missing_fields": z["extra_missing_fields"], "v5_train_records_0.5": z["train_records_0.5"],
       "v4_train_records_0.5": x["train_records_0.5"]}
out["passed"] = same and z["extra_checked"] == z["extra_records_in_epoch"] > 0 and z["extra_fallbacks"] == 0 and z["extra_missing_fields"] == 0
json.dump(out, open(f"{a}/loader-check.json", "w"), indent=1); print(json.dumps(out))
PYEOF
r4=$?
echo "[job] loader checks exit $r1 $r2 $r3 $r4"
gpu() { for attempt in 1 2 3; do t0=$SECONDS; rc=0; "$@" || rc=$?; [ $rc -eq 135 ] && [ $((SECONDS - t0)) -lt 120 ] || return $rc; sleep 20; done; return 135; }
PYTHONPATH=$CODE/src gpu "$PY" -m pkatrain.production_train train abdata-v5-smoke-$J $(recipe "$STORE_V2") "${Q[@]}" "${POOL5[@]}" --smoke --smoke-batches 40; r5=$?
echo "[job] smoke exit $r5"; echo "[job] $(date -Is) done"; [ $r1$r2$r3$r4$r5 = 00000 ]
