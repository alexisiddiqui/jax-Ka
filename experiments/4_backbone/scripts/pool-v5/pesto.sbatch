#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# abdata-v5 (2026-11-16): PeSTo hotspot oracle for the pool-v5 Ab/Ag newcomers -> $P/store-hotspot-v5abag (store-hotspot-v2
# schema). Pipeline = hotspot-oracle-v1 / pesto-full-v1: run_pesto_gpu.py UNCHANGED (sha256 a97bab8c...), PeSTo ba651aa
# i_v4_1_2021-09-07_11-21 + gemmi 0.7.5 from $S/_submission/hotspot-oracle-v1 (read only, unpacked node-local), pkai torch venv.
# Chains were exported on Coulson by export_chains_v5.py (= export_chains.py with the selection "every record of the runtime")
# from the abdata-v5 records (chains-new.zip: newcomers; chains-parity.zip: 35 pool-v4 structures rebuilt on Coulson).
# build_store_v5.py = build_store.py with own-model fields NaN (not computed; recipe reads only pesto) on PKABENCH_RUNTIME=$P/abag-v5.
# Parity: PeSTo of the 35 rebuilt pool-v4 structures vs store-hotspot-v2 (compare_hotspot.py).
#   sbatch -J v5_pesto /scratch/u6xq/hussain.u6xq/_submission/abdata-v5/pesto.sbatch
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
CODE=$S/_submission/abdata-v5; PSRC=$S/_submission/hotspot-oracle-v1; R=$JAXKA_RUNTIME/pkabench; A=$R/audits/abdata-v5
P=$R/training/gqt-production-v2/pinder; X=$P/abag-v5; STORE=$P/store-hotspot-v5abag; SRC=$S/_submission/quartet-rank-v1b/src
export PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=$JAXKA_REPO/.venv/bin/python; TORCHPY=$S/_install/jax-Ka-venvs/pkai/bin/python
ulimit -c 0; cd "$CODE"; echo "[job] $(date -Is) ${SLURMD_NODENAME:-}"; sha256sum "$CODE"/*.py "$0" "$PSRC/pesto-ba651aa-min.tgz"
[ -e "$STORE" ] && { echo "[job] $STORE exists" >&2; exit 3; }
for z in chains-new chains-parity; do [ -f "$A/$z.zip" ] || { echo "missing $A/$z.zip" >&2; exit 3; }; done
L=$(mktemp -d /tmp/v5pesto-XXXX); trap 'rm -rf "$L"' EXIT
tar xzf "$PSRC/pesto-ba651aa-min.tgz" -C "$L"; "$TORCHPY" -m zipfile -e "$PSRC"/gemmi-0.7.5-*.whl "$L/site"
for z in new parity; do
  [ -f "$A/pesto-$z.npz" ] || PYTHONPATH=$L/site PESTO_LOCAL=$L PESTO_WORKERS=64 "$TORCHPY" "$CODE/run_pesto_gpu.py" "$A/chains-$z.zip" "$L/pesto-ba651aa" "$A/pesto-$z"
done
# own-model oracle not computed: NaN arrays in build_store's own npz format
JAX_PLATFORMS=cpu "$PY" - "$A/pesto-new" "$A/own-nan" <<'PYEOF'
import sys, numpy as np
f = np.load(sys.argv[1] + ".npz"); n = np.diff(f["offsets"]); tot = int(n.sum())
np.savez(sys.argv[2] + ".npz", ids=f["ids"], offsets=f["offsets"], own_edgemax=np.full(tot, np.nan, np.float32), own_dburial=np.full(tot, np.nan, np.float32),
         run=np.asarray("none (abdata-v5: own-model oracle not computed; NaN)"))
print("own-nan", len(f["ids"]), tot)
PYEOF
JAX_PLATFORMS=cpu PKABENCH_RUNTIME=$X PYTHONPATH=$SRC "$PY" "$CODE/build_store_v5.py" "$A/pesto-new" "$A/own-nan" "$STORE"
JAX_PLATFORMS=cpu PYTHONPATH=$SRC "$PY" "$CODE/compare_hotspot.py" "$A/pesto-parity" "$P/store-hotspot-v2" "$A/pesto-parity-vs-v2.json"
sha256sum "$STORE"/* | tee "$A/store-hotspot-v5abag.sha256"
echo "[job] $(date -Is) done"
