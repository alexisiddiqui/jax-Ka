#!/usr/bin/env bash
#SBATCH --partition=generalaccess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/abdata-v5/logs/%x-%j.out
# abdata-v5 after the record builds. STEP=export RT=<mini runtime> NAME=<new|parity>: PeSTo chain export (export_chains_v5.py)
# -> $W/pesto/chains-NAME.zip. STEP=parity: compare_parity.py on rt-parity vs the Isambard reference rows.
set -euo pipefail
source /home/coulson/oc/lina4225/_runtime/abdata-v5/jobs/env.sh
cd "$W/jobs"; echo "[job] $(date -Is) $SLURMD_NODENAME step=$STEP"; sha256sum "$W"/jobs/*.py
case "$STEP" in
  export) mkdir -p "$W/pesto"; PKABENCH_RUNTIME=${RT:?} "$PYBUILD" -W ignore export_chains_v5.py "$W/pesto/chains-${NAME:?}.zip" ;;
  parity) PKABENCH_RUNTIME=$W/rt-parity "$PYBUILD" compare_parity.py "$W/rt-parity" "$W/parity/parity-ref.npz" "$W/parity/parity-records.json" ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "[job] $(date -Is) done"
