#!/usr/bin/env bash
#SBATCH --partition=generalaccess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/abdata-v5/logs/%x-%j.out
# abdata-v5 pool-v5. STEP=prepare: candidates + validation + MMseqs2 screens (24 CPUs as pool-v4 job 748141), the pool-v4
# reproduction check and the ANARCI input. STEP=anarci: number_cdrs.py. STEP=finish: the CDR screen -> pool-v5.
#   sbatch -J v5-prepare --export=ALL,STEP=prepare pool.sbatch
set -euo pipefail
source /home/coulson/oc/lina4225/_runtime/abdata-v5/jobs/env.sh
OUT=$W/pool; mkdir -p "$OUT"; cd "$W/jobs"; echo "[job] $(date -Is) $SLURMD_NODENAME step=$STEP"; sha256sum "$W"/jobs/*.py "$W"/src/pkabench/training_pools.py
case "$STEP" in
  prepare) "$PYPOOL" pool_v5.py prepare "$OUT" ;;
  anarci)  PATH=$ANARCI_ENV/bin:$PATH "$ANARCI_ENV/bin/python" number_cdrs.py "$OUT/anarci-input.fasta" "$OUT/anarci.json" "$SLURM_CPUS_PER_TASK" ;;
  finish)  "$PYPOOL" pool_v5.py finish "$OUT" "$OUT/anarci.json" ;;
  ledger)  "$ANARCI_ENV/bin/python" ledger.py "$W/ledger" ;;     # pandas + pyarrow live in the abdata-v5 anarci env
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "[job] $(date -Is) done"
