#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# abdata-v5 pool-v5 on Isambard (moved from Coulson 2026-11-17: generalaccess drained). CPU only (GPU allocated per house rule, unused).
# Inputs: the Coulson runtime files the pool builder and the ledger read, as ONE archive (make_bundle.sh, sha256 checked here),
# extracted node-local (/tmp or /dev/shm, never scratch) and used as PKABENCH_RUNTIME; mmseqs = $MMSEQS (aarch64) linked at the
# builder's relative path. Code: pool_v5.py / number_cdrs.py / ledger.py / build_manifest_v5.py here + pkabench / pkatrain from the
# drsa-within-v1 snapshot (sha256-identical to the Coulson copy used for the records). Python: the abdata-v5 ANARCI env.
#   STEP=prepare  candidates, validation, MMseqs2 screens (24 threads as pool-v4 job 748141) + pool-v4 reproduction check
#   STEP=anarci   ANARCI IMGT numbering of the screen sequences
#   STEP=finish   CDR screen -> pool-v5 tables; pinder-manifest.json (store-v2 + abag-v5 records); publish to $R/training/pool-v5
#   STEP=ledger   per-dimer Ab/Ag ledger + cluster funnel + fixable-cluster plan -> $R/audits/pool-v5-ledger
#   sbatch -J v5_prepare --export=ALL,STEP=prepare v5pool.sbatch   (then anarci / finish / ledger with --dependency=afterok)
set -euo pipefail
S=/scratch/u6xq/hussain.u6xq; CODE=$S/_submission/abdata-v5; SNAP=$S/_submission/drsa-within-v1/src
RT=$S/_runtime/jax-Ka/pkabench; P=$RT/training/gqt-production-v2/pinder; A=$RT/audits/abdata-v5; OUT=$A/pool
BUNDLE=$A/pool-inputs; ENV=$S/_envs/anarci-v1; MMSEQS=${MMSEQS:-$(cat "$CODE/MMSEQS_PATH")}
export PATH=$ENV/bin:$PATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$CODE:$SNAP
PY=$ENV/bin/python; STEP=${STEP:?}
echo "[job] $(date -Is) ${SLURMD_NODENAME:-} step=$STEP"; sha256sum "$CODE"/*.py "$0"; mkdir -p "$OUT"
# node-local runtime: >= 25 GB free on /tmp, else /dev/shm (counts against the job's 144 GB)
need=25000000; base=/tmp; [ "$(df -Pk /tmp | awk 'NR==2{print $4}')" -ge $need ] || base=/dev/shm
L=$(mktemp -d "$base/abdata-v5-XXXX"); trap 'rm -rf "$L"' EXIT
(cd "$BUNDLE" && sha256sum -c pool-inputs.sha256)
zstd -dc "$BUNDLE/pool-inputs.tar.zst" | tar -xf - -C "$L"
mkdir -p "$L/audits/foldbench-full-v1/tools/mmseqs/bin"; ln -s "$MMSEQS" "$L/audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs"; "$MMSEQS" version
export PKABENCH_RUNTIME=$L TMPDIR=$L/tmp; mkdir -p "$TMPDIR"
case "$STEP" in
  prepare) POOL_WORKERS=24 "$PY" "$CODE/pool_v5.py" prepare "$OUT" ;;
  anarci)  "$PY" "$CODE/number_cdrs.py" "$OUT/anarci-input.fasta" "$OUT/anarci.json" 64 ;;
  finish)
    "$PY" -c "import json,sys; c=json.load(open('$OUT/repro-v4-check.json')); print(c); sys.exit(0 if c['passed'] else 4)" \
      || { echo "[job] pool-v4 NOT reproduced with the rule off: stop (see repro-v4-check.json)" >&2; exit 4; }
    "$PY" "$CODE/pool_v5.py" finish "$OUT" "$OUT/anarci.json"
    X=$P/abag-v5/training/gqt-production-v2/pinder; D=$RT/training/pool-v5
    "$PY" "$CODE/build_manifest_v5.py" "$P/manifest-v1.json" "$OUT/pool-v5/pinder.tsv" "$X/manifest-v1.json" "$X/store-v2/verification.json" \
        "$OUT/pinder-manifest.json" "$D/pinder.tsv" "$X/store-v2"
    [ -e "$D" ] && { echo "[job] $D exists" >&2; exit 3; }
    mkdir -p "$D.partial"; cp "$OUT"/pool-v5/{pinder.tsv,pinder-val.tsv,pool-v5.json} "$OUT"/{pinder-manifest.json,cdr-identity.tsv,funnel.json,anarci.json,repro-v4-check.json} "$D.partial/"
    cmp "$D.partial/pinder-val.tsv" "$RT/training/pool-v4/pinder-val.tsv"   # validation identical to pool-v4
    (cd "$D.partial" && sha256sum * > SHA256SUMS); mv "$D.partial" "$D"; cat "$D/SHA256SUMS" ;;
  ledger)
    D=$RT/audits/pool-v5-ledger; [ -e "$D" ] && { echo "[job] $D exists" >&2; exit 3; }
    POOL_OUT=$OUT "$PY" "$CODE/ledger.py" "$D.partial"; (cd "$D.partial" && sha256sum * > SHA256SUMS); mv "$D.partial" "$D"; cat "$D/SHA256SUMS" ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "[job] $(date -Is) done"
