#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
# abdata-v5 publish (2026-11-18): pool-v5 built on Coulson (amd96 jobs 802617-802620), files relayed and sha256-checked
# (audits/abdata-v5/coulson-transfer.sha256). Same manifest + publish commands as v5pool.sbatch STEP=finish / ledger.
set -euo pipefail
S=/scratch/u6xq/hussain.u6xq; CODE=$S/_submission/abdata-v5; RT=$S/_runtime/jax-Ka/pkabench; P=$RT/training/gqt-production-v2/pinder
A=$RT/audits/abdata-v5; OUT=$A/pool; PY=$S/jax-Ka/.venv/bin/python
cd $A && sha256sum -c coulson-transfer.sha256
$PY -c "import json,sys; c=json.load(open(\"$OUT/repro-v4-check.json\")); sys.exit(0 if c[\"passed\"] else 4)"
X=$P/abag-v5/training/gqt-production-v2/pinder; D=$RT/training/pool-v5
$PY "$CODE/build_manifest_v5.py" "$P/manifest-v1.json" "$OUT/pool-v5/pinder.tsv" "$X/manifest-v1.json" "$X/store-v2/verification.json" "$OUT/pinder-manifest.json" "$D/pinder.tsv" "$X/store-v2"
[ -e "$D" ] && { echo "$D exists" >&2; exit 3; }
mkdir -p "$D.partial"; cp "$OUT"/pool-v5/{pinder.tsv,pinder-val.tsv,pool-v5.json} "$OUT"/{pinder-manifest.json,cdr-identity.tsv,funnel.json,anarci.json,repro-v4-check.json} "$D.partial/"
cmp "$D.partial/pinder-val.tsv" "$RT/training/pool-v4/pinder-val.tsv"
(cd "$D.partial" && sha256sum * > SHA256SUMS); mv "$D.partial" "$D"; cat "$D/SHA256SUMS"
L=$RT/audits/pool-v5-ledger; [ -e "$L" ] && { echo "$L exists" >&2; exit 3; }
cp -r "$A/ledger-coulson" "$L.partial"; (cd "$L.partial" && sha256sum * > SHA256SUMS); mv "$L.partial" "$L"; cat "$L/SHA256SUMS"
echo done
