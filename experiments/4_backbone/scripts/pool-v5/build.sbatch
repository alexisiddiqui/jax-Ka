#!/usr/bin/env bash
#SBATCH --partition=generalaccess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/abdata-v5/logs/%x-%j.out
# abdata-v5: training records + side tables for the structures listed in RT (setup_rt.py), with the ORIGINAL builders
# of the drsa-within-v1 snapshot (sha256-identical to Isambard $S/_submission/drsa-within-v1/src), unchanged:
#   production_graphs ids/build/pack -> store-v2; production_loading manifest -> manifest-v1.json;
#   production_inputs build-backbone -> backbone-v1; production_sidechains build -> sidechains-v1;
#   production_aux build -> contacts-v1; production_residue build -> residue-rsa-v1.
#   sbatch -J v5-build-new --export=ALL,RT=/home/coulson/oc/lina4225/_runtime/abdata-v5/rt-new build.sbatch
set -euo pipefail
source /home/coulson/oc/lina4225/_runtime/abdata-v5/jobs/env.sh
export PKABENCH_RUNTIME=${RT:?}            # the mini runtime (builders resolve every path under it)
N=$SLURM_CPUS_PER_TASK; cd "$W/jobs"; echo "[job] $(date -Is) $SLURMD_NODENAME rt=$RT"
"$PYBUILD" -c "import numpy, scipy, biotite, zstandard; print(numpy.__version__, scipy.__version__, biotite.__version__, zstandard.__version__)"
sha256sum "$W"/src/pkatrain/production_{graphs,loading,inputs,sidechains,aux,residue}.py "$W"/src/pkatrain/{gqt_paired_pinder,site_graph_data}.py "$W"/src/pkanet/graph.py "$W"/src/jaxpropka/topology.py
P=$RT/training/gqt-production-v2/pinder
"$PYBUILD" -m pkatrain.production_graphs ids pinder
for t in $(seq 0 $((N - 1))); do "$PYBUILD" -m pkatrain.production_graphs build pinder $t $N > "$W/logs/$SLURM_JOB_ID-shard-$t.log" 2>&1 & done
fail=0; for j in $(jobs -p); do wait $j || fail=1; done
[ $fail = 0 ] || { echo "shard failures:"; cat "$P"/shards/*.failures.json 2>/dev/null | head -c 4000; exit 1; }
"$PYBUILD" -m pkatrain.production_graphs pack pinder
"$PYBUILD" -m pkatrain.production_loading manifest pinder
"$PYBUILD" -m pkatrain.production_inputs build-backbone pinder --workers $N
"$PYBUILD" -m pkatrain.production_sidechains build pinder --workers $N
"$PYBUILD" -m pkatrain.production_aux build --workers $N
"$PYBUILD" -m pkatrain.production_residue build --workers $N
echo "[job] $(date -Is) done"
