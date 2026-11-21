# quartet-rank-v1b jobs: the asym50 dRSA recipe (drsa-within-v1 train.sbatch step drsa = quartet-rank-v1 train.sbatch BASE) and
# the pool-v5 flags. Sourced by check.sbatch / train.sbatch.
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
CODE=$S/_submission/quartet-rank-v1b; R=$JAXKA_RUNTIME/pkabench; P=$R/training/gqt-production-v2/pinder
export PKABENCH_RUNTIME=$R OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PKATRAIN_LOADER_WORKERS=15 PKATRAIN_PREFETCH_DEPTH=4
export PYTHONFAULTHANDLER=1 GQT_PARAMETER_SCALING=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=$JAXKA_REPO/.venv/bin/python
STORE_V2=$P/store-hotspot-v2          # superset of store-hotspot-v1 (bitwise equal on shared records, pesto-full-v1 verify): pool-v5's 50% subset
POOL5=(--pinder-manifest "$R/training/pool-v5/pinder-manifest.json" --pinder-extra "$P/abag-v5" --mask-store-extra "$P/store-hotspot-v5abag")
recipe() {  # recipe MASK_STORE
  echo --fraction 0.5 --shift-weight none --batch 16 --width 64 --dropout 0.1 --aux-weight 1 --aux-heads burial --orientation --dihedrals \
       --variant relations --relation-maps 4,1 --all-residue-tokens --token-cutoff 15 --drop-bonded-cys --gate-weight 0.1 --gate-cys-weight 10 \
       --residue-edge-weight 0.1 --residue-burial-siamese --decoy-retype-weight 0 --within-weight 0.3 \
       --mask-oracle pesto --mask-store "$1" --mask-sides one --mask-prob 0.5 --mask-keep 0.5 --within-target drsa
}
