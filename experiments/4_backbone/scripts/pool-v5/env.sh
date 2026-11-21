# abdata-v5 job environment (Coulson): common.sh equivalent with the abdata-v5 code copy (never ~/jax-Ka's working tree)
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_MEM_PER_CPU:-0} == 2048 ]] || { echo 'Requires a Slurm allocation with --mem-per-cpu=2G' >&2; exit 1; }
export W=/home/coulson/oc/lina4225/_runtime/abdata-v5
export PKABENCH_RUNTIME=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$W/src:$W/jobs"
export TMPDIR="$(mktemp -d /tmp/abdata-v5-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}.XXXXXX)"
export XDG_CACHE_HOME="${TMPDIR}/xdg" MPLCONFIGDIR="${TMPDIR}/mpl"
trap 'rm -rf -- "${TMPDIR}"' EXIT
PYPOOL=$PKABENCH_RUNTIME/envs/train-shared-v1/bin/python      # the env pool-v4 was built with (job 748141)
PYBUILD=$PKABENCH_RUNTIME/envs/ogqt-main-7c3e3e9/bin/python   # numpy 2.5.4 / scipy 1.18.1 / biotite 1.7.1 / zstandard 0.25.0 (~ Isambard venv)
ANARCI_ENV=/home/coulson/oc/lina4225/_runtime/envs/anarci-v1
