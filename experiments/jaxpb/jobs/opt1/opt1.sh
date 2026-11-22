#!/bin/bash
# Submitted via a40.sbatch or coulson_cpu.sbatch; all writes stay in opt1/.
set -euo pipefail
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
T=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline
PY=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python
if [ "$1" = tests ]; then
  export PYTHONPATH="$PWD/src:$T/../jaxpb-pkgtest/pytestshim"
  $PY tests/test_api.py "$T/opt1/unit-api"
  $PY tests/test_assemble.py local
  $PY tests/test_titrate.py local
  $PY tests/test_records.py "$T"/m0/shards/m0_goldens_0{0,1,2}.npz
  echo '{"passed": true, "modes": ["test_api", "test_assemble local", "test_titrate local", "test_records"]}' > "$T/opt1/unit-tests.json"
else
  $PY jobs/opt1.py "$@"
fi
