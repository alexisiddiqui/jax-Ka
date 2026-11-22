#!/bin/bash
set -euo pipefail
T=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline
PY=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python
export PYTHONPATH="$PWD/src:$T/../jaxpb-pkgtest/pytestshim"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
$PY -m pytest -q tests/test_opt1.py
$PY tests/test_api.py "$T/opt1/edge-api-${SLURM_JOB_ID}"
echo '{"passed": true, "checks": ["CSR tie and overflow", "byte budget", "face setup parity", "last class overflow", "lazy import", "bounded driver summary"]}' > "$T/opt1/edge-tests.json"
