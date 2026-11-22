#!/bin/bash
set -euo pipefail
T=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline
PY=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python
export PYTHONPATH="$PWD/src:$T/../jaxpb-pkgtest/pytestshim"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
$PY -m pytest -q tests/test_opt1.py tests/test_opt2.py
$PY tests/test_api.py "$T/opt2/unit-api-${SLURM_JOB_ID}"
$PY tests/test_assemble.py local
$PY tests/test_titrate.py local
$PY tests/test_records.py "$T"/m0/shards/m0_goldens_0{0,1,2}.npz
$PY -c 'import json,os; json.dump(dict(passed=True,job=os.environ["SLURM_JOB_ID"],frozen=os.getcwd()),open("'"$T"'/opt2/unit-tests.json","w"),indent=2)'
