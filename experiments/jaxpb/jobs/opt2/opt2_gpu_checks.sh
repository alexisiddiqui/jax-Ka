#!/bin/bash
set -euo pipefail
T=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline
PY=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python
export PYTHONPATH="$PWD/jobs/a40site:$PWD/src:$T/../jaxpb-pkgtest/pytestshim"
$PY -m pytest -q tests/test_opt2.py
$PY tests/test_sor_pallas.py --n 17 --W 8 --gpu
$PY -c 'import json,os; json.dump(dict(passed=True,job=os.environ["SLURM_JOB_ID"],frozen=os.getcwd()),open("'"$T"'/opt2/gpu-unit-tests.json","w"),indent=2)'
