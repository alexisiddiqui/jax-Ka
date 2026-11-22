#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python
$PY jobs/opt2.py "$@"
