#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python jobs/opt1b_tune.py
