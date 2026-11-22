#!/bin/bash
set -euo pipefail
bash jobs/opt1_checks.sh
bash jobs/opt1.sh tests
PY=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python
$PY jobs/opt1_summary_check.py
