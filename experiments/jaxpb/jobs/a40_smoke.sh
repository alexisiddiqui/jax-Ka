#!/bin/bash
# A40 stage 0: trivial Pallas Triton kernel on sm_86 + persistent-cache tarball write/read (Pallas executables cached).
set -u
O=$A40/m0; mkdir -p $O
$PY -c "import jax; d=jax.devices()[0]; print('jax', jax.__version__, d, getattr(d,'compute_capability',None))"
$PY jobs/m0_cachecheck.py write $O/cc_pallas.tar $O/cachecheck.write.json
$PY jobs/m0_cachecheck.py read $O/cc_pallas.tar $O/cachecheck.read.json
