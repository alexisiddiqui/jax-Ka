#!/bin/bash
# A40 stage 4: M4 closing test vs the GH200 v2 g_results (+ cachecheck in a new process sharing the node-local cache);
# then the v2 prototype gmc.py on this A40 (G50k, G200k) and the same M4 test vs the A40 v2 counts (like with like).
set -u
I=$A40/inputs; O=$A40/m4; mkdir -p $O/vs_gh200 $O/vs_v2a40 $O/v2a40R/audits/mf-surrogate-v2 $CC
ln -sf $B/tierB/mf-surrogate-v2/states-v2.npz $O/v2a40R/audits/mf-surrogate-v2/states-v2.npz
R=$I/Rgh200 JAX_COMPILATION_CACHE_DIR=$CC JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 $PY tests/test_titrate.py m4 $O/vs_gh200
R=$I/Rgh200 JAX_COMPILATION_CACHE_DIR=$CC JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 $PY tests/test_titrate.py cachecheck $O/vs_gh200
echo "[a40] cache files $(find $CC -type f | wc -l)"
(cd $CODE/v2mc && R=$O/v2a40R $PY gmc.py run G50k,G200k)
(cd $CODE/v2mc && R=$O/v2a40R $PY gmc.py test) || true
R=$O/v2a40R $PY tests/test_titrate.py m4 $O/vs_v2a40
