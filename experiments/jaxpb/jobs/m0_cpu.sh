#!/bin/bash
# M0 CPU stand-in of the cache-tarball check (Coulson, JAX 0.11.1 CPU; Pallas in interpret mode -> plain XLA, so this does
# NOT establish Pallas executable caching, which needs the GPU job m0_gpu.sh).
set -u
T=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline; O=$T/m0/cpu; mkdir -p $O
PY=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python
$PY jobs/m0_cachecheck.py write $O/cc_cpu.tar $O/cachecheck.write.json
$PY jobs/m0_cachecheck.py read $O/cc_cpu.tar $O/cachecheck.read.json
rm -f $O/cc_cpu.tar
