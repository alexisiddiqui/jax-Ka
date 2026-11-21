#!/bin/bash
# M0 GPU checks in one job (run via isambard.sbatch <code> bash jobs/m0_gpu.sh): goldens pack, v2 repro with a fresh
# cache, v2 repro of one state from the cache tarball in a new process, Pallas cache check (write + read).
set -u
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
PY=$JAXKA_REPO/.venv/bin/python
R=$S/_runtime/jax-Ka/pkabench; O=$R/audits/jaxpb-pipeline/m0; mkdir -p $O
V1=$R/audits/jaxpb-pilot-v1/ref; V2=$R/audits/jaxpb-pilot-v2/out; V2C=$S/_submission/jaxpb-pilot-v2
# goldens_pilot5.npz was built on Coulson (jobs/m0_goldens.py, sha256 ec537e59...) and relayed with sha256 checks
$PY jobs/m0_cachecheck.py write $O/cc_pallas.tar $O/cachecheck.write.json
$PY jobs/m0_cachecheck.py read $O/cc_pallas.tar $O/cachecheck.read.json
$PY jobs/m0_v2repro.py $V2C $V1 $V2 $O/v2repro run1 --cache $O/cc_v2.tar
$PY jobs/m0_v2repro.py $V2C $V1 $V2 $O/v2repro run2cache --cache $O/cc_v2.tar --states d0d5c619bd23b13d_B,d0d5c619bd23b13d_A
rm -f $O/cc_v2.tar
