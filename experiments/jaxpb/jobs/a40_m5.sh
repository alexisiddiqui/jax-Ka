#!/bin/bash
# M5 on the A40 (run via jobs/a40.sbatch <frozen code> jobs/a40_m5.sh <stage>). Outputs $A40/m5/<stage>.
#  detA : part A -- stage probe, 3 passes + fresh process (from the tarball) + fresh process (no cache, new compiles),
#         then the global-flag reference (frozen M1 code a40-2026-11-19c, XLA deterministic ops) for cause/cost.
#  smoke: part B -- Triton SOR kernel vs XLA pool on synthetic grids (n 17 and 81) + one pilot state both backends.
#  m5   : part B closing -- determinism (3 passes + fresh process) with the Pallas backend on 5 pilot + 10 M0 states,
#         then XLA reference vs Pallas on all 5 pilot + 99 M0 states (sweeps, energies, L2, timing).
set -u
ST=$1; O=$A40/m5/$ST; mkdir -p $O; SH=$T/m0/shards; G=$T/m0/pilot/goldens_pilot5.npz
SHS="$SH/m0_goldens_00.npz $SH/m0_goldens_01.npz $SH/m0_goldens_02.npz"
case $ST in
detA)
  rm -f $O/cc.tar
  $PY tests/test_determinism.py stages $O --goldens $G --tag det
  $PY tests/test_determinism.py passes $O --goldens $G --shards $SHS --cache $O/cc.tar --tag det
  $PY tests/test_determinism.py fresh $O --goldens $G --shards $SHS --cache $O/cc.tar --tag det
  mv $O/det_fresh.json $O/det_fresh_tar.json
  $PY tests/test_determinism.py fresh $O --goldens $G --shards $SHS --tag det
  OLD=$T/a40-2026-11-19c; mkdir -p $O/flag19c
  ( cd $OLD && PYTHONPATH=$OLD/jobs/a40site:$OLD/src XLA_FLAGS=--xla_gpu_deterministic_ops=true \
      $PY tests/test_pb_m1.py $G $O/flag19c main --skip-overflow )
  rm -f $O/cc.tar ;;
detA2)
  # rf-energy probe in two fresh-compile processes, then closing test: 3 passes (cache) + fresh from the tarball +
  # two fresh-compile processes (no cache), all bitwise vs pass 1
  rm -f $O/cc.tar $O/det_rfprobe_ref.npz
  $PY tests/test_determinism.py rfprobe $O --goldens $G --tag det; mv $O/det_rfprobe.json $O/det_rfprobe_1.json
  $PY tests/test_determinism.py rfprobe $O --goldens $G --tag det; mv $O/det_rfprobe.json $O/det_rfprobe_2.json
  $PY tests/test_determinism.py passes $O --goldens $G --shards $SHS --cache $O/cc.tar --tag det
  $PY tests/test_determinism.py fresh $O --goldens $G --shards $SHS --cache $O/cc.tar --tag det; mv $O/det_fresh.json $O/det_fresh_tar.json
  $PY tests/test_determinism.py fresh $O --goldens $G --shards $SHS --tag det; mv $O/det_fresh.json $O/det_fresh_compile1.json
  $PY tests/test_determinism.py fresh $O --goldens $G --shards $SHS --tag det; mv $O/det_fresh.json $O/det_fresh_compile2.json
  rm -f $O/cc.tar ;;
smoke)
  $PY tests/test_sor_pallas.py --gpu
  $PY tests/test_sor_pallas.py --gpu --n 81 --W 64
  $PY tests/test_pb_m5.py $O --goldens $G --limit 1 --tag smoke ;;
m5)
  rm -f $O/cc.tar
  $PY tests/test_determinism.py passes $O --goldens $G --shards $SHS --cache $O/cc.tar --backend pallas --tag detB
  $PY tests/test_determinism.py fresh $O --goldens $G --shards $SHS --cache $O/cc.tar --backend pallas --tag detB; mv $O/detB_fresh.json $O/detB_fresh_tar.json
  $PY tests/test_determinism.py fresh $O --goldens $G --shards $SHS --backend pallas --tag detB; mv $O/detB_fresh.json $O/detB_fresh_compile.json
  $PY tests/test_pb_m5.py $O --goldens $G --shards $SHS --cache $O/cc.tar --tag m5 ;;
*) echo "unknown stage $ST"; exit 2 ;;
esac
