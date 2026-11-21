#!/bin/bash
# Geometry + maps profile on the A40 (BUILD-PROFILE.md; measurement only, src unchanged).
# run via jobs/a40.sbatch <frozen code> jobs/a40_prof.sh <stage>. Outputs $A40/profile/<stage>.
#  smoke: 3 small states (one triple), all passes -> checks trace parsing / clock alignment
#  main : 20 states (5 AB/A/B triples + 5 singles, classes 0-3), all passes (prof_geomaps.py --mode main)
#  conc : concurrency: 1 process alone, then 2 processes on the same A40 (barrier-synchronised timed loops)
set -u
ST=$1; O=$A40/profile/$ST; mkdir -p $O
SH=$T/m0/shards; IN=$A40/m6/inputs
INS="$SH/m0_goldens_00.npz $SH/m0_goldens_01.npz $SH/m0_goldens_02.npz $IN/pilot5_m6.npz"
CCT=$A40/m6/m6/cc.tar
P="$PY jobs/prof_geomaps.py --inputs $INS --cc-tar $CCT"
case $ST in
smoke)
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 $P --mode main --out $O --cc-dir $CC \
    --states 4066edfcdb3ea8e3_AB,4066edfcdb3ea8e3_A,4066edfcdb3ea8e3_B --keep-traces 4066edfcdb3ea8e3_AB ;;
main)
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 $P --mode main --out $O --cc-dir $CC ;;
conc)
  BAR=/tmp/jpa-prof-bar-$SLURM_JOB_ID
  rm -rf $BAR; $P --mode thru --out $O --cc-dir $CC-s --tag single --nproc 1 --repeat ${REPEAT:-2} --barrier $BAR
  rm -rf $BAR
  $P --mode thru --out $O --cc-dir $CC-a --tag dual_a --nproc 2 --repeat ${REPEAT:-2} --barrier $BAR &
  $P --mode thru --out $O --cc-dir $CC-b --tag dual_b --nproc 2 --repeat ${REPEAT:-2} --barrier $BAR &
  wait
  rm -rf $BAR $CC-s $CC-a $CC-b ;;
*) echo "unknown stage $ST"; exit 2 ;;
esac
