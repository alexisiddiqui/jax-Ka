#!/bin/bash
# M6 on the A40 (run via jobs/a40.sbatch <frozen code> jobs/a40_m6.sh <stage>). Outputs $A40/m6/<stage>.
#  smoke: 3 smallest states, driver end to end (warm start from the M5 PB cache), stop after 1 shard + restart, compare, l3
#  m6   : 104 states (5 pilot + 99 M0), shard size 16: run A cold (compiles counted, writes cc.tar); run B fresh process
#         from the tarball (0 compiles; determinism); run C killed with SIGKILL after KILL_AFTER s, restarted (resume);
#         bitwise compare A/B/C; L3 vs the saved PyPKA teacher on run A (+ MC floor, CRN localisation, x64 check)
#  scale: 10 cycles of the 104 records (cycles 1-9 permuted, new MC chain ids) = 1,040 states, shard size 104, cold:
#         compiles per shard (flat), warm throughput, energies bitwise across cycles and vs m6 run A
set -u
ST=$1; O=$A40/m6/$ST; mkdir -p $O; SH=$T/m0/shards; P=$T/m0/pilot; IN=$A40/m6/inputs
STS=$B/envs/pypka/lib/python3.10/site-packages/pypka/G54A7/sts
SHS="$SH/m0_goldens_00.npz $SH/m0_goldens_01.npz $SH/m0_goldens_02.npz"
INS="$SHS $IN/pilot5_m6.npz"
DETREF=$A40/m5/detA2/det_ref.npz
DRV="$PY -m jaxpb.api"
LCC=/tmp/jaxpb-cc-${SLURM_JOB_ID}
[ -f $IN/pilot5_m6.npz ] || $PY tests/test_m6.py prep $IN --goldens $P/goldens_pilot5.npz --pilotjson $P/pilotjson --sts $STS || exit 3
case $ST in
smoke)
  rm -rf $O/a $O/c; cp $A40/m5/m5/cc.tar $O/cc.tar
  $PY tests/test_m6.py items $O/items.json --inputs $INS --smallest 3
  $DRV $O/a --items $O/items.json --shard-size 1 --cache $O/cc.tar
  rm -rf $LCC; $DRV $O/c --items $O/items.json --shard-size 1 --cache $O/cc.tar --max-shards 1
  rm -rf $LCC; $DRV $O/c --items $O/items.json --shard-size 1 --cache $O/cc.tar
  $PY tests/test_m6.py compare $O/compare.json $O/a $O/c
  $PY tests/test_m6.py l3 $O/l3.json --labels $O/a --inputs $INS --det-ref $DETREF ;;
m6)
  rm -rf $O/runA $O/runB $O/runC $O/cc.tar $LCC
  $PY tests/test_m6.py items $O/items.json --inputs $INS
  $DRV $O/runA --items $O/items.json --shard-size 16 --cache $O/cc.tar
  rm -rf $LCC; $DRV $O/runB --items $O/items.json --shard-size 16 --cache $O/cc.tar
  rm -rf $LCC; timeout -s KILL ${KILL_AFTER:-600} $DRV $O/runC --items $O/items.json --shard-size 16 --cache $O/cc.tar
  echo "[m6] runC killed rc=$? shards: $(ls $O/runC | tr '\n' ' ')"
  ls -la $O/runC > $O/runC_after_kill.txt
  rm -rf $LCC; $DRV $O/runC --items $O/items.json --shard-size 16 --cache $O/cc.tar
  $PY tests/test_m6.py compare $O/compare.json $O/runA $O/runB $O/runC
  $PY tests/test_m6.py l3 $O/l3.json --labels $O/runA --inputs $INS --det-ref $DETREF ;;
scale)
  rm -rf $O/run $O/cc_scale.tar $LCC
  $PY tests/test_m6.py items $O/items.json --inputs $INS --cycles 10
  $DRV $O/run --items $O/items.json --shard-size 104 --cache $O/cc_scale.tar
  $PY tests/test_m6.py scale $O/scale.json --labels $O/run --ref-labels $A40/m6/m6/runA
  rm -f $O/cc_scale.tar ;;
l3)
  $PY tests/test_m6.py l3 $A40/m6/m6/l3.json --labels $A40/m6/m6/runA --inputs $INS --det-ref $DETREF ;;
*) echo "unknown stage $ST"; exit 2 ;;
esac
