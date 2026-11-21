#!/bin/bash
# BUILD-SPEED closing tests on the A40 (run via jobs/a40.sbatch <any frozen code> jobs/a40_speed.sh <stage>).
# Code snapshots live in $T/a40-2026-11-20-speed-<L> (L = ref (M6 src) | L1 | L12 | L124 | ...); LEVELS="ref L1 ..." picks them.
#  parity : per level: 104-state parity pass (digests, hash time subtracted) -> $A40/speed/parity/digest_<L>.json; then compare
#           every level against digest_ref.json
#  timing : per level: 20 profile states, untraced timing (REPEAT reps) + TRACE traced states (GPU idle share)
#  det    : level $DET: 2 passes in one process (fresh cache dir) + a fresh process from an empty cache -> digests compared
#  driver : level $DET: production driver on the M6 104 items, labels bitwise vs M6 run A + end-to-end throughput
# levels whose name contains 3 run with --reuse 1 (lever 3 triple scheduling)
set -u
ST=$1; SH=$T/m0/shards; IN=$A40/m6/inputs; LCC=/tmp/jaxpb-cc-${SLURM_JOB_ID}; TSUF=${TSUF:-}
INS="$SH/m0_goldens_00.npz $SH/m0_goldens_01.npz $SH/m0_goldens_02.npz $IN/pilot5_m6.npz"
code() { echo $T/a40-2026-11-20-speed-$1; }
runpy() { local L=$1; shift; ( cd $(code $L) && CODE=$(code $L) PYTHONPATH=$(code $L)/jobs/a40site:$(code $L)/src $PY jobs/speed.py "$@" ); }
case $ST in
parity)
  O=$A40/speed/${SUB:-parity}; mkdir -p $O
  for L in ${LEVELS}; do
    if [ -f $O/digest_$L.json ] && [ "${FORCE:-0}" != 1 ]; then echo "[speed] skip $L (digest exists)"; continue; fi
    RU=0; case $L in *3*) RU=1;; esac
    rm -rf $CC-$L; runpy $L --mode parity --inputs $INS --out $O --tag $L --cc-dir $CC-$L --reuse $RU ${STATES:+--states $STATES}
    rm -rf $CC-$L
  done
  ( cd $(code ${LEVELS%% *}) && $PY jobs/speed.py --mode compare --out $O/compare_$(echo $LEVELS | tr ' ' '_').json $O/digest_ref.json $(for L in $LEVELS; do [ $L != ref ] && echo $O/digest_$L.json; done) ) ;;
timing)
  O=$A40/speed/${SUB:-timing}; mkdir -p $O
  for L in ${LEVELS}; do
    RU=0; case $L in *3*) RU=1;; esac
    rm -rf $CC-t$L; runpy $L --mode timing --inputs $INS --out $O --tag $L$TSUF --cc-dir $CC-t$L --repeat ${REPEAT:-1} --trace ${TRACE:-0} --reuse $RU ${STATES:+--states $STATES}
    rm -rf $CC-t$L
  done ;;
det)
  O=$A40/speed/${SUB:-det}; mkdir -p $O; L=$DET; RU=0; case $L in *3*) RU=1;; esac
  rm -rf $CC-d1 $CC-d2
  runpy $L --mode parity --inputs $INS --out $O --tag ${L}_p12 --cc-dir $CC-d1 --passes 2 --reuse $RU ${STATES:+--states $STATES}
  runpy $L --mode parity --inputs $INS --out $O --tag ${L}_fresh --cc-dir $CC-d2 --warm 0 --reuse $RU ${STATES:+--states $STATES}
  rm -rf $CC-d1 $CC-d2
  ( cd $(code $L) && $PY jobs/speed.py --mode compare --out $O/compare_det_$L.json $A40/speed/parity/digest_ref.json $O/digest_${L}_p12.json $O/digest_${L}_fresh.json ) ;;
driver)
  # production driver (jaxpb.api) on the M6 104-state items, shard size 16, cold cache -> labels bitwise vs M6 run A
  O=$A40/speed/${SUB:-driver}; mkdir -p $O; L=$DET; rm -rf $O/run $LCC
  ( cd $(code $L) && PYTHONPATH=$(code $L)/jobs/a40site:$(code $L)/src $PY -m jaxpb.api $O/run --items $A40/m6/m6/items.json --shard-size 16 )
  ( cd $(code $L) && PYTHONPATH=$(code $L)/jobs/a40site:$(code $L)/src $PY jobs/speed.py --mode labels --out $O/labels_vs_m6.json $A40/m6/m6/runA $O/run ) ;;
*) echo "unknown stage $ST"; exit 2 ;;
esac
