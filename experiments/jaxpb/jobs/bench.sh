#!/bin/bash
# BUILD-BENCH (2026-11-21) stages; run from the frozen code dir via jobs/coulson_cpu.sbatch (CPU) or jobs/a40.sbatch (GPU).
#  pypka <ncpus> <idx|all>  : arm A, native PyPKA per state (bench_pypka.py); with ncpus 1 also times jaxpb's CPU prep
#                             (prep_pypka.py prep) and the input.pdb regeneration. idx = SLURM_ARRAY_TASK_ID when "task".
#  cpu_pb <tag> <states>    : arm B PB + assembly (JAX CPU, XLA threads = allocated cores)
#  cpu_mc <tag> <S> <states> <states200> <energies glob> : arm B titration
#  cpu1 <tag> <states>      : arm B on 1 core: PB + assembly then titration 50k + 200k of the same states
#  gpu                      : arm C (A40): PB 2 passes (cold, warm), titration prod (pKa) + fill (production occupancy)
#  compare                  : sanity diffs
set -u
B=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench; T=$B/tierB/jaxpb-pipeline; O=$T/bench
PYJ=$B/envs/ogqt-main-7c3e3e9/bin/python; PYP=$B/envs/pypka/bin/python; PYR=$B/envs/runner/bin/python
mkdir -p $O/pypka $O/prep $O/jaxcpu $O/gpu
cgmem() { local f=/sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.max_usage_in_bytes
          [ -r $f ] && echo "[cgroup] job memory.max_usage_in_bytes $(cat $f)"; }
lscpu | grep -E "Model name|^CPU\(s\)|Socket" ; echo "[affinity] $(taskset -cp $$)"
ST=$1; shift
case $ST in
pypka)
  N=$1; I=$2; [ "$I" = task ] && I=$SLURM_ARRAY_TASK_ID
  if [ "$I" = all ]; then IDX=$(seq 0 19); else IDX=$I; fi
  mkdir -p $O/pypka/n$N
  for i in $IDX; do
    read SID SRC < <($PYJ jobs/bench_jaxpb.py --mode src --index $i)
    W=/tmp/jpb-${SLURM_JOB_ID}-$SID; rm -rf $W; mkdir -p $W/nat $W/prep
    t0=$(date +%s.%N); $PYR jobs/m0_inputpdb.py $SRC $W/nat > /dev/null; t1=$(date +%s.%N)
    cp $W/nat/input.pdb $W/prep/
    export LD_LIBRARY_PATH=$B/fortran/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    if [ "$N" = 1 ]; then
      t2=$(date +%s.%N)
      /usr/bin/time -v -o $W/prep.time $PYP src/jaxpb/prep_pypka.py prep $SRC $W/prep $W/prep/rec.npz --state-id $SID > $W/prep.log 2>&1 || tail -5 $W/prep.log
      t3=$(date +%s.%N)
      echo "{\"state\": \"$SID\", \"inputpdb_s\": $(echo "$t1-$t0" | bc), \"prep_s\": $(echo "$t3-$t2" | bc), \"maxrss_kb\": $(grep 'Maximum resident' $W/prep.time | awk '{print $NF}'), \"user_s\": $(grep 'User time' $W/prep.time | awk '{print $NF}'), \"sys_s\": $(grep 'System time' $W/prep.time | awk '{print $NF}'), \"host\": \"$(hostname)\"}" > $O/prep/$SID.json
      cat $O/prep/$SID.json
    fi
    /usr/bin/time -v -o $O/pypka/n$N/$SID.time $PYP jobs/bench_pypka.py $SRC $W/nat $O/pypka/n$N/$SID.json --ncpus $N > $W/nat.log 2>&1 || tail -20 $W/nat.log
    grep -E '"wall_titration"|"peak_tree_rss_max"|"mae"|bitwise' $W/nat.log | tr -s ' ' | tr '\n' ' '; echo " <- $SID n$N"
    rm -rf $W
  done ;;
cpu_pb)
  $PYJ jobs/bench_jaxpb.py --mode pb --out $O/jaxcpu --tag $1 --states $2 --passes 1 ;;
cpu_mc)
  $PYJ jobs/bench_jaxpb.py --mode mc --out $O/jaxcpu --tag $1 --S $2 --states $3 --states200 $4 --energies $(ls $O/jaxcpu/$5) ;;
cpu1)
  $PYJ jobs/bench_jaxpb.py --mode pb --out $O/jaxcpu --tag cpu1_$1 --states $2 --passes 1 && \
  $PYJ jobs/bench_jaxpb.py --mode mc --out $O/jaxcpu --tag cpu1_$1 --S tight --states $2 --energies $O/jaxcpu/energies_cpu1_$1.npz ;;
gpu)
  $PYJ jobs/bench_jaxpb.py --mode pb --out $O/gpu --tag gpu --passes 2 && \
  $PYJ jobs/bench_jaxpb.py --mode mc --out $O/gpu --tag gpu_prod --S prod --energies $O/gpu/energies_gpu.npz && \
  $PYJ jobs/bench_jaxpb.py --mode mc --out $O/gpu --tag gpu_fill --S prod --fill 1 --energies $O/gpu/energies_gpu.npz ;;
compare)
  $PYJ jobs/bench_jaxpb.py --mode compare --out $O ;;
*) echo "unknown stage $ST"; exit 2 ;;
esac
rc=$?
cgmem
exit $rc
