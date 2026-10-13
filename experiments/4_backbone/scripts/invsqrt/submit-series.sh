#!/bin/bash
set -euo pipefail
S=/scratch/u6xq/hussain.u6xq
JOBS=$S/_submission/invsqrt-v2/jobs.tsv
test ! -e "$JOBS"
prev=7364848
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-50-s17 "$S/_submission/reference.sbatch" invsqrt-v2-50-s17 0.5 17 invsqrt)
printf "%s\t%s\n" "$jid" "invsqrt-v2-50-s17" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-50-s29 "$S/_submission/reference.sbatch" invsqrt-v2-50-s29 0.5 29 invsqrt)
printf "%s\t%s\n" "$jid" "invsqrt-v2-50-s29" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-50-s43 "$S/_submission/reference.sbatch" invsqrt-v2-50-s43 0.5 43 invsqrt)
printf "%s\t%s\n" "$jid" "invsqrt-v2-50-s43" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-100-s17 "$S/_submission/reference.sbatch" invsqrt-v2-100-s17 1.0 17 invsqrt)
printf "%s\t%s\n" "$jid" "invsqrt-v2-100-s17" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-100-s29 "$S/_submission/reference.sbatch" invsqrt-v2-100-s29 1.0 29 invsqrt)
printf "%s\t%s\n" "$jid" "invsqrt-v2-100-s29" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-100-s43 "$S/_submission/reference.sbatch" invsqrt-v2-100-s43 1.0 43 invsqrt)
printf "%s\t%s\n" "$jid" "invsqrt-v2-100-s43" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=control-v2-50-s17 "$S/_submission/reference.sbatch" control-v2-50-s17 0.5 17 none)
printf "%s\t%s\n" "$jid" "control-v2-50-s17" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=control-v2-50-s29 "$S/_submission/reference.sbatch" control-v2-50-s29 0.5 29 none)
printf "%s\t%s\n" "$jid" "control-v2-50-s29" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=control-v2-50-s43 "$S/_submission/reference.sbatch" control-v2-50-s43 0.5 43 none)
printf "%s\t%s\n" "$jid" "control-v2-50-s43" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-50-fourier-s17 "$S/_submission/ablation.sbatch" invsqrt-v2-50-fourier-s17 0.5 17 invsqrt fourier)
printf "%s\t%s\n" "$jid" "invsqrt-v2-50-fourier-s17" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-50-rope-s17 "$S/_submission/ablation.sbatch" invsqrt-v2-50-rope-s17 0.5 17 invsqrt rope)
printf "%s\t%s\n" "$jid" "invsqrt-v2-50-rope-s17" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-50-shared-projection-s17 "$S/_submission/ablation.sbatch" invsqrt-v2-50-shared-projection-s17 0.5 17 invsqrt shared-projection)
printf "%s\t%s\n" "$jid" "invsqrt-v2-50-shared-projection-s17" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-50-relations-s17 "$S/_submission/ablation.sbatch" invsqrt-v2-50-relations-s17 0.5 17 invsqrt relations)
printf "%s\t%s\n" "$jid" "invsqrt-v2-50-relations-s17" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt-v2-50-paired-delta-s17 "$S/_submission/ablation.sbatch" invsqrt-v2-50-paired-delta-s17 0.5 17 invsqrt paired-delta)
printf "%s\t%s\n" "$jid" "invsqrt-v2-50-paired-delta-s17" | tee -a "$JOBS"
prev=$jid
jid=$(sbatch --parsable --dependency=afterok:$prev --job-name=invsqrt_report "$S/_submission/report.sbatch")
printf "%s\t%s\n" "$jid" "final-report" | tee -a "$JOBS"
