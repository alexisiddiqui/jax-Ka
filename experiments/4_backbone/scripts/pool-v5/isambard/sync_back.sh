#!/bin/bash
# abdata-v5: after the Isambard finish + ledger jobs, copy training/pool-v5 and audits/pool-v5-ledger to Coulson (NEW folders),
# relayed via this Mac; each folder carries SHA256SUMS written by the job, checked on the Mac and on Coulson.
set -euo pipefail
I=u6xq.aip2.isambard; C=Coulson-Chemistry; RI=/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench; RC=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
T=/private/tmp/claude-501/-Users-alexi-isambard-ssh/ff273e5c-e06f-4d00-b3bb-87e52de7a000/scratchpad/syncback
for d in training/pool-v5 audits/pool-v5-ledger; do
  ssh -o BatchMode=yes $C "test ! -e $RC/$d" || { echo "$RC/$d exists on Coulson" >&2; exit 3; }
  rm -rf "$T/$d"; mkdir -p "$T/$d"; rsync -a -e "ssh -o BatchMode=yes -o ServerAliveInterval=30" "$I:$RI/$d/" "$T/$d/"
  (cd "$T/$d" && shasum -a 256 -c SHA256SUMS > /dev/null)
  ssh -o BatchMode=yes $C "mkdir -p $(dirname $RC/$d)"; rsync -a -e "ssh -o BatchMode=yes -o ServerAliveInterval=15" "$T/$d/" "$C:$RC/$d/"
  ssh -o BatchMode=yes $C "cd $RC/$d && sha256sum -c SHA256SUMS > /dev/null" && echo "$d: sha256 equal on Isambard / Mac / Coulson"
done
