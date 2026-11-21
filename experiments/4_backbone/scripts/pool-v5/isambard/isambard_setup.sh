#!/bin/bash
# abdata-v5 Isambard setup, run FROM THIS MAC after `clifton auth` (login-node file ops + the ANARCI env install only; no sbatch).
#  1 upload the pool-input bundle (verified on Coulson and on the Mac: pool-inputs.sha256) to $R/audits/abdata-v5/pool-inputs, re-check
#  2 upload the job code to $S/_submission/abdata-v5 (new files only) and the job script; compare sha256 with the Mac copies
#  3 install the ANARCI env ($S/_envs/anarci-v1) and record the aarch64 mmseqs path in $S/_submission/abdata-v5/MMSEQS_PATH
set -euo pipefail
I=u6xq.aip2.isambard; S=/scratch/u6xq/hussain.u6xq; R=$S/_runtime/jax-Ka/pkabench; A=$R/audits/abdata-v5; C=$S/_submission/abdata-v5
HERE=$(cd "$(dirname "$0")" && pwd); V5=$(dirname "$HERE")
B=/private/tmp/claude-501/-Users-alexi-isambard-ssh/ff273e5c-e06f-4d00-b3bb-87e52de7a000/scratchpad/bundle
ssh() { command ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20 "$@"; }
(cd "$B" && shasum -a 256 -c pool-inputs.sha256)
ssh $I "test ! -e $A/pool-inputs && mkdir -p $A/pool-inputs"
rsync -a --partial -e "ssh -o BatchMode=yes -o ServerAliveInterval=30" "$B/pool-inputs.tar.zst" "$B/pool-inputs.sha256" "$B/files.txt" "$B/aggregate-inputs.sha256" "$I:$A/pool-inputs/"
ssh $I "cd $A/pool-inputs && sha256sum -c pool-inputs.sha256"
for f in pool_v5.py number_cdrs.py ledger.py build_manifest_v5.py; do
  ssh $I "test ! -e $C/$f" || { echo "$C/$f exists (compare by hand)" >&2; exit 3; }
done
scp -q "$V5"/{pool_v5.py,number_cdrs.py,ledger.py,build_manifest_v5.py} "$HERE"/{v5pool.sbatch,install_anarci.sh} "$I:$C/"
(cd "$V5" && shasum -a 256 pool_v5.py number_cdrs.py ledger.py build_manifest_v5.py; cd "$HERE" && shasum -a 256 v5pool.sbatch install_anarci.sh) | sort -k2 > /tmp/abdata-v5-mac.sha
ssh $I "cd $C && sha256sum pool_v5.py number_cdrs.py ledger.py build_manifest_v5.py v5pool.sbatch install_anarci.sh" | sort -k2 | diff - /tmp/abdata-v5-mac.sha && echo "code sha256 equal"
ssh $I "bash $C/install_anarci.sh"
# mmseqs (aarch64): an existing install first, else bioconda mmseqs2 into the new env; version printed for the report
ssh $I "set -e; m=\$(ls $S/_install/*/bin/mmseqs $S/_install/*/*/bin/mmseqs 2>/dev/null | head -1 || true)
  if [ -z \"\$m\" ]; then CONDA_PKGS_DIRS=$S/_envs/anarci-v1-pkgs /home/u6xq/hussain.u6xq/miniconda3/bin/conda install -y -p $S/_envs/anarci-v1 --override-channels -c conda-forge -c bioconda mmseqs2; rm -rf $S/_envs/anarci-v1-pkgs; m=$S/_envs/anarci-v1/bin/mmseqs; fi
  file \$m; \$m version; echo \$m > $C/MMSEQS_PATH"
echo "setup done"
