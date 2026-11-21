#!/bin/bash
# abdata-v5: NEW conda env for ANARCI on Isambard (aarch64), login node (network install only, no compute).
# Same package set as Coulson ~/_runtime/envs/anarci-v1 (anarci 2024.05.21, hmmer 3.4, python 3.11) + pandas / pyarrow (ledger,
# pool builder) + zstd (bundle). Package cache in a new scratch folder (nothing in $HOME), removed afterwards.
set -euo pipefail
S=/scratch/u6xq/hussain.u6xq; ENV=$S/_envs/anarci-v1; export CONDA_PKGS_DIRS=$S/_envs/anarci-v1-pkgs
[ -e "$ENV" ] && { echo "$ENV exists" >&2; exit 3; }
CONDA=/home/u6xq/hussain.u6xq/miniconda3/bin/conda
$CONDA create -y -p "$ENV" --override-channels -c conda-forge -c bioconda python=3.11 anarci=2024.05.21 hmmer=3.4 pandas pyarrow zstd numpy
rm -rf "$CONDA_PKGS_DIRS"
PATH=$ENV/bin:$PATH "$ENV/bin/python" - <<'PY'
from anarci import anarci
seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
num, ali, _ = anarci([("x", seq)], scheme="imgt", allowed_species=None, allow={"H", "K", "L"})
cdr3 = "".join(a for p, a in num[0][0][0] if 105 <= p[0] <= 117 and a != "-")
assert ali[0][0]["chain_type"] == "H" and cdr3 == "SRWGGDGFYAMDY", cdr3
import pandas, pyarrow; print("anarci ok", cdr3, pandas.__version__, pyarrow.__version__)
PY
"$ENV/bin/hmmscan" -h | sed -n 2p; find "$ENV" -type f | wc -l
