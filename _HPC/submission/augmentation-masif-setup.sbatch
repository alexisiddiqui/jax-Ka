#!/bin/bash
#SBATCH --job-name=masif-audit-setup
#SBATCH --partition=amd96
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1/setup-%j.log
set -euo pipefail
root=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export UDOCKER_DIR="$root/udocker" UDOCKER_LOGLEVEL=3
mkdir -p "$root"
if [ ! -x "$root/setup-env/bin/udocker" ]; then
  "$pilot/env/bin/python" -m venv "$root/setup-env"
  "$root/setup-env/bin/pip" install 'udocker==1.3.17'
fi
"$root/setup-env/bin/udocker" install
if [ ! -e "$root/image-manifest.txt" ]; then
  "$root/setup-env/bin/udocker" pull pablogainza/masif:latest
fi
"$root/setup-env/bin/udocker" images -l
if [ ! -e "$root/udocker/containers/masif-audit" ]; then
  "$root/setup-env/bin/udocker" create --name=masif-audit pablogainza/masif:latest
fi
"$root/setup-env/bin/udocker" setup --execmode=P1 masif-audit
mkdir -p "$root/compat"
cp /usr/lib64/libstdc++.so.6.0.25 "$root/compat/libstdc++.so.6"
cp /usr/lib64/libgcc_s-8-20210514.so.1 "$root/compat/libgcc_s.so.1"
"$root/setup-env/bin/udocker" run --env=LD_LIBRARY_PATH=/compat --volume="$root/compat:/compat" --env=OMP_NUM_THREADS=1 --env=OPENBLAS_NUM_THREADS=1 masif-audit /bin/bash -c 'set -e; python -c "import sys, numpy, tensorflow, pymesh; print(sys.version); print(numpy.__version__, tensorflow.__version__)"; ls /masif; command -v msms; command -v apbs; command -v reduce; ls /masif/data/masif_ppi_search/nn_models/sc05/all_feat/model_data'
"$root/setup-env/bin/pip" freeze > "$root/setup-freeze.txt"
if [ ! -e "$root/image-manifest.txt" ]; then
  "$root/setup-env/bin/udocker" manifest inspect pablogainza/masif:latest > "$root/image-manifest.txt"
fi
