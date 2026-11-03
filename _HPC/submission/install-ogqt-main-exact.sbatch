#!/bin/bash
#SBATCH --job-name=install-ogqt-main
#SBATCH --partition=oc
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:45:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/install-ogqt-main-exact-%j.out

set -euo pipefail
source=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/sources/jax-Ka-main-7c3e3e9
environment=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9
/u/oc/lina4225/miniconda3/bin/conda create -y --override-channels -c conda-forge \
  -p "$environment" python=3.12 pip
"$environment/bin/python" -m pip install --upgrade pip
"$environment/bin/python" -m pip install "jax[cuda12]==0.11.1"
"$environment/bin/python" -m pip install -e "$source[train,triton,benchmark]"
"$environment/bin/python" - <<'PY'
import jax, jax_triton, triton
assert jax.__version__ == "0.11.1", jax.__version__
assert triton.__version__ == "3.8.0", triton.__version__
print("jax", jax.__version__)
print("jax_triton", getattr(jax_triton, "__version__", "0.4.1"))
print("triton", triton.__version__)
PY
