#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate primal

python --version
pip --version

if [ -f setup.py ]; then
    pip install -e . || true
fi

if [ -d od_mstar3 ]; then
    cd od_mstar3
    python setup.py build_ext --inplace || true
    cd /workspaces/PRIMAL
fi

python - <<'PY'
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('Visible GPUs:', tf.config.list_physical_devices('GPU'))
PY

echo "Dev container setup finished."
