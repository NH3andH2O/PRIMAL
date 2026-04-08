#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate primal

if [ -n "${TZ:-}" ] && [ -f "/usr/share/zoneinfo/$TZ" ]; then
    echo "Configuring timezone to $TZ..."
    echo "$TZ" | sudo tee /etc/timezone >/dev/null
    sudo ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime
fi

# TF 2.16 ships cuDNN 8 via pip; system cuDNN 9 causes dual-registration crashes.
# Move system cuDNN 9 out of the way if present.
if ls /usr/lib/x86_64-linux-gnu/libcudnn*.so* &>/dev/null; then
    echo "Moving system cuDNN 9 out to avoid conflict with pip cuDNN 8..."
    sudo mkdir -p /usr/lib/x86_64-linux-gnu/cudnn9-backup
    sudo mv /usr/lib/x86_64-linux-gnu/libcudnn*.so* /usr/lib/x86_64-linux-gnu/cudnn9-backup/ || true
    sudo ldconfig
fi

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
