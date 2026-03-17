#!/usr/bin/env bash
set -e

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
fi

echo "Dev container setup finished."