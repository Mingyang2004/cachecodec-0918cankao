#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${C2C_ENV_NAME:-c2c}"
CONDA_BIN="${CONDA_BIN:-}"

if [[ -z "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x /home/ubuntu/anaconda3/bin/conda ]]; then
    CONDA_BIN="/home/ubuntu/anaconda3/bin/conda"
  elif [[ -x /opt/conda/bin/conda ]]; then
    CONDA_BIN="/opt/conda/bin/conda"
  fi
fi

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "找不到 conda: ${CONDA_BIN}" >&2
  echo "请先安装 Miniconda/Anaconda，或设置 CONDA_BIN=/path/to/conda" >&2
  exit 1
fi

CONDA_BASE="$("${CONDA_BIN}" info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "创建 conda 环境 ${ENV_NAME} (Python 3.10)"
  "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.10 pip -y
else
  echo "复用已有 conda 环境 ${ENV_NAME}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements-c2c.txt"
python -m pip install --no-deps -e "${PROJECT_ROOT}"

python - <<'PY'
import torch
import transformers
import datasets
import rosetta

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
print("C2C environment installation complete.")
PY

echo "conda activate ${ENV_NAME}"
echo "cd ${PROJECT_ROOT}"
echo "export PYTHONPATH=${PROJECT_ROOT}:\$PYTHONPATH"
