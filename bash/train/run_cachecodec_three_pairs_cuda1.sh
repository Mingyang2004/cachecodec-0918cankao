#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/c2c/bin/python}"
GLOBAL_LOG="${GLOBAL_LOG:-/data/smy/output/cachecodec/cachecodec_three_pairs_cuda1.log}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CACHECODEC_DETECT_ANOMALY="${CACHECODEC_DETECT_ANOMALY:-0}"
ulimit -c 0

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ' | head -n 1)"
  if [[ -n "${free_mb}" && "${free_mb}" -lt 20000 ]]; then
    echo "GPU1 has only ${free_mb} MiB free; refusing to start training." >&2
    exit 1
  fi
fi

check_gpu1_memory() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local free_mb
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ' | head -n 1)"
    if [[ -n "${free_mb}" && "${free_mb}" -lt 20000 ]]; then
      echo "GPU1 has only ${free_mb} MiB free; refusing to start the next phase." >&2
      exit 1
    fi
  fi
}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "$(dirname "${GLOBAL_LOG}")"
exec > >(tee -a "${GLOBAL_LOG}") 2>&1

echo "[$(date '+%F %T')] Starting Cachecodec three-pair training"
echo "[$(date '+%F %T')] Python: ${PYTHON_BIN}"
echo "[$(date '+%F %T')] PYTHONPATH: ${PYTHONPATH}"
echo "[$(date '+%F %T')] CUDA_VISIBLE_DEVICES=1"
echo "[$(date '+%F %T')] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

PAIRS=(
  qwen25_05b_instruct_to_qwen3_06b
  qwen25_15b_instruct_to_qwen3_06b
  qwen3_4b_base_to_qwen3_06b
)
if [[ -n "${PAIR_FILTER:-}" ]]; then
  PAIRS=("${PAIR_FILTER}")
fi

for pair in "${PAIRS[@]}"; do
  raw_config="recipe/train_recipe/cachecodec_${pair}_raw.json"
  qat_config="recipe/train_recipe/cachecodec_${pair}_adaptive_quant.json"
  raw_log="/data/smy/output/cachecodec/checkpoints/${pair}_raw_50k/launcher.log"
  qat_log="/data/smy/output/cachecodec/checkpoints/${pair}_adaptive_quant_50k/launcher.log"

  mkdir -p "$(dirname "${raw_log}")" "$(dirname "${qat_log}")"
  check_gpu1_memory
  echo "[$(date '+%F %T')] ${pair}: Phase 1 raw projector"
  CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" script/train/SFT_train.py \
    --config "${raw_config}" 2>&1 | tee "${raw_log}"
  check_gpu1_memory
  echo "[$(date '+%F %T')] ${pair}: Phase 2 adaptive QAT"
  CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" script/train/SFT_train.py \
    --config "${qat_config}" 2>&1 | tee "${qat_log}"
  echo "[$(date '+%F %T')] ${pair}: completed"
done

echo "[$(date '+%F %T')] All three model pairs completed"
