#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

: "${CACHECODEC_MODEL_ROOT:?Set CACHECODEC_MODEL_ROOT to the model snapshot root}"
: "${CACHECODEC_DATA_ROOT:?Set CACHECODEC_DATA_ROOT to the dataset root}"
: "${CACHECODEC_RUN_ROOT:?Set CACHECODEC_RUN_ROOT to the external run root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29541}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ "${1:-}" == "--preflight" ]]; then
  exec "${PYTHON_BIN}" tools/preflight_recipes.py
fi

CONFIGS=(
  recipe/train_recipe/cachecodec_concat_external.json
  recipe/train_recipe/cachecodec_concat_jcb_external.json
  recipe/train_recipe/cachecodec_concat_lcf_qat_external.json
  recipe/train_recipe/cachecodec_fusion_external.json
  recipe/train_recipe/cachecodec_fusion_lcf_external.json
  recipe/train_recipe/cachecodec_fusion_lcf_qat_external.json
)
if [[ -n "${TRAIN_CONFIGS:-}" ]]; then
  IFS=',' read -r -a CONFIGS <<< "${TRAIN_CONFIGS}"
fi

for config in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Training ${config} on ${NPROC_PER_NODE} GPUs (${CUDA_VISIBLE_DEVICES})"
  echo "Outputs are under ${CACHECODEC_RUN_ROOT}"
  echo "============================================================"
  torchrun --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    script/train/SFT_train.py \
    --config "${config}"
done
