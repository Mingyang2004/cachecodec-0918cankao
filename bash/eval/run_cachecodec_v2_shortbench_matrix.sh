#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

: "${CACHECODEC_MODEL_ROOT:?Set CACHECODEC_MODEL_ROOT to the model snapshot root}"
: "${CACHECODEC_DATA_ROOT:?Set CACHECODEC_DATA_ROOT to the dataset root}"
: "${CACHECODEC_RUN_ROOT:?Set CACHECODEC_RUN_ROOT to the external run root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
TMP_ROOT="${CACHECODEC_RUN_ROOT}/generated_eval_recipes"
mkdir -p "${TMP_ROOT}"

CONFIGS=(
  recipe/eval_recipe/receiver_only_external.yaml
  recipe/eval_recipe/sharer_only_external.yaml
  recipe/eval_recipe/cachecodec_t2t_external.yaml
  recipe/eval_recipe/cachecodec_concat_external.yaml
  recipe/eval_recipe/cachecodec_concat_jcb_external.yaml
  recipe/eval_recipe/cachecodec_concat_jcb_quant_external.yaml
  recipe/eval_recipe/cachecodec_concat_jcb_quant_scale_external.yaml
  recipe/eval_recipe/cachecodec_concat_lcf_qat_external.yaml
  recipe/eval_recipe/cachecodec_concat_lcf_qat_quant_external.yaml
  recipe/eval_recipe/cachecodec_fusion_external.yaml
  recipe/eval_recipe/cachecodec_fusion_lcf_external.yaml
  recipe/eval_recipe/cachecodec_fusion_lcf_quant_external.yaml
  recipe/eval_recipe/cachecodec_fusion_lcf_quant_scale_external.yaml
  recipe/eval_recipe/cachecodec_fusion_lcf_qat_external.yaml
  recipe/eval_recipe/cachecodec_fusion_lcf_qat_quant_external.yaml
)
if [[ -n "${EVAL_CONFIGS:-}" ]]; then
  IFS=',' read -r -a CONFIGS <<< "${EVAL_CONFIGS}"
fi

DATASETS=(mmlu-redux openbookqa ceval ai2-arc)
if [[ -n "${EVAL_DATASETS:-}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${EVAL_DATASETS}"
fi

for config in "${CONFIGS[@]}"; do
  method="$(basename "${config}" .yaml)"
  for dataset in "${DATASETS[@]}"; do
    generated="${TMP_ROOT}/${method}__${dataset}.yaml"
    "${PYTHON_BIN}" - "${config}" "${generated}" "${dataset}" "${GPU_ID}" <<'PY'
import os
import sys
from pathlib import Path
import yaml

source, target, dataset, gpu_id = sys.argv[1:]
with open(source, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config.setdefault("eval", {})
config.setdefault("output", {})
config["eval"]["dataset"] = dataset
config["eval"]["gpu_ids"] = [int(gpu_id)]
local_files = {
    "mmlu-redux": os.environ.get("MMLU_REDUX_JSONL"),
    "openbookqa": os.environ.get("OPENBOOKQA_JSONL"),
    "ceval": os.environ.get("CEVAL_JSONL"),
    "ai2-arc": os.environ.get("AI2_ARC_JSONL"),
}
local_file = local_files.get(dataset)
if local_file:
    config["eval"]["local_jsonl_file"] = local_file
else:
    config["eval"].pop("local_jsonl_file", None)
output_dir = str(config["output"].get("output_dir", ""))
config["output"]["output_dir"] = f"{output_dir.rstrip('/')}/{dataset}"
Path(target).parent.mkdir(parents=True, exist_ok=True)
with open(target, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
PY
    echo "============================================================"
    echo "Evaluating ${method} on ${dataset} (GPU ${GPU_ID})"
    echo "Config: ${generated}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
      script/evaluation/unified_evaluator.py --config "${generated}"
  done
done
