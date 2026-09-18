#!/usr/bin/env bash
# Source this file before running CacheCodec training or evaluation recipes.
# Existing exported values take precedence over these local defaults.

_cachecodec_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CACHECODEC_REPO_ROOT="${CACHECODEC_REPO_ROOT:-${_cachecodec_env_dir}}"
export CACHECODEC_MODEL_ROOT="${CACHECODEC_MODEL_ROOT:-/data/smy_data/local/models}"
export CACHECODEC_DATA_ROOT="${CACHECODEC_DATA_ROOT:-/data/smy_data/local/data}"
export CACHECODEC_RUN_ROOT="${CACHECODEC_RUN_ROOT:-/data/smy_data/local}"

# Hugging Face cache is distinct from MODEL_ROOT: recipes load local snapshots
# from MODEL_ROOT, while loaders may use this cache for downloaded artifacts.
export HF_HOME="${HF_HOME:-/data/smy_data/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"

# Defaults used by the repository's train/eval launcher scripts.
export PYTHONPATH="${CACHECODEC_REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export GPU_ID="${GPU_ID:-1}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29541}"

# Optional benchmark JSONL overrides consumed by the evaluation matrix script.
# Leave empty until the corresponding local files are available.
: "${MMLU_REDUX_JSONL:=}"
: "${OPENBOOKQA_JSONL:=}"
: "${CEVAL_JSONL:=}"
: "${AI2_ARC_JSONL:=}"

unset _cachecodec_env_dir
