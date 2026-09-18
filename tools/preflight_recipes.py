"""Validate portable CacheCodec recipes without loading models or datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle) if path.suffix == ".json" else yaml.safe_load(handle)


def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def main() -> None:
    train = sorted((ROOT / "recipe/train_recipe").glob("cachecodec_*_external.json"))
    eval_ = sorted((ROOT / "recipe/eval_recipe").glob("*_external.yaml"))
    if not train and not eval_:
        raise SystemExit("No external recipes found")
    outputs: set[str] = set()
    for path in train:
        config = expand(load(path))
        for section in ("model", "training", "output", "data"):
            if section not in config:
                raise SystemExit(f"{path}: missing training section {section!r}")
        output = str(config["output"].get("output_dir", ""))
        if not output:
            raise SystemExit(f"{path}: output.output_dir is empty")
        if output in outputs:
            raise SystemExit(f"duplicate training output directory: {output}")
        outputs.add(output)
    for path in eval_:
        config = expand(load(path))
        for section in ("model", "output", "eval"):
            if section not in config:
                raise SystemExit(f"{path}: missing evaluation section {section!r}")
        if config["eval"].get("answer_method", "generate") != "generate":
            raise SystemExit(f"{path}: shortbench recipes must use answer_method=generate")
    print(f"Validated {len(train)} training and {len(eval_)} evaluation recipes.")
    print("No model weights, tokenizer, or dataset were loaded.")


if __name__ == "__main__":
    main()
