#!/usr/bin/env python3
"""Run raw-KV DCT high-frequency pruning on OpenBookQA and plot accuracy."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


PERCENTAGES = (0, 10, 25, 40, 50, 70, 85, 95)
ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/raw_dct_prune/openbookqa")
    parser.add_argument("--percentages", type=int, nargs="+", default=PERCENTAGES)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def latest_summary(directory: Path) -> Path:
    summaries = sorted(directory.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        raise FileNotFoundError(f"Evaluator produced no summary JSON in {directory}")
    return summaries[-1]


def load_existing_records(output_dir: Path) -> list[dict[str, object]]:
    """Collect completed points so an incremental run preserves the full curve."""
    records: list[dict[str, object]] = []
    for result_path in output_dir.glob("*/pruned_*pct/result.json"):
        records.append(json.loads(result_path.read_text(encoding="utf-8")))
    return records


def main() -> None:
    args = parse_args()
    if any(not 0 <= value <= 100 for value in args.percentages):
        raise ValueError("Pruning percentages must be in [0, 100].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    templates = {
        "Concat": ROOT / "recipe/eval_recipe/raw_dct_prune_concat_openbookqa.yaml",
        "C2C Fusion": ROOT / "recipe/eval_recipe/raw_dct_prune_fusion_openbookqa.yaml",
    }
    for route, template_path in templates.items():
        template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        for percentage in args.percentages:
            run_dir = args.output_dir / route.lower().replace(" ", "_") / f"pruned_{percentage:02d}pct"
            run_dir.mkdir(parents=True, exist_ok=True)
            record_path = run_dir / "result.json"
            if args.skip_existing and record_path.exists():
                continue
            config = json.loads(json.dumps(template))
            config["model"]["cachejpeg_rosetta_config"]["codec"]["raw_dct_prune"] = {
                "enabled": True, "prune_ratio": percentage / 100.0,
            }
            config["model"]["cachejpeg_rosetta_config"]["codec"]["transport"] = {"mode": "socketpair"}
            config["eval"]["gpu_ids"] = [args.gpu]
            config["output"]["output_dir"] = str(run_dir)
            config_path = run_dir / "eval_config.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            subprocess.run(
                [sys.executable, str(ROOT / "script/evaluation/unified_evaluator.py"), "--config", str(config_path)],
                cwd=ROOT, check=True,
            )
            summary_path = latest_summary(run_dir)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            record = {
                "route": route,
                "pruned_high_frequency_pct": percentage,
                "retained_low_frequency_pct": 100 - percentage,
                "accuracy": float(summary["overall_accuracy"]),
                "num_subjects": len(summary.get("subjects", {})),
                "summary": str(summary_path),
            }
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    records = load_existing_records(args.output_dir)
    records.sort(key=lambda row: (str(row["route"]), int(row["pruned_high_frequency_pct"])))
    if not records:
        raise RuntimeError("No pruning results were found to plot.")
    with (args.output_dir / "accuracy_vs_dct_pruning.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    for route, color, marker in (("Concat", "#4C78A8", "o"), ("C2C Fusion", "#E45756", "s")):
        rows = [row for row in records if row["route"] == route]
        axis.plot([row["pruned_high_frequency_pct"] for row in rows],
                  [100 * row["accuracy"] for row in rows], color=color, marker=marker,
                  linewidth=2.2, markersize=6, label=route)
    axis.set(xlabel="Pruned high-frequency DCT coefficients (%)", ylabel="OpenBookQA accuracy (%)",
             title="Accuracy under raw-KV DCT high-frequency pruning")
    axis.set_xticks(sorted({int(row["pruned_high_frequency_pct"]) for row in records}))
    axis.grid(alpha=0.3); axis.legend(frameon=False)
    figure.savefig(args.output_dir / "accuracy_vs_dct_pruning.png", dpi=300)
    figure.savefig(args.output_dir / "accuracy_vs_dct_pruning.pdf")


if __name__ == "__main__":
    main()
