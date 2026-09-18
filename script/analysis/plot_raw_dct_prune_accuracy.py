#!/usr/bin/env python3
"""Plot OpenBookQA accuracy against retained raw-KV DCT frequencies."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ROUTES = {
    "concat": {"label": "Concat", "color": "#4C78A8", "marker": "o"},
    "c2c_fusion": {"label": "C2C Fusion", "color": "#E45756", "marker": "s"},
}


def load_records(route_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in route_dir.glob("pruned_*pct/result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        pruned = float(record["pruned_high_frequency_pct"])
        records.append({
            "route": str(record["route"]),
            "retained_low_frequency_pct": 100.0 - pruned,
            "pruned_high_frequency_pct": pruned,
            "accuracy": float(record["accuracy"]),
            "num_subjects": int(record["num_subjects"]),
            "summary": str(record["summary"]),
        })
    if not records:
        raise FileNotFoundError(f"No result.json files in {route_dir}")
    return sorted(records, key=lambda row: float(row["retained_low_frequency_pct"]))


def write_plot(route_key: str, route_dir: Path) -> list[dict[str, object]]:
    style = ROUTES[route_key]
    records = load_records(route_dir)
    csv_path = route_dir / "accuracy_vs_retained_frequency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    x = [float(row["retained_low_frequency_pct"]) for row in records]
    y = [100.0 * float(row["accuracy"]) for row in records]
    figure, axis = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    axis.plot(x, y, color=style["color"], marker=style["marker"], linewidth=2.2,
              markersize=6, label=style["label"])
    for index, (x_value, y_value) in enumerate(zip(x, y)):
        axis.annotate(f"{y_value:.1f}", (x_value, y_value), textcoords="offset points",
                      xytext=(0, 7 if index % 2 == 0 else 21), ha="center", fontsize=8)
    axis.set(
        title=f"OpenBookQA under raw-KV DCT low-pass retention ({style['label']})",
        xlabel="Retained low-frequency DCT coefficients (%)",
        ylabel="OpenBookQA accuracy (%)",
        xlim=(-1, 101),
        ylim=(0, 50),
    )
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.grid(alpha=0.30)
    axis.legend(frameon=False, loc="lower right")
    for suffix in ("png", "pdf"):
        figure.savefig(route_dir / f"accuracy_vs_retained_frequency.{suffix}", dpi=300)
    plt.close(figure)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/raw_dct_prune/openbookqa")
    args = parser.parse_args()
    for route_key in ROUTES:
        write_plot(route_key, args.output_dir / route_key)


if __name__ == "__main__":
    main()
