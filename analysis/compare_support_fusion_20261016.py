"""Offline, fixed-rule fusion checks on the saved diagnostic subset.

Weights use only annotated support masks. Query labels are used exclusively
to score the resulting predictions. This is exploratory, not test evaluation.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).parent / "small_targets_full_20261016",
    )
    args = parser.parse_args()
    with (args.root / "shots.csv").open() as handle:
        shots = defaultdict(list)
        for row in csv.DictReader(handle):
            shots[row["id"]].append(row)
    with (args.root / "episodes.csv").open() as handle:
        reference = {
            row["id"]: float(row["dice"])
            for row in csv.DictReader(handle) if row["variant"] == "full"
        }
    rows = []
    for episode_id, support_rows in sorted(shots.items()):
        support_rows.sort(key=lambda row: int(row["support_index"]))
        assert [int(row["support_index"]) for row in support_rows] == list(range(4))
        with np.load(args.root / episode_id / "arrays.npz") as arrays:
            logits = arrays["logits"]
            target = arrays["target"].astype(bool)
            np.testing.assert_allclose(logits.mean(axis=0), arrays["mean_logits"], atol=1e-5)
        area = np.array([float(row["support_area"]) for row in support_rows])
        cells = np.array([float(row["support_grid_cells"]) for row in support_rows])
        sqrt_area = np.sqrt(area)
        sqrt_area /= sqrt_area.sum()
        sqrt_cells = np.sqrt(cells)
        sqrt_cells = sqrt_cells / sqrt_cells.sum() if sqrt_cells.sum() else np.full(4, 0.25)
        largest = int(np.argmax(area))
        top2 = np.argsort(-area, kind="stable")[:2]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        predictions = {
            "uniform_logits": logits.mean(axis=0) > 0,
            "uniform_prob": probabilities.mean(axis=0) > 0.5,
            "sqrt_area_logits": np.einsum("s,shw->hw", sqrt_area, logits) > 0,
            "sqrt_area_prob": np.einsum("s,shw->hw", sqrt_area, probabilities) > 0.5,
            "largest_area": logits[largest] > 0,
            "top2_area_logits": logits[top2].mean(axis=0) > 0,
            "sqrt_cells_logits": np.einsum("s,shw->hw", sqrt_cells, logits) > 0,
        }
        for method, pred in predictions.items():
            intersection = int((pred & target).sum())
            pred_area, gt_area = int(pred.sum()), int(target.sum())
            dice = 200.0 * intersection / (pred_area + gt_area)
            if method == "uniform_logits":
                assert abs(dice - reference[episode_id]) < 1e-8
            rows.append({
                "id": episode_id,
                "dataset": support_rows[0]["dataset"],
                "class_name": support_rows[0]["class_name"],
                "patient": support_rows[0]["patient"],
                "method": method,
                "dice": dice,
                "delta": dice - reference[episode_id],
                "intersection": intersection,
                "pred_area": pred_area,
                "gt_area": gt_area,
            })
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["class_name"], row["method"])].append(row)
    summary = []
    for (dataset, class_name, method), entries in sorted(grouped.items()):
        numerator = sum(row["intersection"] for row in entries)
        denominator = sum(row["pred_area"] + row["gt_area"] for row in entries)
        summary.append({
            "dataset": dataset, "class_name": class_name, "method": method,
            "n": len(entries),
            "mean_dice": float(np.mean([row["dice"] for row in entries])),
            "pooled_dice": 200.0 * numerator / denominator,
            "mean_delta": float(np.mean([row["delta"] for row in entries])),
            "improved": sum(row["delta"] > 1e-8 for row in entries),
            "worse": sum(row["delta"] < -1e-8 for row in entries),
        })
    for name, data in [("fusion_episodes.csv", rows), ("fusion_summary.csv", summary)]:
        with (args.root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with (args.root / "fusion_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
