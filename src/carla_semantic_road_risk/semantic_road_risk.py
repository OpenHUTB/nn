"""CARLA semantic camera road occupancy and risk analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "sample_data" / "carla_semantic_segmentation.csv"
LABELS = {0: "unlabeled", 1: "road", 2: "lane_marking", 3: "vehicle", 4: "pedestrian", 5: "sidewalk"}


def generate_sample(path: Path = DATA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.zeros((18, 28), dtype=int)
    grid[7:18, :] = 1
    grid[6, :] = 2
    grid[12:15, 10:14] = 3
    grid[9:11, 20:22] = 3
    grid[8:10, 6:7] = 4
    grid[0:6, :] = 5
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(grid.tolist())


def load_grid(path: Path = DATA) -> np.ndarray:
    if not path.exists():
        generate_sample(path)
    with path.open(newline="", encoding="utf-8") as f:
        return np.array([[int(v) for v in row] for row in csv.reader(f)], dtype=int)


def analyze(grid: np.ndarray) -> dict[str, object]:
    road_mask = np.isin(grid, [1, 2])
    hazard_mask = np.isin(grid, [3, 4])
    lower_half = np.zeros_like(grid, dtype=bool)
    lower_half[grid.shape[0] // 2 :, :] = True
    road_pixels = int(road_mask.sum())
    hazard_pixels = int(hazard_mask.sum())
    drivable_ratio = float(road_pixels / grid.size)
    near_hazard_pixels = int((hazard_mask & lower_half).sum())
    risk_score = min(1.0, 0.55 * near_hazard_pixels / 20 + 0.35 * hazard_pixels / 30 + 0.10 * (1 - drivable_ratio))
    action = "slow_down" if risk_score > 0.55 else "watch" if risk_score > 0.30 else "keep_speed"
    return {
        "source": "CARLA SemanticSegmentation camera label grid",
        "image_shape": list(grid.shape),
        "road_pixels": road_pixels,
        "vehicle_pixels": int((grid == 3).sum()),
        "pedestrian_pixels": int((grid == 4).sum()),
        "near_hazard_pixels": near_hazard_pixels,
        "drivable_ratio": round(drivable_ratio, 4),
        "risk_score": round(risk_score, 4),
        "recommended_action": action,
    }


def plot(grid: np.ndarray, metrics: dict[str, object], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = np.array([
        [30, 30, 30],
        [70, 180, 90],
        [240, 220, 80],
        [230, 70, 70],
        [80, 140, 255],
        [150, 150, 150],
    ]) / 255.0
    rgb = colors[grid]
    p = output / "carla_semantic_camera_view.png"
    plt.figure(figsize=(8.4, 5.2))
    plt.imshow(rgb)
    plt.title("CARLA SemanticSegmentation camera view")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)

    risk = np.zeros_like(grid, dtype=float)
    risk[grid == 3] = 0.75
    risk[grid == 4] = 1.0
    risk[grid == 2] = 0.18
    p = output / "carla_semantic_risk_heatmap.png"
    plt.figure(figsize=(8.4, 5.2))
    plt.imshow(risk, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar(label="risk")
    plt.title(f"Road occupancy risk heatmap: action={metrics['recommended_action']}")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    return paths


def run(output: Path) -> dict[str, object]:
    grid = load_grid()
    metrics = analyze(grid)
    files = plot(grid, metrics, output)
    metrics["generated_files"] = [p.name for p in files]
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/carla_semantic_road_risk"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
