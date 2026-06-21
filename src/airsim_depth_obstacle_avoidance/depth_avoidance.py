"""AirSim DepthPerspective obstacle avoidance advisor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).with_name("sample_data") / "airsim_depth_strips.csv"


def load_rows(path: Path = DATA) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def advise(rows: list[dict[str, float]]) -> list[dict[str, float | str]]:
    out = []
    for r in rows:
        min_depth = min(r["left_depth_m"], r["center_depth_m"], r["right_depth_m"])
        collision_risk = max(0.0, (5.0 - min_depth) / 5.0) * min(r["forward_speed_mps"] / 4.0, 1.2)
        if r["center_depth_m"] < 3.0:
            action = "turn_right" if r["right_depth_m"] > r["left_depth_m"] else "turn_left"
        elif collision_risk > 0.35:
            action = "slow_down"
        else:
            action = "keep_course"
        out.append({**r, "min_depth_m": round(min_depth, 3), "collision_risk": round(float(collision_risk), 4), "avoidance_action": action})
    return out


def plot(rows: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    t = [r["time_s"] for r in rows]
    p = output / "airsim_depth_runtime_view.png"
    depth_image = np.array([[r["left_depth_m"], r["center_depth_m"], r["right_depth_m"]] for r in rows], dtype=float)
    plt.figure(figsize=(8, 5))
    plt.imshow(depth_image.T, cmap="turbo", aspect="auto")
    plt.yticks([0, 1, 2], ["left", "center", "right"])
    plt.xticks(range(len(rows)), [str(r["frame"]) for r in rows])
    plt.colorbar(label="DepthPerspective distance (m)")
    plt.title("AirSim DepthPerspective obstacle view")
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    p = output / "airsim_avoidance_action_curve.png"
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, [r["collision_risk"] for r in rows], marker="o", color="#eb5757", label="risk")
    plt.plot(t, [r["center_depth_m"] for r in rows], marker="s", color="#2f80ed", label="center depth")
    plt.xlabel("time (s)")
    plt.title("AirSim depth risk and avoidance decision")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    paths.append(p)
    return paths


def run(output: Path) -> dict[str, object]:
    rows = advise(load_rows())
    files = plot(rows, output)
    csv_path = output / "airsim_depth_avoidance_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    files.append(csv_path)
    report = {
        "source": "AirSim DepthPerspective front camera strip log",
        "records": len(rows),
        "min_center_depth_m": min(float(r["center_depth_m"]) for r in rows),
        "turn_right_frames": sum(r["avoidance_action"] == "turn_right" for r in rows),
        "generated_files": [p.name for p in files],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/airsim_depth_obstacle_avoidance"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
