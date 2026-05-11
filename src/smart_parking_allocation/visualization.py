"""Visualization utilities for smart parking allocation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from allocator import AllocationResult
from scenario import ParkingScenario


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_parking_layout(scenario: ParkingScenario, optimized: AllocationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "parking_layout_allocation.png"
    assigned_spots = {item.spot_id for item in optimized.assignments if item.spot_id is not None}

    plt.figure(figsize=(10.2, 6.0))
    for spot in scenario.spots:
        color = "#27ae60" if spot.spot_id in assigned_spots else "#bdbdbd"
        marker = "P"
        if spot.has_charger:
            marker = "s"
        if spot.is_accessible:
            marker = "*"
        plt.scatter(spot.x_m, spot.y_m, s=68, marker=marker, color=color, edgecolor="#222222", linewidth=0.4)
        plt.text(spot.x_m + 0.7, spot.y_m + 0.6, spot.zone, fontsize=7)
    plt.scatter([scenario.entrance_xy[0]], [scenario.entrance_xy[1]], color="#2f80ed", s=130, marker=">", label="entrance")
    plt.scatter([scenario.exit_xy[0]], [scenario.exit_xy[1]], color="#eb5757", s=130, marker="<", label="exit")
    plt.title("Parking lot layout and allocated spaces")
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_zone_occupancy(baseline: AllocationResult, optimized: AllocationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "zone_occupancy_curve.png"
    slots = len(baseline.total_occupancy)
    time_hours = np.arange(slots) * 5.0 / 60.0

    plt.figure(figsize=(10.4, 6.0))
    plt.plot(time_hours, baseline.total_occupancy, color="#eb5757", linewidth=2.0, label="nearest total occupancy")
    plt.plot(time_hours, optimized.total_occupancy, color="#27ae60", linewidth=2.0, label="balanced total occupancy")
    for zone, values in optimized.occupancy_by_zone.items():
        plt.plot(time_hours, values, linewidth=1.2, alpha=0.65, label=f"zone {zone}")
    plt.title("Parking occupancy over time")
    plt.xlabel("time / h")
    plt.ylabel("occupied spaces")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_walking_distribution(baseline: AllocationResult, optimized: AllocationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "walking_distance_distribution.png"
    baseline_walk = [item.walking_distance_m for item in baseline.assignments if item.walking_distance_m is not None]
    optimized_walk = [item.walking_distance_m for item in optimized.assignments if item.walking_distance_m is not None]
    bins = np.arange(0, max(max(baseline_walk), max(optimized_walk)) + 20, 15)

    plt.figure(figsize=(9.6, 5.8))
    plt.hist(baseline_walk, bins=bins, alpha=0.65, color="#eb5757", label="nearest")
    plt.hist(optimized_walk, bins=bins, alpha=0.65, color="#27ae60", label="balanced")
    plt.title("Walking distance distribution")
    plt.xlabel("walking distance to exit / m")
    plt.ylabel("vehicle count")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: AllocationResult, optimized: AllocationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"
    labels = ["avg walk", "p90 walk", "zone imbalance", "traffic km"]
    baseline_values = [
        baseline.average_walking_distance_m,
        baseline.p90_walking_distance_m,
        baseline.zone_imbalance_index * 100.0,
        baseline.traffic_distance_m / 1000.0,
    ]
    optimized_values = [
        optimized.average_walking_distance_m,
        optimized.p90_walking_distance_m,
        optimized.zone_imbalance_index * 100.0,
        optimized.traffic_distance_m / 1000.0,
    ]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10.0, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="nearest")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="balanced")
    plt.xticks(x, labels)
    plt.ylabel("metric value")
    plt.title("Parking allocation metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_assignment_csv(baseline: AllocationResult, optimized: AllocationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "parking_assignments.csv"
    optimized_by_id = {item.vehicle_id: item for item in optimized.assignments}
    rows = []
    for base in baseline.assignments:
        opt = optimized_by_id[base.vehicle_id]
        rows.append(
            [
                base.vehicle_id,
                base.arrival_min,
                int(base.needs_charger),
                int(base.needs_accessible),
                -1 if base.spot_id is None else base.spot_id,
                "" if base.zone is None else base.zone,
                -1 if base.walking_distance_m is None else base.walking_distance_m,
                -1 if opt.spot_id is None else opt.spot_id,
                "" if opt.zone is None else opt.zone,
                -1 if opt.walking_distance_m is None else opt.walking_distance_m,
                opt.waiting_min,
            ]
        )
    header = (
        "vehicle_id,arrival_min,needs_charger,needs_accessible,"
        "nearest_spot,nearest_zone,nearest_walk_m,"
        "balanced_spot,balanced_zone,balanced_walk_m,balanced_wait_min"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")
    return path


def create_visualizations(
    scenario: ParkingScenario,
    baseline: AllocationResult,
    optimized: AllocationResult,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_parking_layout(scenario, optimized, output_dir),
        plot_zone_occupancy(baseline, optimized, output_dir),
        plot_walking_distribution(baseline, optimized, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        write_assignment_csv(baseline, optimized, output_dir),
    ]
