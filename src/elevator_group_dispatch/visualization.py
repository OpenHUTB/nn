"""Visualization for elevator group dispatch optimization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispatcher import DispatchResult
from scenario import ElevatorScenario


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_wait_distribution(baseline: DispatchResult, optimized: DispatchResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "wait_time_distribution.png"

    baseline_wait = [record.wait_s for record in baseline.records]
    optimized_wait = [record.wait_s for record in optimized.records]
    bins = np.arange(0, max(max(baseline_wait), max(optimized_wait)) + 30, 20)

    plt.figure(figsize=(9.6, 5.8))
    plt.hist(baseline_wait, bins=bins, color="#eb5757", alpha=0.65, label="nearest car")
    plt.hist(optimized_wait, bins=bins, color="#27ae60", alpha=0.65, label="destination-aware")
    plt.title("Passenger wait time distribution")
    plt.xlabel("waiting time / s")
    plt.ylabel("request count")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_elevator_timelines(scenario: ElevatorScenario, optimized: DispatchResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "elevator_timeline.png"

    plt.figure(figsize=(10.5, 6.2))
    colors = ["#2f80ed", "#27ae60", "#f2994a", "#9b51e0", "#eb5757"]
    for record in optimized.records:
        color = colors[record.elevator_id % len(colors)]
        plt.plot(
            [record.pickup_s / 60.0, record.dropoff_s / 60.0],
            [record.origin_floor, record.destination_floor],
            color=color,
            alpha=0.62,
            linewidth=1.6 + 0.25 * record.passengers,
        )
    plt.title("Destination-aware elevator service timeline")
    plt.xlabel("time / min")
    plt.ylabel("floor")
    plt.yticks(range(1, scenario.floor_count + 1, 2))
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_floor_demand_heatmap(scenario: ElevatorScenario, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "floor_demand_heatmap.png"

    bin_minutes = 5
    bins = scenario.horizon_s // (bin_minutes * 60) + 1
    heatmap = np.zeros((scenario.floor_count, bins))
    for request in scenario.requests:
        time_bin = min(bins - 1, request.arrival_s // (bin_minutes * 60))
        heatmap[request.origin_floor - 1, time_bin] += request.passengers

    plt.figure(figsize=(10.2, 6.0))
    plt.imshow(heatmap, aspect="auto", origin="lower", cmap="YlGnBu")
    plt.colorbar(label="passenger count")
    plt.title("Origin-floor demand heatmap")
    plt.xlabel("time bin / 5 min")
    plt.ylabel("origin floor")
    plt.yticks(np.arange(0, scenario.floor_count, 2), np.arange(1, scenario.floor_count + 1, 2))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: DispatchResult, optimized: DispatchResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"

    labels = ["mean wait", "p90 wait", "long waits", "energy index"]
    baseline_values = [
        baseline.mean_wait_s,
        baseline.p90_wait_s,
        baseline.long_wait_count,
        baseline.energy_index / 10.0,
    ]
    optimized_values = [
        optimized.mean_wait_s,
        optimized.p90_wait_s,
        optimized.long_wait_count,
        optimized.energy_index / 10.0,
    ]

    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10.0, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="nearest car")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="destination-aware")
    plt.xticks(x, labels)
    plt.ylabel("metric value")
    plt.title("Elevator dispatch metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_assignment_csv(baseline: DispatchResult, optimized: DispatchResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "dispatch_assignments.csv"
    optimized_by_id = {record.request_id: record for record in optimized.records}
    rows = []
    for base in baseline.records:
        opt = optimized_by_id[base.request_id]
        rows.append(
            [
                base.request_id,
                base.arrival_s,
                base.origin_floor,
                base.destination_floor,
                base.passengers,
                base.elevator_id,
                base.pickup_s,
                base.dropoff_s,
                base.wait_s,
                opt.elevator_id,
                opt.pickup_s,
                opt.dropoff_s,
                opt.wait_s,
            ]
        )
    header = (
        "request_id,arrival_s,origin_floor,destination_floor,passengers,"
        "nearest_elevator,nearest_pickup_s,nearest_dropoff_s,nearest_wait_s,"
        "optimized_elevator,optimized_pickup_s,optimized_dropoff_s,optimized_wait_s"
    )
    np.savetxt(path, np.array(rows), delimiter=",", header=header, comments="", fmt="%.3f")
    return path


def create_visualizations(
    scenario: ElevatorScenario,
    baseline: DispatchResult,
    optimized: DispatchResult,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_wait_distribution(baseline, optimized, output_dir),
        plot_elevator_timelines(scenario, optimized, output_dir),
        plot_floor_demand_heatmap(scenario, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        write_assignment_csv(baseline, optimized, output_dir),
    ]
