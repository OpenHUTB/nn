"""Visualization helpers for bus headway control."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controller import HeadwayResult
from scenario import BusScenario


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_time_space(scenario: BusScenario, baseline: HeadwayResult, optimized: HeadwayResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "time_space_diagram.png"

    plt.figure(figsize=(10.0, 6.2))
    for run in baseline.runs:
        plt.plot(run.departure_min, scenario.stop_positions_km, color="#eb5757", alpha=0.32, linewidth=1.4)
    for run in optimized.runs:
        plt.plot(run.departure_min, scenario.stop_positions_km, color="#2f80ed", alpha=0.78, linewidth=1.7)
    plt.title("Bus time-space diagram")
    plt.xlabel("time / min")
    plt.ylabel("corridor position / km")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_headway_heatmap(result: HeadwayResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "headway_heatmap.png"

    plt.figure(figsize=(9.8, 5.6))
    plt.imshow(result.headways_by_stop.T, aspect="auto", origin="lower", cmap="RdYlGn", vmin=2.0, vmax=10.0)
    plt.colorbar(label="headway / min")
    plt.title("Adaptive holding headway heatmap")
    plt.xlabel("bus pair index")
    plt.ylabel("stop index")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_passenger_wait_by_stop(baseline: HeadwayResult, optimized: HeadwayResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "passenger_wait_by_stop.png"

    baseline_wait = np.sum([run.wait_cost_by_stop for run in baseline.runs], axis=0)
    optimized_wait = np.sum([run.wait_cost_by_stop for run in optimized.runs], axis=0)
    stops = np.arange(1, len(baseline_wait) + 1)
    width = 0.36

    plt.figure(figsize=(10.0, 5.8))
    plt.bar(stops - width / 2, baseline_wait, width=width, color="#eb5757", label="no control")
    plt.bar(stops + width / 2, optimized_wait, width=width, color="#27ae60", label="adaptive holding")
    plt.title("Passenger waiting cost by stop")
    plt.xlabel("stop index")
    plt.ylabel("passenger waiting cost / min")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: HeadwayResult, optimized: HeadwayResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"
    labels = ["headway std", "bunching", "avg wait", "reliability"]
    baseline_values = [
        baseline.headway_std_min,
        baseline.bunching_count,
        baseline.average_passenger_wait_min,
        baseline.service_reliability_score / 10.0,
    ]
    optimized_values = [
        optimized.headway_std_min,
        optimized.bunching_count,
        optimized.average_passenger_wait_min,
        optimized.service_reliability_score / 10.0,
    ]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10.0, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="no control")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="adaptive holding")
    plt.xticks(x, labels)
    plt.ylabel("metric value")
    plt.title("Bus headway control metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_headway_csv(baseline: HeadwayResult, optimized: HeadwayResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "headway_by_stop.csv"
    rows = []
    for pair in range(optimized.headways_by_stop.shape[0]):
        for stop in range(optimized.headways_by_stop.shape[1]):
            rows.append(
                [
                    pair + 1,
                    stop + 1,
                    baseline.headways_by_stop[pair, stop],
                    optimized.headways_by_stop[pair, stop],
                ]
            )
    header = "bus_pair,stop_index,no_control_headway_min,adaptive_holding_headway_min"
    np.savetxt(path, np.array(rows), delimiter=",", header=header, comments="", fmt="%.4f")
    return path


def create_visualizations(
    scenario: BusScenario,
    baseline: HeadwayResult,
    optimized: HeadwayResult,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_time_space(scenario, baseline, optimized, output_dir),
        plot_headway_heatmap(optimized, output_dir),
        plot_passenger_wait_by_stop(baseline, optimized, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        write_headway_csv(baseline, optimized, output_dir),
    ]
