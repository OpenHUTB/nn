"""Visualization helpers for road-surface slip estimation and ABS control."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controller import BrakingResult
from scenario import BrakingScenario, friction_profile


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _active_slice(result: BrakingResult) -> slice:
    stop_candidates = np.where(result.speed_mps <= 0.05)[0]
    if len(stop_candidates) == 0:
        return slice(None)
    return slice(0, int(stop_candidates[0]) + 1)


def plot_speed_distance(baseline: BrakingResult, optimized: BrakingResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "speed_distance_comparison.png"

    plt.figure(figsize=(9.8, 5.8))
    plt.plot(baseline.position_m[_active_slice(baseline)], baseline.speed_mps[_active_slice(baseline)], color="#eb5757", linewidth=2.2, label="locked brake")
    plt.plot(optimized.position_m[_active_slice(optimized)], optimized.speed_mps[_active_slice(optimized)], color="#27ae60", linewidth=2.2, label="adaptive ABS")
    plt.axvline(baseline.stopping_distance_m, color="#eb5757", linestyle="--", alpha=0.7)
    plt.axvline(optimized.stopping_distance_m, color="#27ae60", linestyle="--", alpha=0.7)
    plt.title("Emergency braking speed-distance comparison")
    plt.xlabel("distance / m")
    plt.ylabel("speed / m/s")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_friction_estimation(scenario: BrakingScenario, optimized: BrakingResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "friction_estimation.png"

    active = _active_slice(optimized)
    distance = optimized.position_m[active]
    road_mu = friction_profile(distance, scenario)
    plt.figure(figsize=(9.8, 5.6))
    plt.plot(distance, road_mu, color="#222222", linewidth=2.2, label="true road friction")
    plt.plot(distance, optimized.estimated_mu[active], color="#2f80ed", linewidth=2.0, label="estimated friction")
    for segment in scenario.road_segments:
        plt.axvspan(segment.start_m, segment.end_m, alpha=0.08, label=segment.name)
    plt.title("Road friction estimation during braking")
    plt.xlabel("distance / m")
    plt.ylabel("friction coefficient mu")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_slip_control(baseline: BrakingResult, optimized: BrakingResult, scenario: BrakingScenario, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "slip_control_response.png"

    plt.figure(figsize=(9.8, 5.8))
    plt.plot(baseline.time[_active_slice(baseline)], baseline.slip_ratio[_active_slice(baseline)], color="#eb5757", linewidth=2.0, label="locked brake slip")
    plt.plot(optimized.time[_active_slice(optimized)], optimized.slip_ratio[_active_slice(optimized)], color="#27ae60", linewidth=2.0, label="adaptive ABS slip")
    plt.axhline(scenario.optimal_slip, color="#222222", linestyle="--", linewidth=1.6, label="optimal slip")
    plt.title("Wheel slip control response")
    plt.xlabel("time / s")
    plt.ylabel("slip ratio")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: BrakingResult, optimized: BrakingResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"

    labels = ["stopping distance", "stopping time", "max slip", "slip error"]
    baseline_values = [
        baseline.stopping_distance_m,
        baseline.stopping_time_s,
        baseline.max_slip_ratio,
        baseline.mean_abs_slip_error,
    ]
    optimized_values = [
        optimized.stopping_distance_m,
        optimized.stopping_time_s,
        optimized.max_slip_ratio,
        optimized.mean_abs_slip_error,
    ]

    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10.0, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="locked brake")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="adaptive ABS")
    plt.xticks(x, labels)
    plt.ylabel("metric value")
    plt.title("ABS braking metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_timeseries_csv(baseline: BrakingResult, optimized: BrakingResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "braking_timeseries.csv"
    table = np.column_stack(
        [
            baseline.time,
            baseline.position_m,
            baseline.speed_mps,
            baseline.slip_ratio,
            optimized.position_m,
            optimized.speed_mps,
            optimized.slip_ratio,
            optimized.estimated_mu,
        ]
    )
    header = (
        "time_s,locked_position_m,locked_speed_mps,locked_slip,"
        "abs_position_m,abs_speed_mps,abs_slip,abs_estimated_mu"
    )
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.6f")
    return path


def create_visualizations(
    scenario: BrakingScenario,
    baseline: BrakingResult,
    optimized: BrakingResult,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_speed_distance(baseline, optimized, output_dir),
        plot_friction_estimation(scenario, optimized, output_dir),
        plot_slip_control(baseline, optimized, scenario, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        write_timeseries_csv(baseline, optimized, output_dir),
    ]
