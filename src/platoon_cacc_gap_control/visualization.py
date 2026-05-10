"""Plots for the platoon CACC project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controller import SimulationResult
from scenario import PlatoonScenario


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_speed_profiles(baseline: SimulationResult, optimized: SimulationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "speed_profiles.png"

    plt.figure(figsize=(10.0, 6.0))
    plt.plot(baseline.time, baseline.speeds[:, 0], color="#222222", linewidth=2.3, label="leader")
    plt.plot(baseline.time, baseline.speeds[:, -1], color="#eb5757", linewidth=2.0, label="last car ACC")
    plt.plot(optimized.time, optimized.speeds[:, -1], color="#27ae60", linewidth=2.0, label="last car CACC")
    plt.title("Speed disturbance propagation")
    plt.xlabel("time / s")
    plt.ylabel("speed / m/s")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_gap_error_heatmap(result: SimulationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "cacc_gap_error_heatmap.png"

    plt.figure(figsize=(10.0, 5.2))
    vmax = max(2.0, float(np.percentile(np.abs(result.gap_errors), 98)))
    plt.imshow(
        result.gap_errors.T,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
        extent=[result.time[0], result.time[-1], 1, result.gap_errors.shape[1]],
    )
    plt.colorbar(label="gap error / m")
    plt.title("CACC follower gap error heatmap")
    plt.xlabel("time / s")
    plt.ylabel("follower index")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: SimulationResult, optimized: SimulationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"

    labels = ["mean gap error", "max gap error", "comfort cost", "speed amplification"]
    baseline_values = [
        baseline.mean_abs_gap_error,
        baseline.max_abs_gap_error,
        baseline.comfort_cost,
        baseline.speed_std_amplification,
    ]
    optimized_values = [
        optimized.mean_abs_gap_error,
        optimized.max_abs_gap_error,
        optimized.comfort_cost,
        optimized.speed_std_amplification,
    ]

    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10.0, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="ACC baseline")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="CACC optimized")
    plt.xticks(x, labels)
    plt.ylabel("metric value")
    plt.title("Platoon control metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_space_time(scenario: PlatoonScenario, result: SimulationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "space_time_trajectories.png"

    plt.figure(figsize=(9.8, 6.0))
    for vehicle in range(result.positions.shape[1]):
        label = "leader" if vehicle == 0 else f"car {vehicle + 1}"
        plt.plot(result.time, result.positions[:, vehicle], linewidth=1.8, label=label)
    for start, end in scenario.dropout_windows:
        plt.axvspan(start, end, color="#f2c94c", alpha=0.22)
    plt.title("CACC space-time trajectories with communication dropout windows")
    plt.xlabel("time / s")
    plt.ylabel("position / m")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_gap_error_csv(result: SimulationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "cacc_gap_errors.csv"
    header = ",".join(["time_s"] + [f"gap_error_car_{index + 2}_m" for index in range(result.gap_errors.shape[1])])
    table = np.column_stack([result.time, result.gap_errors])
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.6f")
    return path


def create_visualizations(scenario: PlatoonScenario, baseline: SimulationResult, optimized: SimulationResult, output_dir: Path) -> list[Path]:
    return [
        plot_speed_profiles(baseline, optimized, output_dir),
        plot_gap_error_heatmap(optimized, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        plot_space_time(scenario, optimized, output_dir),
        write_gap_error_csv(optimized, output_dir),
    ]
