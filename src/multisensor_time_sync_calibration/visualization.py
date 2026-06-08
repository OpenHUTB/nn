"""Visualization helpers for the multi-sensor time sync demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calibration import CalibrationResult
from scenario import SyncScenario, interpolate_positions


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_trajectory_alignment(scenario: SyncScenario, result: CalibrationResult, output_dir: Path) -> Path:
    """Compare raw asynchronous measurements with the synchronized fused track."""
    _prepare_output_dir(output_dir)
    path = output_dir / "trajectory_alignment.png"

    corrected_lidar_time = result.fused_time + result.estimated_lidar_delay
    corrected_lidar = interpolate_positions(scenario.lidar, corrected_lidar_time)

    plt.figure(figsize=(9.6, 6.2))
    plt.plot(scenario.truth_positions[:, 0], scenario.truth_positions[:, 1], color="#222222", linewidth=2.4, label="ground truth")
    plt.scatter(
        scenario.camera.positions[:, 0],
        scenario.camera.positions[:, 1],
        s=13,
        color="#2f80ed",
        alpha=0.42,
        label="camera samples",
    )
    plt.scatter(
        scenario.lidar.positions[:, 0],
        scenario.lidar.positions[:, 1],
        s=14,
        color="#eb5757",
        alpha=0.36,
        label="lidar before sync",
    )
    plt.plot(corrected_lidar[:, 0], corrected_lidar[:, 1], color="#f2994a", linewidth=1.8, label="lidar after sync")
    plt.plot(result.fused_positions[:, 0], result.fused_positions[:, 1], color="#27ae60", linewidth=2.4, label="fused trajectory")
    plt.title("Trajectory alignment after timestamp calibration")
    plt.xlabel("x position / m")
    plt.ylabel("y position / m")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_offset_search_curve(scenario: SyncScenario, result: CalibrationResult, output_dir: Path) -> Path:
    """Show the delay search curve and the selected timestamp offset."""
    _prepare_output_dir(output_dir)
    path = output_dir / "offset_search_curve.png"

    best_error = np.min(result.alignment_errors)
    plt.figure(figsize=(9.6, 5.6))
    plt.plot(result.candidate_offsets, result.alignment_errors, color="#7f52ff", linewidth=2.2)
    plt.axvline(scenario.true_lidar_delay, color="#27ae60", linestyle="--", linewidth=1.8, label="true delay")
    plt.axvline(result.estimated_lidar_delay, color="#eb5757", linestyle=":", linewidth=2.2, label="estimated delay")
    plt.scatter([result.estimated_lidar_delay], [best_error], color="#eb5757", s=52, zorder=3)
    plt.title("Grid search for lidar timestamp delay")
    plt.xlabel("candidate lidar delay / s")
    plt.ylabel("camera-lidar alignment RMSE / m")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_rmse_comparison(result: CalibrationResult, output_dir: Path) -> Path:
    """Compare fusion error before and after timestamp calibration."""
    _prepare_output_dir(output_dir)
    path = output_dir / "rmse_comparison.png"

    values = [result.before_rmse, result.after_rmse]
    labels = ["before sync", "after sync"]
    colors = ["#eb5757", "#27ae60"]

    plt.figure(figsize=(7.2, 5.4))
    bars = plt.bar(labels, values, color=colors, width=0.52)
    plt.ylabel("trajectory RMSE / m")
    plt.title("Fused trajectory error reduction")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value + 0.025, f"{value:.3f} m", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_offset_search_csv(result: CalibrationResult, output_dir: Path) -> Path:
    """Export the offset residual table for repeatable inspection."""
    _prepare_output_dir(output_dir)
    path = output_dir / "offset_search.csv"
    rows = np.column_stack([result.candidate_offsets, result.alignment_errors])
    np.savetxt(path, rows, delimiter=",", header="candidate_delay_s,alignment_rmse_m", comments="", fmt="%.6f")
    return path


def create_visualizations(scenario: SyncScenario, result: CalibrationResult, output_dir: Path) -> list[Path]:
    """Create all visual artifacts for the project report."""
    return [
        plot_trajectory_alignment(scenario, result, output_dir),
        plot_offset_search_curve(scenario, result, output_dir),
        plot_rmse_comparison(result, output_dir),
        write_offset_search_csv(result, output_dir),
    ]
