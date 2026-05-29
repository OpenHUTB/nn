"""Visualization helpers for sensor fusion fault diagnosis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fusion import FusionResult
from scenario import SENSORS, Scenario


COLORS = {
    "gps": "#d95f02",
    "radar": "#7570b3",
    "odometry": "#1b9e77",
}


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_trajectory_plot(scenario: Scenario, result: FusionResult, output_path: str | Path) -> Path:
    """Plot truth, raw sensors, naive average and robust fusion trajectory."""
    output = ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=150)
    ax.plot(scenario.truth[:, 0], scenario.truth[:, 1], color="#111111", linewidth=2.8, label="truth")
    ax.plot(result.naive_estimates[:, 0], result.naive_estimates[:, 1], color="#e6ab02", linewidth=1.8, label="naive average")
    ax.plot(result.estimates[:, 0], result.estimates[:, 1], color="#1f77b4", linewidth=2.4, label="robust fusion")

    for sensor in SENSORS:
        measurement = scenario.measurements[sensor]
        ax.scatter(measurement[::5, 0], measurement[::5, 1], s=9, alpha=0.35, color=COLORS[sensor], label=f"{sensor} measurements")

    ax.set_title("Robust sensor fusion trajectory under injected sensor faults")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_residual_plot(scenario: Scenario, result: FusionResult, output_path: str | Path) -> Path:
    """Plot residual timelines and mark injected fault windows."""
    output = ensure_parent(output_path)
    time = np.arange(scenario.steps) * scenario.dt
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.2), dpi=150, sharex=True)

    for ax, sensor in zip(axes, SENSORS):
        ax.plot(time, result.residuals[sensor], color=COLORS[sensor], linewidth=1.8, label=f"{sensor} residual")
        flags = result.fault_flags[sensor].astype(bool)
        ax.scatter(time[flags], result.residuals[sensor][flags], s=12, color="#c63737", label="diagnosed fault")
        for fault in scenario.faults:
            if fault.sensor == sensor:
                ax.axvspan(fault.start * scenario.dt, fault.end * scenario.dt, color="#f4a261", alpha=0.22, label=fault.fault_type)
        ax.set_ylabel("Residual")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Sensor residuals and detected fault intervals")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_rmse_chart(result: FusionResult, output_path: str | Path) -> Path:
    """Compare naive averaging and robust fusion RMSE."""
    output = ensure_parent(output_path)
    names = ["naive average", "robust fusion"]
    values = [result.rmse_naive, result.rmse_robust]

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=150)
    bars = ax.bar(names, values, color=["#e6ab02", "#1f77b4"], width=0.55)
    ax.set_ylabel("Position RMSE")
    ax.set_title("Fusion accuracy improvement")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
