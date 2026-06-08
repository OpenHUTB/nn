"""Visualization for eco-driving speed advisor."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from advisor import Trajectory, signal_compliance
from scenario import CorridorScenario


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_speed_profile(eco: Trajectory, baseline: Trajectory, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=150)
    ax.plot(eco.position_m, eco.speed_mps, linewidth=2.4, label="eco speed advisor", color="#1f77b4")
    ax.plot(baseline.position_m, baseline.speed_mps, linewidth=1.8, label="fixed cruise baseline", color="#d95f02")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed profile along signalized corridor")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_time_space_diagram(scenario: CorridorScenario, eco: Trajectory, baseline: Trajectory, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=150)
    ax.plot(eco.time_s, eco.position_m, linewidth=2.5, label="eco speed advisor", color="#1f77b4")
    ax.plot(baseline.time_s, baseline.position_m, linewidth=1.9, label="fixed cruise baseline", color="#d95f02")

    for light in scenario.lights:
        ax.axhline(light.position_m, color="#555555", linestyle="--", linewidth=0.8)
        max_time = max(float(eco.time_s[-1]), float(baseline.time_s[-1]))
        cycle = 0
        while cycle * light.cycle_s + light.offset_s < max_time + light.cycle_s:
            start = cycle * light.cycle_s + light.offset_s
            end = start + light.green_s
            ax.fill_between([start, end], light.position_m - 10, light.position_m + 10, color="#2ca02c", alpha=0.16)
            cycle += 1
        ax.text(0.5, light.position_m + 8, light.light_id, fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Time-space diagram with green signal windows")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_metrics_chart(scenario: CorridorScenario, eco: Trajectory, baseline: Trajectory, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    eco_green = sum(signal_compliance(scenario, eco).values())
    base_green = sum(signal_compliance(scenario, baseline).values())

    names = ["energy", "travel time", "stops", "green passes"]
    eco_values = [eco.energy, eco.time_s[-1], eco.stops, eco_green]
    base_values = [baseline.energy, baseline.time_s[-1], baseline.stops, base_green]
    normalized_eco = np.asarray(eco_values) / np.maximum(base_values, 1e-6)
    normalized_base = np.ones(len(names))

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=150)
    ax.bar(x - width / 2, normalized_base, width, label="fixed cruise baseline", color="#d95f02")
    ax.bar(x + width / 2, normalized_eco, width, label="eco speed advisor", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("Normalized value (baseline = 1.0)")
    ax.set_title("Eco-driving performance comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
