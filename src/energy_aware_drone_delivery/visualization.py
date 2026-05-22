"""Visualization for energy-aware drone delivery."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from planner import MissionPlan, battery_trace
from scenario import DeliveryScenario


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def draw_city_background(ax, scenario: DeliveryScenario) -> None:
    ax.imshow(scenario.grid, cmap="Greys", origin="upper", vmin=0, vmax=1, alpha=0.88)
    ax.set_xticks(np.arange(-0.5, scenario.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, scenario.height, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=0.3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def save_route_map(scenario: DeliveryScenario, plan: MissionPlan, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=150)
    draw_city_background(ax, scenario)

    # Wind field sampled for readability.
    step = 4
    rows, cols = np.mgrid[0 : scenario.height : step, 0 : scenario.width : step]
    ax.quiver(cols, rows, scenario.wind_u[::step, ::step], -scenario.wind_v[::step, ::step], color="#7f8c8d", alpha=0.55)

    for station in scenario.stations:
        ax.scatter(station.cell[1], station.cell[0], marker="P", s=150, c="#2ca02c", edgecolors="white", linewidths=0.8, label="charging station" if station == scenario.stations[0] else None)
        ax.text(station.cell[1] + 0.4, station.cell[0] + 0.3, station.station_id, fontsize=8)

    for task in scenario.tasks:
        ax.scatter(task.pickup[1], task.pickup[0], marker="s", s=100, c="#f0a43a", edgecolors="black", linewidths=0.4)
        ax.scatter(task.dropoff[1], task.dropoff[0], marker="*", s=170, c="#c63737", edgecolors="black", linewidths=0.4)
        ax.text(task.pickup[1] + 0.2, task.pickup[0] - 0.5, task.task_id, fontsize=8)

    route = plan.route
    if route:
        xs = [cell[1] for cell in route]
        ys = [cell[0] for cell in route]
        ax.plot(xs, ys, color="#1f77b4", linewidth=2.6, label="energy-aware route")
    ax.scatter(scenario.drone.start[1], scenario.drone.start[0], marker="o", s=130, c="#1f77b4", edgecolors="white", linewidths=0.9, label="drone start")

    ax.set_title("Energy-aware drone delivery route with wind field and charging stations")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_battery_profile(scenario: DeliveryScenario, plan: MissionPlan, baseline: MissionPlan, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    smart = battery_trace(scenario, plan)
    naive = battery_trace(scenario, baseline)
    reserve = scenario.drone.battery_capacity * scenario.drone.reserve_ratio

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=150)
    ax.plot(range(len(smart)), smart, marker="o", linewidth=2.4, color="#1f77b4", label="energy-aware plan")
    ax.plot(range(len(naive)), naive, marker="x", linewidth=2.0, color="#d95f02", label="naive task order")
    ax.axhline(reserve, color="#c63737", linestyle="--", label="safety reserve")
    ax.set_xlabel("Mission segment")
    ax.set_ylabel("Remaining battery")
    ax.set_title("Battery profile with reserve-aware charging")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_energy_breakdown(plan: MissionPlan, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    labels = [segment.label for segment in plan.segments]
    energies = [segment.energy for segment in plan.segments]
    colors = ["#2ca02c" if segment.recharged else "#1f77b4" for segment in plan.segments]

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=150)
    ax.bar(range(len(labels)), energies, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Energy cost")
    ax.set_title("Energy cost by mission segment")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
