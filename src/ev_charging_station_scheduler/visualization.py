"""Visualization utilities for EV charging station scheduling."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scheduler import ScheduleResult
from scenario import ChargingScenario


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _hour_ticks(time_min: np.ndarray) -> np.ndarray:
    return np.arange(0, int(time_min[-1]) + 1, 180)


def plot_load_profiles(scenario: ChargingScenario, baseline: ScheduleResult, optimized: ScheduleResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "load_profile_comparison.png"
    hours = baseline.time_min / 60.0

    plt.figure(figsize=(10.5, 6.0))
    plt.plot(hours, baseline.net_grid_load_kw, color="#eb5757", linewidth=2.0, label="FCFS net grid load")
    plt.plot(hours, optimized.net_grid_load_kw, color="#27ae60", linewidth=2.0, label="load-aware net grid load")
    plt.plot(hours, scenario.base_load_kw, color="#555555", linewidth=1.4, alpha=0.75, label="base load")
    plt.axhline(scenario.grid_limit_kw, color="#222222", linestyle="--", linewidth=1.6, label="grid limit")
    plt.fill_between(hours, 0, scenario.solar_kw, color="#f2c94c", alpha=0.25, label="solar generation")
    plt.title("Charging station net grid load")
    plt.xlabel("time / h")
    plt.ylabel("power / kW")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_charger_occupancy(baseline: ScheduleResult, optimized: ScheduleResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "charger_occupancy.png"
    hours = baseline.time_min / 60.0

    plt.figure(figsize=(10.5, 5.6))
    plt.step(hours, baseline.charger_occupancy, where="post", color="#eb5757", linewidth=1.8, label="FCFS")
    plt.step(hours, optimized.charger_occupancy, where="post", color="#2f80ed", linewidth=1.8, label="load-aware")
    plt.title("Charger occupancy over the day")
    plt.xlabel("time / h")
    plt.ylabel("active chargers")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_waiting_time_distribution(baseline: ScheduleResult, optimized: ScheduleResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "waiting_time_distribution.png"

    baseline_wait = [item.waiting_min for item in baseline.vehicle_schedules if item.waiting_min is not None]
    optimized_wait = [item.waiting_min for item in optimized.vehicle_schedules if item.waiting_min is not None]
    bins = np.arange(0, max(max(baseline_wait), max(optimized_wait)) + 35, 30)

    plt.figure(figsize=(9.4, 5.6))
    plt.hist(baseline_wait, bins=bins, alpha=0.65, color="#eb5757", label="FCFS")
    plt.hist(optimized_wait, bins=bins, alpha=0.65, color="#27ae60", label="load-aware")
    plt.title("Vehicle waiting time distribution")
    plt.xlabel("waiting time / min")
    plt.ylabel("vehicle count")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: ScheduleResult, optimized: ScheduleResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"

    labels = ["avg wait", "peak load", "overload min", "load variance"]
    baseline_values = [
        baseline.average_wait_min,
        baseline.peak_grid_load_kw,
        baseline.overload_minutes,
        baseline.load_variance / 10.0,
    ]
    optimized_values = [
        optimized.average_wait_min,
        optimized.peak_grid_load_kw,
        optimized.overload_minutes,
        optimized.load_variance / 10.0,
    ]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10.0, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="FCFS")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="load-aware")
    plt.xticks(x, labels)
    plt.ylabel("metric value")
    plt.title("Scheduling metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_schedule_csv(baseline: ScheduleResult, optimized: ScheduleResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "vehicle_schedule.csv"
    rows = []
    by_vehicle = {item.vehicle_id: item for item in optimized.vehicle_schedules}
    for base in baseline.vehicle_schedules:
        opt = by_vehicle[base.vehicle_id]
        rows.append(
            [
                base.vehicle_id,
                base.arrival_min,
                base.deadline_min,
                base.energy_kwh,
                base.priority,
                -1 if base.start_min is None else base.start_min,
                -1 if base.finish_min is None else base.finish_min,
                -1 if base.waiting_min is None else base.waiting_min,
                -1 if opt.start_min is None else opt.start_min,
                -1 if opt.finish_min is None else opt.finish_min,
                -1 if opt.waiting_min is None else opt.waiting_min,
            ]
        )
    header = (
        "vehicle_id,arrival_min,deadline_min,energy_kwh,priority,"
        "fcfs_start_min,fcfs_finish_min,fcfs_wait_min,"
        "optimized_start_min,optimized_finish_min,optimized_wait_min"
    )
    np.savetxt(path, np.array(rows), delimiter=",", header=header, comments="", fmt="%.3f")
    return path


def create_visualizations(
    scenario: ChargingScenario,
    baseline: ScheduleResult,
    optimized: ScheduleResult,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_load_profiles(scenario, baseline, optimized, output_dir),
        plot_charger_occupancy(baseline, optimized, output_dir),
        plot_waiting_time_distribution(baseline, optimized, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        write_schedule_csv(baseline, optimized, output_dir),
    ]
