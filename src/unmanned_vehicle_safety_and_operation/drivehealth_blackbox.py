"""DriveHealth integrated safety blackbox.

This module turns the small single-purpose demos in this folder into one
simulator-free project entrypoint. It generates a synthetic vehicle health
telemetry stream, scores subsystem risks, records blackbox events, and writes a
visual report pack under ``drivehealth_visualizations``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "drivehealth_visualizations"

RISK_COLUMNS = (
    "thermal_risk",
    "smoke_risk",
    "speed_risk",
    "brake_risk",
    "takeover_risk",
    "efficiency_risk",
)

RISK_NAMES = {
    "thermal_risk": "Thermal",
    "smoke_risk": "Smoke",
    "speed_risk": "Speed sensor",
    "brake_risk": "Brake margin",
    "takeover_risk": "Takeover",
    "efficiency_risk": "Efficiency",
}

RISK_WEIGHTS = {
    "thermal_risk": 0.18,
    "smoke_risk": 0.17,
    "speed_risk": 0.15,
    "brake_risk": 0.22,
    "takeover_risk": 0.16,
    "efficiency_risk": 0.12,
}

EVENT_ACTIONS = {
    "thermal_risk": "Reduce speed, enable cooling, inspect battery/motor loop.",
    "smoke_risk": "Open ventilation, isolate power module, prepare safe stop.",
    "speed_risk": "Cross-check wheel/IMU/GNSS speed and mark sensor unreliable.",
    "brake_risk": "Increase following distance and prepare emergency braking.",
    "takeover_risk": "Issue takeover warning and switch to conservative mode.",
    "efficiency_risk": "Check route deviation, energy consumption, and control smoothness.",
}


@dataclass(frozen=True)
class DriveHealthConfig:
    duration_s: int = 420
    sample_hz: int = 1
    seed: int = 42
    event_threshold: float = 62.0


def clip(value: np.ndarray | float, low: float = 0.0, high: float = 100.0) -> np.ndarray:
    return np.clip(value, low, high)


def severity_from_risk(risk: float) -> str:
    if risk >= 88:
        return "CRITICAL"
    if risk >= 74:
        return "WARNING"
    if risk >= 55:
        return "WATCH"
    return "NORMAL"


def level_from_health(health_score: float, max_risk: float) -> str:
    if health_score < 35 or max_risk >= 90:
        return "CRITICAL"
    if health_score < 55 or max_risk >= 76:
        return "WARNING"
    if health_score < 75 or max_risk >= 58:
        return "WATCH"
    return "NORMAL"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def contiguous_regions(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    """Yield inclusive start/end indexes for True regions."""

    if not np.any(mask):
        return
    indexes = np.flatnonzero(mask)
    start = prev = int(indexes[0])
    for idx in indexes[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        yield start, prev
        start = prev = idx
    yield start, prev


class DriveHealthBlackbox:
    """Generate telemetry, score health, and write blackbox artifacts."""

    def __init__(self, config: DriveHealthConfig, output_dir: Path = DEFAULT_OUTPUT_DIR):
        self.config = config
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(config.seed)

    def run(self) -> Dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        telemetry = self.generate_telemetry()
        risks = self.score_risks(telemetry)
        health = self.compute_health(risks)
        events = self.extract_events(telemetry, risks, health)
        summary = self.build_summary(telemetry, risks, health, events)

        artifacts: List[Path] = []
        artifacts.append(self.write_csv(telemetry, risks, health))
        artifacts.append(self.write_events(events))
        artifacts.append(self.write_summary(summary))
        artifacts.extend(self.write_visualizations(telemetry, risks, health, events, summary))
        artifacts.append(self.write_html_report(summary, artifacts))

        summary["artifact_count"] = len(artifacts)
        summary["artifacts"] = [str(path) for path in artifacts]
        self.write_summary(summary)
        return summary

    def generate_telemetry(self) -> Dict[str, np.ndarray]:
        cfg = self.config
        t = np.arange(0, cfg.duration_s, 1 / cfg.sample_hz, dtype=float)
        n = len(t)
        rng = self.rng

        speed = 43 + 8 * np.sin(t / 34) + 4 * np.sin(t / 9) + rng.normal(0, 1.4, n)
        speed[230:237] *= np.linspace(0.25, 0.12, 7)
        speed[300:306] += np.linspace(18, 32, 6)
        speed = clip(speed, 0, 120)

        motor_temp = 58 + 0.018 * t + 4 * np.sin(t / 50) + rng.normal(0, 1.2, n)
        battery_temp = 42 + 0.012 * t + 2 * np.sin(t / 45) + rng.normal(0, 0.9, n)
        controller_temp = 50 + 0.014 * t + 3 * np.sin(t / 55) + rng.normal(0, 1.0, n)
        thermal_slice = slice(96, 150)
        motor_temp[thermal_slice] += np.linspace(0, 34, thermal_slice.stop - thermal_slice.start)
        battery_temp[thermal_slice] += np.linspace(0, 18, thermal_slice.stop - thermal_slice.start)
        controller_temp[thermal_slice] += np.linspace(0, 25, thermal_slice.stop - thermal_slice.start)

        smoke_ppm = np.maximum(0, rng.normal(2.5, 0.45, n))
        smoke_ppm[182:207] += np.hanning(25) * 42
        smoke_ppm[330:350] += np.hanning(20) * 15

        obstacle_distance = 62 + 12 * np.sin(t / 27) + rng.normal(0, 3.0, n)
        obstacle_distance[270:305] = np.linspace(44, 8, 35) + rng.normal(0, 1.0, 35)
        obstacle_distance = clip(obstacle_distance, 4, 120)
        road_friction = clip(0.78 - 0.18 * np.sin(t / 80) + rng.normal(0, 0.03, n), 0.25, 0.95)
        road_friction[260:315] -= 0.20
        road_friction = clip(road_friction, 0.18, 0.95)

        driver_fatigue = clip(18 + 0.09 * t + 8 * np.sin(t / 70) + rng.normal(0, 2.0, n), 0, 100)
        driver_attention = clip(92 - 0.42 * driver_fatigue + rng.normal(0, 3.0, n), 0, 100)
        driver_fatigue[310:365] += np.linspace(0, 25, 55)
        driver_attention[310:365] -= np.linspace(0, 28, 55)
        driver_fatigue = clip(driver_fatigue, 0, 100)
        driver_attention = clip(driver_attention, 0, 100)

        autopilot_confidence = clip(92 - 0.22 * smoke_ppm - 0.12 * np.maximum(motor_temp - 75, 0) + rng.normal(0, 2.0, n), 0, 100)
        takeover_time = clip(
            0.9
            + 0.045 * driver_fatigue
            + 0.035 * (100 - driver_attention)
            + 0.032 * (100 - autopilot_confidence)
            + rng.normal(0, 0.22, n),
            0.2,
            12.0,
        )

        path_deviation = clip(0.18 + 0.015 * np.abs(speed - rolling_mean(speed, 15)) + rng.normal(0, 0.04, n), 0, 3.0)
        energy_kwh_100km = clip(18 + 0.11 * speed + 0.10 * np.maximum(motor_temp - 65, 0) + rng.normal(0, 1.3, n), 12, 42)
        hard_brake_count = np.cumsum((np.r_[0, -np.diff(speed)] > 10).astype(int))
        efficiency_score = clip(
            100
            - 7.0 * path_deviation
            - 1.45 * np.maximum(energy_kwh_100km - 20, 0)
            - 0.8 * hard_brake_count
            - 0.08 * driver_fatigue
            + rng.normal(0, 1.0, n),
            0,
            100,
        )

        return {
            "time_s": t,
            "speed_kmh": speed,
            "motor_temp_c": motor_temp,
            "battery_temp_c": battery_temp,
            "controller_temp_c": controller_temp,
            "smoke_ppm": smoke_ppm,
            "obstacle_distance_m": obstacle_distance,
            "road_friction": road_friction,
            "driver_fatigue": driver_fatigue,
            "driver_attention": driver_attention,
            "autopilot_confidence": autopilot_confidence,
            "takeover_time_s": takeover_time,
            "path_deviation_m": path_deviation,
            "energy_kwh_100km": energy_kwh_100km,
            "efficiency_score": efficiency_score,
        }

    def score_risks(self, telemetry: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        speed_mps = telemetry["speed_kmh"] / 3.6
        stopping_distance = (speed_mps**2) / (2 * 9.81 * telemetry["road_friction"]) + 0.25 * speed_mps
        brake_margin = telemetry["obstacle_distance_m"] - stopping_distance

        thermal_risk = clip(
            np.maximum.reduce(
                [
                    (telemetry["motor_temp_c"] - 70) / 35,
                    (telemetry["battery_temp_c"] - 50) / 25,
                    (telemetry["controller_temp_c"] - 62) / 30,
                ]
            )
            * 100
        )

        smoke = telemetry["smoke_ppm"]
        smoke_risk = np.where(
            smoke < 5,
            smoke * 4,
            np.where(smoke < 20, 20 + (smoke - 5) * 3.2, 68 + (smoke - 20) * 1.1),
        )
        smoke_risk = clip(smoke_risk)

        speed = telemetry["speed_kmh"]
        smooth_speed = rolling_mean(speed, 19)
        accel = np.r_[0, np.diff(speed)]
        speed_risk = clip(5.0 * np.abs(speed - smooth_speed) + 3.6 * np.abs(accel))

        brake_risk = clip((14 - brake_margin) / 22 * 100)
        takeover_risk = clip((telemetry["takeover_time_s"] - 2.6) / 5.2 * 100)
        efficiency_risk = clip(100 - telemetry["efficiency_score"])

        return {
            "thermal_risk": thermal_risk,
            "smoke_risk": smoke_risk,
            "speed_risk": speed_risk,
            "brake_risk": brake_risk,
            "takeover_risk": takeover_risk,
            "efficiency_risk": efficiency_risk,
            "brake_margin_m": brake_margin,
            "stopping_distance_m": stopping_distance,
        }

    def compute_health(self, risks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        weighted_risk = np.zeros_like(risks["thermal_risk"], dtype=float)
        for column, weight in RISK_WEIGHTS.items():
            weighted_risk += risks[column] * weight
        max_risk = np.maximum.reduce([risks[column] for column in RISK_COLUMNS])
        health_score = clip(100 - weighted_risk - 0.08 * max_risk)
        levels = np.array([level_from_health(float(score), float(risk)) for score, risk in zip(health_score, max_risk)])
        return {
            "weighted_risk": weighted_risk,
            "max_risk": max_risk,
            "health_score": health_score,
            "level": levels,
        }

    def extract_events(
        self,
        telemetry: Dict[str, np.ndarray],
        risks: Dict[str, np.ndarray],
        health: Dict[str, np.ndarray],
    ) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        times = telemetry["time_s"]
        event_id = 1
        for column in RISK_COLUMNS:
            mask = risks[column] >= self.config.event_threshold
            for start, end in contiguous_regions(mask) or []:
                region = risks[column][start : end + 1]
                peak_offset = int(np.argmax(region))
                peak_idx = start + peak_offset
                event = {
                    "id": f"DH-{event_id:03d}",
                    "type": column.replace("_risk", "").upper(),
                    "subsystem": RISK_NAMES[column],
                    "severity": severity_from_risk(float(risks[column][peak_idx])),
                    "start_time_s": round(float(times[start]), 2),
                    "end_time_s": round(float(times[end]), 2),
                    "peak_time_s": round(float(times[peak_idx]), 2),
                    "peak_risk": round(float(risks[column][peak_idx]), 2),
                    "min_health_score": round(float(np.min(health["health_score"][start : end + 1])), 2),
                    "recommended_action": EVENT_ACTIONS[column],
                }
                events.append(event)
                event_id += 1
        events.sort(key=lambda item: (item["peak_time_s"], item["peak_risk"]))
        return events

    def build_summary(
        self,
        telemetry: Dict[str, np.ndarray],
        risks: Dict[str, np.ndarray],
        health: Dict[str, np.ndarray],
        events: List[Dict[str, object]],
    ) -> Dict[str, object]:
        final_idx = len(telemetry["time_s"]) - 1
        peak_risk_column = max(RISK_COLUMNS, key=lambda column: float(np.max(risks[column])))
        severity_counts: Dict[str, int] = {}
        for event in events:
            severity_counts[event["severity"]] = severity_counts.get(event["severity"], 0) + 1
        return {
            "project": "DriveHealth",
            "description": "Simulator-free multi-source anomaly diagnosis and autonomous vehicle health blackbox.",
            "duration_s": int(self.config.duration_s),
            "sample_hz": int(self.config.sample_hz),
            "sample_count": int(len(telemetry["time_s"])),
            "mean_health_score": round(float(np.mean(health["health_score"])), 2),
            "minimum_health_score": round(float(np.min(health["health_score"])), 2),
            "final_health_score": round(float(health["health_score"][final_idx]), 2),
            "final_level": str(health["level"][final_idx]),
            "event_count": len(events),
            "severity_counts": severity_counts,
            "peak_risk_subsystem": RISK_NAMES[peak_risk_column],
            "peak_risk_value": round(float(np.max(risks[peak_risk_column])), 2),
            "top_actions": [event["recommended_action"] for event in sorted(events, key=lambda item: item["peak_risk"], reverse=True)[:5]],
        }

    def write_csv(
        self,
        telemetry: Dict[str, np.ndarray],
        risks: Dict[str, np.ndarray],
        health: Dict[str, np.ndarray],
    ) -> Path:
        path = self.output_dir / "drivehealth_blackbox_log.csv"
        fieldnames = [
            "time_s",
            "speed_kmh",
            "motor_temp_c",
            "battery_temp_c",
            "controller_temp_c",
            "smoke_ppm",
            "obstacle_distance_m",
            "road_friction",
            "driver_fatigue",
            "driver_attention",
            "autopilot_confidence",
            "takeover_time_s",
            "efficiency_score",
            "brake_margin_m",
            "stopping_distance_m",
            *RISK_COLUMNS,
            "weighted_risk",
            "max_risk",
            "health_score",
            "level",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(telemetry["time_s"])):
                row = {}
                for key in fieldnames:
                    if key in telemetry:
                        row[key] = round(float(telemetry[key][i]), 4)
                    elif key in risks:
                        row[key] = round(float(risks[key][i]), 4)
                    elif key in health:
                        value = health[key][i]
                        row[key] = str(value) if key == "level" else round(float(value), 4)
                writer.writerow(row)
        return path

    def write_events(self, events: List[Dict[str, object]]) -> Path:
        path = self.output_dir / "drivehealth_events.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(events, handle, ensure_ascii=False, indent=2)
        return path

    def write_summary(self, summary: Dict[str, object]) -> Path:
        path = self.output_dir / "drivehealth_summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return path

    def write_visualizations(
        self,
        telemetry: Dict[str, np.ndarray],
        risks: Dict[str, np.ndarray],
        health: Dict[str, np.ndarray],
        events: List[Dict[str, object]],
        summary: Dict[str, object],
    ) -> List[Path]:
        paths = [
            self.plot_overview_dashboard(telemetry, risks, health, events, summary),
            self.plot_health_timeline(telemetry, health, events),
            self.plot_sensor_streams(telemetry, risks),
            self.plot_risk_heatmap(telemetry, risks),
            self.plot_event_timeline(events),
            self.plot_risk_radar(risks),
            self.plot_event_distribution(events),
            self.plot_correlation_matrix(telemetry, risks, health),
            self.plot_health_histogram(health),
            self.plot_action_priority(events),
        ]
        return paths

    def plot_overview_dashboard(
        self,
        telemetry: Dict[str, np.ndarray],
        risks: Dict[str, np.ndarray],
        health: Dict[str, np.ndarray],
        events: List[Dict[str, object]],
        summary: Dict[str, object],
    ) -> Path:
        path = self.output_dir / "01_system_overview_dashboard.png"
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        times = telemetry["time_s"]

        axes[0, 0].plot(times, health["health_score"], color="#0f766e", linewidth=2)
        axes[0, 0].axhspan(0, 35, color="#dc2626", alpha=0.12)
        axes[0, 0].axhspan(35, 55, color="#f97316", alpha=0.10)
        axes[0, 0].axhspan(55, 75, color="#facc15", alpha=0.10)
        axes[0, 0].set_title("Health score timeline")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].grid(alpha=0.25)

        risk_max = [float(np.max(risks[column])) for column in RISK_COLUMNS]
        axes[0, 1].barh([RISK_NAMES[column] for column in RISK_COLUMNS], risk_max, color="#2563eb")
        axes[0, 1].set_xlim(0, 100)
        axes[0, 1].set_title("Peak risk by subsystem")
        axes[0, 1].grid(axis="x", alpha=0.25)

        axes[1, 0].plot(times, telemetry["speed_kmh"], label="Speed km/h", color="#334155")
        axes[1, 0].plot(times, telemetry["obstacle_distance_m"], label="Obstacle m", color="#ea580c")
        axes[1, 0].plot(times, risks["brake_margin_m"], label="Brake margin m", color="#16a34a")
        axes[1, 0].axhline(0, color="#dc2626", linestyle="--", linewidth=1)
        axes[1, 0].set_title("Motion and brake margin")
        axes[1, 0].legend(loc="best", fontsize=8)
        axes[1, 0].grid(alpha=0.25)

        cards = [
            ("Mean health", summary["mean_health_score"]),
            ("Min health", summary["minimum_health_score"]),
            ("Events", summary["event_count"]),
            ("Peak risk", summary["peak_risk_value"]),
            ("Final level", summary["final_level"]),
        ]
        axes[1, 1].axis("off")
        y = 0.88
        for label, value in cards:
            axes[1, 1].text(0.05, y, f"{label}", fontsize=13, weight="bold", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.70, y, f"{value}", fontsize=13, transform=axes[1, 1].transAxes)
            y -= 0.16
        axes[1, 1].set_title("Blackbox summary")

        fig.suptitle("DriveHealth simulator-free autonomous vehicle health blackbox", fontsize=16, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_health_timeline(self, telemetry: Dict[str, np.ndarray], health: Dict[str, np.ndarray], events: List[Dict[str, object]]) -> Path:
        path = self.output_dir / "02_health_score_timeline.png"
        fig, ax = plt.subplots(figsize=(14, 5))
        times = telemetry["time_s"]
        ax.plot(times, health["health_score"], color="#0f766e", linewidth=2.2, label="Health score")
        ax.axhline(75, color="#22c55e", linestyle="--", linewidth=1, label="Normal threshold")
        ax.axhline(55, color="#f59e0b", linestyle="--", linewidth=1, label="Warning threshold")
        ax.axhline(35, color="#ef4444", linestyle="--", linewidth=1, label="Critical threshold")
        for event in events:
            ax.axvline(float(event["peak_time_s"]), color="#64748b", alpha=0.15, linewidth=1)
        ax.set_title("DriveHealth score with event markers")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Health score")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left", ncol=4)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_sensor_streams(self, telemetry: Dict[str, np.ndarray], risks: Dict[str, np.ndarray]) -> Path:
        path = self.output_dir / "03_sensor_streams_overview.png"
        times = telemetry["time_s"]
        fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
        axes = axes.ravel()

        axes[0].plot(times, telemetry["motor_temp_c"], label="Motor")
        axes[0].plot(times, telemetry["battery_temp_c"], label="Battery")
        axes[0].plot(times, telemetry["controller_temp_c"], label="Controller")
        axes[0].set_title("Thermal stream")
        axes[0].set_ylabel("Celsius")
        axes[0].legend(fontsize=8)

        axes[1].plot(times, telemetry["smoke_ppm"], color="#7c2d12")
        axes[1].axhline(5, color="#f59e0b", linestyle="--", linewidth=1)
        axes[1].axhline(20, color="#dc2626", linestyle="--", linewidth=1)
        axes[1].set_title("Smoke concentration")
        axes[1].set_ylabel("ppm")

        axes[2].plot(times, telemetry["speed_kmh"], color="#334155")
        axes[2].set_title("Speed stream")
        axes[2].set_ylabel("km/h")

        axes[3].plot(times, telemetry["obstacle_distance_m"], label="Obstacle distance")
        axes[3].plot(times, risks["stopping_distance_m"], label="Stopping distance")
        axes[3].plot(times, risks["brake_margin_m"], label="Margin")
        axes[3].axhline(0, color="#dc2626", linestyle="--", linewidth=1)
        axes[3].set_title("Brake geometry")
        axes[3].legend(fontsize=8)

        axes[4].plot(times, telemetry["driver_fatigue"], label="Fatigue")
        axes[4].plot(times, telemetry["driver_attention"], label="Attention")
        axes[4].plot(times, telemetry["autopilot_confidence"], label="Autopilot confidence")
        axes[4].set_title("Human-machine readiness")
        axes[4].legend(fontsize=8)

        axes[5].plot(times, telemetry["efficiency_score"], color="#0891b2")
        axes[5].set_title("Efficiency score")
        axes[5].set_ylabel("Score")

        for ax in axes:
            ax.grid(alpha=0.25)
            ax.set_xlabel("Time (s)")

        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_risk_heatmap(self, telemetry: Dict[str, np.ndarray], risks: Dict[str, np.ndarray]) -> Path:
        path = self.output_dir / "04_subsystem_risk_heatmap.png"
        risk_matrix = np.vstack([risks[column] for column in RISK_COLUMNS])
        fig, ax = plt.subplots(figsize=(14, 5.5))
        image = ax.imshow(risk_matrix, aspect="auto", cmap="magma", vmin=0, vmax=100, extent=[0, telemetry["time_s"][-1], len(RISK_COLUMNS), 0])
        ax.set_yticks(np.arange(len(RISK_COLUMNS)) + 0.5)
        ax.set_yticklabels([RISK_NAMES[column] for column in RISK_COLUMNS])
        ax.set_xlabel("Time (s)")
        ax.set_title("Subsystem risk heatmap")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Risk score")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_event_timeline(self, events: List[Dict[str, object]]) -> Path:
        path = self.output_dir / "05_blackbox_event_timeline.png"
        fig, ax = plt.subplots(figsize=(14, 5))
        subsystems = list(dict.fromkeys(event["subsystem"] for event in events)) or ["No events"]
        y_map = {name: i for i, name in enumerate(subsystems)}
        if events:
            x = [float(event["peak_time_s"]) for event in events]
            y = [y_map[event["subsystem"]] for event in events]
            sizes = [40 + float(event["peak_risk"]) * 2.0 for event in events]
            colors = [float(event["peak_risk"]) for event in events]
            scatter = ax.scatter(x, y, s=sizes, c=colors, cmap="Reds", vmin=55, vmax=100, edgecolor="#111827", linewidth=0.5)
            fig.colorbar(scatter, ax=ax, label="Peak risk")
            for event in events:
                ax.text(float(event["peak_time_s"]) + 2, y_map[event["subsystem"]] + 0.05, event["id"], fontsize=8)
        else:
            ax.text(0.5, 0.5, "No blackbox events detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks(range(len(subsystems)))
        ax.set_yticklabels(subsystems)
        ax.set_xlabel("Peak time (s)")
        ax.set_title("Blackbox event timeline")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_risk_radar(self, risks: Dict[str, np.ndarray]) -> Path:
        path = self.output_dir / "06_risk_radar_summary.png"
        labels = [RISK_NAMES[column] for column in RISK_COLUMNS]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.r_[angles, angles[0]]
        mean_values = np.r_[[float(np.mean(risks[column])) for column in RISK_COLUMNS], float(np.mean(risks[RISK_COLUMNS[0]]))]
        peak_values = np.r_[[float(np.max(risks[column])) for column in RISK_COLUMNS], float(np.max(risks[RISK_COLUMNS[0]]))]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, mean_values, color="#2563eb", linewidth=2, label="Mean risk")
        ax.fill(angles, mean_values, color="#2563eb", alpha=0.15)
        ax.plot(angles, peak_values, color="#dc2626", linewidth=2, label="Peak risk")
        ax.fill(angles, peak_values, color="#dc2626", alpha=0.10)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_ylim(0, 100)
        ax.set_title("Risk radar")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_event_distribution(self, events: List[Dict[str, object]]) -> Path:
        path = self.output_dir / "07_event_distribution.png"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        subsystem_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        for event in events:
            subsystem_counts[event["subsystem"]] = subsystem_counts.get(event["subsystem"], 0) + 1
            severity_counts[event["severity"]] = severity_counts.get(event["severity"], 0) + 1

        if subsystem_counts:
            axes[0].bar(subsystem_counts.keys(), subsystem_counts.values(), color="#2563eb")
            axes[0].tick_params(axis="x", rotation=30)
        axes[0].set_title("Events by subsystem")
        axes[0].set_ylabel("Count")
        axes[0].grid(axis="y", alpha=0.25)

        severity_order = ["WATCH", "WARNING", "CRITICAL"]
        values = [severity_counts.get(name, 0) for name in severity_order]
        axes[1].bar(severity_order, values, color=["#facc15", "#f97316", "#dc2626"])
        axes[1].set_title("Events by severity")
        axes[1].grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_correlation_matrix(
        self,
        telemetry: Dict[str, np.ndarray],
        risks: Dict[str, np.ndarray],
        health: Dict[str, np.ndarray],
    ) -> Path:
        path = self.output_dir / "08_feature_health_correlation.png"
        feature_map = {
            "speed": telemetry["speed_kmh"],
            "motor_temp": telemetry["motor_temp_c"],
            "battery_temp": telemetry["battery_temp_c"],
            "smoke": telemetry["smoke_ppm"],
            "brake_margin": risks["brake_margin_m"],
            "takeover": telemetry["takeover_time_s"],
            "efficiency": telemetry["efficiency_score"],
            "health": health["health_score"],
        }
        labels = list(feature_map.keys())
        data = np.vstack([feature_map[label] for label in labels])
        corr = np.corrcoef(data)
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Feature-health correlation")
        fig.colorbar(image, ax=ax, label="Correlation")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_health_histogram(self, health: Dict[str, np.ndarray]) -> Path:
        path = self.output_dir / "09_health_score_distribution.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(health["health_score"], bins=24, color="#14b8a6", edgecolor="white")
        ax.axvline(75, color="#22c55e", linestyle="--", label="Normal")
        ax.axvline(55, color="#f59e0b", linestyle="--", label="Warning")
        ax.axvline(35, color="#ef4444", linestyle="--", label="Critical")
        ax.set_title("Health score distribution")
        ax.set_xlabel("Health score")
        ax.set_ylabel("Samples")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_action_priority(self, events: List[Dict[str, object]]) -> Path:
        path = self.output_dir / "10_action_priority_board.png"
        top_events = sorted(events, key=lambda event: float(event["peak_risk"]), reverse=True)[:8]
        fig, ax = plt.subplots(figsize=(13, 6))
        if top_events:
            labels = [f"{event['id']} {event['subsystem']}" for event in top_events]
            values = [float(event["peak_risk"]) for event in top_events]
            y = np.arange(len(labels))
            ax.barh(y, values, color="#dc2626")
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            for i, event in enumerate(top_events):
                ax.text(values[i] + 1, i, event["severity"], va="center", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No priority actions", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Action priority board")
        ax.set_xlabel("Peak risk")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def write_html_report(self, summary: Dict[str, object], artifacts: List[Path]) -> Path:
        path = self.output_dir / "drivehealth_report.html"
        images = [artifact for artifact in artifacts if artifact.suffix.lower() == ".png"]
        event_rows = ""
        events_path = self.output_dir / "drivehealth_events.json"
        events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []
        for event in sorted(events, key=lambda item: item["peak_risk"], reverse=True)[:12]:
            event_rows += (
                "<tr>"
                f"<td>{event['id']}</td><td>{event['subsystem']}</td><td>{event['severity']}</td>"
                f"<td>{event['peak_time_s']}</td><td>{event['peak_risk']}</td>"
                f"<td>{event['recommended_action']}</td>"
                "</tr>"
            )

        image_cards = "\n".join(
            f"<section><h2>{image.stem.replace('_', ' ').title()}</h2><img src='{image.name}' alt='{image.stem}'></section>"
            for image in images
        )

        html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>DriveHealth Blackbox Report</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #0f172a; background: #f8fafc; }}
    header {{ background: #0f766e; color: white; padding: 24px; border-radius: 10px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 18px 0; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
    .card strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    section {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; margin: 18px 0; padding: 18px; }}
    img {{ width: 100%; max-width: 1200px; display: block; border: 1px solid #e5e7eb; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #ecfeff; }}
  </style>
</head>
<body>
  <header>
    <h1>DriveHealth: 自健康黑匣子报告</h1>
    <p>无需 CARLA/AirSim/MuJoCo，基于多源传感器流完成运行安全诊断、事件记录和可视化复盘。</p>
  </header>
  <div class="cards">
    <div class="card">平均健康分<strong>{summary.get('mean_health_score')}</strong></div>
    <div class="card">最低健康分<strong>{summary.get('minimum_health_score')}</strong></div>
    <div class="card">黑匣子事件<strong>{summary.get('event_count')}</strong></div>
    <div class="card">峰值风险<strong>{summary.get('peak_risk_value')}</strong></div>
    <div class="card">最终等级<strong>{summary.get('final_level')}</strong></div>
  </div>
  <section>
    <h2>Top Blackbox Events</h2>
    <table>
      <thead><tr><th>ID</th><th>Subsystem</th><th>Severity</th><th>Peak Time(s)</th><th>Peak Risk</th><th>Recommended Action</th></tr></thead>
      <tbody>{event_rows}</tbody>
    </table>
  </section>
  {image_cards}
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")
        return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DriveHealth simulator-free vehicle health blackbox")
    parser.add_argument("--duration", type=int, default=420, help="telemetry duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible telemetry")
    parser.add_argument("--event-threshold", type=float, default=62.0, help="risk threshold used to extract events")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="artifact directory")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DriveHealthConfig(duration_s=args.duration, seed=args.seed, event_threshold=args.event_threshold)
    blackbox = DriveHealthBlackbox(config=config, output_dir=Path(args.output_dir))
    summary = blackbox.run()
    print("DriveHealth blackbox finished.")
    print(f"Output directory: {blackbox.output_dir}")
    print(f"Samples: {summary['sample_count']} | Events: {summary['event_count']} | Min health: {summary['minimum_health_score']}")
    print(f"Peak risk: {summary['peak_risk_subsystem']} {summary['peak_risk_value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
