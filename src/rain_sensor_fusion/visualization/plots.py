"""Matplotlib visualizations for rain-aware sensor fusion results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ..fusion.deep_fusion import FusionResult, RainFusionEngine
from ..sensor.synthetic import DEMO_SCENARIOS, SensorSnapshot, make_demo_snapshot
from ..sensor.synthetic import make_sensor_snapshot


ResultRow = tuple[SensorSnapshot, FusionResult]

SENSOR_COLORS = {
    "camera": "#3B82F6",
    "lidar": "#10B981",
    "radar": "#F59E0B",
}
RISK_COLOR = "#EF4444"
TEXT_COLOR = "#172033"
GRID_COLOR = "#D7DEE8"


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pct_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA7B7")
    ax.spines["bottom"].set_color("#9AA7B7")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.title.set_color(TEXT_COLOR)


def _save_csv(rows: Sequence[ResultRow], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "scenario",
                "rain",
                "fog",
                "wet_road",
                "camera_weight",
                "lidar_weight",
                "radar_weight",
                "object_confidence",
                "risk_score",
                "recommended_action",
                "box_x_m",
                "box_y_m",
            ]
        )
        for snapshot, result in rows:
            writer.writerow(
                [
                    snapshot.scenario_id,
                    float(snapshot.weather[0]),
                    float(snapshot.weather[1]),
                    float(snapshot.weather[2]),
                    result.modality_weights["camera"],
                    result.modality_weights["lidar"],
                    result.modality_weights["radar"],
                    result.object_confidence,
                    result.risk_score,
                    result.recommended_action,
                    result.box_3d["center_x_m"],
                    result.box_3d["center_y_m"],
                ]
            )


def save_scenario_dashboard(
    snapshot: SensorSnapshot,
    result: FusionResult,
    output_dir: str | Path,
) -> Path:
    """Save a four-panel visual explanation for one scenario."""

    output_dir = _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle(
        f"Rain Sensor Fusion Dashboard: {snapshot.scenario_id}",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    ax = axes[0, 0]
    names = ["camera", "lidar", "radar"]
    values = [result.modality_weights[name] for name in names]
    ax.bar(names, values, color=[SENSOR_COLORS[name] for name in names], width=0.58)
    _pct_axis(ax)
    _style_axes(ax)
    ax.set_title("Deep Fusion Modality Weights")
    for idx, value in enumerate(values):
        ax.text(
            idx,
            value + 0.025,
            f"{value * 100:.1f}%",
            ha="center",
            color=TEXT_COLOR,
            fontsize=10,
        )

    ax = axes[0, 1]
    labels = ["rain", "fog", "wet road", "network risk", "physical risk", "final risk"]
    risk_values = [
        float(snapshot.weather[0]),
        float(snapshot.weather[1]),
        float(snapshot.weather[2]),
        result.network_risk,
        result.physical_risk,
        result.risk_score,
    ]
    colors = ["#60A5FA", "#94A3B8", "#38BDF8", "#FB7185", "#F97316", RISK_COLOR]
    ax.barh(labels, risk_values, color=colors, height=0.52)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.invert_yaxis()
    _style_axes(ax)
    ax.set_title(f"Risk Breakdown -> {result.recommended_action}")
    for label_y, value in enumerate(risk_values):
        ax.text(
            min(value + 0.025, 0.94),
            label_y,
            f"{value * 100:.1f}%",
            va="center",
            color=TEXT_COLOR,
            fontsize=9,
        )

    ax = axes[1, 0]
    sensor_matrix = np.array(
        [
            snapshot.camera,
            np.pad(snapshot.lidar, (0, 1), constant_values=np.nan),
            np.pad(snapshot.radar, (0, 1), constant_values=np.nan),
        ],
        dtype=float,
    )
    masked = np.ma.masked_invalid(sensor_matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#EDF2F7")
    image = ax.imshow(masked, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["camera", "lidar", "radar"])
    ax.set_xticks(range(5))
    ax.set_xticklabels(["cue 1", "cue 2", "cue 3", "cue 4", "cue 5"])
    ax.set_title("Normalized Sensor Feature Heatmap")
    _style_axes(ax)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for y in range(sensor_matrix.shape[0]):
        for x in range(sensor_matrix.shape[1]):
            value = sensor_matrix[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)

    ax = axes[1, 1]
    ax.set_title("Top-View 3D Detection Sketch")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 60)
    ax.set_xlabel("lateral offset (m)")
    ax.set_ylabel("forward distance (m)")
    ax.grid(color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.add_patch(Rectangle((-0.9, 0.5), 1.8, 3.8, facecolor="#1F2937", alpha=0.88))
    ax.text(0, 2.5, "ego", ha="center", va="center", color="white", fontsize=9)
    box = result.box_3d
    width = float(box["width_m"])
    length = float(box["length_m"])
    x = float(box["center_y_m"]) - width / 2.0
    y = float(box["center_x_m"]) - length / 2.0
    ax.add_patch(
        Rectangle((x, y), width, length, facecolor="#F97316", edgecolor="#9A3412", alpha=0.75)
    )
    ax.plot([0, float(box["center_y_m"])], [4.3, float(box["center_x_m"])], color=RISK_COLOR, linewidth=2)
    ax.text(
        float(box["center_y_m"]),
        float(box["center_x_m"]) + 3.5,
        f"conf {result.object_confidence * 100:.1f}%\nrisk {result.risk_score * 100:.1f}%",
        ha="center",
        color=TEXT_COLOR,
        fontsize=9,
    )
    _style_axes(ax)

    path = output_dir / f"{snapshot.scenario_id}_dashboard.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_scenario_comparison(rows: Sequence[ResultRow], output_dir: str | Path) -> Path:
    """Save a compact comparison across demo scenarios."""

    output_dir = _ensure_output_dir(output_dir)
    scenarios = [snapshot.scenario_id for snapshot, _result in rows]
    x = np.arange(len(rows))
    width = 0.22

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), constrained_layout=True)
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle("Rain Sensor Fusion Scenario Comparison", fontsize=17, fontweight="bold", color=TEXT_COLOR)

    ax = axes[0]
    for idx, sensor in enumerate(("camera", "lidar", "radar")):
        values = [result.modality_weights[sensor] for _snapshot, result in rows]
        ax.bar(x + (idx - 1) * width, values, width=width, label=sensor, color=SENSOR_COLORS[sensor])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_title("Modality Weight Shift")
    ax.legend(frameon=False, ncols=3)
    _pct_axis(ax)
    _style_axes(ax)

    ax = axes[1]
    risks = [result.risk_score for _snapshot, result in rows]
    confidences = [result.object_confidence for _snapshot, result in rows]
    rains = [float(snapshot.weather[0]) for snapshot, _result in rows]
    ax.plot(scenarios, risks, marker="o", linewidth=2.2, color=RISK_COLOR, label="risk")
    ax.plot(scenarios, confidences, marker="s", linewidth=2.2, color="#2563EB", label="object confidence")
    ax.plot(scenarios, rains, marker="^", linewidth=2.2, color="#0EA5E9", label="rain intensity")
    ax.set_title("Risk, Confidence, and Weather")
    ax.legend(frameon=False, ncols=3)
    _pct_axis(ax)
    _style_axes(ax)

    path = output_dir / "scenario_comparison.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_rain_sweep(
    engine: RainFusionEngine,
    output_dir: str | Path,
    *,
    samples: int = 16,
) -> Path:
    """Save a rain-intensity sweep showing dynamic fusion behavior."""

    output_dir = _ensure_output_dir(output_dir)
    rains = np.linspace(0.0, 1.0, samples)
    results: List[FusionResult] = []
    for idx, rain in enumerate(rains):
        snapshot = make_sensor_snapshot(
            rain=float(rain),
            fog=float(0.18 + 0.42 * rain),
            wet_road=float(min(1.0, 0.15 + 0.80 * rain)),
            obstacle_distance_m=26.0 - 9.0 * float(rain),
            closing_speed_mps=3.0 + 6.0 * float(rain),
            lateral_offset_m=-0.15,
            scenario_id=f"rain_{idx:02d}",
            seed=100 + idx,
        )
        results.append(engine.analyze(snapshot))

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), constrained_layout=True)
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle("Rain Intensity Sweep", fontsize=17, fontweight="bold", color=TEXT_COLOR)

    ax = axes[0]
    for sensor in ("camera", "lidar", "radar"):
        ax.plot(
            rains,
            [result.modality_weights[sensor] for result in results],
            marker="o",
            linewidth=2.0,
            label=sensor,
            color=SENSOR_COLORS[sensor],
        )
    ax.set_xlabel("rain intensity")
    ax.set_title("Fusion Weight Response")
    ax.legend(frameon=False, ncols=3)
    _pct_axis(ax)
    _style_axes(ax)

    ax = axes[1]
    ax.plot(rains, [result.risk_score for result in results], marker="o", linewidth=2.3, color=RISK_COLOR, label="risk")
    ax.plot(
        rains,
        [result.object_confidence for result in results],
        marker="s",
        linewidth=2.0,
        color="#2563EB",
        label="object confidence",
    )
    ax.axhline(engine.config.slow_down_threshold, color="#F97316", linestyle="--", linewidth=1.6, label="slow down")
    ax.axhline(engine.config.brake_threshold, color="#DC2626", linestyle="--", linewidth=1.6, label="brake")
    ax.set_xlabel("rain intensity")
    ax.set_title("Risk Response")
    ax.legend(frameon=False, ncols=4)
    _pct_axis(ax)
    _style_axes(ax)

    path = output_dir / "rain_intensity_sweep.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def analyze_scenarios(
    engine: RainFusionEngine,
    scenarios: Iterable[str] = DEMO_SCENARIOS,
) -> List[ResultRow]:
    """Analyze named demo scenarios."""

    rows: List[ResultRow] = []
    for name in scenarios:
        snapshot = make_demo_snapshot(name)
        rows.append((snapshot, engine.analyze(snapshot)))
    return rows


def save_visualization_suite(
    engine: RainFusionEngine,
    output_dir: str | Path,
    *,
    scenario: str = "heavy_rain",
    include_comparison: bool = True,
) -> list[Path]:
    """Save all project visualizations and a CSV summary."""

    output_dir = _ensure_output_dir(output_dir)
    saved: list[Path] = []
    snapshot = make_demo_snapshot(scenario)
    result = engine.analyze(snapshot)
    saved.append(save_scenario_dashboard(snapshot, result, output_dir))

    if include_comparison:
        rows = analyze_scenarios(engine)
        for row_snapshot, row_result in rows:
            if row_snapshot.scenario_id != scenario:
                saved.append(save_scenario_dashboard(row_snapshot, row_result, output_dir))
        saved.append(save_scenario_comparison(rows, output_dir))
        saved.append(save_rain_sweep(engine, output_dir))
        csv_path = output_dir / "scenario_results.csv"
        _save_csv(rows, csv_path)
        saved.append(csv_path)

    return saved
