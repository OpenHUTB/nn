"""Visualization utilities for the signal optimization project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from optimizer import OptimizationResult
from scenario import IntersectionScenario
from simulator import SimulationResult

COLORS = {
    "north_south_through": "#2d6cdf",
    "east_west_through": "#ef8a17",
    "north_south_left": "#2a9d62",
    "east_west_left": "#c44e52",
}


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_queue_profiles(result: SimulationResult, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150)
    for name, queue in result.queues.items():
        ax.plot(result.time / 60, queue, label=name, color=COLORS.get(name), linewidth=1.8)
    ax.set_title("Optimized signal queue profile")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Queue length (veh)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_plan_comparison(scenario: IntersectionScenario, result: OptimizationResult, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    names = scenario.names
    x = np.arange(len(names))
    width = 0.36
    baseline = [result.baseline.plan.green_for(name) for name in names]
    optimized = [result.optimized.plan.green_for(name) for name in names]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    ax.bar(x - width / 2, baseline, width, label="Webster baseline", color="#7b8794")
    ax.bar(x + width / 2, optimized, width, label="GA optimized", color="#2d6cdf")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in names], fontsize=8)
    ax.set_ylabel("Green time (s)")
    ax.set_title("Signal green split comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_metric_comparison(result: OptimizationResult, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    labels = ["avg delay", "max queue", "stops/100", "score"]
    baseline = [
        result.baseline.average_delay,
        result.baseline.max_queue,
        result.baseline.stops / 100,
        result.baseline.score,
    ]
    optimized = [
        result.optimized.average_delay,
        result.optimized.max_queue,
        result.optimized.stops / 100,
        result.optimized.score,
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.1), dpi=150)
    ax.bar(x - width / 2, baseline, width, label="baseline", color="#7b8794")
    ax.bar(x + width / 2, optimized, width, label="optimized", color="#2a9d62")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Optimization effect on traffic metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_convergence(result: OptimizationResult, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=150)
    ax1.plot(result.history.generation, result.history.best_score, color="#2d6cdf", label="best score", linewidth=2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best score")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(result.history.generation, result.history.best_delay, color="#c44e52", label="avg delay", linewidth=1.7)
    ax2.set_ylabel("Average delay")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax1.set_title("Genetic algorithm convergence")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_intersection_animation(
    scenario: IntersectionScenario,
    result: SimulationResult,
    output_path: str | Path,
    frame_step: int = 24,
) -> Path:
    output = ensure_parent(output_path)
    frames = []
    for index in range(0, len(result.time), frame_step):
        frames.append(draw_intersection_frame(scenario, result, index))
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=130, loop=0)
    return output


def draw_intersection_frame(scenario: IntersectionScenario, result: SimulationResult, index: int) -> Image.Image:
    image = Image.new("RGB", (760, 520), (244, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([285, 0, 475, 520], fill=(58, 65, 77))
    draw.rectangle([0, 175, 760, 345], fill=(58, 65, 77))
    draw.rectangle([305, 195, 455, 325], fill=(80, 90, 104))
    draw.line([380, 0, 380, 520], fill=(225, 225, 210), width=3)
    draw.line([0, 260, 760, 260], fill=(225, 225, 210), width=3)
    active = result.active_phase[index]
    queue_values = {name: float(values[index]) for name, values in result.queues.items()}
    draw.text((18, 18), f"time={int(result.time[index])}s  green={active}", fill=(20, 28, 36))
    draw_signal(draw, (350, 150), active == "north_south_through" or active == "north_south_left")
    draw_signal(draw, (485, 290), active == "east_west_through" or active == "east_west_left")
    draw_queue(draw, "north_south_through", queue_values["north_south_through"], (332, 20), vertical=True)
    draw_queue(draw, "east_west_through", queue_values["east_west_through"], (600, 220), vertical=False)
    draw_queue(draw, "north_south_left", queue_values["north_south_left"], (420, 430), vertical=True)
    draw_queue(draw, "east_west_left", queue_values["east_west_left"], (45, 290), vertical=False)
    draw_metrics(draw, result)
    return image


def draw_signal(draw: ImageDraw.ImageDraw, origin: tuple[int, int], is_green: bool) -> None:
    x, y = origin
    draw.rectangle([x, y, x + 34, y + 72], fill=(30, 35, 42), outline=(15, 18, 22))
    green = (39, 174, 96) if is_green else (60, 76, 88)
    red = (200, 56, 56) if not is_green else (92, 72, 72)
    draw.ellipse([x + 7, y + 8, x + 27, y + 28], fill=red)
    draw.ellipse([x + 7, y + 43, x + 27, y + 63], fill=green)


def draw_queue(draw: ImageDraw.ImageDraw, name: str, queue: float, origin: tuple[int, int], vertical: bool) -> None:
    x, y = origin
    count = int(min(14, round(queue)))
    color = hex_to_rgb(COLORS[name])
    for idx in range(count):
        if vertical:
            box = [x, y + idx * 23, x + 36, y + idx * 23 + 15]
        else:
            box = [x + idx * 39, y, x + idx * 39 + 28, y + 17]
        draw.rounded_rectangle(box, radius=4, fill=color)
    label = name.replace("_", " ")
    draw.text((x, y - 16), f"{label}: {queue:.1f}", fill=(20, 28, 36))


def draw_metrics(draw: ImageDraw.ImageDraw, result: SimulationResult) -> None:
    draw.rectangle([16, 380, 250, 500], fill=(255, 255, 255), outline=(210, 216, 224))
    rows = [
        f"delay: {result.average_delay:.2f}",
        f"max queue: {result.max_queue:.2f}",
        f"throughput: {result.throughput:.3f}",
        f"cycle: {result.plan.cycle_length}s",
    ]
    for idx, row in enumerate(rows):
        draw.text((32, 398 + idx * 24), row, fill=(20, 28, 36))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
