"""Visualization helpers for adaptive grid navigation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from grid_map import Cell, GridMap
from planners import PlanResult
from simulator import SimulationFrame, SimulationResult

FREE = (248, 248, 244)
OBSTACLE = (40, 43, 48)
START = (40, 128, 70)
GOAL = (198, 55, 55)
PATH = (42, 119, 190)
EXPANDED = (244, 189, 76)
DYNAMIC = (143, 75, 180)
ROBOT = (24, 99, 160)
GRID = (215, 215, 210)


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def draw_grid(
    grid_map: GridMap,
    path: Iterable[Cell] | None = None,
    expanded: Iterable[Cell] | None = None,
    dynamic_obstacles: Iterable[Cell] | None = None,
    robot: Cell | None = None,
    cell_size: int = 20,
) -> Image.Image:
    image = Image.new("RGB", (grid_map.width * cell_size, grid_map.height * cell_size), FREE)
    draw = ImageDraw.Draw(image)

    for row in range(grid_map.height):
        for col in range(grid_map.width):
            cell = (row, col)
            x0, y0 = col * cell_size, row * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            fill = OBSTACLE if grid_map.is_obstacle(cell) else FREE
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=GRID)

    for cell in expanded or []:
        _draw_cell(draw, cell, cell_size, EXPANDED, inset=5)
    for cell in path or []:
        _draw_cell(draw, cell, cell_size, PATH, inset=6)
    for cell in dynamic_obstacles or []:
        _draw_cell(draw, cell, cell_size, DYNAMIC, inset=3)

    _draw_cell(draw, grid_map.start, cell_size, START, inset=2)
    _draw_cell(draw, grid_map.goal, cell_size, GOAL, inset=2)
    if robot is not None:
        _draw_cell(draw, robot, cell_size, ROBOT, inset=1)
    return image


def _draw_cell(draw: ImageDraw.ImageDraw, cell: Cell, cell_size: int, color: tuple[int, int, int], inset: int) -> None:
    row, col = cell
    x0, y0 = col * cell_size + inset, row * cell_size + inset
    x1, y1 = (col + 1) * cell_size - inset, (row + 1) * cell_size - inset
    draw.rectangle([x0, y0, x1, y1], fill=color)


def save_plan_image(grid_map: GridMap, result: PlanResult, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    image = draw_grid(grid_map, path=result.path, expanded=result.expanded)
    image.save(output)
    return output


def save_comparison_chart(results: list[PlanResult], output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    names = [result.planner for result in results]
    lengths = [result.path_length if result.success else 0 for result in results]
    expanded = [result.expanded_count for result in results]

    x = np.arange(len(names))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=140)
    bars1 = ax1.bar(x - width / 2, lengths, width, label="path length", color="#2a77be")
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, expanded, width, label="expanded nodes", color="#f0a43a")

    ax1.set_ylabel("Path length")
    ax2.set_ylabel("Expanded nodes")
    ax1.set_title("Planner comparison on the same grid map")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    labels = [bars1, bars2]
    ax1.legend(labels, [bar.get_label() for bar in labels], loc="upper right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_animation(grid_map: GridMap, simulation: SimulationResult, output_path: str | Path, cell_size: int = 16) -> Path:
    output = ensure_parent(output_path)
    frames: list[Image.Image] = []
    for frame in simulation.frames:
        image = draw_frame(grid_map, frame, cell_size=cell_size)
        frames.append(image)
    if not frames:
        raise ValueError("simulation has no frames")
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=140, loop=0)
    return output


def draw_frame(grid_map: GridMap, frame: SimulationFrame, cell_size: int = 16) -> Image.Image:
    image = draw_grid(
        grid_map,
        path=frame.planned_path,
        dynamic_obstacles=frame.dynamic_obstacles,
        robot=frame.robot,
        cell_size=cell_size,
    )
    draw = ImageDraw.Draw(image)
    label = f"step={frame.step}  replan={'yes' if frame.replanned else 'no'}"
    draw.rectangle([2, 2, 210, 22], fill=(255, 255, 255))
    draw.text((6, 6), label, fill=(0, 0, 0))
    return image
