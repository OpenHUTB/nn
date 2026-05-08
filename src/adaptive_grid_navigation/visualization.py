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

PLANNER_COLORS = {
    "Dijkstra": "#6c757d",
    "A*": "#1f77b4",
    "Greedy best-first": "#d95f02",
}


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


def _cell_centers(path: Iterable[Cell]) -> tuple[list[float], list[float]]:
    cols = [cell[1] + 0.5 for cell in path]
    rows = [cell[0] + 0.5 for cell in path]
    return cols, rows


def _draw_map_background(ax, grid_map: GridMap) -> None:
    ax.imshow(grid_map.occupancy, cmap="Greys", origin="upper", vmin=0, vmax=1, alpha=0.9)
    ax.set_xticks(np.arange(-0.5, grid_map.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_map.height, 1), minor=True)
    ax.grid(which="minor", color="#d6d6d0", linewidth=0.35)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.scatter(grid_map.start[1], grid_map.start[0], marker="o", s=90, c="#2a8046", label="start", zorder=5)
    ax.scatter(grid_map.goal[1], grid_map.goal[0], marker="*", s=160, c="#c63737", label="goal", zorder=5)


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


def save_path_overlay(grid_map: GridMap, results: list[PlanResult], output_path: str | Path) -> Path:
    """Draw all planner paths on the same map for visual comparison."""
    output = ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=150)
    _draw_map_background(ax, grid_map)

    for result in results:
        if not result.success:
            continue
        xs, ys = _cell_centers(result.path)
        ax.plot(
            xs,
            ys,
            color=PLANNER_COLORS.get(result.planner, "#333333"),
            linewidth=2.4,
            label=f"{result.planner} ({result.path_length})",
            alpha=0.9,
        )

    ax.set_title("Planner route overlay: shortest path vs fast search")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_expansion_heatmap(grid_map: GridMap, result: PlanResult, output_path: str | Path) -> Path:
    """Visualize how quickly a planner expands cells before reaching the goal."""
    output = ensure_parent(output_path)
    heat = np.full((grid_map.height, grid_map.width), np.nan)
    for order, cell in enumerate(result.expanded, start=1):
        heat[cell] = order

    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=150)
    _draw_map_background(ax, grid_map)
    masked = np.ma.masked_invalid(heat)
    im = ax.imshow(masked, cmap="YlOrRd", origin="upper", alpha=0.78)
    if result.path:
        xs, ys = _cell_centers(result.path)
        ax.plot(xs, ys, color="#0f4c81", linewidth=2.4, label="final path")
    ax.set_title(f"{result.planner} search expansion heatmap")
    ax.legend(loc="lower right", framealpha=0.95)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Expansion order")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_robot_trace(grid_map: GridMap, simulation: SimulationResult, output_path: str | Path) -> Path:
    """Draw the actually travelled robot trajectory after dynamic replanning."""
    output = ensure_parent(output_path)
    trace = [frame.robot for frame in simulation.frames]

    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=150)
    _draw_map_background(ax, grid_map)
    if trace:
        xs, ys = _cell_centers(trace)
        ax.plot(xs, ys, color="#1b6ca8", linewidth=2.7, label="travelled trace")
        replan_cells = [frame.robot for frame in simulation.frames if frame.replanned]
        if replan_cells:
            rx, ry = _cell_centers(replan_cells)
            ax.scatter(rx, ry, marker="D", s=75, c="#8f4bb4", label="replan point", zorder=6)
    ax.set_title("Executed trajectory under moving obstacles")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_replanning_timeline(simulation: SimulationResult, output_path: str | Path) -> Path:
    """Plot replanning events and distance-to-goal trend over time."""
    output = ensure_parent(output_path)
    steps = [frame.step for frame in simulation.frames]
    goal = simulation.frames[0].goal if simulation.frames else (0, 0)
    distances = [abs(frame.robot[0] - goal[0]) + abs(frame.robot[1] - goal[1]) for frame in simulation.frames]
    replan_steps = [frame.step for frame in simulation.frames if frame.replanned]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    ax.plot(steps, distances, color="#1f77b4", linewidth=2.5, label="Manhattan distance to goal")
    for index, step in enumerate(replan_steps):
        ax.axvline(step, color="#8f4bb4", linestyle="--", linewidth=1.6, alpha=0.8, label="replan" if index == 0 else None)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Distance to goal")
    ax.set_title("Dynamic replanning timeline")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
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
