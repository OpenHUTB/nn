"""Visualization for warehouse robot coordination."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from planner import AssignmentResult, RobotPlan
from scheduler import Schedule, pad_position, trajectory_loads
from warehouse import Cell, WarehouseMap

ROBOT_COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def draw_background(ax, warehouse: WarehouseMap) -> None:
    ax.imshow(warehouse.occupancy, cmap="Greys", origin="upper", vmin=0, vmax=1, alpha=0.9)
    ax.set_xticks(np.arange(-0.5, warehouse.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, warehouse.height, 1), minor=True)
    ax.grid(which="minor", color="#d7d7d0", linewidth=0.35)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def centers(path: list[Cell]) -> tuple[list[float], list[float]]:
    return [cell[1] for cell in path], [cell[0] for cell in path]


def save_task_map(warehouse: WarehouseMap, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    draw_background(ax, warehouse)

    for idx, robot in enumerate(warehouse.robots):
        ax.scatter(robot.start[1], robot.start[0], s=120, c=ROBOT_COLORS[idx], marker="o", label=robot.robot_id, zorder=5)
        ax.text(robot.start[1] + 0.3, robot.start[0] + 0.3, robot.robot_id, fontsize=9, weight="bold")

    for task in warehouse.tasks:
        ax.scatter(task.pickup[1], task.pickup[0], marker="s", s=95, c="#f0a43a", zorder=5)
        ax.scatter(task.dropoff[1], task.dropoff[0], marker="*", s=145, c="#c63737", zorder=5)
        ax.plot([task.pickup[1], task.dropoff[1]], [task.pickup[0], task.dropoff[0]], "--", color="#777777", linewidth=1.0, alpha=0.65)
        ax.text(task.pickup[1] + 0.2, task.pickup[0] - 0.3, task.task_id, fontsize=8)

    ax.set_title("Warehouse layout: robots, pickups and drop-offs")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_route_plan(warehouse: WarehouseMap, result: AssignmentResult, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    draw_background(ax, warehouse)

    for idx, plan in enumerate(result.plans):
        route = plan.route
        xs, ys = centers(route)
        color = ROBOT_COLORS[idx % len(ROBOT_COLORS)]
        ax.plot(xs, ys, linewidth=2.4, color=color, label=f"{plan.robot.robot_id}: {plan.task_count} tasks")
        ax.scatter(xs[0], ys[0], s=95, c=color, marker="o", zorder=5)
        ax.scatter(xs[-1], ys[-1], s=95, c=color, marker="X", zorder=5)

    ax.set_title("Assigned multi-robot routes before time conflict resolution")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_load_chart(result: AssignmentResult, schedule: Schedule, baseline_distance: int, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    robot_ids = [plan.robot.robot_id for plan in result.plans]
    planned = [plan.route_length for plan in result.plans]
    scheduled_loads = trajectory_loads(schedule.trajectories)
    scheduled = [scheduled_loads[robot_id] for robot_id in robot_ids]

    x = np.arange(len(robot_ids))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=150)
    ax.bar(x - width / 2, planned, width, label="planned distance", color="#1f77b4")
    ax.bar(x + width / 2, scheduled, width, label="after waits", color="#f0a43a")
    ax.axhline(baseline_distance, color="#c63737", linestyle="--", linewidth=1.8, label="single-robot baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(robot_ids)
    ax.set_ylabel("Path steps")
    ax.set_title("Workload balance and single-robot baseline")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_conflict_chart(schedule: Schedule, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    labels = ["before scheduling", "after scheduling"]
    counts = [len(schedule.conflicts_before), len(schedule.conflicts_after)]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    bars = ax.bar(labels, counts, color=["#c63737", "#2a8046"], width=0.55)
    ax.set_ylabel("Detected conflicts")
    ax.set_title("Conflict reduction after wait insertion")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def save_gantt(schedule: Schedule, output_path: str | Path) -> Path:
    output = ensure_parent(output_path)
    robot_ids = sorted(schedule.trajectories)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)

    for idx, robot_id in enumerate(robot_ids):
        length = len(schedule.trajectories[robot_id]) - 1
        ax.barh(idx, length, left=0, height=0.45, color=ROBOT_COLORS[idx % len(ROBOT_COLORS)], label=robot_id)
        waits = count_wait_steps(schedule.trajectories[robot_id])
        if waits:
            ax.text(length + 0.5, idx, f"waits={waits}", va="center", fontsize=9)

    ax.set_yticks(range(len(robot_ids)))
    ax.set_yticklabels(robot_ids)
    ax.set_xlabel("Time step")
    ax.set_title("Robot execution timeline after conflict avoidance")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def count_wait_steps(path: list[Cell]) -> int:
    return sum(1 for prev, curr in zip(path, path[1:]) if prev == curr)


def save_animation(warehouse: WarehouseMap, schedule: Schedule, output_path: str | Path, cell_size: int = 18) -> Path:
    output = ensure_parent(output_path)
    max_time = max((len(path) for path in schedule.trajectories.values()), default=0)
    frames: list[Image.Image] = []
    for time in range(max_time):
        frames.append(draw_frame(warehouse, schedule, time, cell_size))
    if not frames:
        raise ValueError("schedule has no frames")
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=150, loop=0)
    return output


def draw_frame(warehouse: WarehouseMap, schedule: Schedule, time: int, cell_size: int) -> Image.Image:
    width = warehouse.width * cell_size
    height = warehouse.height * cell_size
    image = Image.new("RGB", (width, height), (248, 248, 244))
    draw = ImageDraw.Draw(image)

    for row in range(warehouse.height):
        for col in range(warehouse.width):
            x0, y0 = col * cell_size, row * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            fill = (45, 48, 54) if warehouse.is_obstacle((row, col)) else (248, 248, 244)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(218, 218, 212))

    for task in warehouse.tasks:
        draw_cell(draw, task.pickup, cell_size, (240, 164, 58), inset=4)
        draw_cell(draw, task.dropoff, cell_size, (198, 55, 55), inset=4)

    for idx, robot_id in enumerate(sorted(schedule.trajectories)):
        cell = pad_position(schedule.trajectories[robot_id], time)
        color = hex_to_rgb(ROBOT_COLORS[idx % len(ROBOT_COLORS)])
        draw_cell(draw, cell, cell_size, color, inset=2)
        draw.text((cell[1] * cell_size + 3, cell[0] * cell_size + 3), robot_id, fill=(255, 255, 255))

    draw.rectangle([2, 2, 150, 22], fill=(255, 255, 255))
    draw.text((6, 6), f"time={time}", fill=(0, 0, 0))
    return image


def draw_cell(draw: ImageDraw.ImageDraw, cell: Cell, cell_size: int, color: tuple[int, int, int], inset: int) -> None:
    row, col = cell
    draw.rectangle(
        [col * cell_size + inset, row * cell_size + inset, (col + 1) * cell_size - inset, (row + 1) * cell_size - inset],
        fill=color,
    )


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
