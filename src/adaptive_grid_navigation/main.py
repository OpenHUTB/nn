"""Command line entry point for adaptive grid navigation demo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from grid_map import make_demo_map, make_random_map
from planners import default_planners
from simulator import run_dynamic_replanning
from visualization import (
    save_animation,
    save_comparison_chart,
    save_expansion_heatmap,
    save_path_overlay,
    save_plan_image,
    save_replanning_timeline,
    save_robot_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive grid navigation demo")
    parser.add_argument("--map", choices=["demo", "random"], default="demo", help="map type")
    parser.add_argument("--seed", type=int, default=7, help="random map seed")
    parser.add_argument("--output", default="assets", help="output directory for figures and metrics")
    parser.add_argument("--max-steps", type=int, default=120, help="maximum dynamic simulation steps")
    return parser.parse_args()


def run(output_dir: str | Path = "assets", map_type: str = "demo", seed: int = 7, max_steps: int = 120) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_map = make_demo_map() if map_type == "demo" else make_random_map(seed=seed)
    planners = default_planners()
    results = [planner.plan(grid_map) for planner in planners]

    best_result = next(result for result in results if result.planner == "A*")
    plan_image = save_plan_image(grid_map, best_result, output_dir / "astar_route.png")
    chart_image = save_comparison_chart(results, output_dir / "planner_comparison.png")
    overlay_image = save_path_overlay(grid_map, results, output_dir / "planner_route_overlay.png")
    heatmap_image = save_expansion_heatmap(grid_map, best_result, output_dir / "astar_expansion_heatmap.png")

    simulation = run_dynamic_replanning(grid_map, max_steps=max_steps)
    trace_image = save_robot_trace(grid_map, simulation, output_dir / "robot_trace.png")
    timeline_image = save_replanning_timeline(simulation, output_dir / "replanning_timeline.png")
    gif_path = save_animation(grid_map, simulation, output_dir / "dynamic_replanning.gif")

    metrics = {
        "map": map_type,
        "start": grid_map.start,
        "goal": grid_map.goal,
        "planners": [
            {
                "name": result.planner,
                "success": result.success,
                "path_length": result.path_length,
                "expanded_nodes": result.expanded_count,
            }
            for result in results
        ],
        "dynamic_replanning": {
            "reached_goal": simulation.reached_goal,
            "replans": simulation.replans,
            "travelled": simulation.travelled,
            "frames": len(simulation.frames),
            "final_position": simulation.final_position,
        },
        "outputs": {
            "route_image": str(plan_image),
            "comparison_chart": str(chart_image),
            "route_overlay": str(overlay_image),
            "expansion_heatmap": str(heatmap_image),
            "robot_trace": str(trace_image),
            "replanning_timeline": str(timeline_image),
            "animation": str(gif_path),
        },
    }

    write_metrics(output_dir, metrics)
    print_summary(metrics)
    return metrics


def write_metrics(output_dir: Path, metrics: dict) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    with open(output_dir / "planner_metrics.csv", "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["name", "success", "path_length", "expanded_nodes"])
        writer.writeheader()
        writer.writerows(metrics["planners"])


def print_summary(metrics: dict) -> None:
    print("Adaptive grid navigation demo finished")
    print(f"Map: {metrics['map']}  Start: {metrics['start']}  Goal: {metrics['goal']}")
    for planner in metrics["planners"]:
        print(
            f"{planner['name']}: success={planner['success']} "
            f"path_length={planner['path_length']} expanded_nodes={planner['expanded_nodes']}"
        )
    dynamic = metrics["dynamic_replanning"]
    print(
        "Dynamic replanning: "
        f"reached_goal={dynamic['reached_goal']} replans={dynamic['replans']} "
        f"travelled={dynamic['travelled']} frames={dynamic['frames']}"
    )
    print("Generated outputs:")
    for name, path in metrics["outputs"].items():
        print(f"- {name}: {path}")


def main() -> None:
    args = parse_args()
    run(args.output, args.map, args.seed, args.max_steps)


if __name__ == "__main__":
    main()
