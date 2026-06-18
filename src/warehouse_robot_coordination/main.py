"""Run the warehouse robot coordination demo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from planner import baseline_single_robot_plan, greedy_assign_tasks
from scheduler import resolve_conflicts, trajectory_loads
from visualization import (
    save_animation,
    save_conflict_chart,
    save_gantt,
    save_load_chart,
    save_route_plan,
    save_task_map,
)
from warehouse import make_demo_warehouse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warehouse multi-robot coordination demo")
    parser.add_argument("--output", default="assets", help="output directory")
    return parser.parse_args()


def run(output_dir: str | Path = "assets") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warehouse = make_demo_warehouse()
    assignment = greedy_assign_tasks(warehouse)
    baseline = baseline_single_robot_plan(warehouse)
    schedule = resolve_conflicts(assignment)

    task_map = save_task_map(warehouse, output_dir / "warehouse_task_map.png")
    route_plan = save_route_plan(warehouse, assignment, output_dir / "assigned_routes.png")
    load_chart = save_load_chart(assignment, schedule, baseline.route_length, output_dir / "workload_balance.png")
    conflict_chart = save_conflict_chart(schedule, output_dir / "conflict_reduction.png")
    gantt_chart = save_gantt(schedule, output_dir / "robot_timeline.png")
    animation = save_animation(warehouse, schedule, output_dir / "warehouse_coordination.gif")

    metrics = {
        "robots": len(warehouse.robots),
        "tasks": len(warehouse.tasks),
        "total_planned_distance": assignment.total_distance,
        "scheduled_makespan": schedule.makespan,
        "single_robot_baseline_distance": baseline.route_length,
        "inserted_waits": schedule.inserted_waits,
        "conflicts_before": len(schedule.conflicts_before),
        "conflicts_after": len(schedule.conflicts_after),
        "robot_loads": trajectory_loads(schedule.trajectories),
        "assignments": {
            plan.robot.robot_id: [task.task_id for task in plan.tasks]
            for plan in assignment.plans
        },
        "outputs": {
            "task_map": str(task_map),
            "assigned_routes": str(route_plan),
            "workload_balance": str(load_chart),
            "conflict_reduction": str(conflict_chart),
            "robot_timeline": str(gantt_chart),
            "animation": str(animation),
        },
    }

    write_metrics(output_dir, metrics)
    print_summary(metrics)
    return metrics


def write_metrics(output_dir: Path, metrics: dict) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    with open(output_dir / "robot_loads.csv", "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["robot_id", "scheduled_steps"])
        writer.writeheader()
        for robot_id, steps in metrics["robot_loads"].items():
            writer.writerow({"robot_id": robot_id, "scheduled_steps": steps})


def print_summary(metrics: dict) -> None:
    print("Warehouse robot coordination demo finished")
    print(f"Robots: {metrics['robots']}  Tasks: {metrics['tasks']}")
    print(f"Total planned distance: {metrics['total_planned_distance']}")
    print(f"Scheduled makespan: {metrics['scheduled_makespan']}")
    print(f"Single robot baseline distance: {metrics['single_robot_baseline_distance']}")
    print(f"Conflicts before/after: {metrics['conflicts_before']} -> {metrics['conflicts_after']}")
    print(f"Inserted waits: {metrics['inserted_waits']}")
    for robot_id, tasks in metrics["assignments"].items():
        print(f"{robot_id}: {', '.join(tasks) if tasks else 'no tasks'}")


def main() -> None:
    args = parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
