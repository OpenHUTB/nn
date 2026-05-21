"""Run the smart intersection signal optimization demo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from optimizer import optimize_signal_plan
from scenario import make_demo_scenario
from simulator import summarize_result
from visualization import (
    save_convergence,
    save_intersection_animation,
    save_metric_comparison,
    save_plan_comparison,
    save_queue_profiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart intersection signal optimization")
    parser.add_argument("--output", default="assets", help="output directory")
    parser.add_argument("--generations", type=int, default=36, help="GA generations")
    parser.add_argument("--population", type=int, default=42, help="GA population size")
    return parser.parse_args()


def run(output_dir: str | Path = "assets", generations: int = 36, population: int = 42) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = make_demo_scenario()
    result = optimize_signal_plan(scenario, population_size=population, generations=generations)

    queue_profiles = save_queue_profiles(result.optimized, output_dir / "optimized_queue_profiles.png")
    plan_comparison = save_plan_comparison(scenario, result, output_dir / "green_split_comparison.png")
    metric_comparison = save_metric_comparison(result, output_dir / "metric_comparison.png")
    convergence = save_convergence(result, output_dir / "ga_convergence.png")
    animation = save_intersection_animation(scenario, result.optimized, output_dir / "intersection_signal_optimization.gif")

    baseline_summary = summarize_result(result.baseline)
    optimized_summary = summarize_result(result.optimized)
    improvements = {
        "delay_reduction_percent": percent_reduction(result.baseline.average_delay, result.optimized.average_delay),
        "max_queue_reduction_percent": percent_reduction(result.baseline.max_queue, result.optimized.max_queue),
        "score_reduction_percent": percent_reduction(result.baseline.score, result.optimized.score),
        "throughput_gain_percent": round((result.optimized.throughput - result.baseline.throughput) * 100, 3),
    }
    metrics = {
        "scenario": {
            "horizon_seconds": scenario.horizon_seconds,
            "approaches": [
                {
                    "name": approach.name,
                    "demand_vph": approach.demand_vph,
                    "saturation_vph": approach.saturation_vph,
                    "priority": approach.priority,
                }
                for approach in scenario.approaches
            ],
        },
        "optimizer": {
            "algorithm": "elitist genetic algorithm",
            "population_size": result.population_size,
            "generations": result.generations,
        },
        "baseline": baseline_summary,
        "optimized": optimized_summary,
        "improvements": improvements,
        "optimized_greens": result.optimized.plan.greens,
        "outputs": {
            "queue_profiles": str(queue_profiles),
            "green_split_comparison": str(plan_comparison),
            "metric_comparison": str(metric_comparison),
            "ga_convergence": str(convergence),
            "animation": str(animation),
        },
    }
    write_outputs(output_dir, metrics, result)
    print_summary(metrics)
    return metrics


def percent_reduction(before: float, after: float) -> float:
    return round((before - after) / max(before, 1e-9) * 100, 3)


def write_outputs(output_dir: Path, metrics: dict, result) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    with open(output_dir / "optimized_greens.csv", "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["phase", "green_seconds"])
        writer.writeheader()
        for phase, green_seconds in metrics["optimized_greens"].items():
            writer.writerow({"phase": phase, "green_seconds": green_seconds})

    with open(output_dir / "convergence.csv", "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["generation", "best_score", "best_delay", "best_queue"])
        writer.writeheader()
        for generation, score, delay, queue in zip(
            result.history.generation,
            result.history.best_score,
            result.history.best_delay,
            result.history.best_queue,
        ):
            writer.writerow(
                {
                    "generation": generation,
                    "best_score": round(score, 6),
                    "best_delay": round(delay, 6),
                    "best_queue": round(queue, 6),
                }
            )


def print_summary(metrics: dict) -> None:
    print("Smart intersection signal optimization finished")
    print(f"Algorithm: {metrics['optimizer']['algorithm']}")
    print(f"Baseline delay: {metrics['baseline']['average_delay']}  Optimized delay: {metrics['optimized']['average_delay']}")
    print(f"Baseline max queue: {metrics['baseline']['max_queue']}  Optimized max queue: {metrics['optimized']['max_queue']}")
    print(f"Delay reduction: {metrics['improvements']['delay_reduction_percent']}%")
    print(f"Score reduction: {metrics['improvements']['score_reduction_percent']}%")
    print(f"Optimized cycle: {metrics['optimized']['cycle_length']}s")
    for phase, green in metrics["optimized_greens"].items():
        print(f"{phase}: {green}s")


def main() -> None:
    args = parse_args()
    run(args.output, generations=args.generations, population=args.population)


if __name__ == "__main__":
    main()
