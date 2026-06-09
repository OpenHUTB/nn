"""Command-line demo for the rain sensor fusion project."""

from __future__ import annotations

import argparse
import os
import sys


if __package__ in (None, ""):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.rain_sensor_fusion.fusion.deep_fusion import RainFusionEngine
from src.rain_sensor_fusion.sensor.synthetic import DEMO_SCENARIOS, make_demo_snapshot
from src.rain_sensor_fusion.training import train_on_synthetic_rain
from src.rain_sensor_fusion.visualization.plots import save_visualization_suite
from src.rain_sensor_fusion.visualization.report import format_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rain-aware deep sensor fusion demo")
    parser.add_argument(
        "--scenario",
        default="heavy_rain",
        choices=DEMO_SCENARIOS,
        help="synthetic weather scenario to analyze",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help="optional quick synthetic training steps before inference",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save PNG visualizations for the selected scenario",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also save all demo scenario dashboards, a comparison plot, sweep plot, and CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rain_sensor_fusion",
        help="directory for visualization artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = RainFusionEngine()
    if args.train_steps > 0:
        history = train_on_synthetic_rain(engine.model, steps=args.train_steps)
        print(f"Synthetic training finished: final_loss={history[-1]:.4f}")

    snapshot = make_demo_snapshot(args.scenario)
    result = engine.analyze(snapshot)
    print(format_result(snapshot, result))
    if args.visualize or args.compare:
        saved = save_visualization_suite(
            engine,
            args.output_dir,
            scenario=args.scenario,
            include_comparison=args.compare,
        )
        print("\nSaved visualizations:")
        for path in saved:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
