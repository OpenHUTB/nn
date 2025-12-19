#!/usr/bin/env python3
"""
beatsvr_bimanual_eval - single entrypoint

This script is the ONLY runnable entry for:
- training
- evaluation
- bimanual + envCamera/mainCamera evaluation visualization
"""

from __future__ import annotations
import argparse
import sys

from pipelines.train import run_train
from pipelines.eval import run_eval
from pipelines.record_multiview import run_record
from utils.export_gifs import run_export_gifs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beatsvr_bimanual_eval",
        description="Single entrypoint for BeatSVR bimanual evaluation and visualization."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---------------- train ----------------
    p_train = sub.add_parser("train", help="Train BeatSVR bimanual policy")
    p_train.add_argument("--config", required=True, help="Path to training config")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--total_steps", type=int, default=1_000_000)
    p_train.set_defaults(func=run_train)

    # ---------------- eval ----------------
    p_eval = sub.add_parser("eval", help="Evaluate trained policy")
    p_eval.add_argument("--simulator_dir", required=True)
    p_eval.add_argument("--checkpoint", default="", help="Optional model checkpoint")
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.set_defaults(func=run_eval)

    # ---------------- record ----------------
    p_rec = sub.add_parser(
        "record",
        help="Run bimanual evaluation and record envCamera + mainCamera"
    )
    p_rec.add_argument("--simulator_dir", required=True)
    p_rec.add_argument("--episodes", type=int, default=1)
    p_rec.add_argument("--max_steps", type=int, default=2000)
    p_rec.add_argument("--fps", type=int, default=20)
    p_rec.add_argument("--out_dir", default="evaluate")
    p_rec.add_argument("--checkpoint", default="", help="Optional model checkpoint")
    p_rec.set_defaults(func=run_record)

    # ---------------- export gif ----------------
    p_gif = sub.add_parser("export_gif", help="Convert evaluation MP4 to GIF")
    p_gif.add_argument("--inputs", nargs="+", required=True)
    p_gif.add_argument("--out_dir", default=".")
    p_gif.add_argument("--fps", type=int, default=15)
    p_gif.add_argument("--width", type=int, default=640)
    p_gif.set_defaults(func=run_export_gifs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
