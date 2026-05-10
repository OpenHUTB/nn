import argparse
import sys
from pathlib import Path

from PyQt5 import QtWidgets

try:
    from scripts.utils.thread_evaluation import EvaluateThread
    from scripts.utils.ui_train import TrainingUi
except ImportError:
    from utils.thread_evaluation import EvaluateThread
    from utils.ui_train import TrainingUi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_PATH = PROJECT_ROOT / "logs" / "NH_center" / "2026_05_07_22_43_Multirotor_mlp_PPO"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start model evaluation with visualization UI")
    parser.add_argument(
        "--eval-path",
        default=str(DEFAULT_EVAL_PATH),
        help="Path to a training run folder containing config/ and models/.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config file path. Defaults to <eval-path>/config/config.ini",
    )
    parser.add_argument(
        "--model-file",
        default=None,
        help="Optional model file path. Defaults to <eval-path>/models/model_sb3.zip",
    )
    parser.add_argument("--eval-eps", type=int, default=50, help="Evaluation episodes.")
    parser.add_argument("--eval-env", default=None, help="Optional override for env_name.")
    parser.add_argument(
        "--eval-dynamics", default=None, help="Optional override for dynamic_name."
    )
    return parser


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main() -> None:
    args = get_parser().parse_args()

    eval_path = resolve_path(args.eval_path)
    config_file = resolve_path(args.config) if args.config else eval_path / "config" / "config.ini"
    model_file = (
        resolve_path(args.model_file)
        if args.model_file
        else eval_path / "models" / "model_sb3.zip"
    )

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config=str(config_file))
    gui.show()

    evaluate_thread = EvaluateThread(
        eval_path=str(eval_path),
        config=str(config_file),
        model_file=str(model_file),
        eval_ep_num=args.eval_eps,
        eval_env=args.eval_env,
        eval_dynamics=args.eval_dynamics,
    )
    evaluate_thread.env.action_signal.connect(gui.action_cb)
    evaluate_thread.env.state_signal.connect(gui.state_cb)
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)

    if evaluate_thread.cfg.has_option("options", "perception"):
        if evaluate_thread.cfg.get("options", "perception") == "lgmd":
            evaluate_thread.env.lgmd_signal.connect(gui.lgmd_plot_cb)

    evaluate_thread.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
