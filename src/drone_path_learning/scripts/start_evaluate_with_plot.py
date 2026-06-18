<<<<<<< HEAD
import sys
import argparse
from PyQt5 import QtWidgets

from utils.thread_evaluation import EvaluateThread
from utils.ui_train import TrainingUi
from configparser import ConfigParser


def get_parser():
    parser = argparse.ArgumentParser(description="trained model evaluation with plot")
    parser.add_argument(
        "-model_path",
        required=True,
        help="model path to be evaluated, \
                                            just copy the relative path of the log",
    )
    parser.add_argument(
        "-eval_eps", required=True, type=int, help="evaluation episode number"
    )

    return parser


def main():

    # 设置评估模型路径
    eval_path = r"logs\NH_center\2026_04_16_16_37_Multirotor_CNN_FC_PPO"

    # 选择配置文件和模型名称
    config_file = eval_path + "/config/config.ini"
    model_file = eval_path + "/models/model.zip"
    total_eval_episodes = 50

    # 1. 创建Qt线程
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config=config_file)
    gui.show()

    # 2. 启动评估线程
    evaluate_thread = EvaluateThread(
        eval_path, config_file, model_file, total_eval_episodes
=======
﻿import argparse
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
>>>>>>> upstream/main
    )
    evaluate_thread.env.action_signal.connect(gui.action_cb)
    evaluate_thread.env.state_signal.connect(gui.state_cb)
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)

<<<<<<< HEAD
    cfg = ConfigParser()
    cfg.read(config_file)
    if cfg.has_option("options", "perception"):
        if cfg.get("options", "perception") == "lgmd":
            evaluate_thread.env.lgmd_signal.connect(gui.lgmd_plot_cb)

    evaluate_thread.start()

    # 程序会在关闭GUI后才退出
    sys.exit(app.exec_())
    print("Exiting program")
=======
    if evaluate_thread.cfg.has_option("options", "perception"):
        if evaluate_thread.cfg.get("options", "perception") == "lgmd":
            evaluate_thread.env.lgmd_signal.connect(gui.lgmd_plot_cb)

    evaluate_thread.start()
    sys.exit(app.exec_())
>>>>>>> upstream/main


if __name__ == "__main__":
    main()
