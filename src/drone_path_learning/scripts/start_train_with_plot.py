import argparse
import sys
from pathlib import Path

from PyQt5 import QtWidgets

try:
    from scripts.utils.thread_train import TrainingThread
    from scripts.utils.ui_train import TrainingUi
except ImportError:
    from utils.thread_train import TrainingThread
    from utils.ui_train import TrainingUi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "configs\config_NH_center_Multirotor_Phase3C_Yaw90_Goal180_20m.ini"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start training with visualization UI")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to config .ini file. Supports absolute or project-relative path.",
    )
    return parser


def resolve_config_path(raw_path: str) -> Path:
    config_path = Path(raw_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path.resolve()


def main() -> None:
    args = get_parser().parse_args()
    config_file = resolve_config_path(args.config)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(str(config_file))
    gui.show()

    training_thread = TrainingThread(str(config_file))
    training_thread.env.action_signal.connect(gui.action_cb)
    training_thread.env.state_signal.connect(gui.state_cb)
    training_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    training_thread.env.reward_signal.connect(gui.reward_plot_cb)
    training_thread.env.pose_signal.connect(gui.traj_plot_cb)

    training_thread.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
