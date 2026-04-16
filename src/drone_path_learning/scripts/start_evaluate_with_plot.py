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
    )
    evaluate_thread.env.action_signal.connect(gui.action_cb)
    evaluate_thread.env.state_signal.connect(gui.state_cb)
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)

    cfg = ConfigParser()
    cfg.read(config_file)
    if cfg.has_option("options", "perception"):
        if cfg.get("options", "perception") == "lgmd":
            evaluate_thread.env.lgmd_signal.connect(gui.lgmd_plot_cb)

    evaluate_thread.start()

    # 程序会在关闭GUI后才退出
    sys.exit(app.exec_())
    print("Exiting program")


if __name__ == "__main__":
    main()
