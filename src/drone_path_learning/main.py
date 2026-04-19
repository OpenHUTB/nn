import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MenuEntry:
    label: str
    script: Path


MENU_ENTRIES = {
    "train": MenuEntry(
        label="训练（可视化）",
        script=PROJECT_ROOT / "scripts" / "start_train_with_plot.py",
    ),
    "eval": MenuEntry(
        label="评估（可视化）",
        script=PROJECT_ROOT / "scripts" / "start_evaluate_with_plot.py",
    ),
    "torch_check": MenuEntry(
        label="Torch/CUDA 检查",
        script=PROJECT_ROOT / "tools" / "test" / "torch_gpu_cpu_test.py",
    ),
    "env_test": MenuEntry(
        label="环境快速测试",
        script=PROJECT_ROOT / "tools" / "test" / "env_test.py",
    ),
}

MENU_ORDER = [
    ("1", "train"),
    ("2", "eval"),
    ("3", "torch_check"),
    ("4", "env_test"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="drone_path_learning 启动器")
    parser.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=sorted(MENU_ENTRIES.keys()),
        help="启动模式（可选）: train / eval / torch_check / env_test",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="透传给目标启动脚本的参数",
    )
    return parser


def _interactive_select_mode() -> str | None:
    print("\ndrone_path_learning 启动菜单")
    for idx, key in MENU_ORDER:
        print(f"{idx}) {MENU_ENTRIES[key].label} ({key})")
    print("0) 退出")

    mapping = {idx: key for idx, key in MENU_ORDER}
    mapping.update({"0": None, "q": None, "quit": None, "exit": None})

    while True:
        choice = input("请选择功能编号（或输入功能键）: ").strip().lower()
        if choice in MENU_ENTRIES:
            return choice
        if choice in mapping:
            return mapping[choice]
        print("输入无效，请重新输入。")


def _run_entry(mode: str, forwarded_args: list[str]) -> None:
    target = MENU_ENTRIES[mode]
    if not target.script.exists():
        raise FileNotFoundError(f"未找到启动脚本: {target.script}")

    rel_path = target.script.relative_to(PROJECT_ROOT)
    print(f"\n正在启动: {target.label}")
    print(f"脚本路径: {rel_path}")

    cmd = [sys.executable, str(target.script), *forwarded_args]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    args = _build_parser().parse_args()

    if args.mode is None:
        mode = _interactive_select_mode()
        if mode is None:
            print("已退出启动器。")
            return
        forwarded_args = []
    else:
        mode = args.mode
        forwarded_args = list(args.script_args)
        if forwarded_args and forwarded_args[0] == "--":
            forwarded_args = forwarded_args[1:]

    _run_entry(mode, forwarded_args)


if __name__ == "__main__":
    main()
