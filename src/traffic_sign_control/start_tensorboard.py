"""
启动TensorBoard可视化工具
监控YOLO模型训练过程
"""
import os
import subprocess
import argparse

def start_tensorboard(logdir='runs/detect'):
    """
    启动TensorBoard

    Args:
        logdir: TensorBoard日志目录
    """
    if not os.path.exists(logdir):
        print(f"警告: 日志目录不存在: {logdir}")
        print("提示: 请先运行 train.py 开始训练")
        return

    print(f"启动TensorBoard...")
    print(f"日志目录: {logdir}")
    print(f"访问地址: http://localhost:6006")
    print(f"按 Ctrl+C 停止\n")

    try:
        subprocess.run(['tensorboard', '--logdir', logdir, '--port', '6006'])
    except KeyboardInterrupt:
        print("\nTensorBoard已停止")
    except FileNotFoundError:
        print("错误: TensorBoard未安装")
        print("安装命令: pip install tensorboard")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动TensorBoard监控训练")
    parser.add_argument('--logdir', type=str, default='runs/detect',
                       help='TensorBoard日志目录 (默认: runs/detect)')

    args = parser.parse_args()
    start_tensorboard(args.logdir)
