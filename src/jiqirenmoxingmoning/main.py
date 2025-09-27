import os
import sys
import time
import argparse
import threading
import numpy as np
import mujoco
from mujoco import viewer

def load_model(model_path):
    """加载模型（支持XML和MJB格式）"""
    try:
        if model_path.endswith('.mjb'):
            model = mujoco.MjModel.from_binary_path(model_path)
        else:
            model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"成功加载模型: {model_path}")
        return model, data
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None, None

def convert_model(input_path, output_path):
    """模型格式转换（XML↔MJB）"""
    model, _ = load_model(input_path)
    if not model:
        return False

    try:
        if output_path.endswith('.mjb'):
            mujoco.save_model(model, output_path)
        else:
            with open(output_path, 'w') as f:
                f.write(mujoco.save_last_xml(output_path, model))
        print(f"模型已转换至: {output_path}")
        return True
    except Exception as e:
        print(f"模型转换失败: {str(e)}")
        return False

def test_speed(model_path, nstep=10000, nthread=1, ctrlnoise=0.01):
    """测试模拟速度"""
    model, data = load_model(model_path)
    if not model:
        return

    # 生成控制噪声
    ctrl = ctrlnoise * np.random.randn(nstep, model.nu)

    def simulate_thread(thread_id):
        """线程模拟函数"""
        start = time.time()
        mj_data = mujoco.MjData(model)
        for i in range(nstep):
            mj_data.ctrl[:] = ctrl[i]
            mujoco.mj_step(model, mj_data)
        end = time.time()
        return end - start

    # 多线程模拟
    threads = []
    start_time = time.time()
    for i in range(nthread):
        t = threading.Thread(target=simulate_thread, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    total_time = time.time() - start_time

    # 计算性能指标
    total_steps = nstep * nthread
    steps_per_sec = total_steps / total_time
    realtime_factor = (total_steps * model.opt.timestep) / total_time

    print("\n速度测试结果:")
    print(f"总步数: {total_steps}, 总时间: {total_time:.2f}s")
    print(f"每秒步数: {steps_per_sec:.0f}")
    print(f"实时因子: {realtime_factor:.2f}x")

def visualize(model_path):
    """可视化模拟"""
    model, data = load_model(model_path)
    if not model:
        return

    # 启动交互式查看器
    with viewer.launch_passive(model, data) as v:
        print("可视化窗口已启动 (按ESC退出)")
        while True:
            mujoco.mj_step(model, data)
            v.sync()
            # 检查退出条件（窗口关闭）
            if not v.is_running():
                break

def main():
    parser = argparse.ArgumentParser(description="MuJoCo功能整合工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 可视化命令
    viz_parser = subparsers.add_parser("visualize", help="可视化模型")
    viz_parser.add_argument("model", help="模型文件路径（XML或MJB）")

    # 速度测试命令
    speed_parser = subparsers.add_parser("testspeed", help="测试模拟速度")
    speed_parser.add_argument("model", help="模型文件路径")
    speed_parser.add_argument("--nstep", type=int, default=10000, help="每轮步数")
    speed_parser.add_argument("--nthread", type=int, default=1, help="线程数")
    speed_parser.add_argument("--ctrlnoise", type=float, default=0.01, help="控制噪声强度")

    # 模型转换命令
    convert_parser = subparsers.add_parser("convert", help="转换模型格式")
    convert_parser.add_argument("input", help="输入模型路径")
    convert_parser.add_argument("output", help="输出模型路径（.xml或.mjb）")

    args = parser.parse_args()

    if args.command == "visualize":
        visualize(args.model)
    elif args.command == "testspeed":
        test_speed(args.model, args.nstep, args.nthread, args.ctrlnoise)
    elif args.command == "convert":
        convert_model(args.input, args.output)

if __name__ == "__main__":
    main()