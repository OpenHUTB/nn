import os
import sys
import time
import argparse
import numpy as np
import mujoco
from mujoco import viewer
from typing import Tuple, List, Optional, NoReturn
from concurrent.futures import ThreadPoolExecutor
import tqdm

def load_model(model_path: str) -> Tuple[Optional[mujoco.MjModel], Optional[mujoco.MjData]]:
    """加载模型（支持XML和MJB格式）"""
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在 - {model_path}")
        return None, None

    try:
        if model_path.endswith('.mjb'):
            model = mujoco.MjModel.from_binary_path(model_path)
        else:
            model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"[成功] 加载模型: {os.path.basename(model_path)}")
        print_model_info(model)
        return model, data
    except Exception as e:
        print(f"[错误] 模型加载失败: {str(e)}")
        return None, None

def print_model_info(model: mujoco.MjModel) -> None:
    """打印模型基本信息"""
    info = [
        f"  自由度: {model.nq}",
        f"  关节数量: {model.njnt}",
        f"  身体数量: {model.nbody}",
        f"  传感器数量: {model.nsensor}",
        f"  控制维度: {model.nu}",
        f"  时间步长: {model.opt.timestep:.6f}s"
    ]
    print("模型信息:")
    print("\n".join(info))

def convert_model(input_path: str, output_path: str) -> bool:
    """模型格式转换（XML↔MJB）"""
    # 验证输出路径：若目录不存在则创建
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"[错误] 无法创建输出目录: {str(e)}")
            return False

    # 加载输入模型
    model, _ = load_model(input_path)
    if not model:
        return False

    try:
        if output_path.endswith('.mjb'):
            mujoco.save_model(model, output_path)
        else:
            xml_content = mujoco.save_last_xml(output_path, model)
            with open(output_path, 'w') as f:
                f.write(xml_content)
        
        print(f"[成功] 模型已转换至: {output_path}")
        print(f"  文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        return True
    except Exception as e:
        print(f"[错误] 模型转换失败: {str(e)}")
        return False

def simulate_worker(model: mujoco.MjModel, nstep: int, ctrl: np.ndarray) -> float:
    """模拟工作线程：执行单线程的模拟计算"""
    data = mujoco.MjData(model)
    start = time.time()
    for i in range(nstep):
        data.ctrl[:] = ctrl[i]
        mujoco.mj_step(model, data)
    return time.time() - start

def test_speed(model_path: str, nstep: int = 10000, nthread: int = 1, ctrlnoise: float = 0.01) -> NoReturn:
    """测试模拟速度：支持多线程，输出性能指标"""
    model, _ = load_model(model_path)
    if not model:
        return

    # 参数修正：确保线程数不超过CPU核心数，步数不小于100
    nthread = max(1, min(nthread, os.cpu_count() or 1))
    nstep = max(100, nstep)
    
    print(f"\n[信息] 开始速度测试: {nstep}步 × {nthread}线程")
    print(f"  控制噪声强度: {ctrlnoise}")

    # 生成控制噪声（模拟真实控制信号波动）
    ctrl = ctrlnoise * np.random.randn(nstep, model.nu)

    # 使用线程池执行多线程模拟，添加进度条
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=nthread) as executor:
        futures = [executor.submit(simulate_worker, model, nstep, ctrl) for _ in range(nthread)]
        
        # 进度条显示：等待所有线程完成
        for _ in tqdm.tqdm(range(nthread), desc="模拟进度"):
            for future in futures:
                if future.done():
                    futures.remove(future)
                    break
            time.sleep(0.1)

    total_time = time.time() - start_time

    # 计算核心性能指标
    total_steps = nstep * nthread
    steps_per_sec = total_steps / total_time
    realtime_factor = (total_steps * model.opt.timestep) / total_time

    print("\n[测试结果] 速度测试结果:")
    print(f"  总步数: {total_steps:,}")  # 千位分隔符提升可读性
    print(f"  总时间: {total_time:.2f}s")
    print(f"  每秒步数: {steps_per_sec:.0f}")
    print(f"  实时因子: {realtime_factor:.2f}倍")  # 明确标注“倍”，避免误解

def visualize(model_path: str) -> NoReturn:
    """可视化模拟：支持视角控制、暂停/继续等交互操作"""
    model, data = load_model(model_path)
    if not model or not data:
        return

    print("\n[可视化] 启动可视化窗口")
    print("  操作说明:")
    print("  - 鼠标拖动: 旋转/平移视角")
    print("  - 滚轮: 缩放视角")
    print("  - 空格键: 暂停/继续模拟")
    print("  - ESC键: 退出窗口")

    # 启动被动式查看器（支持自定义交互）
    with viewer.launch_passive(model, data) as v:
        paused = False  # 暂停状态标记
        while v.is_running():
            # 未暂停时执行模拟步
            if not paused:
                mujoco.mj_step(model, data)
            # 同步查看器与模拟数据
            v.sync()
            
            # 处理键盘事件
            for event in v.window.events:
                if event.key == ord(' '):  # 空格键切换暂停状态
                    paused = not paused
                    event.consumed = True  # 标记事件已处理，避免重复响应
                elif event.key == 27:  # ESC键退出
                    v.close()
                    event.consumed = True

def main():
    # 命令行参数解析：使用默认值提示，提升易用性
    parser = argparse.ArgumentParser(
        description="MuJoCo功能整合工具（支持模型加载、可视化、速度测试、格式转换）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="子命令（选择需执行的功能）")

    # 1. 可视化子命令
    viz_parser = subparsers.add_parser("visualize", help="可视化模型模拟过程")
    viz_parser.add_argument("model", help="模型文件路径（支持XML或MJB格式）")

    # 2. 速度测试子命令
    speed_parser = subparsers.add_parser("testspeed", help="测试模型模拟速度")
    speed_parser.add_argument("model", help="模型文件路径")
    speed_parser.add_argument("--nstep", type=int, default=10000, help="每线程执行的模拟步数")
    speed_parser.add_argument("--nthread", type=int, default=1, help="用于模拟的线程数（建议不超过CPU核心数）")
    speed_parser.add_argument("--ctrlnoise", type=float, default=0.01, help="控制信号的噪声强度（模拟真实场景波动）")

    # 3. 模型转换子命令
    convert_parser = subparsers.add_parser("convert", help="转换模型格式（XML与MJB互转）")
    convert_parser.add_argument("input", help="输入模型路径（XML或MJB格式）")
    convert_parser.add_argument("output", help="输出模型路径（需指定格式：.xml或.mjb）")

    args = parser.parse_args()

    # 执行对应功能，捕获异常并友好提示
    try:
        if args.command == "visualize":
            visualize(args.model)
        elif args.command == "testspeed":
            test_speed(args.model, args.nstep, args.nthread, args.ctrlnoise)
        elif args.command == "convert":
            convert_model(args.input, args.output)
        print("\n[完成] 操作执行完毕")
    except KeyboardInterrupt:
        print("\n[提示] 操作被用户手动中断（Ctrl+C）")
    except Exception as e:
        print(f"\n[错误] 发生未预期错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()