"""机械臂抓取器控制演示

演示 MuJoCo 中三指抓取器的开合运动，支持命令行参数指定模型路径。

用法：
    python grasp.py                                    # 使用默认模型路径
    python grasp.py --model path/to/arm.xml            # 指定模型路径
    python grasp.py --model path/to/arm.xml --cycles 3 # 指定抓取循环次数
"""

import os
import shutil
import time
import argparse
import sys
import numpy as np
import mujoco

# ── 默认配置 ──────────────────────────────────────────────
DEFAULT_MODEL_PATH = r"C:\Users\龙忠梁\Downloads\mujoco-3.3.7-windows-x86_64\model\robotic_arm\arm_with_gripper.xml"
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_model")

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ARM_ANGLES = [0.0, 0.785, -1.57, 0.0, 0.0, 0.0]

FINGER_JOINTS = [
    "finger_1_joint", "finger_2_joint", "finger_3_joint",
    "finger_1_proximal_joint", "finger_2_proximal_joint", "finger_3_proximal_joint",
    "finger_1_distal_joint", "finger_2_distal_joint", "finger_3_distal_joint",
]

# 相机默认视角
CAMERA = {"azimuth": 180, "elevation": -20, "distance": 0.8, "lookat": [0.25, 0.0, 0.15]}


# ── 工具函数 ──────────────────────────────────────────────

def resolve_model_path(path: str) -> str:
    """处理包含非ASCII字符的模型路径"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    if any(ord(ch) > 127 for ch in path):
        os.makedirs(TEMP_DIR, exist_ok=True)
        temp_path = os.path.join(TEMP_DIR, os.path.basename(path))
        shutil.copy2(path, temp_path)
        print(f"模型已复制到临时路径: {temp_path}")
        return temp_path
    return path


def sim_step(model, data, viewer, ctrl_values: dict, steps: int = 1, speed: float = 1.0):
    """统一的仿真步进函数

    Args:
        model: MuJoCo 模型
        data: MuJoCo 数据
        viewer: 可视化窗口
        ctrl_values: 控制通道到目标值的映射 {channel_index: value}
        steps: 仿真步数
        speed: 速度倍率（<1 加速）
    """
    for _ in range(steps):
        for idx, val in ctrl_values.items():
            if 0 <= idx < model.nu:
                data.ctrl[idx] = val
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        time.sleep(model.opt.timestep * speed)


def set_arm_control(data, values: list):
    """设置机械臂前6个关节的控制信号"""
    for i, val in enumerate(values):
        if i < data.ctrl.shape[0]:
            data.ctrl[i] = val


def set_gripper_control(data, model, value: float):
    """统一设置所有抓取器关节的控制信号"""
    for j in range(6, model.nu):
        data.ctrl[j] = value


# ── 初始化与控制 ──────────────────────────────────────────

def set_initial_pose(model, data):
    """设置机械臂的初始姿态"""
    initial_qpos = np.zeros(model.nq)

    for i, joint_name in enumerate(ARM_JOINTS):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            initial_qpos[model.jnt_qposadr[joint_id]] = ARM_ANGLES[i]

    for joint_name in FINGER_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            addr = model.jnt_qposadr[joint_id]
            if "slide" in joint_name:
                initial_qpos[addr] = 0.02
            elif "proximal" in joint_name:
                initial_qpos[addr] = 0.3
            elif "distal" in joint_name:
                initial_qpos[addr] = 0.2

    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)


def setup_camera(viewer):
    """设置相机视角到抓取器"""
    if hasattr(viewer, 'cam'):
        viewer.cam.azimuth = CAMERA["azimuth"]
        viewer.cam.elevation = CAMERA["elevation"]
        viewer.cam.distance = CAMERA["distance"]
        viewer.cam.lookat[:] = CAMERA["lookat"]


def stabilize(model, data, viewer, steps: int = 150):
    """让机械臂稳定一段时间"""
    sim_step(model, data, viewer, {}, steps=steps, speed=2.0)


# ── 演示函数 ──────────────────────────────────────────────

def print_gripper_info(model):
    """打印抓取器关节信息"""
    print("\n抓取器关节信息:")
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        if joint_id != -1:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name and "finger" in name:
                print(f"  通道 {i}: 关节 '{name}' (ID: {joint_id})")


def build_finger_map(model):
    """构建控制通道到手指编号的映射"""
    mapping = {}
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        if joint_id != -1:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                for finger_num in [1, 2, 3]:
                    if f"finger_{finger_num}" in name:
                        mapping[i] = finger_num
                        break
    return mapping


def classify_joints(model):
    """分类抓取器关节类型"""
    slides, proximals, distals = [], [], []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        if jid != -1:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name and "finger" in name:
                if "slide" in name:
                    slides.append(i)
                elif "proximal" in name:
                    proximals.append(i)
                elif "distal" in name:
                    distals.append(i)
    return slides, proximals, distals


def demo_open_close(model, data, viewer):
    """演示基本开合动作"""
    print("\n1. 基本开合测试...")

    print("   张开 (负控制信号)...")
    sim_step(model, data, viewer, {j: -1.0 for j in range(6, model.nu)}, steps=300)

    print("   闭合 (正控制信号)...")
    sim_step(model, data, viewer, {j: 1.0 for j in range(6, model.nu)}, steps=300)

    print("   重置为张开...")
    sim_step(model, data, viewer, {j: -1.0 for j in range(6, model.nu)}, steps=300)


def demo_individual_fingers(model, data, viewer):
    """演示单独控制每个手指"""
    print("\n2. 分别控制三个手指...")
    finger_map = build_finger_map(model)

    for finger_num in [1, 2, 3]:
        print(f"   手指{finger_num}闭合...")
        ctrl = {}
        for idx, fn in finger_map.items():
            ctrl[idx] = 1.5 if fn == finger_num else 0.0
        sim_step(model, data, viewer, ctrl, steps=200)

    print("   所有手指张开...")
    sim_step(model, data, viewer, {j: -1.0 for j in range(6, model.nu)}, steps=300)


def demo_grasp_cycles(model, data, viewer, num_cycles: int = 3):
    """演示多次抓取循环"""
    print(f"\n3. 多次抓取循环 ({num_cycles} 次)...")

    for cycle in range(num_cycles):
        print(f"   循环 {cycle + 1}: 闭合...")
        sim_step(model, data, viewer, {j: 2.0 for j in range(6, model.nu)}, steps=200)
        sim_step(model, data, viewer, {}, steps=100)  # 保持

        print(f"   循环 {cycle + 1}: 张开...")
        sim_step(model, data, viewer, {j: -2.0 for j in range(6, model.nu)}, steps=200)
        sim_step(model, data, viewer, {}, steps=100)  # 保持

    print("   最终张开...")
    sim_step(model, data, viewer, {j: -1.0 for j in range(6, model.nu)}, steps=200)


# ── 参数解析 ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo 机械臂抓取器控制演示")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="MuJoCo XML 模型文件路径")
    parser.add_argument("--cycles", type=int, default=3,
                        help="抓取循环次数 (默认: 3)")
    parser.add_argument("--demo", type=str, default="all",
                        choices=["all", "basic", "fingers", "cycles"],
                        help="演示模式 (默认: all)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="无GUI模式（仅测试加载）")
    return parser.parse_args()


# ── 主函数 ────────────────────────────────────────────────

def main():
    args = parse_args()

    model_path = resolve_model_path(args.model)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print("=" * 50)
    print("  机械臂抓取器控制演示")
    print("=" * 50)

    if args.no_viewer:
        print("✓ 模型加载成功！")
        set_initial_pose(model, data)
        mujoco.mj_forward(model, data)
        print(f"  关节数: {model.nq}, 执行器数: {model.nu}")
        return 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        setup_camera(viewer)

        print("\n阶段 1: 初始化...")
        set_initial_pose(model, data)
        stabilize(model, data, viewer)
        print("✓ 初始化完成")

        print_gripper_info(model)

        mode = args.demo
        if mode in ("all", "basic"):
            demo_open_close(model, data, viewer)
        if mode in ("all", "fingers"):
            demo_individual_fingers(model, data, viewer)
        if mode in ("all", "cycles"):
            demo_grasp_cycles(model, data, viewer, args.cycles)

        print(f"\n✓ 演示完成! 抓取循环次数: {args.cycles}")
        print("  关闭窗口以退出...")

        # 保持窗口
        while viewer.is_running():
            sim_step(model, data, viewer, {}, steps=1)

    return 0


if __name__ == "__main__":
    sys.exit(main())