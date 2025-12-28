import mujoco
import mujoco_viewer
import numpy as np
import os
import warnings
import time
from contextlib import suppress

# ===================== 极简配置（剔除冗余，确保自动运行） =====================
warnings.filterwarnings('ignore')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")

# 核心参数（极简+强制）
GRASP_FORCE = 3.8
IK_GAIN = 1.0  # 极低增益，确保稳定
JOINT_LIMITS = np.array([[-1.5, 1.5], [-1.2, 1.2], [-1.0, 1.0]])
# 自动任务参数（极简流程）
AUTO_TARGETS = [
    np.array([0.2, 0.0, 0.08]),  # 物体位置
    np.array([-0.1, 0.0, 0.08]),  # 放置位置
    np.array([0.0, 0.0, 0.1])  # 归位位置
]
STEP_PER_TARGET = 800  # 每个目标点执行步数（缩短，快速看到效果）

# ===================== 全局变量（极简自动运行） =====================
current_target_idx = 0  # 当前目标点索引
task_step = 0  # 当前目标点内步数
grasp_state = False  # 抓取状态
viewer = None  # 全局viewer，确保可访问


# ===================== 核心逆运动学控制（极简版） =====================
def simple_ik_control(model, data, ee_id, target_pos):
    """极简逆运动学：只保留核心，确保不转圈+快速响应"""
    # 获取当前末端位置
    current_pos = data.site_xpos[ee_id] if ee_id >= 0 else np.array([0.0, 0.0, 0.1])

    # 计算误差并限制
    error = target_pos - current_pos
    error = np.clip(error, -0.03, 0.03)

    # 简易关节控制（直接映射，快速生效）
    for i in range(min(3, model.njnt)):
        # 直接更新关节角度（限制范围）
        data.qpos[i] += error[i] * IK_GAIN * model.opt.timestep
        data.qpos[i] = np.clip(data.qpos[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])

    mujoco.mj_forward(model, data)


# ===================== 强制自动运行逻辑（核心） =====================
def run_auto_task(model, data, ee_id, obj_id):
    """强制自动运行：启动即执行，无复杂判断"""
    global current_target_idx, task_step, grasp_state

    # 1. 执行当前目标点的控制
    target = AUTO_TARGETS[current_target_idx]
    simple_ik_control(model, data, ee_id, target)

    # 2. 抓取/释放逻辑（极简）
    if current_target_idx == 0 and task_step > STEP_PER_TARGET * 0.7:
        # 到达物体位置，闭合夹爪
        if model.nu >= 4:
            data.ctrl[3] = min(data.ctrl[3] + 0.05, GRASP_FORCE)
            data.ctrl[4] = max(data.ctrl[4] - 0.05, -GRASP_FORCE)
        grasp_state = True
    elif current_target_idx == 1 and task_step > STEP_PER_TARGET * 0.7:
        # 到达放置位置，释放夹爪
        if model.nu >= 4:
            data.ctrl[3] = max(data.ctrl[3] - 0.05, 0.0)
            data.ctrl[4] = min(data.ctrl[4] + 0.05, 0.0)
        grasp_state = False

    # 3. 切换目标点（步数到即切换）
    task_step += 1
    if task_step >= STEP_PER_TARGET:
        print(f"✅ 完成目标点 {current_target_idx + 1}/{len(AUTO_TARGETS)}")
        task_step = 0
        current_target_idx += 1

        # 所有目标点完成，退出
        if current_target_idx >= len(AUTO_TARGETS):
            print("\n🎉 所有自动任务强制完成！")
            return False  # 任务完成，返回False
    return True  # 任务继续


# ===================== 初始化+主程序（强制自动） =====================
def init():
    """极简初始化：确保快速启动"""
    global viewer
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"请确保robot.xml在当前目录：{MODEL_PATH}")

    # 加载模型
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 初始化关节到中间位置（避免初始转圈）
    for i in range(min(3, model.njnt)):
        data.qpos[i] = (JOINT_LIMITS[i][0] + JOINT_LIMITS[i][1]) / 2
    mujoco.mj_forward(model, data)

    # 初始化Viewer（强制显示）
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    viewer.cam.distance = 1.5
    viewer.cam.elevation = 20
    viewer.cam.azimuth = 70
    viewer.cam.lookat = [0.1, 0.0, 0.1]

    # 极简ID识别（只找关键ID）
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")

    # 打印强制启动提示
    print("=" * 50)
    print("🚨 强制自动运行模式启动！")
    print("📌 无需任何按键，立刻执行抓取任务")
    print("🎯 目标点：物体位置→放置位置→归位")
    print("=" * 50)
    return model, data, ee_id, obj_id


def main():
    global viewer
    try:
        # 初始化
        model, data, ee_id, obj_id = init()

        # 强制自动运行核心循环（无任何按键依赖）
        while viewer.is_alive:
            # 执行自动任务，返回False则退出
            if not run_auto_task(model, data, ee_id, obj_id):
                break

            # 仿真步进（快速渲染）
            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.005)

        # 任务完成后，保持窗口3秒
        print("\n⏳ 任务完成，3秒后自动退出...")
        for _ in range(3):
            viewer.render()
            time.sleep(1)

    except Exception as e:
        print(f"\n❌ 错误：{e}")
    finally:
        with suppress(Exception):
            viewer.close()
        print("🔚 强制自动运行结束")


if __name__ == "__main__":
    # 强制检查依赖并启动
    try:
        import mujoco, mujoco_viewer
    except ImportError:
        print("❌ 缺少依赖！执行：pip install mujoco mujoco-viewer numpy")
        exit(1)
    main()