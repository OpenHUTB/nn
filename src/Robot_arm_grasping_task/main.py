import mujoco
import mujoco_viewer
import numpy as np
import os
import warnings
import time
from contextlib import suppress

# ===================== 配置（已根据你的模型定制） =====================
warnings.filterwarnings('ignore')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")

# --- 1. 任务清单（已使用正确的物体名称 'target_object'） ---
TASK_QUEUE = [
    # 将名为 'target_object' 的物体移动到 (-0.3, 0, 0.05)
    ["target_object", [-0.3, 0, 0.05]],
]

# --- 2. 核心控制参数 ---
IK_GAIN = 1.5
GRASP_FORCE = -8.0  # 夹爪闭合的力（负值表示向左/右）
CLEARANCE_HEIGHT = 0.25  # 移动时的安全高度
STEP_PER_MOVE = 1200  # 移动到一个新位置所需的步数
STEP_PER_GRASP = 400  # 抓取/释放动作所需的步数

# ===================== 全局状态机 =====================
viewer = None
current_task_index = 0
task_step = 0


class TaskState:
    MOVE_TO_OBJECT_ABOVE = 1
    MOVE_DOWN_TO_GRASP = 2
    GRASP_OBJECT = 3
    MOVE_UP_AFTER_GRASP = 4
    MOVE_TO_TARGET_ABOVE = 5
    MOVE_DOWN_TO_PLACE = 6
    RELEASE_OBJECT = 7
    MOVE_UP_AFTER_RELEASE = 8
    FINISHED_ALL = 9


current_state = TaskState.MOVE_TO_OBJECT_ABOVE


# ===================== 核心功能函数 =====================
def simple_ik_control(model, data, ee_id, target_pos):
    """逆运动学控制，让末端执行器移动到目标位置"""
    current_pos = data.site_xpos[ee_id]
    error = target_pos - current_pos
    error = np.clip(error, -0.05, 0.05)

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, None, current_pos, ee_id)
    jnt_vel = np.dot(jacp[:, :3].T, error * IK_GAIN)
    jnt_vel = np.clip(jnt_vel, -0.5, 0.5)

    # 注意：这里控制的是关节力矩（motor），而不是直接设置角度
    for i in range(min(3, model.nu - 2)):  # 减去夹爪的两个控制
        data.ctrl[i] = jnt_vel[i] * 100  # 乘以一个系数来放大控制信号


def run_smart_grasp_task(model, data, ee_id):
    """智能抓取任务的状态机逻辑"""
    global current_task_index, task_step, current_state

    if current_task_index >= len(TASK_QUEUE):
        if current_state != TaskState.FINISHED_ALL:
            print("\n🎉🎉🎉 所有抓取任务已成功完成！🎉🎉🎉")
            current_state = TaskState.FINISHED_ALL
        return False

    obj_name, target_place_pos = TASK_QUEUE[current_task_index]
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)

    if obj_id == -1:
        print(f"❌ 错误：未在模型中找到物体 '{obj_name}'，请检查XML文件。")
        current_task_index += 1
        return True

    # --- 状态机逻辑 ---
    if current_state == TaskState.MOVE_TO_OBJECT_ABOVE:
        if task_step == 0:
            print(f"\n[任务 {current_task_index + 1}/{len(TASK_QUEUE)}] 开始处理物体: {obj_name}")
            print("-> 状态: 移动到物体上方...")
        target_pos = data.xpos[obj_id].copy()
        target_pos[2] = CLEARANCE_HEIGHT
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.01:
            task_step = 0
            current_state = TaskState.MOVE_DOWN_TO_GRASP

    elif current_state == TaskState.MOVE_DOWN_TO_GRASP:
        if task_step == 0:
            print("-> 状态: 下降以抓取物体...")
        target_pos = data.xpos[obj_id].copy()
        target_pos[2] += 0.05  # 停在物体表面上方一点
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.005:
            task_step = 0
            current_state = TaskState.GRASP_OBJECT

    elif current_state == TaskState.GRASP_OBJECT:
        if task_step == 0:
            print("-> 状态: 正在抓取...")
        # 闭合夹爪: 左爪左移(负), 右爪右移(正)
        data.ctrl[3] = GRASP_FORCE
        data.ctrl[4] = -GRASP_FORCE
        if task_step > STEP_PER_GRASP:
            task_step = 0
            current_state = TaskState.MOVE_UP_AFTER_GRASP

    elif current_state == TaskState.MOVE_UP_AFTER_GRASP:
        if task_step == 0:
            print("-> 状态: 抓取成功，上升...")
        target_pos = data.site_xpos[ee_id].copy()
        target_pos[2] = CLEARANCE_HEIGHT
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.01:
            task_step = 0
            current_state = TaskState.MOVE_TO_TARGET_ABOVE

    elif current_state == TaskState.MOVE_TO_TARGET_ABOVE:
        if task_step == 0:
            print(f"-> 状态: 移动到目标放置区上方 {target_place_pos[:2]}...")
        target_pos = np.array(target_place_pos)
        target_pos[2] = CLEARANCE_HEIGHT
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.01:
            task_step = 0
            current_state = TaskState.MOVE_DOWN_TO_PLACE

    elif current_state == TaskState.MOVE_DOWN_TO_PLACE:
        if task_step == 0:
            print("-> 状态: 下降以放置物体...")
        target_pos = np.array(target_place_pos)
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.005:
            task_step = 0
            current_state = TaskState.RELEASE_OBJECT

    elif current_state == TaskState.RELEASE_OBJECT:
        if task_step == 0:
            print("-> 状态: 正在释放物体...")
        # 打开夹爪: 左右爪都回中
        data.ctrl[3] = 0
        data.ctrl[4] = 0
        if task_step > STEP_PER_GRASP:
            task_step = 0
            current_state = TaskState.MOVE_UP_AFTER_RELEASE

    elif current_state == TaskState.MOVE_UP_AFTER_RELEASE:
        if task_step == 0:
            print("-> 状态: 释放成功，上升并准备下一个任务...")
        target_pos = data.site_xpos[ee_id].copy()
        target_pos[2] = CLEARANCE_HEIGHT
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.01:
            current_task_index += 1
            task_step = 0
            current_state = TaskState.MOVE_TO_OBJECT_ABOVE

    task_step += 1
    return True


# ===================== 主程序 =====================
def init():
    global viewer
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"请确保 'robot.xml' 文件在当前目录: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    viewer.cam.lookat = [0.2, 0.0, 0.1]

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if ee_id == -1:
        raise ValueError("模型中必须包含一个名为 'ee_site' 的site。")

    print("=" * 60)
    print("🚀 全自动智能抓取系统启动！")
    print(f"📋 任务清单: 共 {len(TASK_QUEUE)} 个物体需要处理。")
    print("💡 正在连接到模型 'simple_arm'...")
    print("=" * 60)
    return model, data, ee_id


def main():
    global viewer
    try:
        model, data, ee_id = init()

        while viewer.is_alive:
            if not run_smart_grasp_task(model, data, ee_id):
                break

            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.005)

        print("\n⏳ 所有任务已完成，窗口将在5秒后自动关闭。")
        for _ in range(5):
            viewer.render()
            time.sleep(1)

    except Exception as e:
        print(f"\n❌ 程序发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with suppress(Exception):
            viewer.close()
        print("🔚 程序已退出。")


if __name__ == "__main__":
    try:
        import mujoco, mujoco_viewer
    except ImportError:
        print("❌ 缺少依赖！请运行: pip install mujoco mujoco-viewer numpy")
        exit(1)
    main()