import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import warnings
import time
import glfw
from contextlib import suppress
from enum import Enum  # 新增枚举，简化模式管理

# ===================== 基础配置 =====================
warnings.filterwarnings('ignore')
mpl.use('TkAgg')
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")


# ===================== 新增：操作模式枚举（易管理） =====================
class ControlMode(Enum):
    MANUAL = 1  # 基础手动控制（原有）
    PRECISE = 2  # 精准微调模式（新增）
    AUTO_SIMPLE = 3  # 简易自动抓取（原有）
    AUTO_COMPLEX = 4  # 复杂任务流程（新增）
    CIRCLE_TASK = 5  # 画圆任务（新增）
    BACK_FORTH = 6  # 往复运动（新增）


# ===================== 核心参数（保留流畅性+新增功能） =====================
# 基础控制参数
MANUAL_SPEED = 0.025
PRECISE_SPEED = 0.01  # 精准模式速度（新增）
GRASP_FORCE = 3.8
AUTO_LIFT_HEIGHT = 0.12
AUTO_TRANSPORT_X = -0.15
SMOOTH_GAIN = 3.0
SMOOTH_CLIP = 1.0
ACCEL_FACTOR = 0.05

# 新增任务参数
CIRCLE_RADIUS = 0.1  # 画圆半径
CIRCLE_SPEED = 0.005  # 画圆速度
BACK_FORTH_DIST = 0.2  # 往复运动距离

# ===================== 全局变量（丰富功能） =====================
control_cmd = {
    'forward': 0, 'backward': 0, 'left': 0, 'right': 0,
    'up': 0, 'down': 0, 'grasp': 0, 'release': 0,
    'auto_simple': False,  # Z：简易自动
    'auto_complex': False,  # X：复杂任务
    'circle_task': False,  # V：画圆任务
    'back_forth': False,  # B：往复运动
    'switch_precise': False,  # P：切换精准模式
    'reset': False
}
last_ctrl = np.zeros(10)
current_mode = ControlMode.MANUAL  # 当前控制模式
task_step = 0  # 任务步数计数器


# ===================== 兼容版按键检测（新增操作按键） =====================
def check_keyboard_input(viewer):
    global current_mode
    # 重置基础指令
    for key in control_cmd.keys():
        if key not in ['auto_simple', 'auto_complex', 'circle_task', 'back_forth', 'switch_precise', 'reset']:
            control_cmd[key] = 0

    if hasattr(viewer, 'window') and viewer.window is not None:
        window = viewer.window
        # 基础移动按键
        control_cmd['forward'] = 1 if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS else 0
        control_cmd['backward'] = 1 if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS else 0
        control_cmd['left'] = 1 if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS else 0
        control_cmd['right'] = 1 if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS else 0
        control_cmd['up'] = 1 if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS else 0
        control_cmd['down'] = 1 if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS else 0
        # 抓取/释放
        control_cmd['grasp'] = 1 if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS else 0
        control_cmd['release'] = 1 if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS else 0
        # 新增：多模式任务按键
        control_cmd['auto_simple'] = True if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS else False
        control_cmd['auto_complex'] = True if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS else False
        control_cmd['circle_task'] = True if glfw.get_key(window, glfw.KEY_V) == glfw.PRESS else False
        control_cmd['back_forth'] = True if glfw.get_key(window, glfw.KEY_B) == glfw.PRESS else False
        control_cmd['switch_precise'] = True if glfw.get_key(window, glfw.KEY_P) == glfw.PRESS else False
        control_cmd['reset'] = True if glfw.get_key(window, glfw.KEY_C) == glfw.PRESS else False
        # ESC退出
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        # 切换精准模式（新增）
        if control_cmd['switch_precise']:
            current_mode = ControlMode.PRECISE if current_mode != ControlMode.PRECISE else ControlMode.MANUAL
            mode_name = "精准微调" if current_mode == ControlMode.PRECISE else "基础手动"
            print(
                f"\n🔄 切换到【{mode_name}】模式（速度：{PRECISE_SPEED if current_mode == ControlMode.PRECISE else MANUAL_SPEED}）")
            control_cmd['switch_precise'] = False
    else:
        print("\n⚠️ 旧版mujoco-viewer，支持：Z(简易自动)、X(复杂任务)、C(重置)")
        control_cmd['auto_simple'] = True


# ===================== 核心控制函数（保留平滑+新增任务） =====================
def smooth_control(target_ctrl, last_ctrl, joint_idx):
    delta = target_ctrl - last_ctrl[joint_idx]
    smoothed = last_ctrl[joint_idx] + delta * ACCEL_FACTOR
    smoothed = np.clip(smoothed, -SMOOTH_CLIP, SMOOTH_CLIP)
    last_ctrl[joint_idx] = smoothed
    return smoothed


def manual_control(model, data, ee_id):
    """手动控制（新增精准模式）"""
    global last_ctrl, current_mode
    # 选择速度（基础/精准）
    speed = PRECISE_SPEED if current_mode == ControlMode.PRECISE else MANUAL_SPEED

    # 安全获取末端位置
    ee_pos = np.array([0.0, 0.0, 0.1])
    if ee_id >= 0:
        try:
            ee_pos = data.site_xpos[ee_id].copy()
        except:
            ee_pos = data.xpos[ee_id].copy()

    # 计算目标位置
    target_pos = ee_pos.copy()
    target_pos[0] += (control_cmd['forward'] - control_cmd['backward']) * speed
    target_pos[1] += (control_cmd['left'] - control_cmd['right']) * speed
    target_pos[2] += (control_cmd['up'] - control_cmd['down']) * speed

    # 平滑控制
    error = target_pos - ee_pos
    for i in range(min(3, model.njnt)):
        target_ctrl = error[i] * SMOOTH_GAIN
        data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)

    # 渐进抓取/释放
    if control_cmd['grasp']:
        if model.nu >= 4:
            data.ctrl[3] = min(data.ctrl[3] + 0.1, GRASP_FORCE)
        if model.nu >= 5:
            data.ctrl[4] = max(data.ctrl[4] - 0.1, -GRASP_FORCE)
    elif control_cmd['release']:
        if model.nu >= 4:
            data.ctrl[3] = max(data.ctrl[3] - 0.1, 0.0)
        if model.nu >= 5:
            data.ctrl[4] = min(data.ctrl[4] + 0.1, 0.0)


# ===================== 新增：丰富的自动任务函数 =====================
def auto_simple_grasp(model, data, ee_id, obj_id):
    """原有简易自动抓取（保留）"""
    global last_ctrl
    print("🔄 开始【简易自动抓取】任务...")
    last_ctrl = np.zeros(10)
    obj_pos = np.array([0.2, 0.0, 0.05])
    if obj_id >= 0:
        try:
            obj_pos = data.xpos[obj_id].copy()
        except:
            pass

    # 阶段1：移动到物体上方
    step = 0
    while step < 800 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
        target = obj_pos + [0, 0, 0.08]
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.8
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段2：下降抓取
    step = 0
    while step < 600 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
        target = obj_pos + [0, 0, 0.01]
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.5
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)
        if model.nu >= 4:
            data.ctrl[3] = min(data.ctrl[3] + 0.05, GRASP_FORCE)
        if model.nu >= 5:
            data.ctrl[4] = max(data.ctrl[4] - 0.05, -GRASP_FORCE)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段3-6：抬升→搬运→下放→归位（保留流畅性）
    step = 0
    while step < 500 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
        target = obj_pos + [0, 0, AUTO_LIFT_HEIGHT] if step > 100 else obj_pos + [0, 0, 0.01]
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.7
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    step = 0
    while step < 800 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
        target = obj_pos + [AUTO_TRANSPORT_X, 0, AUTO_LIFT_HEIGHT]
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.6
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    step = 0
    while step < 600 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
        target = obj_pos + [AUTO_TRANSPORT_X, 0, 0.03]
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.5
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)
        if step > 300:
            if model.nu >= 4:
                data.ctrl[3] = max(data.ctrl[3] - 0.05, 0.0)
            if model.nu >= 5:
                data.ctrl[4] = min(data.ctrl[4] + 0.05, 0.0)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    step = 0
    while step < 700 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
        target = np.array([0.0, 0.0, 0.15])
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.7
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    print("🎉 【简易自动抓取】任务完成！")


def auto_complex_task(model, data, ee_id, obj_id):
    """新增：复杂任务流程（多物体抓取+多位置放置）"""
    global last_ctrl
    print("🔄 开始【复杂任务】：抓取→搬运→放置→返回→二次抓取...")
    last_ctrl = np.zeros(10)
    # 定义多个目标位置（丰富任务）
    target_positions = [
        np.array([0.2, 0.0, 0.05]),  # 初始物体位置
        np.array([-0.15, 0.1, 0.05]),  # 第一个放置点
        np.array([-0.15, -0.1, 0.05]),  # 第二个放置点
        np.array([0.2, 0.0, 0.05])  # 回到初始位置
    ]

    for idx, target in enumerate(target_positions):
        if not viewer.is_alive:
            break
        print(f"📌 复杂任务阶段 {idx + 1}/{len(target_positions)}：移动到 {target[:2]} 位置")

        # 阶段1：移动到目标上方
        step = 0
        while step < 700 and viewer.is_alive:
            ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
            target_above = target + [0, 0, 0.08]
            error = target_above - ee_pos
            for i in range(min(3, model.njnt)):
                data.ctrl[i] = smooth_control(error[i] * SMOOTH_GAIN * 0.7, last_ctrl, i)
            mujoco.mj_step(model, data)
            viewer.render()
            step += 1

        # 阶段2：下降（仅第一阶段抓取，其他阶段放置）
        step = 0
        while step < 500 and viewer.is_alive:
            ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
            error = target - ee_pos
            for i in range(min(3, model.njnt)):
                data.ctrl[i] = smooth_control(error[i] * SMOOTH_GAIN * 0.5, last_ctrl, i)

            # 第一阶段抓取，第二/三阶段释放，第四阶段准备二次抓取
            if idx == 0:  # 抓取
                if model.nu >= 4:
                    data.ctrl[3] = min(data.ctrl[3] + 0.05, GRASP_FORCE)
                if model.nu >= 5:
                    data.ctrl[4] = max(data.ctrl[4] - 0.05, -GRASP_FORCE)
            elif idx in [1, 2]:  # 释放
                if model.nu >= 4:
                    data.ctrl[3] = max(data.ctrl[3] - 0.05, 0.0)
                if model.nu >= 5:
                    data.ctrl[4] = min(data.ctrl[4] + 0.05, 0.0)

            mujoco.mj_step(model, data)
            viewer.render()
            step += 1

        # 阶段3：抬升
        step = 0
        while step < 400 and viewer.is_alive:
            ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
            target_up = target + [0, 0, AUTO_LIFT_HEIGHT]
            error = target_up - ee_pos
            for i in range(min(3, model.njnt)):
                data.ctrl[i] = smooth_control(error[i] * SMOOTH_GAIN * 0.6, last_ctrl, i)
            mujoco.mj_step(model, data)
            viewer.render()
            step += 1

    # 归位
    step = 0
    while step < 600 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
        target = np.array([0.0, 0.0, 0.15])
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            data.ctrl[i] = smooth_control(error[i] * SMOOTH_GAIN * 0.7, last_ctrl, i)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    print("🎉 【复杂任务】全流程完成！（多位置抓取+放置）")


def circle_task(model, data, ee_id):
    """新增：画圆任务（机械臂末端画圆，丰富操作）"""
    global last_ctrl, task_step
    print("🔄 开始【画圆任务】：末端以原点为中心画圆（按ESC停止）")
    last_ctrl = np.zeros(10)
    center = np.array([0.1, 0.0, 0.1])  # 圆心位置

    while viewer.is_alive and task_step < 1500:  # 画2圈左右
        # 计算圆上的目标点（三角函数生成圆形轨迹）
        angle = task_step * CIRCLE_SPEED
        target_x = center[0] + CIRCLE_RADIUS * np.cos(angle)
        target_y = center[1] + CIRCLE_RADIUS * np.sin(angle)
        target_z = center[2]
        target_pos = np.array([target_x, target_y, target_z])

        # 安全获取末端位置
        ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
        error = target_pos - ee_pos

        # 平滑控制画圆
        for i in range(min(3, model.njnt)):
            data.ctrl[i] = smooth_control(error[i] * SMOOTH_GAIN * 0.8, last_ctrl, i)

        # 实时反馈画圆进度
        if task_step % 100 == 0:
            print(f"📈 画圆进度：{int(task_step / 1500 * 100)}%（角度：{int(angle * 180 / np.pi)}°）")

        mujoco.mj_step(model, data)
        viewer.render()
        task_step += 1

    task_step = 0
    print("🎉 【画圆任务】完成！机械臂末端画出完整圆形轨迹")


def back_forth_task(model, data, ee_id):
    """新增：往复运动任务（前后/左右往复，丰富操作）"""
    global last_ctrl, task_step
    print("🔄 开始【往复运动任务】：前后往复移动（按ESC停止）")
    last_ctrl = np.zeros(10)
    start_pos = np.array([0.0, 0.0, 0.1])  # 起始位置

    while viewer.is_alive and task_step < 2000:
        # 生成往复轨迹（正弦函数实现平滑往复）
        cycle = np.sin(task_step * 0.01)  # -1~1的周期变化
        target_x = start_pos[0] + cycle * BACK_FORTH_DIST
        target_pos = np.array([target_x, start_pos[1], start_pos[2]])

        # 平滑控制往复运动
        ee_pos = np.array([0.0, 0.0, 0.1]) if ee_id < 0 else data.site_xpos[ee_id].copy()
        error = target_pos - ee_pos
        for i in range(min(3, model.njnt)):
            data.ctrl[i] = smooth_control(error[i] * SMOOTH_GAIN * 0.7, last_ctrl, i)

        # 实时反馈往复进度
        if task_step % 200 == 0:
            direction = "前" if cycle > 0 else "后"
            print(f"📌 往复运动：当前方向【{direction}】（位置X：{target_x:.2f}）")

        mujoco.mj_step(model, data)
        viewer.render()
        task_step += 1

    task_step = 0
    print("🎉 【往复运动任务】完成！机械臂完成多次平滑往复")


# ===================== 初始化+主程序（整合所有功能） =====================
def init_model_and_viewer():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到robot.xml: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    viewer.cam.distance = 1.8
    viewer.cam.elevation = 12
    viewer.cam.azimuth = 50
    viewer.cam.lookat = [0.15, 0.0, 0.12]

    # 兼容原有模型ID
    ee_id, obj_id = -1, -1
    for name in ["ee_site", "ee", "end_effector"]:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if ee_id >= 0: break
    if ee_id < 0:
        for name in ["ee", "end_effector"]:
            ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if ee_id >= 0: break
    for name in ["target_object", "object", "ball"]:
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if obj_id >= 0: break
    if obj_id < 0:
        for name in ["object_geom", "ball_geom"]:
            obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if obj_id >= 0: break

    # 新增：打印丰富的操作指南
    print("=" * 50)
    print("✅ 多功能机械臂控制程序初始化完成！")
    print("🎮 基础操作：")
    print("   W/S/A/D/Q/E：移动   空格：抓取   R：释放   P：切换精准/基础模式")
    print("🎯 自动任务（新增）：")
    print("   Z：简易自动抓取   X：复杂多位置任务")
    print("   V：画圆任务       B：往复运动任务")
    print("🔧 其他：C-重置   ESC-退出")
    print("=" * 50)
    return model, data, viewer, ee_id, obj_id


def main():
    global viewer, last_ctrl, task_step, current_mode
    last_ctrl = np.zeros(10)
    task_step = 0
    current_mode = ControlMode.MANUAL
    model, data, viewer, ee_id, obj_id = init_model_and_viewer()

    try:
        while viewer.is_alive:
            check_keyboard_input(viewer)

            # 重置功能
            if control_cmd['reset']:
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                last_ctrl = np.zeros(10)
                task_step = 0
                current_mode = ControlMode.MANUAL
                print("\n🔄 模型完全重置：位置、缓存、任务、模式均已恢复初始状态")
                control_cmd['reset'] = False

            # 执行各类自动任务（新增）
            elif control_cmd['auto_simple']:
                auto_simple_grasp(model, data, ee_id, obj_id)
                control_cmd['auto_simple'] = False
            elif control_cmd['auto_complex']:
                auto_complex_task(model, data, ee_id, obj_id)
                control_cmd['auto_complex'] = False
            elif control_cmd['circle_task']:
                circle_task(model, data, ee_id)
                control_cmd['circle_task'] = False
            elif control_cmd['back_forth']:
                back_forth_task(model, data, ee_id)
                control_cmd['back_forth'] = False

            # 手动控制（基础/精准）
            else:
                manual_control(model, data, ee_id)

            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.005)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with suppress(Exception):
            viewer.close()
        print("\n🔚 多功能机械臂程序退出（未修改robot.xml）")


if __name__ == "__main__":
    try:
        import mujoco, mujoco_viewer, glfw
    except ImportError as e:
        print(f"❌ 缺少依赖 {str(e).split()[-1]}！执行：")
        print("   pip install mujoco mujoco-viewer glfw numpy matplotlib")
        exit(1)
    main()