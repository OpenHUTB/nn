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
from enum import Enum

# ===================== 基础配置 =====================
warnings.filterwarnings('ignore')
mpl.use('TkAgg')
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")


# ===================== 操作模式枚举 =====================
class ControlMode(Enum):
    MANUAL = 1  # 基础手动控制
    PRECISE = 2  # 精准微调模式
    AUTO_SIMPLE = 3  # 简易自动抓取
    AUTO_COMPLEX = 4  # 复杂任务流程
    CIRCLE_TASK = 5  # 画圆任务
    BACK_FORTH = 6  # 往复运动


# ===================== 核心参数（修复转圈关键） =====================
# 控制参数（降低增益，杜绝转圈）
MANUAL_SPEED = 0.015  # 进一步降低速度，减少误差累积
PRECISE_SPEED = 0.008  # 精准模式速度
GRASP_FORCE = 3.8
AUTO_LIFT_HEIGHT = 0.10
AUTO_TRANSPORT_X = -0.12
# 逆运动学参数（核心：限制关节范围，杜绝转圈）
IK_GAIN = 1.5  # 逆运动学增益（大幅降低）
JOINT_LIMITS = np.array([
    [-1.2, 1.2],  # joint1范围
    [-1.0, 1.0],  # joint2范围
    [-0.8, 0.8]  # joint3范围
])
# 任务参数
CIRCLE_RADIUS = 0.08  # 缩小画圆半径，避免超出关节范围
CIRCLE_SPEED = 0.004
BACK_FORTH_DIST = 0.15

# ===================== 全局变量 =====================
control_cmd = {
    'forward': 0, 'backward': 0, 'left': 0, 'right': 0,
    'up': 0, 'down': 0, 'grasp': 0, 'release': 0,
    'auto_simple': False, 'auto_complex': False,
    'circle_task': False, 'back_forth': False,
    'switch_precise': False, 'reset': False
}
current_mode = ControlMode.MANUAL
task_step = 0
# 新增：目标位置缓存（避免突变）
target_ee_pos = np.array([0.0, 0.0, 0.1])  # 初始末端目标位置


# ===================== 兼容版按键检测 =====================
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
        # 多模式任务按键
        control_cmd['auto_simple'] = True if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS else False
        control_cmd['auto_complex'] = True if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS else False
        control_cmd['circle_task'] = True if glfw.get_key(window, glfw.KEY_V) == glfw.PRESS else False
        control_cmd['back_forth'] = True if glfw.get_key(window, glfw.KEY_B) == glfw.PRESS else False
        control_cmd['switch_precise'] = True if glfw.get_key(window, glfw.KEY_P) == glfw.PRESS else False
        control_cmd['reset'] = True if glfw.get_key(window, glfw.KEY_C) == glfw.PRESS else False
        # ESC退出
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        # 切换精准模式
        if control_cmd['switch_precise']:
            current_mode = ControlMode.PRECISE if current_mode != ControlMode.PRECISE else ControlMode.MANUAL
            mode_name = "精准微调" if current_mode == ControlMode.PRECISE else "基础手动"
            print(
                f"\n🔄 切换到【{mode_name}】模式（速度：{PRECISE_SPEED if current_mode == ControlMode.PRECISE else MANUAL_SPEED}）")
            control_cmd['switch_precise'] = False
    else:
        print("\n⚠️ 旧版mujoco-viewer，支持：Z(简易自动)、X(复杂任务)、C(重置)")
        control_cmd['auto_simple'] = True


# ===================== 核心修复：逆运动学控制（杜绝转圈） =====================
def ik_control(model, data, ee_id, target_pos):
    """
    逆运动学控制：让机械臂末端精准跟随目标位置，杜绝转圈
    :param model: MuJoCo模型
    :param data: MuJoCo数据
    :param ee_id: 末端ID
    :param target_pos: 末端目标位置
    """
    # 1. 获取当前末端位置
    current_pos = np.array([0.0, 0.0, 0.1])
    if ee_id >= 0:
        try:
            current_pos = data.site_xpos[ee_id].copy()
        except:
            current_pos = data.xpos[ee_id].copy()

    # 2. 计算位置误差（限制误差范围，避免过大）
    error = target_pos - current_pos
    error = np.clip(error, -0.05, 0.05)  # 单次最大误差不超过0.05

    # 3. 计算关节雅可比矩阵（核心：关联末端位置和关节角度）
    jacp = np.zeros((3, model.nv))  # 位置雅可比
    jacr = np.zeros((3, model.nv))  # 旋转雅可比
    if ee_id >= 0:
        mujoco.mj_jac(model, data, jacp, jacr, current_pos, ee_id)

    # 4. 提取前3个关节的雅可比（机械臂主关节）
    jacp_joints = jacp[:, :3]

    # 5. 计算关节速度指令（伪逆求解，避免无解转圈）
    jnt_vel = np.dot(jacp_joints.T, error * IK_GAIN)
    jnt_vel = np.clip(jnt_vel, -0.2, 0.2)  # 限制关节速度

    # 6. 积分得到关节角度，并限制关节范围（核心：杜绝转圈）
    for i in range(min(3, model.njnt)):
        # 积分更新关节角度
        data.qpos[i] += jnt_vel[i] * model.opt.timestep
        # 限制关节在安全范围（彻底杜绝转圈）
        data.qpos[i] = np.clip(data.qpos[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])

    # 7. 更新关节数据，应用限制
    mujoco.mj_forward(model, data)


def manual_control(model, data, ee_id):
    """手动控制（基于逆运动学，杜绝转圈）"""
    global target_ee_pos
    # 选择速度（基础/精准）
    speed = PRECISE_SPEED if current_mode == ControlMode.PRECISE else MANUAL_SPEED

    # 1. 更新目标位置（渐进式，避免突变）
    target_ee_pos[0] += (control_cmd['forward'] - control_cmd['backward']) * speed
    target_ee_pos[1] += (control_cmd['left'] - control_cmd['right']) * speed
    target_ee_pos[2] += (control_cmd['up'] - control_cmd['down']) * speed

    # 2. 限制目标位置在安全范围（避免超出关节可达范围）
    target_ee_pos = np.clip(target_ee_pos,
                            np.array([-0.2, -0.15, 0.05]),
                            np.array([0.3, 0.15, 0.2]))

    # 3. 逆运动学控制（核心：让末端精准跟随目标，不转圈）
    ik_control(model, data, ee_id, target_ee_pos)

    # 4. 渐进抓取/释放
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


# ===================== 自动任务（适配逆运动学，杜绝转圈） =====================
def auto_simple_grasp(model, data, ee_id, obj_id):
    """简易自动抓取（基于逆运动学）"""
    global target_ee_pos
    print("🔄 开始【简易自动抓取】任务...")
    # 重置目标位置
    target_ee_pos = np.array([0.0, 0.0, 0.1])
    # 获取物体位置
    obj_pos = np.array([0.2, 0.0, 0.05])
    if obj_id >= 0:
        try:
            obj_pos = data.xpos[obj_id].copy()
        except:
            pass

    # 阶段1：移动到物体上方（安全位置）
    step = 0
    while step < 1000 and viewer.is_alive:
        target = obj_pos + [0, 0, 0.07]  # 降低高度，避免超出范围
        ik_control(model, data, ee_id, target)
        # 渐进闭合夹爪（提前准备）
        if step > 800 and model.nu >= 4:
            data.ctrl[3] = min(data.ctrl[3] + 0.03, GRASP_FORCE)
            data.ctrl[4] = max(data.ctrl[4] - 0.03, -GRASP_FORCE)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段2：下降抓取
    step = 0
    while step < 800 and viewer.is_alive:
        target = obj_pos + [0, 0, 0.02]  # 贴近物体但不碰撞
        ik_control(model, data, ee_id, target)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段3：抬升
    step = 0
    while step < 800 and viewer.is_alive:
        target = obj_pos + [0, 0, AUTO_LIFT_HEIGHT]
        ik_control(model, data, ee_id, target)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段4：搬运
    step = 0
    while step < 1000 and viewer.is_alive:
        target = obj_pos + [AUTO_TRANSPORT_X, 0, AUTO_LIFT_HEIGHT]
        ik_control(model, data, ee_id, target)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段5：下放释放
    step = 0
    while step < 800 and viewer.is_alive:
        target = obj_pos + [AUTO_TRANSPORT_X, 0, 0.03]
        ik_control(model, data, ee_id, target)
        # 渐进释放
        if step > 400:
            if model.nu >= 4:
                data.ctrl[3] = max(data.ctrl[3] - 0.03, 0.0)
            if model.nu >= 5:
                data.ctrl[4] = min(data.ctrl[4] + 0.03, 0.0)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段6：归位
    step = 0
    while step < 1000 and viewer.is_alive:
        target = np.array([0.0, 0.0, 0.12])
        ik_control(model, data, ee_id, target)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    print("🎉 【简易自动抓取】任务完成！（无转圈）")


def auto_complex_task(model, data, ee_id, obj_id):
    """复杂任务流程（多位置，无转圈）"""
    global target_ee_pos
    print("🔄 开始【复杂任务】：多位置抓取+放置...")
    target_ee_pos = np.array([0.0, 0.0, 0.1])
    # 定义安全的目标位置（避免超出关节范围）
    target_positions = [
        np.array([0.18, 0.0, 0.05]),  # 初始物体位置
        np.array([-0.10, 0.08, 0.05]),  # 第一个放置点
        np.array([-0.10, -0.08, 0.05]),  # 第二个放置点
        np.array([0.18, 0.0, 0.05])  # 回到初始位置
    ]

    for idx, target in enumerate(target_positions):
        if not viewer.is_alive:
            break
        print(f"📌 复杂任务阶段 {idx + 1}/{len(target_positions)}：移动到 {target[:2]} 位置")

        # 阶段1：移动到目标上方
        step = 0
        while step < 900 and viewer.is_alive:
            target_above = target + [0, 0, 0.06]
            ik_control(model, data, ee_id, target_above)
            mujoco.mj_step(model, data)
            viewer.render()
            step += 1

        # 阶段2：下降（抓取/释放）
        step = 0
        while step < 700 and viewer.is_alive:
            ik_control(model, data, ee_id, target + [0, 0, 0.02])
            # 第一阶段抓取，其他阶段释放
            if idx == 0:  # 抓取
                if model.nu >= 4:
                    data.ctrl[3] = min(data.ctrl[3] + 0.03, GRASP_FORCE)
                if model.nu >= 5:
                    data.ctrl[4] = max(data.ctrl[4] - 0.03, -GRASP_FORCE)
            elif idx in [1, 2]:  # 释放
                if model.nu >= 4:
                    data.ctrl[3] = max(data.ctrl[3] - 0.03, 0.0)
                if model.nu >= 5:
                    data.ctrl[4] = min(data.ctrl[4] + 0.03, 0.0)
            mujoco.mj_step(model, data)
            viewer.render()
            step += 1

        # 阶段3：抬升
        step = 0
        while step < 700 and viewer.is_alive:
            ik_control(model, data, ee_id, target + [0, 0, AUTO_LIFT_HEIGHT])
            mujoco.mj_step(model, data)
            viewer.render()
            step += 1

    # 归位
    step = 0
    while step < 900 and viewer.is_alive:
        ik_control(model, data, ee_id, np.array([0.0, 0.0, 0.12]))
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    print("🎉 【复杂任务】全流程完成！（无转圈）")


def circle_task(model, data, ee_id):
    """画圆任务（限制范围，无转圈）"""
    global task_step
    print("🔄 开始【画圆任务】：末端画圆（无转圈）")
    center = np.array([0.08, 0.0, 0.10])  # 缩小圆心范围

    while viewer.is_alive and task_step < 2000:
        # 计算圆上的目标点（限制在关节可达范围）
        angle = task_step * CIRCLE_SPEED
        target_x = center[0] + CIRCLE_RADIUS * np.cos(angle)
        target_y = center[1] + CIRCLE_RADIUS * np.sin(angle)
        target_pos = np.array([target_x, target_y, center[2]])
        # 限制目标位置
        target_pos = np.clip(target_pos,
                             np.array([-0.1, -0.1, 0.08]),
                             np.array([0.2, 0.1, 0.15]))

        # 逆运动学控制画圆
        ik_control(model, data, ee_id, target_pos)

        # 实时反馈
        if task_step % 200 == 0:
            print(f"📈 画圆进度：{int(task_step / 2000 * 100)}%（角度：{int(angle * 180 / np.pi)}°）")

        mujoco.mj_step(model, data)
        viewer.render()
        task_step += 1

    task_step = 0
    print("🎉 【画圆任务】完成！（无转圈）")


def back_forth_task(model, data, ee_id):
    """往复运动任务（无转圈）"""
    global task_step
    print("🔄 开始【往复运动任务】：前后往复（无转圈）")
    start_pos = np.array([0.05, 0.0, 0.10])

    while viewer.is_alive and task_step < 2500:
        # 生成往复轨迹（限制范围）
        cycle = np.sin(task_step * 0.008)
        target_x = start_pos[0] + cycle * BACK_FORTH_DIST
        # 限制X轴范围，避免超出关节
        target_x = np.clip(target_x, -0.1, 0.2)
        target_pos = np.array([target_x, start_pos[1], start_pos[2]])

        # 逆运动学控制往复
        ik_control(model, data, ee_id, target_pos)

        # 实时反馈
        if task_step % 300 == 0:
            direction = "前" if cycle > 0 else "后"
            print(f"📌 往复运动：当前方向【{direction}】（位置X：{target_x:.2f}）")

        mujoco.mj_step(model, data)
        viewer.render()
        task_step += 1

    task_step = 0
    print("🎉 【往复运动任务】完成！（无转圈）")


# ===================== 初始化+主程序 =====================
def init_model_and_viewer():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到robot.xml: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 初始化关节位置（重置到中间位置，避免初始转圈）
    for i in range(min(3, model.njnt)):
        data.qpos[i] = (JOINT_LIMITS[i][0] + JOINT_LIMITS[i][1]) / 2
    mujoco.mj_forward(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    viewer.cam.distance = 1.8
    viewer.cam.elevation = 15  # 调整视角，更清楚看关节
    viewer.cam.azimuth = 60
    viewer.cam.lookat = [0.1, 0.0, 0.1]

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

    # 打印修复后的操作指南
    print("=" * 60)
    print("✅ 机械臂控制程序（修复转圈问题）初始化完成！")
    print("🔧 核心修复：逆运动学控制+关节范围限制，彻底杜绝转圈")
    print("🎮 操作指南：")
    print("   W/S/A/D/Q/E：移动（精准不转圈）   空格：抓取   R：释放   P：精准模式")
    print("   Z：简易抓取   X：复杂任务   V：画圆   B：往复运动")
    print("   C：重置   ESC：退出")
    print("=" * 60)
    return model, data, viewer, ee_id, obj_id


def main():
    global viewer, task_step, current_mode, target_ee_pos
    task_step = 0
    current_mode = ControlMode.MANUAL
    target_ee_pos = np.array([0.0, 0.0, 0.1])
    model, data, viewer, ee_id, obj_id = init_model_and_viewer()

    try:
        while viewer.is_alive:
            check_keyboard_input(viewer)

            # 重置功能（恢复初始关节位置）
            if control_cmd['reset']:
                # 重置关节到中间位置
                for i in range(min(3, model.njnt)):
                    data.qpos[i] = (JOINT_LIMITS[i][0] + JOINT_LIMITS[i][1]) / 2
                mujoco.mj_forward(model, data)
                target_ee_pos = np.array([0.0, 0.0, 0.1])
                task_step = 0
                current_mode = ControlMode.MANUAL
                print("\n🔄 模型重置完成：关节回到中间位置，彻底杜绝初始转圈")
                control_cmd['reset'] = False

            # 执行自动任务
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

            # 手动控制
            else:
                manual_control(model, data, ee_id)

            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.006)  # 稍慢帧率，更稳定

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with suppress(Exception):
            viewer.close()
        print("\n🔚 机械臂程序退出（已修复转圈问题）")


if __name__ == "__main__":
    try:
        import mujoco, mujoco_viewer, glfw
    except ImportError as e:
        print(f"❌ 缺少依赖 {str(e).split()[-1]}！执行：")
        print("   pip install mujoco mujoco-viewer glfw numpy matplotlib")
        exit(1)
    main()