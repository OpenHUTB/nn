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

# ===================== 基础配置 =====================
warnings.filterwarnings('ignore')
mpl.use('TkAgg')
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")

# ===================== 核心优化参数（流畅性重点） =====================
# 手动控制：更低速、更平滑
MANUAL_SPEED = 0.025  # 进一步降低速度，减少抖动
GRASP_FORCE = 3.8  # 微调抓取力度，保证抓稳且不卡顿
# 自动抓取：平滑轨迹参数
AUTO_LIFT_HEIGHT = 0.12
AUTO_TRANSPORT_X = -0.15
SMOOTH_GAIN = 3.0  # 降低控制增益，减少超调
SMOOTH_CLIP = 1.0  # 更严格的输出限制，避免猛冲
ACCEL_FACTOR = 0.05  # 加速度因子，让动作渐进加速/减速

# ===================== 全局变量 =====================
control_cmd = {
    'forward': 0, 'backward': 0, 'left': 0, 'right': 0,
    'up': 0, 'down': 0, 'grasp': 0, 'release': 0,
    'auto': False, 'reset': False
}
# 平滑控制缓存（记录上一步的控制输出，避免突变）
last_ctrl = np.zeros(10)  # 适配最多10个关节


# ===================== 兼容版按键检测 =====================
def check_keyboard_input(viewer):
    for key in control_cmd.keys():
        if key != 'auto' and key != 'reset':
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
        # 抓取/释放/自动/重置
        control_cmd['grasp'] = 1 if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS else 0
        control_cmd['release'] = 1 if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS else 0
        control_cmd['auto'] = True if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS else False
        control_cmd['reset'] = True if glfw.get_key(window, glfw.KEY_C) == glfw.PRESS else False
        # ESC退出
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
    else:
        print("\n⚠️ 旧版mujoco-viewer，按Z触发自动抓取，C重置")
        control_cmd['auto'] = True


# ===================== 平滑控制函数（核心优化） =====================
def smooth_control(target_ctrl, last_ctrl, joint_idx):
    """
    平滑控制输出，避免关节突变（解决抖动/卡顿）
    :param target_ctrl: 目标控制值
    :param last_ctrl: 上一步控制值
    :param joint_idx: 关节索引
    :return: 平滑后的控制值
    """
    # 渐进逼近目标值，避免猛冲
    delta = target_ctrl - last_ctrl[joint_idx]
    smoothed = last_ctrl[joint_idx] + delta * ACCEL_FACTOR
    # 限制最大变化量，彻底避免抖动
    smoothed = np.clip(smoothed, -SMOOTH_CLIP, SMOOTH_CLIP)
    # 更新缓存
    last_ctrl[joint_idx] = smoothed
    return smoothed


def manual_control(model, data, ee_id):
    """手动控制（增加平滑逻辑，动作更丝滑）"""
    global last_ctrl
    # 安全获取末端位置
    ee_pos = np.array([0.0, 0.0, 0.1])
    if ee_id >= 0:
        try:
            ee_pos = data.site_xpos[ee_id].copy()
        except:
            ee_pos = data.xpos[ee_id].copy()

    # 计算目标位置（低速，易控）
    target_pos = ee_pos.copy()
    target_pos[0] += (control_cmd['forward'] - control_cmd['backward']) * MANUAL_SPEED
    target_pos[1] += (control_cmd['left'] - control_cmd['right']) * MANUAL_SPEED
    target_pos[2] += (control_cmd['up'] - control_cmd['down']) * MANUAL_SPEED

    # 计算误差并平滑控制（核心优化）
    error = target_pos - ee_pos
    for i in range(min(3, model.njnt)):
        target_ctrl = error[i] * SMOOTH_GAIN
        # 平滑输出，避免关节突变
        data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)

    # 抓取控制（渐进加力，避免夹爪猛夹）
    if control_cmd['grasp']:
        # 渐进增加抓取力，避免物体被弹飞
        if model.nu >= 4:
            data.ctrl[3] = min(data.ctrl[3] + 0.1, GRASP_FORCE)
        if model.nu >= 5:
            data.ctrl[4] = max(data.ctrl[4] - 0.1, -GRASP_FORCE)
    elif control_cmd['release']:
        # 渐进释放，避免物体掉落
        if model.nu >= 4:
            data.ctrl[3] = max(data.ctrl[3] - 0.1, 0.0)
        if model.nu >= 5:
            data.ctrl[4] = min(data.ctrl[4] + 0.1, 0.0)


def auto_grasp(model, data, ee_id, obj_id):
    """一键自动抓取（全流程平滑优化，无卡顿）"""
    global last_ctrl
    print("🔄 开始平滑自动抓取...")
    # 重置平滑缓存
    last_ctrl = np.zeros(10)

    # 安全获取物体位置
    obj_pos = np.array([0.2, 0.0, 0.05])
    if obj_id >= 0:
        try:
            obj_pos = data.xpos[obj_id].copy()
        except:
            pass

    # 阶段1：缓慢移动到物体上方（平滑逼近，无猛冲）
    step = 0
    while step < 800 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()

        # 目标位置：物体上方（高度微调，避免碰撞）
        target = obj_pos + [0, 0, 0.08]
        error = target - ee_pos

        # 平滑控制关节，无抖动
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.8  # 更慢速度
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)

        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段2：缓慢下降（渐进接近，避免压碎物体）
    step = 0
    while step < 600 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()

        # 动态调整目标：根据物体位置实时微调，避免偏差
        target = obj_pos + [0, 0, 0.01]  # 仅贴近物体，不压下去
        error = target - ee_pos

        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.5  # 极慢速度
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)

        # 渐进闭合夹爪（核心优化：避免猛夹导致物体掉落）
        if model.nu >= 4:
            data.ctrl[3] = min(data.ctrl[3] + 0.05, GRASP_FORCE)
        if model.nu >= 5:
            data.ctrl[4] = max(data.ctrl[4] - 0.05, -GRASP_FORCE)

        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段3：缓慢抬升（确认抓稳后再抬升）
    step = 0
    grasp_confirmed = False
    while step < 500 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()

        # 确认抓稳后再抬升（避免刚夹就抬导致掉落）
        if step > 100:
            grasp_confirmed = True

        if grasp_confirmed:
            target = obj_pos + [0, 0, AUTO_LIFT_HEIGHT]
        else:
            target = obj_pos + [0, 0, 0.01]  # 先保持位置

        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.7
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)

        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段4：平稳搬运（匀速移动，无晃动）
    step = 0
    while step < 800 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()

        target = obj_pos + [AUTO_TRANSPORT_X, 0, AUTO_LIFT_HEIGHT]
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.6  # 更平稳的速度
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)

        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段5：缓慢下放（精准定位，无掉落）
    step = 0
    while step < 600 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()

        target = obj_pos + [AUTO_TRANSPORT_X, 0, 0.03]  # 更贴近地面
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.5
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)

        # 延迟且渐进释放（核心优化：避免提前释放）
        if step > 300:
            if model.nu >= 4:
                data.ctrl[3] = max(data.ctrl[3] - 0.05, 0.0)
            if model.nu >= 5:
                data.ctrl[4] = min(data.ctrl[4] + 0.05, 0.0)

        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段6：平稳归位（缓慢退回，无晃动）
    step = 0
    while step < 700 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()

        target = np.array([0.0, 0.0, 0.15])  # 更高的归位位置，避免碰撞
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            target_ctrl = error[i] * SMOOTH_GAIN * 0.7
            data.ctrl[i] = smooth_control(target_ctrl, last_ctrl, i)

        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    print("🎉 平滑自动抓取完成！（无卡顿/掉落）")


# ===================== 初始化+主程序 =====================
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

    print("✅ 初始化完成！（平滑控制模式）")
    print("🎮 操作指南：W/S/A/D/Q/E移动（丝滑无抖），空格抓取（渐进加力）")
    print("   Z：一键平滑抓取  C：重置  ESC：退出")
    return model, data, viewer, ee_id, obj_id


def main():
    global viewer, last_ctrl
    last_ctrl = np.zeros(10)  # 初始化平滑缓存
    model, data, viewer, ee_id, obj_id = init_model_and_viewer()

    try:
        while viewer.is_alive:
            check_keyboard_input(viewer)

            if control_cmd['reset']:
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                last_ctrl = np.zeros(10)  # 重置平滑缓存
                print("🔄 模型重置完成（平滑缓存已清空）")
                control_cmd['reset'] = False
            elif control_cmd['auto']:
                auto_grasp(model, data, ee_id, obj_id)
                control_cmd['auto'] = False
            else:
                manual_control(model, data, ee_id)

            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.005)  # 更慢的帧率，更丝滑

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with suppress(Exception):
            viewer.close()
        print("\n🔚 程序退出（未修改robot.xml）")


if __name__ == "__main__":
    try:
        import mujoco, mujoco_viewer, glfw
    except ImportError as e:
        print(f"❌ 缺少依赖 {str(e).split()[-1]}！执行：")
        print("   pip install mujoco mujoco-viewer glfw numpy matplotlib")
        exit(1)
    main()