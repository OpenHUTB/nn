import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import warnings
import time
import glfw  # 直接用glfw检测按键，兼容所有版本
from contextlib import suppress

# ===================== 基础配置（消除警告） =====================
warnings.filterwarnings('ignore')
mpl.use('TkAgg')
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 路径配置（适配你的原有robot.xml路径）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")

# ===================== 核心控制参数（微调适配原有模型） =====================
# 手动控制参数（适配原有模型的关节范围，低速易控）
MANUAL_SPEED = 0.03  # 比之前略小，适配原有模型的关节灵敏度
GRASP_FORCE = 3.5  # 微调力度，适配原有夹爪尺寸
# 自动控制参数（适配原有模型的物体位置）
AUTO_LIFT_HEIGHT = 0.12  # 适配原有模型的抬升范围
AUTO_TRANSPORT_X = -0.15  # 适配原有模型的搬运范围

# ===================== 全局控制变量 =====================
control_cmd = {
    'forward': 0,  # 前（W）
    'backward': 0,  # 后（S）
    'left': 0,  # 左（A）
    'right': 0,  # 右（D）
    'up': 0,  # 上（Q）
    'down': 0,  # 下（E）
    'grasp': 0,  # 抓取（空格）
    'release': 0,  # 释放（R）
    'auto': False,  # 一键自动抓取（Z）
    'reset': False  # 重置（C）
}


# ===================== 兼容版按键检测函数（核心修复） =====================
def check_keyboard_input(viewer):
    """
    兼容所有版本mujoco-viewer的按键检测
    替代原有get_key()方法，解决属性不存在问题
    """
    # 重置所有指令（避免按键粘连）
    for key in control_cmd.keys():
        if key != 'auto' and key != 'reset':
            control_cmd[key] = 0

    # 方式1：适配新版mujoco-viewer（有window属性）
    if hasattr(viewer, 'window') and viewer.window is not None:
        window = viewer.window
        # W键 - 前
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            control_cmd['forward'] = 1
        # S键 - 后
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            control_cmd['backward'] = 1
        # A键 - 左
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            control_cmd['left'] = 1
        # D键 - 右
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            control_cmd['right'] = 1
        # Q键 - 上
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            control_cmd['up'] = 1
        # E键 - 下
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            control_cmd['down'] = 1
        # 空格键 - 抓取
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            control_cmd['grasp'] = 1
        # R键 - 释放
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            control_cmd['release'] = 1
        # Z键 - 一键自动抓取
        if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS:
            control_cmd['auto'] = True
        # C键 - 重置
        if glfw.get_key(window, glfw.KEY_C) == glfw.PRESS:
            control_cmd['reset'] = True
        # ESC键 - 关闭窗口
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    # 方式2：适配旧版mujoco-viewer（无window属性，备用方案）
    else:
        # 旧版无法实时检测按键，提供替代操作方式
        print("\n⚠️ 检测到旧版mujoco-viewer，按键控制受限！")
        print("   替代操作：按Z键（一键自动抓取）或C键（重置）继续")
        # 仅保留核心功能（自动抓取/重置）
        # 按任意键触发自动抓取（简化适配）
        control_cmd['auto'] = True


# ===================== 核心控制函数（仅微调适配原有模型） =====================
def init_model_and_viewer():
    """初始化模型（完全适配原有robot.xml，不修改模型）"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到原有robot.xml文件: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 初始化Viewer（微调视角，适配原有模型的显示）
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    viewer.cam.distance = 1.8  # 微调视角距离，看清原有模型
    viewer.cam.elevation = 12  # 微调仰角，适配原有模型的高度
    viewer.cam.azimuth = 50  # 微调方位角，看清物体位置
    viewer.cam.lookat = [0.15, 0.0, 0.12]  # 适配原有模型的物体位置

    # 兼容原有模型的ID命名（不修改模型，仅适配识别）
    ee_id = -1
    obj_id = -1
    # 尝试所有可能的末端命名（适配原有模型）
    for name in ["ee_site", "ee", "end_effector"]:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if ee_id >= 0:
            break
    if ee_id < 0:
        for name in ["ee", "end_effector"]:
            ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if ee_id >= 0:
                break
    # 尝试所有可能的物体命名（适配原有模型）
    for name in ["target_object", "object", "ball"]:
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if obj_id >= 0:
            break
    if obj_id < 0:
        for name in ["object_geom", "ball_geom"]:
            obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if obj_id >= 0:
                break

    print("✅ 适配原有robot.xml完成！")
    print("🎮 操作指南（适配原有模型）：")
    print("   W/S：前后移动   A/D：左右移动   Q/E：上下移动（低速易控）")
    print("   空格：抓取      R：释放        Z：一键自动抓取（适配原有模型）")
    print("   C：重置        ESC：退出")

    return model, data, viewer, ee_id, obj_id


def manual_control(model, data, ee_id):
    """手动控制（仅微调参数，适配原有模型的关节响应）"""
    # 安全获取末端位置（适配原有模型）
    ee_pos = np.array([0.0, 0.0, 0.1])
    if ee_id >= 0:
        try:
            ee_pos = data.site_xpos[ee_id].copy()
        except:
            ee_pos = data.xpos[ee_id].copy()

    # 计算目标位置（微调速度，适配原有模型）
    target_pos = ee_pos.copy()
    target_pos[0] += control_cmd['forward'] * MANUAL_SPEED
    target_pos[0] -= control_cmd['backward'] * MANUAL_SPEED
    target_pos[1] += control_cmd['left'] * MANUAL_SPEED
    target_pos[1] -= control_cmd['right'] * MANUAL_SPEED
    target_pos[2] += control_cmd['up'] * MANUAL_SPEED
    target_pos[2] -= control_cmd['down'] * MANUAL_SPEED

    # 微调控制增益（适配原有模型的关节传动比，避免转圈）
    error = target_pos - ee_pos
    gain = 4.0  # 微调增益，适配原有模型的关节灵敏度
    for i in range(min(3, model.njnt)):
        # 更严格的输出限制，彻底避免转圈
        data.ctrl[i] = np.clip(error[i] * gain, -1.8, 1.8)

    # 抓取控制（微调力度，适配原有夹爪）
    if control_cmd['grasp']:
        # 适配原有模型的夹爪控制维度
        if model.nu >= 4:
            data.ctrl[3] = GRASP_FORCE
        if model.nu >= 5:
            data.ctrl[4] = -GRASP_FORCE
    elif control_cmd['release']:
        if model.nu >= 4:
            data.ctrl[3] = 0.0
        if model.nu >= 5:
            data.ctrl[4] = 0.0


def auto_grasp(model, data, ee_id, obj_id):
    """一键自动抓取（仅微调轨迹，适配原有模型的物体位置）"""
    print("🔄 开始适配原有模型的一键自动抓取...")
    # 安全获取物体位置（适配原有模型）
    obj_pos = np.array([0.2, 0.0, 0.05])  # 适配原有模型的默认物体位置
    if obj_id >= 0:
        try:
            obj_pos = data.xpos[obj_id].copy()
        except:
            pass

    # 阶段1：移动到物体上方（微调距离，适配原有模型）
    step = 0
    while step < 600 and viewer.is_alive:  # 增加窗口存活检测
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()
        target = obj_pos + [0, 0, 0.07]  # 微调高度，适配原有模型
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            data.ctrl[i] = np.clip(error[i] * 3.5, -1.2, 1.2)
        mujoco.mj_step(model, data)
        viewer.render()  # 自动抓取时也渲染，避免窗口卡死
        step += 1

    # 阶段2：下降抓取（微调力度，适配原有夹爪）
    step = 0
    while step < 400 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()
        target = obj_pos
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            data.ctrl[i] = np.clip(error[i] * 2.8, -1.0, 1.0)
        # 适配原有模型的夹爪控制
        if model.nu >= 4:
            data.ctrl[3] = GRASP_FORCE
        if model.nu >= 5:
            data.ctrl[4] = -GRASP_FORCE
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段3：抬升（微调高度，适配原有模型）
    step = 0
    while step < 450 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()
        target = obj_pos + [0, 0, AUTO_LIFT_HEIGHT]
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            data.ctrl[i] = np.clip(error[i] * 3.2, -1.1, 1.1)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段4：搬运（微调距离，适配原有模型）
    step = 0
    while step < 700 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()
        target = obj_pos + [AUTO_TRANSPORT_X, 0, AUTO_LIFT_HEIGHT]
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            data.ctrl[i] = np.clip(error[i] * 3.5, -1.2, 1.2)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段5：下放释放（适配原有模型）
    step = 0
    while step < 450 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()
        target = obj_pos + [AUTO_TRANSPORT_X, 0, 0.04]  # 微调下放高度
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            data.ctrl[i] = np.clip(error[i] * 2.8, -1.0, 1.0)
        # 延迟释放，适配原有模型
        if step > 250:
            if model.nu >= 4:
                data.ctrl[3] = 0.0
            if model.nu >= 5:
                data.ctrl[4] = 0.0
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    # 阶段6：归位（适配原有模型的初始位置）
    step = 0
    while step < 600 and viewer.is_alive:
        ee_pos = np.array([0.0, 0.0, 0.1])
        if ee_id >= 0:
            try:
                ee_pos = data.site_xpos[ee_id].copy()
            except:
                ee_pos = data.xpos[ee_id].copy()
        target = np.array([0.0, 0.0, 0.12])  # 微调归位位置
        error = target - ee_pos
        for i in range(min(3, model.njnt)):
            data.ctrl[i] = np.clip(error[i] * 3.5, -1.2, 1.2)
        mujoco.mj_step(model, data)
        viewer.render()
        step += 1

    print("🎉 适配原有模型的自动抓取完成！")


# ===================== 主程序（修复后版本） =====================
def main():
    global viewer  # 声明全局变量，让auto_grasp能访问
    model, data, viewer, ee_id, obj_id = init_model_and_viewer()

    try:
        while viewer.is_alive:
            # 核心修复：用兼容版按键检测替代get_key()
            check_keyboard_input(viewer)

            # 执行控制（适配原有模型）
            if control_cmd['reset']:
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                print("🔄 原有模型已重置到初始状态！")
                control_cmd['reset'] = False
            elif control_cmd['auto']:
                auto_grasp(model, data, ee_id, obj_id)
                control_cmd['auto'] = False
            else:
                manual_control(model, data, ee_id)

            # 仿真步进（微调延迟，适配原有模型的帧率）
            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.004)  # 微调延迟，适配原有模型的流畅度

    except Exception as e:
        print(f"\n❌ 运行出错（适配原有模型时）: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误栈，方便排查
    finally:
        with suppress(Exception):
            viewer.close()
        print("\n🔚 程序退出（未修改任何robot.xml内容）")


# ===================== 运行入口 =====================
if __name__ == "__main__":
    # 检查依赖（新增glfw检查）
    try:
        import mujoco
        import mujoco_viewer
        import glfw
    except ImportError as e:
        missing_lib = str(e).split()[-1]
        print(f"❌ 缺少依赖 {missing_lib}！执行以下命令安装：")
        print(f"   pip install mujoco mujoco-viewer glfw numpy matplotlib")
        exit(1)

    main()