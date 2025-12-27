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


# ===================== 自动任务枚举（按执行顺序） =====================
class AutoTask(Enum):
    INIT_MOVE = 1  # 初始精准移动（热身）
    SIMPLE_GRASP = 2  # 简易自动抓取
    COMPLEX_TASK = 3  # 复杂多位置任务
    CIRCLE_TASK = 4  # 画圆任务
    BACK_FORTH = 5  # 往复运动
    FINISH = 6  # 任务完成


# ===================== 核心参数（自动运行适配） =====================
# 控制参数（无转圈）
MANUAL_SPEED = 0.015
PRECISE_SPEED = 0.008
GRASP_FORCE = 3.8
AUTO_LIFT_HEIGHT = 0.10
AUTO_TRANSPORT_X = -0.12
# 逆运动学参数
IK_GAIN = 1.5
JOINT_LIMITS = np.array([
    [-1.2, 1.2],  # joint1范围
    [-1.0, 1.0],  # joint2范围
    [-0.8, 0.8]  # joint3范围
])
# 自动任务参数
CIRCLE_RADIUS = 0.08
CIRCLE_SPEED = 0.004
BACK_FORTH_DIST = 0.15
# 自动运行参数（新增）
TASK_DELAY = 2.0  # 任务间等待时间（秒）
AUTO_MOVE_POINTS = [  # 初始自动移动的目标点
    np.array([0.1, 0.0, 0.1]),
    np.array([0.1, 0.05, 0.12]),
    np.array([0.05, -0.05, 0.08]),
    np.array([0.0, 0.0, 0.1])
]

# ===================== 全局变量（自动运行核心） =====================
current_task = AutoTask.INIT_MOVE  # 当前执行的自动任务
task_step = 0  # 任务内部步数
target_ee_pos = np.array([0.0, 0.0, 0.1])  # 末端目标位置
init_move_idx = 0  # 初始移动的目标点索引
task_finished = False  # 所有任务是否完成


# ===================== 核心逆运动学控制（无转圈） =====================
def ik_control(model, data, ee_id, target_pos):
    """逆运动学控制：精准跟随目标位置，杜绝转圈"""
    # 1. 获取当前末端位置
    current_pos = np.array([0.0, 0.0, 0.1])
    if ee_id >= 0:
        try:
            current_pos = data.site_xpos[ee_id].copy()
        except:
            current_pos = data.xpos[ee_id].copy()

    # 2. 计算位置误差（限制误差范围）
    error = target_pos - current_pos
    error = np.clip(error, -0.05, 0.05)

    # 3. 计算关节雅可比矩阵
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    if ee_id >= 0:
        mujoco.mj_jac(model, data, jacp, jacr, current_pos, ee_id)

    # 4. 提取前3个关节的雅可比
    jacp_joints = jacp[:, :3]

    # 5. 计算关节速度指令（伪逆求解）
    jnt_vel = np.dot(jacp_joints.T, error * IK_GAIN)
    jnt_vel = np.clip(jnt_vel, -0.2, 0.2)

    # 6. 积分得到关节角度，并限制范围
    for i in range(min(3, model.njnt)):
        data.qpos[i] += jnt_vel[i] * model.opt.timestep
        data.qpos[i] = np.clip(data.qpos[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])

    # 7. 更新关节数据
    mujoco.mj_forward(model, data)


# ===================== 自动任务实现（按顺序执行） =====================
def auto_init_move(model, data, ee_id):
    """自动任务1：初始精准移动（热身）"""
    global task_step, init_move_idx, current_task, target_ee_pos
    # 到达当前目标点后，切换下一个目标点
    if task_step == 0:
        print(f"\n🎯 开始初始自动移动：目标点 {init_move_idx + 1}/{len(AUTO_MOVE_POINTS)}")
        target_ee_pos = AUTO_MOVE_POINTS[init_move_idx]

    # 逆运动学控制移动到目标点
    ik_control(model, data, ee_id, target_ee_pos)

    # 检查是否到达目标点（误差小于0.005）
    current_pos = np.array([0.0, 0.0, 0.1])
    if ee_id >= 0:
        try:
            current_pos = data.site_xpos[ee_id].copy()
        except:
            current_pos = data.xpos[ee_id].copy()
    error = np.linalg.norm(target_ee_pos - current_pos)

    if error < 0.005:
        task_step = 0
        init_move_idx += 1
        if init_move_idx >= len(AUTO_MOVE_POINTS):
            print("✅ 初始自动移动完成！")
            time.sleep(TASK_DELAY)  # 任务间等待
            current_task = AutoTask.SIMPLE_GRASP  # 切换到下一个任务
    else:
        task_step += 1


def auto_simple_grasp(model, data, ee_id, obj_id):
    """自动任务2：简易抓取（无需按键）"""
    global task_step, current_task, target_ee_pos
    # 获取物体位置
    obj_pos = np.array([0.2, 0.0, 0.05])
    if obj_id >= 0:
        try:
            obj_pos = data.xpos[obj_id].copy()
        except:
            pass

    # 阶段1：移动到物体上方
    if task_step < 1000:
        if task_step == 0:
            print("\n🎯 开始自动简易抓取任务...")
        target = obj_pos + [0, 0, 0.07]
        ik_control(model, data, ee_id, target)
        # 渐进闭合夹爪
        if task_step > 800 and model.nu >= 4:
            data.ctrl[3] = min(data.ctrl[3] + 0.03, GRASP_FORCE)
            data.ctrl[4] = max(data.ctrl[4] - 0.03, -GRASP_FORCE)
    # 阶段2：下降抓取
    elif task_step < 1800:
        target = obj_pos + [0, 0, 0.02]
        ik_control(model, data, ee_id, target)
    # 阶段3：抬升
    elif task_step < 2600:
        target = obj_pos + [0, 0, AUTO_LIFT_HEIGHT]
        ik_control(model, data, ee_id, target)
    # 阶段4：搬运
    elif task_step < 3600:
        target = obj_pos + [AUTO_TRANSPORT_X, 0, AUTO_LIFT_HEIGHT]
        ik_control(model, data, ee_id, target)
    # 阶段5：下放释放
    elif task_step < 4400:
        target = obj_pos + [AUTO_TRANSPORT_X, 0, 0.03]
        ik_control(model, data, ee_id, target)
        # 渐进释放
        if task_step > 4000:
            if model.nu >= 4:
                data.ctrl[3] = max(data.ctrl[3] - 0.03, 0.0)
            if model.nu >= 5:
                data.ctrl[4] = min(data.ctrl[4] + 0.03, 0.0)
    # 阶段6：归位
    elif task_step < 5400:
        target = np.array([0.0, 0.0, 0.12])
        ik_control(model, data, ee_id, target)
    # 任务完成
    else:
        print("✅ 自动简易抓取任务完成！")
        task_step = 0
        time.sleep(TASK_DELAY)
        current_task = AutoTask.COMPLEX_TASK  # 切换到复杂任务


def auto_complex_task(model, data, ee_id, obj_id):
    """自动任务3：复杂多位置抓取+放置"""
    global task_step, current_task
    # 定义安全的目标位置
    target_positions = [
        np.array([0.18, 0.0, 0.05]),
        np.array([-0.10, 0.08, 0.05]),
        np.array([-0.10, -0.08, 0.05]),
        np.array([0.18, 0.0, 0.05])
    ]
    stage = task_step // 2300  # 每个阶段2300步

    if stage < len(target_positions):
        if task_step % 2300 == 0:
            print(f"\n🎯 复杂任务阶段 {stage + 1}/{len(target_positions)}：移动到 {target_positions[stage][:2]}")
        sub_step = task_step % 2300

        # 阶段1：移动到目标上方（0-900步）
        if sub_step < 900:
            target = target_positions[stage] + [0, 0, 0.06]
            ik_control(model, data, ee_id, target)
        # 阶段2：下降（抓取/释放）（900-1600步）
        elif sub_step < 1600:
            target = target_positions[stage] + [0, 0, 0.02]
            ik_control(model, data, ee_id, target)
            # 第一阶段抓取，其他阶段释放
            if stage == 0:
                if model.nu >= 4:
                    data.ctrl[3] = min(data.ctrl[3] + 0.03, GRASP_FORCE)
                    data.ctrl[4] = max(data.ctrl[4] - 0.03, -GRASP_FORCE)
            elif stage in [1, 2]:
                if model.nu >= 4:
                    data.ctrl[3] = max(data.ctrl[3] - 0.03, 0.0)
                    data.ctrl[4] = min(data.ctrl[4] + 0.03, 0.0)
        # 阶段3：抬升（1600-2300步）
        else:
            target = target_positions[stage] + [0, 0, AUTO_LIFT_HEIGHT]
            ik_control(model, data, ee_id, target)
    else:
        # 归位（额外1000步）
        if task_step < 5600:
            target = np.array([0.0, 0.0, 0.12])
            ik_control(model, data, ee_id, target)
        else:
            print("✅ 自动复杂任务完成！")
            task_step = 0
            time.sleep(TASK_DELAY)
            current_task = AutoTask.CIRCLE_TASK  # 切换到画圆任务

    task_step += 1


def auto_circle_task(model, data, ee_id):
    """自动任务4：画圆任务"""
    global task_step, current_task
    center = np.array([0.08, 0.0, 0.10])

    if task_step < 2000:
        # 计算圆上目标点
        angle = task_step * CIRCLE_SPEED
        target_x = center[0] + CIRCLE_RADIUS * np.cos(angle)
        target_y = center[1] + CIRCLE_RADIUS * np.sin(angle)
        target_pos = np.array([target_x, target_y, center[2]])
        # 限制范围
        target_pos = np.clip(target_pos,
                             np.array([-0.1, -0.1, 0.08]),
                             np.array([0.2, 0.1, 0.15]))
        # 逆运动学控制画圆
        ik_control(model, data, ee_id, target_pos)
        # 实时反馈
        if task_step % 400 == 0:
            print(f"\n📈 自动画圆进度：{int(task_step / 2000 * 100)}%")
    else:
        print("✅ 自动画圆任务完成！")
        task_step = 0
        time.sleep(TASK_DELAY)
        current_task = AutoTask.BACK_FORTH  # 切换到往复运动

    task_step += 1


def auto_back_forth(model, data, ee_id):
    """自动任务5：往复运动"""
    global task_step, current_task, task_finished
    start_pos = np.array([0.05, 0.0, 0.10])

    if task_step < 2500:
        # 生成往复轨迹
        cycle = np.sin(task_step * 0.008)
        target_x = start_pos[0] + cycle * BACK_FORTH_DIST
        target_x = np.clip(target_x, -0.1, 0.2)
        target_pos = np.array([target_x, start_pos[1], start_pos[2]])
        # 逆运动学控制往复
        ik_control(model, data, ee_id, target_pos)
        # 实时反馈
        if task_step % 600 == 0:
            direction = "前" if cycle > 0 else "后"
            print(f"\n📌 自动往复运动：当前方向【{direction}】（X：{target_x:.2f}）")
    else:
        print("✅ 自动往复运动任务完成！")
        task_step = 0
        time.sleep(TASK_DELAY)
        current_task = AutoTask.FINISH  # 所有任务完成
        task_finished = True

    task_step += 1


# ===================== 初始化+主程序（自动运行核心） =====================
def init_model_and_viewer():
    """初始化模型和Viewer，自动运行准备"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到robot.xml: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 初始化关节到中间位置
    for i in range(min(3, model.njnt)):
        data.qpos[i] = (JOINT_LIMITS[i][0] + JOINT_LIMITS[i][1]) / 2
    mujoco.mj_forward(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    viewer.cam.distance = 1.8
    viewer.cam.elevation = 15
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

    # 打印自动运行提示
    print("=" * 60)
    print("🚀 机械臂自动运行程序启动！")
    print("🔧 自动执行流程：初始移动→简易抓取→复杂任务→画圆→往复运动")
    print("⏱ 任务间等待时间：{}秒".format(TASK_DELAY))
    print("💡 按ESC可随时退出程序")
    print("=" * 60)
    return model, data, viewer, ee_id, obj_id


def main():
    global viewer, current_task, task_step, task_finished
    model, data, viewer, ee_id, obj_id = init_model_and_viewer()

    try:
        while viewer.is_alive and not task_finished:
            # 根据当前任务执行对应逻辑（自动运行核心）
            if current_task == AutoTask.INIT_MOVE:
                auto_init_move(model, data, ee_id)
            elif current_task == AutoTask.SIMPLE_GRASP:
                auto_simple_grasp(model, data, ee_id, obj_id)
            elif current_task == AutoTask.COMPLEX_TASK:
                auto_complex_task(model, data, ee_id, obj_id)
            elif current_task == AutoTask.CIRCLE_TASK:
                auto_circle_task(model, data, ee_id)
            elif current_task == AutoTask.BACK_FORTH:
                auto_back_forth(model, data, ee_id)
            elif current_task == AutoTask.FINISH:
                print("\n🎉 所有自动任务执行完成！")
                task_finished = True

            # 仿真步进
            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.006)

        # 所有任务完成后，保持窗口5秒再退出
        if task_finished:
            print("\n⏳ 所有任务完成，5秒后自动退出...")
            for i in range(5):
                viewer.render()
                time.sleep(1)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with suppress(Exception):
            viewer.close()
        print("\n🔚 机械臂自动运行程序退出")


if __name__ == "__main__":
    # 检查依赖
    try:
        import mujoco, mujoco_viewer, glfw
    except ImportError as e:
        print(f"❌ 缺少依赖 {str(e).split()[-1]}！执行：")
        print("   pip install mujoco mujoco-viewer glfw numpy matplotlib")
        exit(1)
    # 启动自动运行
    main()