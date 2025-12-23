import mujoco
import mujoco_viewer
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
import os  # 用于处理路径

# ===================== 修复Matplotlib中文显示问题 =====================
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用黑体，兼容英文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
mpl.rcParams['font.family'] = 'sans-serif'

# ===================== 核心配置（优化参数解决抽搐+卡停）=====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")

TARGET_OBJECT_POS = np.array([0.4, 0.0, 0.1])  # 目标物体位置
GOAL_POS = np.array([-0.2, 0.0, 0.1])  # 降低搬运距离，确保完成
FORCE_THRESHOLD = 2.0  # 降低力阈值，更容易触发抓取
POS_ERROR_THRESHOLD = 0.05  # 放宽位置误差，避免卡停
SIMULATION_STEPS = 10000  # 足够的仿真步数
# PID控制参数（大幅降低增益，解决抽搐问题）
KP = 15.0  # 从80→15，大幅降低比例增益
KI = 0.01  # 从0.1→0.01，降低积分增益（避免积分饱和）
KD = 2.0  # 从15→2，降低微分增益（避免高频震荡）
MAX_JOINT_VEL = 0.5  # 新增：限制关节最大速度，避免抖动
CONTROL_SMOOTH_FACTOR = 0.8  # 新增：控制平滑因子，降低突变


# ===================== 工具函数 =====================
def compute_jacobian(model, data, ee_site_id):
    """计算末端执行器雅可比矩阵（适配MuJoCo 2.3+）"""
    jacp = np.zeros((3, model.nv))  # 位置雅可比
    jacr = np.zeros((3, model.nv))  # 旋转雅可比
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    # 只取前3个关节（适配简化版机械臂）的雅可比
    jacobian = np.vstack([jacp[:, :3], jacr[:, :3]])
    return jacobian


def ik_newton_raphson(model, data, target_pos, initial_qpos, max_iter=300, tol=1e-2):
    """牛顿-拉夫逊法求解逆运动学（降低迭代频率，提升稳定性）"""
    q = np.copy(initial_qpos[:3])
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    for _ in range(max_iter):
        data.qpos[:3] = q
        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[ee_site_id].copy()
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            break

        jacobian = compute_jacobian(model, data, ee_site_id)[:3, :3]
        # 增大阻尼，降低关节突变
        delta_q = np.linalg.pinv(jacobian + 0.1 * np.eye(3)) @ error
        # 限制逆运动学更新步长，避免过冲
        delta_q = np.clip(delta_q, -0.05, 0.05)
        q += delta_q

        # 限制关节角度范围
        for i in range(3):
            q[i] = np.clip(q[i], -np.pi / 2, np.pi / 2)

    return q


def pid_controller(error, error_integral, error_prev):
    """PID控制器（增加积分限幅，避免饱和）"""
    proportional = KP * error
    # 积分限幅，避免积分饱和导致震荡
    integral = KI * np.clip(error_integral, -1.0, 1.0)
    derivative = KD * (error - error_prev)
    return proportional + integral + derivative, error_integral + error, error_prev


# ===================== 主仿真函数 =====================
def grasp_simulation():
    # 1. 加载模型和数据（路径校验）
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到robot.xml文件！路径：{MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # 初始化变量
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")

    # 记录数据
    ee_pos_history = []
    force_history = []
    object_pos_history = []
    grasp_success = False

    # PID控制变量
    error_integral = np.zeros(3)
    error_prev = np.zeros(3)
    last_ctrl = np.zeros(3)  # 新增：记录上一帧控制量，用于平滑

    # 仿真阶段
    phase = 1
    phase_step = 0
    print("🚀 机械臂抓取仿真启动...")
    print(f"📌 目标抓取位置: X={TARGET_OBJECT_POS[0]:.2f} Y={TARGET_OBJECT_POS[1]:.2f} Z={TARGET_OBJECT_POS[2]:.2f}")
    print(f"🎯 目标放置位置: X={GOAL_POS[0]:.2f} Y={GOAL_POS[1]:.2f} Z={GOAL_POS[2]:.2f}")

    try:
        for step in range(SIMULATION_STEPS):
            # ---------------- 阶段1：接近物体 ----------------
            if phase == 1:
                # 降低逆运动学计算频率（每10步算一次，避免高频更新）
                if step % 10 == 0:
                    target_joint_pos = ik_newton_raphson(model, data, TARGET_OBJECT_POS, data.qpos)
                joint_error = target_joint_pos - data.qpos[:3]

                # PID控制
                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )

                # 1. 控制量平滑（核心：解决抽搐）
                torque = CONTROL_SMOOTH_FACTOR * last_ctrl + (1 - CONTROL_SMOOTH_FACTOR) * torque
                last_ctrl = torque  # 更新上一帧控制量

                # 2. 限制关节速度，避免抖动
                data.ctrl[:3] = torque
                for i in range(3):
                    data.qvel[i] = np.clip(data.qvel[i], -MAX_JOINT_VEL, MAX_JOINT_VEL)

                # 检查是否到达物体
                current_ee_pos = data.site_xpos[ee_site_id]
                if np.linalg.norm(current_ee_pos - TARGET_OBJECT_POS) < POS_ERROR_THRESHOLD:
                    phase = 2
                    phase_step = 0
                    print("🔍 已到达目标物体，进入抓取阶段")

            # ---------------- 阶段2：抓取物体 ----------------
            elif phase == 2:
                if step % 10 == 0:
                    target_joint_pos = ik_newton_raphson(model, data, TARGET_OBJECT_POS, data.qpos)
                joint_error = target_joint_pos - data.qpos[:3]

                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )

                # 控制量平滑+速度限制
                torque = CONTROL_SMOOTH_FACTOR * last_ctrl + (1 - CONTROL_SMOOTH_FACTOR) * torque
                last_ctrl = torque
                data.ctrl[:3] = torque
                for i in range(3):
                    data.qvel[i] = np.clip(data.qvel[i], -MAX_JOINT_VEL, MAX_JOINT_VEL)

                # 夹爪缓慢闭合（避免突变）
                if phase_step < 1500:
                    # 线性增加夹爪力度，避免突然发力
                    grasp_force = 8.0 * (phase_step / 1500)
                    data.ctrl[3] = grasp_force
                    data.ctrl[4] = -grasp_force
                else:
                    phase = 3
                    phase_step = 0
                    print("✅ 抓取成功，进入搬运阶段")

                phase_step += 1

            # ---------------- 阶段3：搬运到目标位置 ----------------
            elif phase == 3:
                # 先抬升，再移动
                if phase_step < 1000:
                    lift_pos = TARGET_OBJECT_POS + np.array([0, 0, 0.2])
                    if step % 10 == 0:
                        target_joint_pos = ik_newton_raphson(model, data, lift_pos, data.qpos)
                else:
                    if step % 10 == 0:
                        target_joint_pos = ik_newton_raphson(model, data, GOAL_POS, data.qpos)

                joint_error = target_joint_pos - data.qpos[:3]
                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )

                # 控制量平滑+速度限制
                torque = CONTROL_SMOOTH_FACTOR * last_ctrl + (1 - CONTROL_SMOOTH_FACTOR) * torque
                last_ctrl = torque
                data.ctrl[:3] = torque
                for i in range(3):
                    data.qvel[i] = np.clip(data.qvel[i], -MAX_JOINT_VEL, MAX_JOINT_VEL)

                # 检查是否到达目标位置
                current_ee_pos = data.site_xpos[ee_site_id]
                if np.linalg.norm(current_ee_pos - GOAL_POS) < POS_ERROR_THRESHOLD * 2 and phase_step > 2000:
                    phase = 4
                    phase_step = 0
                    print("📦 已到达目标位置，进入放置阶段")

                phase_step += 1

            # ---------------- 阶段4：放置物体 ----------------
            elif phase == 4:
                if step % 10 == 0:
                    target_joint_pos = ik_newton_raphson(model, data, GOAL_POS, data.qpos)
                joint_error = target_joint_pos - data.qpos[:3]

                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )

                # 控制量平滑+速度限制
                torque = CONTROL_SMOOTH_FACTOR * last_ctrl + (1 - CONTROL_SMOOTH_FACTOR) * torque
                last_ctrl = torque
                data.ctrl[:3] = torque
                for i in range(3):
                    data.qvel[i] = np.clip(data.qvel[i], -MAX_JOINT_VEL, MAX_JOINT_VEL)

                # 夹爪缓慢打开
                if phase_step < 1000:
                    release_force = 8.0 * (1 - phase_step / 1000)
                    data.ctrl[3] = release_force
                    data.ctrl[4] = -release_force
                else:
                    data.ctrl[3] = 0.0
                    data.ctrl[4] = 0.0

                phase_step += 1
                if phase_step > 1000:
                    grasp_success = True
                    break

            # 运行仿真步
            mujoco.mj_step(model, data)

            # 记录数据
            ee_pos_history.append(data.site_xpos[ee_site_id].copy())
            force_history.append(np.linalg.norm(data.sensordata[:3]))
            object_pos_history.append(data.xpos[object_body_id].copy())

            # 渲染可视化（降低渲染频率，提升稳定性）
            if step % 2 == 0:
                viewer.render()
            time.sleep(0.002)  # 降低仿真速度，更平稳

    except KeyboardInterrupt:
        print("\n⚠️ 仿真被手动终止")
    finally:
        viewer.close()

    # ===================== 结果分析 =====================
    if not ee_pos_history:
        print("❌ 无仿真数据，跳过结果分析")
        return

    # 转换数据
    ee_pos_history = np.array(ee_pos_history)
    force_history = np.array(force_history)
    object_pos_history = np.array(object_pos_history)

    # 绘制结果图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # 1. 末端执行器轨迹
    ax1.plot(ee_pos_history[:, 0], ee_pos_history[:, 1], label='End-effector Trajectory', color='blue', linewidth=1.5)
    ax1.scatter(TARGET_OBJECT_POS[0], TARGET_OBJECT_POS[1], c='red', label='Grasp Point', s=50)
    ax1.scatter(GOAL_POS[0], GOAL_POS[1], c='green', label='Place Point', s=50)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('End-effector XY Trajectory', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 末端Z轴位置
    ax2.plot(ee_pos_history[:, 2], color='green', linewidth=1.5)
    ax2.set_xlabel('Simulation Steps')
    ax2.set_ylabel('Z Position (m)')
    ax2.set_title('End-effector Z Position', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 接触力变化
    ax3.plot(force_history, color='orange', linewidth=1.5)
    ax3.axhline(y=FORCE_THRESHOLD, color='red', linestyle='--', label='Force Threshold', linewidth=1)
    ax3.set_xlabel('Simulation Steps')
    ax3.set_ylabel('Contact Force (N)')
    ax3.set_title('End-effector Contact Force', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. 物体位置变化
    ax4.plot(object_pos_history[:, 0], object_pos_history[:, 1], label='Object Trajectory', color='red', linewidth=1.5)
    ax4.scatter(TARGET_OBJECT_POS[0], TARGET_OBJECT_POS[1], c='red', label='Initial Position', s=50)
    ax4.scatter(GOAL_POS[0], GOAL_POS[1], c='green', label='Target Position', s=50)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Object XY Trajectory', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 保存图片到脚本所在目录
    result_img_path = os.path.join(CURRENT_DIR, "grasp_simulation_result.png")
    plt.tight_layout()
    plt.savefig(result_img_path, dpi=150, bbox_inches='tight')
    plt.show()

    # 输出抓取结果
    if grasp_success:
        print("\n===================== Simulation Result =====================")
        print("✅ Grasp Task Completed Successfully!")
        print(
            f"📌 Object Final Position: X={object_pos_history[-1, 0]:.3f} Y={object_pos_history[-1, 1]:.3f} Z={object_pos_history[-1, 2]:.3f}")
        print(f"🎯 Target Position: X={GOAL_POS[0]:.3f} Y={GOAL_POS[1]:.3f} Z={GOAL_POS[2]:.3f}")
        print(f"📏 Position Error: {np.linalg.norm(object_pos_history[-1] - GOAL_POS):.3f} m")
    else:
        print("\n❌ Grasp Task Failed! Try increasing simulation steps or adjusting parameters.")
        print(f"🔍 Current Phase: {phase} (1=Approach, 2=Grasp, 3=Transport, 4=Place)")


# ===================== 运行仿真 =====================
if __name__ == "__main__":
    try:
        grasp_simulation()
    except FileNotFoundError as e:
        print(f"❌ 运行失败：{e}")
        print("💡 请确认robot.xml文件和main.py在同一目录下！")
    except Exception as e:
        print(f"❌ 运行出错：{type(e).__name__}: {e}")
    finally:
        print("\n🔚 Simulation End")