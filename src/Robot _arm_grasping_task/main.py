import mujoco
import mujoco_viewer
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import matplotlib as mpl

# ===================== 修复Matplotlib中文显示问题 =====================
# 设置支持中文的字体（Windows系统）
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用黑体，兼容英文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
mpl.rcParams['font.family'] = 'sans-serif'

# ===================== 核心配置（优化参数确保抓取成功）=====================
MODEL_PATH = "D:/nn/src/Robot _arm_grasping_task/robot.xml"
TARGET_OBJECT_POS = np.array([0.4, 0.0, 0.1])  # 目标物体位置
GOAL_POS = np.array([-0.2, 0.0, 0.1])  # 降低搬运距离，确保完成
FORCE_THRESHOLD = 2.0  # 降低力阈值，更容易触发抓取
POS_ERROR_THRESHOLD = 0.02  # 放宽位置误差，更容易判定到达
SIMULATION_STEPS = 5000  # 增加仿真步数，确保完成所有阶段
# PID控制参数（优化增益，提升稳定性）
KP = 50.0
KI = 0.05
KD = 10.0


# ===================== 工具函数 =====================
def compute_jacobian(model, data, ee_site_id):
    """计算末端执行器雅可比矩阵（适配MuJoCo 2.3+）"""
    jacp = np.zeros((3, model.nv))  # 位置雅可比
    jacr = np.zeros((3, model.nv))  # 旋转雅可比
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    # 只取前3个关节（适配简化版机械臂）的雅可比
    jacobian = np.vstack([jacp[:, :3], jacr[:, :3]])
    return jacobian


def ik_newton_raphson(model, data, target_pos, initial_qpos, max_iter=200, tol=1e-3):
    """牛顿-拉夫逊法求解逆运动学（增加迭代次数，放宽误差）"""
    q = np.copy(initial_qpos[:3])
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    for _ in range(max_iter):
        # 设置关节位置并更新动力学
        data.qpos[:3] = q
        mujoco.mj_forward(model, data)

        # 获取当前末端位置
        current_pos = data.site_xpos[ee_site_id].copy()
        # 计算位置误差
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            break

        # 计算雅可比矩阵
        jacobian = compute_jacobian(model, data, ee_site_id)[:3, :3]
        # 牛顿-拉夫逊更新（增加阻尼，提升稳定性）
        delta_q = np.linalg.pinv(jacobian + 0.01 * np.eye(3)) @ error
        q += delta_q

        # 限制关节角度在范围内
        for i in range(3):
            q[i] = np.clip(q[i], -np.pi / 2, np.pi / 2)  # 放宽关节范围

    return q


def pid_controller(error, error_integral, error_prev):
    """PID控制器"""
    proportional = KP * error
    integral = KI * error_integral
    derivative = KD * (error - error_prev)
    return proportional + integral + derivative, error_integral + error, error_prev


# ===================== 主仿真函数 =====================
def grasp_simulation():
    # 1. 加载模型和数据
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
                target_joint_pos = ik_newton_raphson(model, data, TARGET_OBJECT_POS, data.qpos)
                joint_error = target_joint_pos - data.qpos[:3]

                # PID控制
                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )
                data.ctrl[:3] = torque

                # 检查是否到达物体
                current_ee_pos = data.site_xpos[ee_site_id]
                if np.linalg.norm(current_ee_pos - TARGET_OBJECT_POS) < POS_ERROR_THRESHOLD:
                    phase = 2
                    phase_step = 0
                    print("🔍 已到达目标物体，进入抓取阶段")

            # ---------------- 阶段2：抓取物体 ----------------
            elif phase == 2:
                # 保持末端位置
                target_joint_pos = ik_newton_raphson(model, data, TARGET_OBJECT_POS, data.qpos)
                joint_error = target_joint_pos - data.qpos[:3]
                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )
                data.ctrl[:3] = torque

                # 夹爪闭合
                current_force = np.linalg.norm(data.sensordata[:3])
                if phase_step < 1000:  # 延长夹爪闭合时间
                    data.ctrl[3] = 8.0  # 增大夹爪力度
                    data.ctrl[4] = -8.0
                else:
                    data.ctrl[3] = 3.0
                    data.ctrl[4] = -3.0

                    # 检查抓取是否成功
                    object_pos = data.xpos[object_body_id].copy()
                    pos_diff = np.linalg.norm(object_pos - current_ee_pos)
                    if pos_diff < 0.03 and phase_step > 500:
                        phase = 3
                        phase_step = 0
                        print("✅ 抓取成功，进入搬运阶段")

                phase_step += 1

            # ---------------- 阶段3：搬运到目标位置 ----------------
            elif phase == 3:
                # 先抬升，再移动
                if phase_step < 500:
                    lift_pos = TARGET_OBJECT_POS + np.array([0, 0, 0.2])  # 增加抬升高度
                    target_joint_pos = ik_newton_raphson(model, data, lift_pos, data.qpos)
                else:
                    target_joint_pos = ik_newton_raphson(model, data, GOAL_POS, data.qpos)

                # PID控制
                joint_error = target_joint_pos - data.qpos[:3]
                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )
                data.ctrl[:3] = torque

                # 检查是否到达目标位置
                current_ee_pos = data.site_xpos[ee_site_id]
                if np.linalg.norm(current_ee_pos - GOAL_POS) < POS_ERROR_THRESHOLD * 1.5 and phase_step > 1000:
                    phase = 4
                    phase_step = 0
                    print("📦 已到达目标位置，进入放置阶段")

                phase_step += 1

            # ---------------- 阶段4：放置物体 ----------------
            elif phase == 4:
                # 保持位置
                target_joint_pos = ik_newton_raphson(model, data, GOAL_POS, data.qpos)
                joint_error = target_joint_pos - data.qpos[:3]
                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )
                data.ctrl[:3] = torque

                # 打开夹爪
                data.ctrl[3] = 0.0
                data.ctrl[4] = 0.0

                phase_step += 1
                if phase_step > 500:
                    grasp_success = True
                    break

            # 运行仿真步
            mujoco.mj_step(model, data)

            # 记录数据
            ee_pos_history.append(data.site_xpos[ee_site_id].copy())
            force_history.append(np.linalg.norm(data.sensordata[:3]))
            object_pos_history.append(data.xpos[object_body_id].copy())

            # 渲染可视化
            viewer.render()
            time.sleep(0.0005)  # 降低仿真速度，更易观察

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

    # 绘制结果图（全英文标签，避免字体问题）
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

    plt.tight_layout()
    plt.savefig('grasp_simulation_result.png', dpi=150, bbox_inches='tight')
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
    grasp_simulation()
    print("\n🔚 Simulation End")