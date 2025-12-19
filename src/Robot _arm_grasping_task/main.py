import mujoco
import mujoco_viewer
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

# ===================== 核心配置（绝对路径，零出错）=====================
MODEL_PATH = "D:/nn/src/Robot _arm_grasping_task/robot.xml"
TARGET_OBJECT_POS = np.array([0.4, 0.0, 0.1])  # 目标物体位置
GOAL_POS = np.array([-0.4, 0.0, 0.1])  # 放置目标位置
FORCE_THRESHOLD = 5.0  # 抓取力阈值（N）
POS_ERROR_THRESHOLD = 0.01  # 位置误差阈值（m）
SIMULATION_STEPS = 3000  # 总仿真步数
# PID控制参数
KP = 80.0
KI = 0.1
KD = 5.0


# ===================== 工具函数 =====================
def compute_jacobian(model, data, ee_site_id):
    """计算末端执行器雅可比矩阵（适配MuJoCo 2.3+）"""
    jacp = np.zeros((3, model.nv))  # 位置雅可比
    jacr = np.zeros((3, model.nv))  # 旋转雅可比
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    # 只取前3个关节（适配简化版机械臂）的雅可比
    jacobian = np.vstack([jacp[:, :3], jacr[:, :3]])
    return jacobian


def ik_newton_raphson(model, data, target_pos, initial_qpos, max_iter=100, tol=1e-4):
    """牛顿-拉夫逊法求解逆运动学（适配3关节机械臂+MuJoCo 2.3+）"""
    q = np.copy(initial_qpos[:3])
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    for _ in range(max_iter):
        # 设置关节位置并更新动力学
        data.qpos[:3] = q
        mujoco.mj_forward(model, data)

        # 获取当前末端位置（site_xpos 是2.3+保留属性）
        current_pos = data.site_xpos[ee_site_id].copy()
        # 计算位置误差
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            break

        # 计算雅可比矩阵
        jacobian = compute_jacobian(model, data, ee_site_id)[:3, :3]
        # 牛顿-拉夫逊更新
        delta_q = np.linalg.pinv(jacobian) @ error
        q += delta_q

        # 限制关节角度在范围内
        for i in range(3):
            q[i] = np.clip(q[i], model.jnt_range[i][0], model.jnt_range[i][1])

    return q


def pid_controller(error, error_integral, error_prev):
    """PID控制器"""
    proportional = KP * error
    integral = KI * error_integral
    derivative = KD * (error - error_prev)
    return proportional + integral + derivative, error_integral + error, error_prev


# ===================== 主仿真函数（修复所有废弃属性）=====================
def grasp_simulation():
    # 1. 加载模型和数据（MuJoCo 2.3+ 标准写法）
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    # 初始化可视化器（兼容0.1.4版本）
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # 初始化变量（适配2.3+ ID查询）
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")

    # 记录数据
    ee_pos_history = []
    force_history = []
    object_pos_history = []
    grasp_success = False

    # PID控制积分项和前一误差
    error_integral = np.zeros(3)
    error_prev = np.zeros(3)

    # 仿真阶段：1-接近物体 2-抓取 3-搬运 4-放置
    phase = 1
    phase_step = 0

    try:
        for step in range(SIMULATION_STEPS):
            # ---------------- 阶段1：接近目标物体 ----------------
            if phase == 1:
                # 求解IK得到目标关节角度（适配3关节）
                target_joint_pos = ik_newton_raphson(model, data, TARGET_OBJECT_POS, data.qpos)

                # PID控制关节力矩（适配3关节）
                joint_error = target_joint_pos - data.qpos[:3]
                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )

                # 设置关节力矩
                data.ctrl[:3] = torque

                # 检查是否到达物体位置
                current_ee_pos = data.site_xpos[ee_site_id]
                if np.linalg.norm(current_ee_pos - TARGET_OBJECT_POS) < POS_ERROR_THRESHOLD:
                    phase = 2
                    phase_step = 0
                    print("进入抓取阶段")

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

                # 夹爪闭合（力反馈控制）
                current_force = np.linalg.norm(data.sensordata[:3])  # 读取力传感器数据
                if current_force < FORCE_THRESHOLD and phase_step < 800:
                    # 逐渐闭合夹爪（控制第4、5个执行器）
                    data.ctrl[3] = 5.0  # 左夹爪闭合
                    data.ctrl[4] = -5.0  # 右夹爪闭合
                else:
                    # 保持夹爪力度
                    data.ctrl[3] = 2.0
                    data.ctrl[4] = -2.0
                    # 检查抓取是否成功（物体随末端移动）
                    # 关键修复：data.body_xpos → data.xpos（MuJoCo 2.3+ 标准属性）
                    object_pos = data.xpos[object_body_id].copy()
                    pos_diff = np.linalg.norm(object_pos - current_ee_pos)
                    if pos_diff < 0.02 and phase_step > 400:
                        phase = 3
                        phase_step = 0
                        print("抓取成功，进入搬运阶段")

                phase_step += 1

            # ---------------- 阶段3：搬运到目标位置 ----------------
            elif phase == 3:
                # 抬升并移动到目标位置
                lift_pos = TARGET_OBJECT_POS + np.array([0, 0, 0.1])
                if phase_step < 400:
                    # 先抬升
                    target_joint_pos = ik_newton_raphson(model, data, lift_pos, data.qpos)
                else:
                    # 移动到目标位置
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
                if np.linalg.norm(current_ee_pos - GOAL_POS) < POS_ERROR_THRESHOLD and phase_step > 800:
                    phase = 4
                    phase_step = 0
                    print("到达目标位置，进入放置阶段")

                phase_step += 1

            # ---------------- 阶段4：放置物体 ----------------
            elif phase == 4:
                # 保持位置，打开夹爪
                target_joint_pos = ik_newton_raphson(model, data, GOAL_POS, data.qpos)
                joint_error = target_joint_pos - data.qpos[:3]
                torque = np.zeros(3)
                for i in range(3):
                    torque[i], error_integral[i], error_prev[i] = pid_controller(
                        joint_error[i], error_integral[i], error_prev[i]
                    )
                data.ctrl[:3] = torque

                # 打开夹爪
                data.ctrl[3] = 0.0  # 左夹爪打开
                data.ctrl[4] = 0.0  # 右夹爪打开

                phase_step += 1
                if phase_step > 400:
                    grasp_success = True
                    break

            # 2. 运行仿真步（MuJoCo 2.3+ 标准写法）
            mujoco.mj_step(model, data)

            # 3. 记录数据（修复所有废弃属性）
            ee_pos_history.append(data.site_xpos[ee_site_id].copy())
            force_history.append(np.linalg.norm(data.sensordata[:3]))
            # 核心修复：data.body_xpos → data.xpos
            object_pos_history.append(data.xpos[object_body_id].copy())

            # 4. 渲染可视化（兼容0.1.4版本）
            viewer.render()

            # 控制仿真速度
            time.sleep(0.001)

    except KeyboardInterrupt:
        # 捕获窗口关闭/键盘中断，正常退出
        print("\n⚠️ 仿真被手动终止")
    finally:
        # 确保可视化器正常关闭
        viewer.close()

    # ===================== 结果分析 =====================
    # 转换记录数据为numpy数组
    ee_pos_history = np.array(ee_pos_history)
    force_history = np.array(force_history)
    object_pos_history = np.array(object_pos_history)

    # 绘制结果图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # 1. 末端执行器轨迹
    ax1.plot(ee_pos_history[:, 0], ee_pos_history[:, 1], label='末端轨迹', color='blue', linewidth=1.5)
    ax1.scatter(TARGET_OBJECT_POS[0], TARGET_OBJECT_POS[1], c='red', label='抓取点', s=50)
    ax1.scatter(GOAL_POS[0], GOAL_POS[1], c='green', label='放置点', s=50)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('末端执行器XY平面轨迹', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 末端Z轴位置
    ax2.plot(ee_pos_history[:, 2], color='green', linewidth=1.5)
    ax2.set_xlabel('仿真步数')
    ax2.set_ylabel('Z位置 (m)')
    ax2.set_title('末端执行器Z轴位置变化', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 接触力变化
    ax3.plot(force_history, color='orange', linewidth=1.5)
    ax3.axhline(y=FORCE_THRESHOLD, color='red', linestyle='--', label='力阈值', linewidth=1)
    ax3.set_xlabel('仿真步数')
    ax3.set_ylabel('接触力 (N)')
    ax3.set_title('末端接触力变化', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. 物体位置变化
    ax4.plot(object_pos_history[:, 0], object_pos_history[:, 1], label='物体轨迹', color='red', linewidth=1.5)
    ax4.scatter(TARGET_OBJECT_POS[0], TARGET_OBJECT_POS[1], c='red', label='初始位置', s=50)
    ax4.scatter(GOAL_POS[0], GOAL_POS[1], c='green', label='目标位置', s=50)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('物体XY平面轨迹', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grasp_simulation_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 输出抓取结果
    if grasp_success:
        print("\n===================== 仿真结果 =====================")
        print("✅ 抓取任务成功完成！")
        print(
            f"📌 物体最终位置: X={object_pos_history[-1, 0]:.3f} Y={object_pos_history[-1, 1]:.3f} Z={object_pos_history[-1, 2]:.3f}")
        print(f"🎯 目标放置位置: X={GOAL_POS[0]:.3f} Y={GOAL_POS[1]:.3f} Z={GOAL_POS[2]:.3f}")
        print(f"📏 位置误差: {np.linalg.norm(object_pos_history[-1] - GOAL_POS):.3f} m")
    else:
        print("\n❌ 抓取任务未完成，请检查参数或仿真步数！")


# ===================== 运行仿真 =====================
if __name__ == "__main__":
    print("🚀 机械臂抓取仿真启动...")
    grasp_simulation()
    print("\n🔚 仿真程序结束")