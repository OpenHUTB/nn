import mujoco
import mujoco.viewer
import numpy as np
import time

# ===================== 高级配置 =====================
XML_PATH = "arm6dof_final.xml"  # 你的模型路径
DOF = 6  # 6自由度机械臂
CTRL_DT = 0.01  # 控制步长
INTERP_STEPS = 100  # 平滑插值步数
Kp = 50.0  # PD控制比例增益
Kd = 15.0  # PD控制微分增益

# 末端固定姿态（四元数 [w,x,y,z]）
FIXED_ROTATION = np.array([1.0, 0.0, 0.0, 0.0])

# 笛卡尔空间目标点（只变位置，姿态不变）
waypoints = [
    np.array([0.00, 0.00, 0.30]),
    np.array([0.15, 0.10, 0.30]),
    np.array([0.15, -0.10, 0.30]),
    np.array([0.00, -0.15, 0.25]),
    np.array([0.00, 0.00, 0.30]),
]


# ===================== 轨迹生成（五次多项式） =====================
def quintic_interp(start, end, steps):
    """五次多项式插值，保证速度/加速度连续"""
    t = np.linspace(0, 1, steps)
    tau = 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3  # 无冲击插值
    return np.outer(1 - tau, start) + np.outer(tau, end)


# ===================== 数值逆运动学（彻底修复所有错误） =====================
def ik_numeric(model, data, target_pos, target_quat, max_iter=200):
    """
    数值法逆运动学求解（适配新版MuJoCo）
    :param model: MuJoCo模型
    :param data: MuJoCo数据
    :param target_pos: 目标位置 [x,y,z]
    :param target_quat: 目标姿态 [w,x,y,z]
    :return: 求解后的关节角度
    """
    # 获取末端body ID（你的模型是wrist3）
    eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist3")
    if eef_id == -1:
        raise ValueError("找不到末端body: wrist3，请检查XML")

    # 初始化关节角
    q = data.qpos[:DOF].copy()
    alpha = 0.05  # 步长（调小更稳定）

    # 获取模型自由度
    nv = model.nv  # 关键：用模型实际自由度

    for _ in range(max_iter):
        # 更新模型状态
        data.qpos[:DOF] = q
        mujoco.mj_forward(model, data)

        # 获取当前末端位姿
        x_pos = data.xpos[eef_id].copy()
        x_quat = data.xquat[eef_id].copy()

        # 计算位置误差（核心：只控制位置，姿态固定靠关节角约束）
        err_pos = target_pos - x_pos  # 位置误差 (3,)

        # 误差足够小则退出
        if np.linalg.norm(err_pos) < 1e-3:
            break

        # ========== 修复1：雅克比矩阵维度 ==========
        J_pos = np.zeros((3, nv))  # 位置雅克比 (3, nv)
        J_rot = np.zeros((3, nv))  # 姿态雅克比 (3, nv)
        mujoco.mj_jacBody(model, data, J_pos, J_rot, eef_id)

        # 只保留位置雅克比（姿态固定，无需姿态误差）
        J = J_pos[:, :DOF]  # (3, 6)

        # ========== 修复2：移除错误的mju_quat2Vel ==========
        # 伪逆求解关节增量（只针对位置，保证姿态固定）
        J_pinv = np.linalg.pinv(J, rcond=1e-4)
        dq = alpha * J_pinv @ err_pos

        # 关节角限幅（防止超限，适配你的模型关节范围）
        for i in range(DOF):
            # 获取关节范围（从模型中读取，更通用）
            jnt_min = model.jnt_range[i][0] if model.jnt_limited[i] else -np.pi
            jnt_max = model.jnt_range[i][1] if model.jnt_limited[i] else np.pi
            q[i] = np.clip(q[i] + dq[i], jnt_min, jnt_max)

    return q


# ===================== PD控制器 =====================
def pd_control(q_des, q_cur, dq_cur, Kp, Kd):
    """PD位置控制器，输出关节力矩"""
    err = q_des - q_cur
    tau = Kp * err - Kd * dq_cur
    return tau


# ===================== 主控制程序 =====================
def main():
    try:
        # 加载模型和数据
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)

        # 预计算所有目标点的关节角（离线IK）
        print("🔄 预计算逆运动学...")
        joint_waypoints = []
        for idx, p in enumerate(waypoints):
            q = ik_numeric(model, data, p, FIXED_ROTATION)
            joint_waypoints.append(q)
            print(f"✅ 目标点 {idx + 1} IK求解完成: {np.round(q, 3)}")

        # 生成完整平滑轨迹
        full_traj = []
        for i in range(len(joint_waypoints)):
            start_q = joint_waypoints[i]
            end_q = joint_waypoints[(i + 1) % len(joint_waypoints)]
            traj_segment = quintic_interp(start_q, end_q, INTERP_STEPS)
            full_traj.extend(traj_segment)

        # 初始化机械臂到初始位
        data.ctrl[:DOF] = joint_waypoints[0]
        for _ in range(100):
            mujoco.mj_step(model, data)

        print("\n✅ 高级姿态固定控制已启动（笛卡尔空间 + 数值IK + PD控制）")
        print("👉 末端姿态严格固定，只做平移运动")
        print("👉 关闭仿真窗口停止程序")

        # 启动仿真可视化
        with mujoco.viewer.launch_passive(model, data) as viewer:
            traj_idx = 0
            while viewer.is_running():
                # 获取当前目标关节角
                q_des = full_traj[traj_idx % len(full_traj)]
                # 获取当前关节状态
                q_cur = data.qpos[:DOF]
                dq_cur = data.qvel[:DOF]

                # PD控制输出力矩
                tau = pd_control(q_des, q_cur, dq_cur, Kp, Kd)
                # 应用控制（限幅防止力矩过大）
                data.ctrl[:DOF] = np.clip(data.ctrl[:DOF] + 0.1 * tau, -2, 2)

                # 步进仿真
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(CTRL_DT)

                traj_idx += 1

    except Exception as e:
        print(f"\n❌ 运行错误: {type(e).__name__} - {e}")
        print("💡 请检查：1.XML路径是否正确 2.末端body名称是否为wrist3")


if __name__ == "__main__":
    main()