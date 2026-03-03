import mujoco
import mujoco.viewer
import time
import numpy as np

# ======================
# 核心修复：正确初始化模型和数据
# ======================
# 加载机械臂模型
model = mujoco.MjModel.from_xml_path("arm6dof_final.xml")
# 关键修复：data 必须用 MjData 创建（之前错写成 MjModel）
data = mujoco.MjData(model)

# 关节执行器ID获取（简洁写法）
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
act = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{j}")
    for j in joint_names
]

viewer = None

# ======================
# 关节安全限位配置（工业级范围）
# ======================
JOINT_LIMITS = np.array([
    [-2.6, 2.6],  # joint1 基座旋转
    [-1.6, 1.6],  # joint2 肩关节
    [-1.8, 1.8],  # joint3 肘关节
    [-2.3, 2.3],  # joint4 腕关节1
    [-1.7, 1.7],  # joint5 腕关节2
    [-2.3, 2.3]  # joint6 末端旋转
])


def get_current_joint_pos():
    """获取当前关节角度（简洁封装）"""
    return data.qpos[:6].copy()


def get_safe_control_cmd(target_cmd, current_joints):
    """安全限位：超限自动回退到安全区"""
    safe_cmd = target_cmd.copy()
    for i in range(6):
        # 低于下限 → 往正方向回退
        if current_joints[i] < JOINT_LIMITS[i, 0]:
            safe_cmd[i] = 0.2
            print(f"⚠️  关节{i + 1} 低于安全下限 {JOINT_LIMITS[i, 0]:.2f}，自动回退")
        # 高于上限 → 往负方向回退
        elif current_joints[i] > JOINT_LIMITS[i, 1]:
            safe_cmd[i] = -0.2
            print(f"⚠️  关节{i + 1} 高于安全上限 {JOINT_LIMITS[i, 1]:.2f}，自动回退")
    return safe_cmd


def is_joint_over_limit(current_joints):
    """检测是否有关节超限"""
    for i in range(6):
        if current_joints[i] < JOINT_LIMITS[i, 0] or current_joints[i] > JOINT_LIMITS[i, 1]:
            return True
    return False


# ======================
# 平滑轨迹运动（带实时安全保护）
# ======================
def move_smooth_with_safety(target_pose, duration_sec=1.5):
    """
    带安全保护的平滑移动
    :param target_pose: 目标关节姿态 [j1,j2,j3,j4,j5,j6]
    :param duration_sec: 运动时长（秒），控制速度
    """
    steps = int(duration_sec * 1000)  # 按时间拆分步数，速度更可控
    start_pose = data.ctrl[act].copy()  # 起始姿态

    for i in range(steps):
        # 平滑插值（S型曲线，无冲击）
        t = i / steps
        smooth_t = t * t * (3 - 2 * t)

        # 计算目标指令
        target_cmd = start_pose + smooth_t * (target_pose - start_pose)
        # 获取当前关节角度
        current_joints = get_current_joint_pos()

        # 安全保护逻辑：超限则执行回退，否则执行目标指令
        if is_joint_over_limit(current_joints):
            data.ctrl[act] = get_safe_control_cmd(np.zeros(6), current_joints)
        else:
            data.ctrl[act] = target_cmd

        # 执行仿真步
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        time.sleep(0.001)


# ======================
# 完整演示流程（限位保护+安全回退）
# ======================
def joint_limit_safety_demo():
    print("=" * 60)
    print("🛡️  机械臂关节限位与安全保护（最终优化版）")
    print("=" * 60)

    # 定义关键姿态（HOME位=安全位，risky位=故意超限测试位）
    home_pose = np.array([0.0, -0.7, 0.5, 0.0, -0.3, 0.0])  # 安全待命位
    risky_pose = np.array([3.0, -1.8, 1.9, 0.0, 0.0, 0.0])  # 故意超限的测试位

    print("\n👉 步骤1：回到安全HOME姿态")
    move_smooth_with_safety(home_pose, duration_sec=1.2)
    time.sleep(0.5)  # 停顿，便于观察

    print("\n👉 步骤2：尝试向超限位置运动（触发安全保护）")
    move_smooth_with_safety(risky_pose, duration_sec=2.0)  # 放慢速度，保护过程更清晰
    time.sleep(0.5)

    print("\n👉 步骤3：安全退回HOME姿态")
    move_smooth_with_safety(home_pose, duration_sec=1.2)

    print("\n✅ 关节限位安全保护演示完成！")


# ======================
# 主程序运行
# ======================
if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as v:
        viewer = v
        # 相机视角优化（看得更清楚）
        viewer.cam.distance = 2.2
        viewer.cam.lookat = [0.2, 0.0, 0.3]
        viewer.cam.azimuth = 60

        # 执行演示
        joint_limit_safety_demo()

        # 保持窗口打开
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()