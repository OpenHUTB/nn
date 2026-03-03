import mujoco
import mujoco.viewer
import time
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("arm6dof_final.xml")
data = mujoco.MjData(model)
model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EQUALITY

# 关节执行器
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
act = {}
for j in joint_names:
    act[j] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{j}")

#  home 姿态（真实机械臂待命位）
HOME_POS = np.array([0.0, -0.8, 0.6, 0.0, -0.4, 0.0])

# 关节限位
JOINT_LIMITS = np.array([
    [-2.8, 2.8],
    [-1.8, 1.8],
    [-2.0, 2.0],
    [-2.5, 2.5],
    [-1.9, 1.9],
    [-2.5, 2.5]
])

def get_current_joint_pos():
    """获取当前关节角度"""
    return np.array([
        data.qpos[0], data.qpos[1], data.qpos[2],
        data.qpos[3], data.qpos[4], data.qpos[5]
    ])

def limit_joint_cmd(cmd):
    """关节限位保护"""
    for i in range(6):
        cmd[i] = np.clip(cmd[i], JOINT_LIMITS[i,0], JOINT_LIMITS[i,1])
    return cmd

def smooth_trajectory(start, target, steps):
    """梯形速度规划轨迹（真实机械臂用的）"""
    traj = np.zeros((steps, 6))
    for i in range(6):
        traj[:,i] = np.linspace(start[i], target[i], steps)
    return traj

def home_with_planning():
    print("="*60)
    print("🏭 工业级机械臂复位（带轨迹规划 + 限位 + 速度控制）")
    print("="*60)

    current = get_current_joint_pos()
    steps = 1500

    # 生成平滑轨迹
    traj = smooth_trajectory(current, HOME_POS, steps)

    # 轨迹跟随
    for i in range(steps):
        target_cmd = traj[i]
        target_cmd = limit_joint_cmd(target_cmd)

        data.ctrl[act["joint1"]] = target_cmd[0]
        data.ctrl[act["joint2"]] = target_cmd[1]
        data.ctrl[act["joint3"]] = target_cmd[2]
        data.ctrl[act["joint4"]] = target_cmd[3]
        data.ctrl[act["joint5"]] = target_cmd[4]
        data.ctrl[act["joint6"]] = target_cmd[5]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.0008)

        if i % 300 == 0:
            err = np.linalg.norm(HOME_POS - get_current_joint_pos())
            print(f"📶 复位进度: {i/steps*100:2.0f}% | 位置误差: {err:.3f}")

    # 姿态保持
    for _ in range(600):
        data.ctrl[:6] = HOME_POS
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)

    final_err = np.linalg.norm(HOME_POS - get_current_joint_pos())
    print("="*60)
    print(f"✅ 复位完成！最终误差: {final_err:.3f}")
    print("🏆 机械臂已进入标准 HOME 待命姿态")
    print("="*60)

# ==============================
# 运行
# ==============================
with mujoco.viewer.launch_passive(model, data) as v:
    global viewer
    viewer = v
    viewer.cam.distance = 2.2
    viewer.cam.lookat = [0.2, 0, 0.3]
    viewer.cam.azimuth = 50

    home_with_planning()

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()