import mujoco
import mujoco.viewer
import numpy as np
import time

# ===================== 配置 =====================
XML_PATH = "arm6dof_final.xml"
DOF = 6
CTRL_DT = 0.005
INTERP_STEPS = 60

# 大幅度、动作明显的关节点位
joint_waypoints = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([1.57, -1.05, 1.05, 0.79, -0.79, 0.52]),
    np.array([-1.57, -0.9, 0.8, -0.79, 0.79, -0.52]),
    np.array([0.8, -1.2, 1.0, 0.5, -0.5, 0.3]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
]


# 平滑插值
def smooth_joint_interp(q_start, q_end, steps):
    t = np.linspace(0, 1, steps)
    t = 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    return np.outer(1 - t, q_start) + np.outer(t, q_end)


# 主程序
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    data.ctrl[:DOF] = joint_waypoints[0]
    for _ in range(50):
        mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        current_idx = 0
        current_joint = joint_waypoints[current_idx]
        print("✅ 大幅度快速点到点循环运动已启动")

        while viewer.is_running():
            next_idx = (current_idx + 1) % len(joint_waypoints)
            next_joint = joint_waypoints[next_idx]

            joint_traj = smooth_joint_interp(current_joint, next_joint, INTERP_STEPS)

            for q in joint_traj:
                data.ctrl[:DOF] = q
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(CTRL_DT)

            current_joint = next_joint
            current_idx = next_idx


if __name__ == "__main__":
    main()