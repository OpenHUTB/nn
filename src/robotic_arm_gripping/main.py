import mujoco
import mujoco.viewer
import numpy as np
import time


def main():
    model = mujoco.MjModel.from_xml_path("arm_model.xml")
    data = mujoco.MjData(model)

    # 仅获取小臂关节（joint3-joint6）
    joint_names = []
    n_joints = 0
    for i in range(model.njnt):
        if model.joint(i).type != 0:  # 排除固定关节
            joint_names.append(model.joint(i).name)
            n_joints += 1
    print(f"可动小臂关节列表: {joint_names}")
    print(f"可动关节数量: {n_joints}")

    # 初始化小臂目标位置
    target_pos = np.zeros(n_joints)
    target_pos[0] = np.pi / 6  # joint3 初始旋转30度
    target_pos[1] = np.pi / 8  # joint4 初始旋转22.5度

    # 启动可视化
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat = [0, 0, 0.3]

        start_time = time.time()
        while viewer.is_running():
            current_time = time.time() - start_time

            # 仅控制小臂摆动
            target_pos[0] = np.pi / 6 + 0.3 * np.sin(current_time)  # joint3
            target_pos[1] = np.pi / 8 + 0.2 * np.cos(current_time)  # joint4
            target_pos[2] = 0.2 * np.sin(current_time * 1.2)  # joint5
            target_pos[3] = 0.1 * np.cos(current_time * 1.5)  # joint6

            # 设置执行器控制量
            for i in range(n_joints):
                data.actuator(i).ctrl = target_pos[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    print(f"MuJoCo 版本: {mujoco.__version__}")
    try:
        main()
    except Exception as e:
        print(f"运行出错: {e}")