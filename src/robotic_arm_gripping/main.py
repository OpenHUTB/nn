import mujoco
import mujoco.viewer
import numpy as np
import time


def main():
    model = mujoco.MjModel.from_xml_path("arm_model.xml")
    data = mujoco.MjData(model)

    # 获取可动关节列表（聚焦右臂抓取关节）
    joint_names = []
    n_joints = 0
    for i in range(model.njnt):
        if model.joint(i).type != 0:  # 排除固定关节
            joint_names.append(model.joint(i).name)
            n_joints += 1
    print(f"可动关节列表: {joint_names}")
    print(f"可动关节数量: {n_joints}")

    # 预获取抓取相关geom的ID（避免每次循环重复查询）
    hand_grasp_geom_id = model.geom('hand_right_grasp').id
    target_ball_geom_id = model.geom('target_ball_geom').id

    # 定义动作阶段和目标关节角度（重点控制右臂：shoulder1_right/shoulder2_right/elbow_right）
    # 阶段0：初始姿态 → 阶段1：向前伸臂抓取 → 阶段2：举起 → 阶段3：保持
    phase_targets = {
        0: [0, 0, 0, 0, 0, 0],  # 初始姿态
        1: [np.pi / 3, -np.pi / 4, -np.pi / 3, 0, 0, 0],  # 向前抓取（右臂前伸）
        2: [np.pi / 6, -np.pi / 6, np.pi / 6, 0, 0, 0],  # 抓取后举起
        3: [np.pi / 6, -np.pi / 6, np.pi / 6, 0, 0, 0]  # 保持举升状态
    }
    phase_durations = [2, 3, 2, 3]  # 每个阶段持续时间（秒）：初始2s→抓取3s→举起2s→保持3s
    phase_start_time = 0
    current_phase = 0

    # 启动可视化
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -15
        viewer.cam.distance = 1.2
        viewer.cam.lookat = [0.2, 0, 0.3]  # 视角聚焦抓取目标

        start_time = time.time()
        while viewer.is_running():
            current_time = time.time() - start_time
            elapsed_phase_time = current_time - phase_start_time

            # 切换动作阶段
            if current_phase < len(phase_durations) and elapsed_phase_time > phase_durations[current_phase]:
                phase_start_time = current_time
                current_phase = min(current_phase + 1, len(phase_durations) - 1)
                print(f"\n切换到动作阶段 {current_phase}: {['初始姿态', '向前抓取', '举起', '保持'][current_phase]}")

            # 平滑插值到目标关节角度（避免动作卡顿）
            target_pos = np.array(phase_targets[current_phase])
            current_pos = data.qpos[:n_joints]
            # 用指数平滑实现流畅的动作过渡
            smooth_pos = 0.98 * current_pos + 0.02 * target_pos

            # 设置执行器控制量（重点控制右臂抓取关节）
            for i in range(n_joints):
                data.actuator(i).ctrl = smooth_pos[i]

            # 执行物理步
            mujoco.mj_step(model, data)
            viewer.sync()

            # 修正：MuJoCo 3.x获取接触状态的正确方式
            grasp_contact = False
            if current_phase >= 1:
                # 遍历所有接触对，检测手部和目标球的接触
                for i in range(data.ncon):
                    contact = data.contact[i]
                    if (contact.geom1 == hand_grasp_geom_id and contact.geom2 == target_ball_geom_id) or \
                            (contact.geom1 == target_ball_geom_id and contact.geom2 == hand_grasp_geom_id):
                        grasp_contact = True
                        break

                # 打印抓取状态
                if grasp_contact:
                    print("\r抓取成功！正在举起...", end="")
                else:
                    print("\r正在向前抓取...", end="")

            time.sleep(1 / 60)  # 60fps刷新率


if __name__ == "__main__":
    print(f"MuJoCo 版本: {mujoco.__version__}")
    try:
        main()
    except Exception as e:
        print(f"\n运行出错: {e}")