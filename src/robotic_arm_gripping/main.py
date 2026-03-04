import mujoco
import mujoco.viewer
import numpy as np
import time


def main():
    model = mujoco.MjModel.from_xml_path("arm_model.xml")
    data = mujoco.MjData(model)

    # 获取可动关节列表（包含手指关节）
    joint_names = []
    n_joints = 0
    finger_joint_indices = []  # 存储手指关节的索引
    finger_joint_names = [
        "finger1_joint", "finger1_tip_joint",
        "finger2_joint", "finger2_tip_joint",
        "finger3_joint", "finger3_tip_joint"
    ]

    for i in range(model.njnt):
        if model.joint(i).type != 0:
            joint_name = model.joint(i).name
            joint_names.append(joint_name)
            n_joints += 1
            # 记录手指关节的索引
            if joint_name in finger_joint_names:
                finger_joint_indices.append(i)

    print(f"可动关节列表: {joint_names}")
    print(f"可动关节数量: {n_joints}")
    print(f"手指关节索引: {finger_joint_indices}")

    # 预获取抓取目标和手指的geom ID
    target_ball_geom_id = model.geom('target_ball_geom').id
    finger_geom_ids = [
        model.geom('finger1_geom').id,
        model.geom('finger2_geom').id,
        model.geom('finger3_geom').id
    ]

    # 定义动作阶段
    # 阶段0：初始姿态（手指张开）→ 阶段1：向前伸臂（手指保持张开）→ 阶段2：闭合手指抓取 → 阶段3：举起 → 阶段4：保持
    phase_targets = {
        # [肩1, 肩2, 肘, 指1根, 指1尖, 指2根, 指2尖, 指3根, 指3尖, 左肩1, 左肩2, 左肘]
        0: [0, 0, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0, 0, 0],  # 初始（手指张开）
        1: [np.pi / 3, -np.pi / 4, -np.pi / 3, 0.2, 0, 0.2, 0, 0.2, 0, 0, 0, 0],  # 前伸（手指张开）
        2: [np.pi / 3, -np.pi / 4, -np.pi / 3, -1.0, -0.6, -1.0, -0.6, -1.0, -0.6, 0, 0, 0],  # 闭合手指
        3: [np.pi / 6, -np.pi / 6, np.pi / 6, -1.0, -0.6, -1.0, -0.6, -1.0, -0.6, 0, 0, 0],  # 举起（手指闭合）
        4: [np.pi / 6, -np.pi / 6, np.pi / 6, -1.0, -0.6, -1.0, -0.6, -1.0, -0.6, 0, 0, 0]  # 保持
    }
    phase_durations = [2, 3, 1.5, 2, 3]  # 各阶段时长（秒）
    phase_start_time = 0
    current_phase = 0
    grasp_detected = False  # 标记是否检测到抓取

    # 启动可视化
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -15
        viewer.cam.distance = 1.2
        viewer.cam.lookat = [0.2, 0, 0.3]

        start_time = time.time()
        while viewer.is_running():
            current_time = time.time() - start_time
            elapsed_phase_time = current_time - phase_start_time

            # 切换动作阶段
            if current_phase < len(phase_durations) and elapsed_phase_time > phase_durations[current_phase]:
                phase_start_time = current_time
                current_phase = min(current_phase + 1, len(phase_durations) - 1)
                phase_names = ['初始姿态', '向前伸臂', '闭合手指抓取', '举起', '保持']
                print(f"\n切换到动作阶段 {current_phase}: {phase_names[current_phase]}")

            # 检测手指与目标球的接触（触发抓取）
            if current_phase == 1 and not grasp_detected:
                for i in range(data.ncon):
                    contact = data.contact[i]
                    if (contact.geom1 in finger_geom_ids and contact.geom2 == target_ball_geom_id) or \
                            (contact.geom1 == target_ball_geom_id and contact.geom2 in finger_geom_ids):
                        grasp_detected = True
                        print("\n检测到手指接触目标，准备闭合抓取！")
                        # 提前切换到闭合手指阶段
                        current_phase = 2
                        phase_start_time = current_time
                        break

            # 平滑插值到目标关节角度
            target_pos = np.array(phase_targets[current_phase])
            current_pos = data.qpos[:n_joints]
            smooth_pos = 0.98 * current_pos + 0.02 * target_pos

            # 设置执行器控制量
            for i in range(n_joints):
                data.actuator(i).ctrl = smooth_pos[i]

            # 执行物理步
            mujoco.mj_step(model, data)
            viewer.sync()

            # 打印状态信息
            if current_phase == 1:
                print("\r正在向前伸臂（手指张开）...", end="")
            elif current_phase == 2:
                print("\r手指闭合抓取中...", end="")
            elif current_phase >= 3:
                print("\r抓取成功！已举起目标物体...", end="")

            time.sleep(1 / 60)


if __name__ == "__main__":
    print(f"MuJoCo 版本: {mujoco.__version__}")
    try:
        main()
    except Exception as e:
        print(f"\n运行出错: {e}")