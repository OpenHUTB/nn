import mujoco
from mujoco import viewer
import time
import numpy as np


def control_robot(model_path):
    """
    控制优化后的拟人化机器人模型行走，适配墙壁障碍环境。
    包含碰撞检测和避障逻辑。
    """
    # 加载带墙壁障碍的机器人模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 获取墙壁和机器人躯干的ID（用于碰撞检测）
    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wall")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # 避障状态标记
    avoid_obstacle = False
    obstacle_distance_threshold = 1.0  # 距离墙壁小于1米时开始避障
    obstacle_avoidance_time = 0
    obstacle_avoidance_duration = 3.0  # 避障持续时间（秒）

    # 启动可视化器
    with viewer.launch_passive(model, data) as viewer_instance:
        print("仿真开始。按ESC或关闭窗口停止...")
        start_time = time.time()

        try:
            while True:
                if not viewer_instance.is_running():
                    break

                # 获取机器人躯干与墙壁的距离
                torso_pos = data.xpos[torso_id]
                wall_pos = data.xpos[wall_id]
                distance_to_wall = np.linalg.norm(torso_pos[:2] - wall_pos[:2])  # 只考虑xy平面距离

                # 碰撞/障碍检测
                if distance_to_wall < obstacle_distance_threshold and not avoid_obstacle:
                    avoid_obstacle = True
                    obstacle_avoidance_time = time.time()
                    print(f"\n检测到墙壁障碍！距离：{distance_to_wall:.2f}米，开始避障...")

                # 重置避障状态
                if avoid_obstacle and (time.time() - obstacle_avoidance_time) > obstacle_avoidance_duration:
                    avoid_obstacle = False
                    print("避障完成，恢复正常行走")

                # 步态周期：2.5秒一步，保持稳定性
                elapsed_time = time.time() - start_time
                cycle = elapsed_time % 2.5

                # -------------------------- 核心步态逻辑 --------------------------
                if not avoid_obstacle:
                    # 正常行走模式
                    if cycle < 1.25:
                        # swing_phase 从 0 变化到 1，表示摆动腿的一个完整摆动过程
                        swing_phase = (cycle / 1.25)

                        # --- 腿部控制 ---
                        # 左腿（摆动腿）：髋关节前摆，膝关节弯曲
                        data.ctrl[0] = 0.05 + 0.25 * np.sin(swing_phase * np.pi)  # left_hip
                        data.ctrl[1] = -0.15 - 0.3 * np.sin(swing_phase * np.pi)  # left_knee
                        data.ctrl[2] = 0.0 + 0.1 * np.cos(swing_phase * np.pi)  # left_ankle

                        # 右腿（支撑腿）：保持稳定，轻微调整以平衡身体
                        data.ctrl[3] = -0.05 - 0.05 * np.sin(swing_phase * np.pi)  # right_hip
                        data.ctrl[4] = -0.15 + 0.05 * np.sin(swing_phase * np.pi)  # right_knee
                        data.ctrl[5] = 0.0 - 0.05 * np.cos(swing_phase * np.pi)  # right_ankle

                        # --- 手臂协同控制 ---
                        # 左臂（与摆动腿反向）：肩关节后摆，肘关节随动弯曲
                        data.ctrl[6] = 0.0 - 0.2 * np.sin(swing_phase * np.pi)  # left_shoulder
                        data.ctrl[7] = -0.8 + 0.15 * np.sin(swing_phase * np.pi)  # left_elbow

                        # 右臂（与摆动腿同向）：肩关节前摆，肘关节随动弯曲
                        data.ctrl[9] = 0.0 + 0.2 * np.sin(swing_phase * np.pi)  # right_shoulder
                        data.ctrl[10] = -0.8 - 0.15 * np.sin(swing_phase * np.pi)  # right_elbow

                    else:
                        # swing_phase 从 0 变化到 1
                        swing_phase = ((cycle - 1.25) / 1.25)

                        # --- 腿部控制 ---
                        # 右腿（摆动腿）
                        data.ctrl[3] = -0.05 - 0.25 * np.sin(swing_phase * np.pi)  # right_hip
                        data.ctrl[4] = -0.15 - 0.3 * np.sin(swing_phase * np.pi)  # right_knee
                        data.ctrl[5] = 0.0 + 0.1 * np.cos(swing_phase * np.pi)  # right_ankle

                        # 左腿（支撑腿）
                        data.ctrl[0] = 0.05 + 0.05 * np.sin(swing_phase * np.pi)  # left_hip
                        data.ctrl[1] = -0.15 + 0.05 * np.sin(swing_phase * np.pi)  # left_knee
                        data.ctrl[2] = 0.0 - 0.05 * np.cos(swing_phase * np.pi)  # left_ankle

                        # --- 手臂协同控制 ---
                        # 右臂（与摆动腿反向）
                        data.ctrl[9] = 0.0 - 0.2 * np.sin(swing_phase * np.pi)  # right_shoulder
                        data.ctrl[10] = -0.8 + 0.15 * np.sin(swing_phase * np.pi)  # right_elbow

                        # 左臂（与摆动腿同向）
                        data.ctrl[6] = 0.0 + 0.2 * np.sin(swing_phase * np.pi)  # left_shoulder
                        data.ctrl[7] = -0.8 - 0.15 * np.sin(swing_phase * np.pi)  # left_elbow

                else:
                    # 避障模式：停止前进，改为原地转向（向左转）
                    avoid_phase = (time.time() - obstacle_avoidance_time) / obstacle_avoidance_duration

                    # 原地左转控制
                    data.ctrl[0] = 0.1 + 0.1 * np.sin(avoid_phase * 2 * np.pi)  # left_hip
                    data.ctrl[1] = -0.15  # left_knee
                    data.ctrl[2] = 0.0  # left_ankle

                    data.ctrl[3] = -0.2 - 0.1 * np.sin(avoid_phase * 2 * np.pi)  # right_hip（反向转）
                    data.ctrl[4] = -0.15  # right_knee
                    data.ctrl[5] = 0.0  # right_ankle

                    # 手臂配合转向动作
                    data.ctrl[6] = 0.1 + 0.1 * np.sin(avoid_phase * 2 * np.pi)  # left_shoulder
                    data.ctrl[7] = -0.8  # left_elbow
                    data.ctrl[9] = -0.1 - 0.1 * np.sin(avoid_phase * 2 * np.pi)  # right_shoulder
                    data.ctrl[10] = -0.8  # right_elbow

                # 固定关节：腕关节保持下垂，颈部保持中立
                data.ctrl[8] = -0.2  # left_wrist (保持微微下垂)
                data.ctrl[11] = -0.2  # right_wrist (保持微微下垂)
                data.ctrl[12] = 0.5  # neck (中间位置)

                # -------------------------- 仿真推进 --------------------------
                mujoco.mj_step(model, data)
                viewer_instance.sync()

                # 实时显示距离信息
                if int(elapsed_time * 10) % 10 == 0:  # 每0.1秒显示一次
                    print(f"\r距离墙壁：{distance_to_wall:.2f}米 | 状态：{'避障中' if avoid_obstacle else '正常行走'}",
                          end="")

                # 控制仿真速度，使其更易于观察
                time.sleep(model.opt.timestep * 1.5)

        except KeyboardInterrupt:
            print("\n\n仿真被用户中断")


if __name__ == "__main__":
    # 确保模型文件名与实际保存的一致（带墙壁的模型）
    model_file = "Robot_move_straight.xml"  # 如果墙壁模型另存为其他名称，请修改此处
    control_robot(model_file)