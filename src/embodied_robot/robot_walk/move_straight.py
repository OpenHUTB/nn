import mujoco
from mujoco import viewer
import time
import numpy as np
import random  # 新增：随机转向控制


def control_robot(model_path):
    """
    控制DeepMind Humanoid模型：向前行走 → 随机转向避障 → 回归路径 → 停止
    """
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 打印电机数量（调试用）
    print(f"模型电机数量：{model.nu}，data.ctrl长度：{len(data.ctrl)}")

    # -------------------------- 兼容低版本MuJoCo的ID查询 --------------------------
    wall_id = -1
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name == "wall":
            wall_id = i
            break
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # -------------------------- 核心参数配置 --------------------------
    # 避障参数
    avoid_obstacle = False
    obstacle_distance_threshold = 1.5  # 触发避障距离
    obstacle_avoidance_time = 0
    obstacle_avoidance_duration = 4.0  # 转向避障持续时间
    turn_direction = 0  # 0:未选择, 1:右转, -1:左转（随机赋值）
    return_to_path = False  # 是否进入回归路径阶段
    return_time = 0
    return_duration = 3.0  # 回归路径持续时间
    stop_walk = False  # 是否停止行走

    # 步态控制参数
    gait_period = 2.0
    swing_gain = 1.0  # 降低增益，提升稳定性
    stance_gain = 0.8
    forward_speed = 0.5  # 前进速度系数

    # 姿态稳定参数
    torso_pitch_target = 0.0
    torso_roll_target = 0.0
    torso_yaw_target = 0.0  # 新增：偏航角目标（x轴正方向）
    balance_kp = 80.0
    balance_kd = 10.0
    yaw_kp = 60.0  # 偏航角回正比例增益

    # 启动可视化器
    mujoco.set_mjcb_control(None)
    with viewer.launch_passive(model, data) as viewer_instance:
        print("DeepMind Humanoid仿真启动！")
        print("控制逻辑：向前行走 → 检测障碍 → 随机转向避障 → 回归路径 → 停止")
        start_time = time.time()

        try:
            while True:
                if not viewer_instance.is_running():
                    break

                # -------------------------- 1. 障碍检测与状态切换 --------------------------
                distance_to_wall = 0.0
                if wall_id != -1 and torso_id != -1 and not stop_walk:
                    torso_pos = data.xpos[torso_id]
                    wall_pos = data.xpos[wall_id]
                    distance_to_wall = np.linalg.norm(torso_pos[:2] - wall_pos[:2])

                    # 1.1 触发避障（随机选择转向方向）
                    if distance_to_wall < obstacle_distance_threshold and not avoid_obstacle and not return_to_path:
                        avoid_obstacle = True
                        obstacle_avoidance_time = time.time()
                        turn_direction = random.choice([-1, 1])  # 随机左转/右转
                        dir_name = "左转" if turn_direction == -1 else "右转"
                        print(f"\n检测到墙壁障碍！距离：{distance_to_wall:.2f}米，开始{dir_name}避障...")

                    # 1.2 避障完成，进入回归路径阶段
                    if avoid_obstacle and (time.time() - obstacle_avoidance_time) > obstacle_avoidance_duration:
                        avoid_obstacle = False
                        return_to_path = True
                        return_time = time.time()
                        print(f"{dir_name}避障完成，开始回归原前进方向...")

                    # 1.3 回归路径完成，停止行走
                    if return_to_path and (time.time() - return_time) > return_duration:
                        return_to_path = False
                        stop_walk = True
                        print(f"\n已回归原前进方向，停止行走！最终位置：x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f}")

                # -------------------------- 2. 步态周期计算 --------------------------
                elapsed_time = time.time() - start_time
                cycle = elapsed_time % gait_period
                phase = cycle / gait_period

                # -------------------------- 3. 关节控制核心逻辑 --------------------------
                data.ctrl[:] = 0.0  # 重置控制指令

                if stop_walk:
                    # 4. 停止状态：所有关节归零，保持站立
                    continue

                elif return_to_path:
                    # 3. 回归路径模式：反向转向，回到x轴正方向
                    return_phase = (time.time() - return_time) / return_duration
                    # 减速回归，避免过冲
                    return_speed = 1.0 * np.cos(return_phase * np.pi)

                    # 3.1 躯干偏航角回正（关键：回到x轴正方向）
                    torso_quat = data.xquat[torso_id]
                    # 四元数转偏航角（yaw）
                    yaw = np.arctan2(2 * (torso_quat[0] * torso_quat[3] + torso_quat[1] * torso_quat[2]),
                                     1 - 2 * (torso_quat[2] ** 2 + torso_quat[3] ** 2))
                    yaw_error = torso_yaw_target - yaw

                    # 3.2 转向回正控制
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")

                    if 0 <= abdomen_z_act_id < len(data.ctrl):
                        data.ctrl[abdomen_z_act_id] = yaw_kp * yaw_error * return_speed
                    if 0 <= hip_z_right_act_id < len(data.ctrl):
                        data.ctrl[hip_z_right_act_id] = -yaw_error * return_speed * 0.5
                    if 0 <= hip_z_left_act_id < len(data.ctrl):
                        data.ctrl[hip_z_left_act_id] = yaw_error * return_speed * 0.5

                    # 3.3 保持基本站立姿态
                    for side in ["right", "left"]:
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        if 0 <= hip_y_act_id < len(data.ctrl):
                            data.ctrl[hip_y_act_id] = -0.5
                        if 0 <= knee_act_id < len(data.ctrl):
                            data.ctrl[knee_act_id] = 0.8

                elif avoid_obstacle:
                    # 2. 避障模式：转向绕开障碍
                    avoid_phase = (time.time() - obstacle_avoidance_time) / obstacle_avoidance_duration
                    # 先加速后减速转向，提升稳定性
                    turn_speed = 1.2 * np.sin(avoid_phase * np.pi)

                    # 2.1 转向控制
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")

                    if 0 <= hip_z_right_act_id < len(data.ctrl):
                        data.ctrl[hip_z_right_act_id] = turn_direction * turn_speed * 0.8
                    if 0 <= hip_z_left_act_id < len(data.ctrl):
                        data.ctrl[hip_z_left_act_id] = -turn_direction * turn_speed * 0.8
                    if 0 <= abdomen_z_act_id < len(data.ctrl):
                        data.ctrl[abdomen_z_act_id] = turn_direction * turn_speed * 1.5

                    # 2.2 保持平衡
                    for side in ["right", "left"]:
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        if 0 <= hip_y_act_id < len(data.ctrl):
                            data.ctrl[hip_y_act_id] = -0.8
                        if 0 <= knee_act_id < len(data.ctrl):
                            data.ctrl[knee_act_id] = 1.0

                else:
                    # 1. 正常向前行走模式
                    for side, sign in [("right", 1), ("left", -1)]:
                        swing_phase = (phase + 0.5 * sign) % 1.0

                        # 查询电机ID
                        hip_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_x_{side}")
                        hip_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_z_{side}")
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                        ankle_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")

                        # 腿部关节控制（向前行走）
                        if 0 <= hip_x_act_id < len(data.ctrl):
                            # 新增前进速度系数，确保向前走
                            data.ctrl[hip_x_act_id] = swing_gain * np.sin(2 * np.pi * swing_phase) * forward_speed
                        if 0 <= hip_z_act_id < len(data.ctrl):
                            data.ctrl[hip_z_act_id] = stance_gain * np.cos(2 * np.pi * swing_phase) * 0.3
                        if 0 <= hip_y_act_id < len(data.ctrl):
                            data.ctrl[hip_y_act_id] = -1.2 * np.sin(2 * np.pi * swing_phase) - 0.5
                        if 0 <= knee_act_id < len(data.ctrl):
                            data.ctrl[knee_act_id] = 1.5 * np.sin(2 * np.pi * swing_phase) + 0.8
                        if 0 <= ankle_y_act_id < len(data.ctrl):
                            data.ctrl[ankle_y_act_id] = 0.4 * np.cos(2 * np.pi * swing_phase)
                        if 0 <= ankle_x_act_id < len(data.ctrl):
                            data.ctrl[ankle_x_act_id] = 0.2 * np.sin(2 * np.pi * swing_phase)

                    # 躯干稳定控制（防摔倒+保持前进方向）
                    abdomen_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    abdomen_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_y")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_x")

                    if torso_id != -1:
                        torso_quat = data.xquat[torso_id]
                        # 四元数转欧拉角（俯仰/侧翻）
                        pitch = 2 * (torso_quat[1] * torso_quat[3] - torso_quat[0] * torso_quat[2])
                        roll = 2 * (torso_quat[0] * torso_quat[1] + torso_quat[2] * torso_quat[3])

                        # 平衡补偿
                        if 0 <= abdomen_x_act_id < len(data.ctrl):
                            data.ctrl[abdomen_x_act_id] = balance_kp * (torso_roll_target - roll) - balance_kd * \
                                                          data.qvel[abdomen_x_act_id]
                        if 0 <= abdomen_y_act_id < len(data.ctrl):
                            data.ctrl[abdomen_y_act_id] = balance_kp * (torso_pitch_target - pitch) - balance_kd * \
                                                          data.qvel[abdomen_y_act_id]
                        if 0 <= abdomen_z_act_id < len(data.ctrl):
                            data.ctrl[abdomen_z_act_id] = 0.1 * np.sin(elapsed_time * 0.5)  # 小幅扭腰稳定

                    # 手臂自然摆动
                    for side, sign in [("right", 1), ("left", -1)]:
                        shoulder1_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder1_{side}")
                        shoulder2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder2_{side}")
                        elbow_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"elbow_{side}")

                        shoulder1_cmd = 0.3 * np.sin(2 * np.pi * (phase + 0.5 * sign))
                        shoulder2_cmd = 0.2 * np.cos(2 * np.pi * (phase + 0.5 * sign))
                        elbow_cmd = -0.5 * np.sin(2 * np.pi * (phase + 0.5 * sign)) - 0.3

                        if 0 <= shoulder1_act_id < len(data.ctrl):
                            data.ctrl[shoulder1_act_id] = shoulder1_cmd
                        if 0 <= shoulder2_act_id < len(data.ctrl):
                            data.ctrl[shoulder2_act_id] = shoulder2_cmd
                        if 0 <= elbow_act_id < len(data.ctrl):
                            data.ctrl[elbow_act_id] = elbow_cmd

                # -------------------------- 4. 仿真推进 --------------------------
                mujoco.mj_step(model, data)
                viewer_instance.sync()

                # 实时状态输出
                if wall_id != -1 and torso_id != -1 and int(elapsed_time * 2) % 2 == 0:
                    if stop_walk:
                        status = "已停止"
                    elif return_to_path:
                        status = "回归路径中"
                    elif avoid_obstacle:
                        status = f"避障中（{dir_name}）"
                    else:
                        status = "正常行走"
                    print(f"\r时间：{elapsed_time:.1f}s | 距离墙壁：{distance_to_wall:.2f}m | 状态：{status}", end="")

                time.sleep(model.opt.timestep * 2)

        except KeyboardInterrupt:
            print("\n\n仿真被用户中断")
        except Exception as e:
            print(f"\n\n运行错误：{e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    model_file = "Robot_move_straight.xml"  # 你的模型文件名
    control_robot(model_file)