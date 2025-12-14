import mujoco
from mujoco import viewer
import time
import numpy as np
import random


def control_robot(model_path):
    """
    控制DeepMind Humanoid模型：目标导航（固定目标点）→ 动态避障 → 到达目标点停止
    """
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # -------------------------- 核心配置：固定导航目标点 --------------------------
    TARGET_POS = np.array([8.0, 0.0])  # 固定目标点坐标 (x=8, y=0)
    target_reached_threshold = 0.5  # 到达目标点的距离阈值（米）
    navigation_target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "navigation_target")

    # -------------------------- 动态障碍初始化 --------------------------
    # 获取动态障碍关节和电机ID
    wall2_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wall2_slide_y")
    wall2_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall2_motor")
    wall2_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wall2")

    # 动态障碍参数
    wall2_speed = 0.8  # 移动速度
    wall2_amplitude = 1.8  # 移动幅度（Y轴-1.8~1.8米）
    wall2_phase = random.uniform(0, 2 * np.pi)  # 随机初始相位

    # 打印电机数量（调试用）
    print(f"模型电机数量：{model.nu}，data.ctrl长度：{len(data.ctrl)}")
    print(f"动态障碍配置：关节ID={wall2_joint_id}，电机ID={wall2_motor_id}，初始相位={wall2_phase:.2f}")
    print(f"导航目标点：x={TARGET_POS[0]}, y={TARGET_POS[1]} | 到达阈值：{target_reached_threshold}米")

    # -------------------------- 兼容低版本MuJoCo的ID查询 --------------------------
    wall_ids = []
    wall_names = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and body_name.startswith("wall"):
            wall_ids.append(i)
            wall_names.append(body_name)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # 打印检测到的障碍物
    if wall_ids:
        print(f"检测到障碍物：{wall_names}（ID：{wall_ids}）")
    else:
        print("警告：未检测到任何障碍物！")

    # -------------------------- 核心参数配置 --------------------------
    # 避障参数（适配导航+动态障碍）
    avoid_obstacle = False
    obstacle_distance_threshold = 1.8  # 触发避障的距离阈值
    obstacle_avoidance_time = 0
    obstacle_avoidance_duration = 4.5  # 避障持续时间
    turn_direction = 0
    return_to_path = False
    return_time = 0
    return_duration = 3.5  # 回归导航路径的持续时间
    stop_walk = False
    target_reached = False  # 目标点到达标记
    closest_wall_id = -1
    last_closest_wall_pos = np.zeros(2)  # 记录上一帧最近障碍位置

    # 步态控制参数（适配导航）
    gait_period = 2.0
    swing_gain = 0.85
    stance_gain = 0.8
    forward_speed = 0.35  # 前进速度
    heading_kp = 80.0  # 朝向目标点的比例增益

    # 姿态稳定参数
    torso_pitch_target = 0.0
    torso_roll_target = 0.0
    balance_kp = 100.0
    balance_kd = 15.0

    # 启动可视化器
    mujoco.set_mjcb_control(None)
    with viewer.launch_passive(model, data) as viewer_instance:
        print("\nDeepMind Humanoid目标导航仿真启动！")
        print(
            f"导航逻辑：向目标点({TARGET_POS[0]},{TARGET_POS[1]})移动 → 检测动态障碍 → 避障 → 回归导航路径 → 到达目标点停止")
        start_time = time.time()

        try:
            while True:
                if not viewer_instance.is_running():
                    break

                # -------------------------- 1. 驱动动态障碍移动 --------------------------
                elapsed_time = time.time() - start_time
                # 正弦曲线控制动态障碍Y轴往复移动
                wall2_target_pos = wall2_amplitude * np.sin(wall2_speed * elapsed_time + wall2_phase)
                # 控制动态障碍关节
                if 0 <= wall2_motor_id < len(data.ctrl):
                    data.ctrl[wall2_motor_id] = (wall2_target_pos - data.qpos[wall2_joint_id]) * 2.0  # PD控制

                # -------------------------- 2. 导航状态计算 --------------------------
                if torso_id != -1 and not target_reached:
                    # 获取机器人躯干位置和朝向
                    torso_pos = data.xpos[torso_id]
                    robot_xy = torso_pos[:2]

                    # 计算到目标点的距离和方向
                    target_vector = TARGET_POS - robot_xy
                    distance_to_target = np.linalg.norm(target_vector)

                    # 检查是否到达目标点
                    if distance_to_target < target_reached_threshold:
                        target_reached = True
                        stop_walk = True
                        print(
                            f"\n\n✅ 到达目标点！最终位置：x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f} | 目标点：x={TARGET_POS[0]}, y={TARGET_POS[1]}")
                        continue

                    # 计算机器人朝向（偏航角）
                    torso_quat = data.xquat[torso_id]
                    robot_yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                                           torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 + torso_quat[
                                               3] ** 2)
                    # 计算目标方向的角度
                    target_yaw = np.arctan2(target_vector[1], target_vector[0])
                    # 计算朝向误差（归一化到[-π, π]）
                    yaw_error = target_yaw - robot_yaw
                    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                # -------------------------- 3. 实时检测动态障碍 --------------------------
                distance_to_closest_wall = float('inf')
                closest_wall_name = ""
                closest_wall_pos = np.zeros(2)

                if wall_ids and torso_id != -1 and not stop_walk and not target_reached:
                    torso_pos = data.xpos[torso_id]
                    for idx, wall_id in enumerate(wall_ids):
                        wall_pos = data.xpos[wall_id]
                        distance = np.linalg.norm(torso_pos[:2] - wall_pos[:2])
                        # 动态障碍预判：如果是wall2，额外计算未来0.5秒的位置
                        if wall_names[idx] == "wall2":
                            # 预估0.5秒后障碍位置
                            future_wall2_pos = wall2_amplitude * np.sin(
                                wall2_speed * (elapsed_time + 0.5) + wall2_phase)
                            future_distance = np.linalg.norm(np.array([torso_pos[0], torso_pos[1]]) -
                                                             np.array([wall_pos[0], future_wall2_pos]))
                            distance = min(distance, future_distance)  # 取当前/未来距离最小值

                        if distance < distance_to_closest_wall:
                            distance_to_closest_wall = distance
                            closest_wall_id = wall_id
                            closest_wall_name = wall_names[idx]
                            closest_wall_pos = wall_pos[:2]

                # -------------------------- 4. 导航+避障状态切换 --------------------------
                if closest_wall_id != -1 and torso_id != -1 and not stop_walk and not target_reached:
                    # 触发避障
                    if (distance_to_closest_wall < obstacle_distance_threshold and
                            not avoid_obstacle and not return_to_path):
                        avoid_obstacle = True
                        obstacle_avoidance_time = time.time()
                        # 基于目标方向和障碍位置选择转向方向（更智能的避障）
                        torso_pos = data.xpos[torso_id]
                        wall_relative_y = closest_wall_pos[1] - torso_pos[1]
                        turn_direction = -1 if wall_relative_y > 0 else 1  # 障碍在右侧则左转，左侧则右转
                        dir_name = "左转" if turn_direction == -1 else "右转"
                        print(
                            f"\n⚠️  检测到最近障碍【{closest_wall_name}】！距离：{distance_to_closest_wall:.2f}米，开始{dir_name}避障...")

                    # 避障完成，回归导航路径
                    if avoid_obstacle and (time.time() - obstacle_avoidance_time) > obstacle_avoidance_duration:
                        avoid_obstacle = False
                        return_to_path = True
                        return_time = time.time()
                        print(f"✅ 避障完成，开始回归导航路径...")

                    # 回归完成，继续向目标点移动
                    if return_to_path and (time.time() - return_time) > return_duration:
                        return_to_path = False
                        print(f"✅ 已回归导航路径，继续向目标点移动...")

                # -------------------------- 5. 步态周期计算 --------------------------
                cycle = elapsed_time % gait_period
                phase = cycle / gait_period

                # -------------------------- 6. 关节控制核心逻辑 --------------------------
                data.ctrl[:len(data.ctrl) - 1] = 0.0  # 重置机器人控制指令（保留动态障碍电机）

                if stop_walk or target_reached:
                    # 停止状态：所有关节归零，保持站立
                    continue

                elif return_to_path:
                    # 回归导航路径模式：转向回到目标方向
                    return_phase = (time.time() - return_time) / return_duration
                    return_speed = 1.3 * np.cos(return_phase * np.pi)

                    # 朝向目标点回正
                    torso_quat = data.xquat[torso_id]
                    robot_yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                                           torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 + torso_quat[
                                               3] ** 2)
                    target_vector = TARGET_POS - data.xpos[torso_id][:2]
                    target_yaw = np.arctan2(target_vector[1], target_vector[0])
                    yaw_error = target_yaw - robot_yaw
                    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                    # 转向回正控制
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")

                    if 0 <= abdomen_z_act_id < len(data.ctrl) - 1:
                        data.ctrl[abdomen_z_act_id] = heading_kp * yaw_error * return_speed
                    if 0 <= hip_z_right_act_id < len(data.ctrl) - 1:
                        data.ctrl[hip_z_right_act_id] = -yaw_error * return_speed * 0.7
                    if 0 <= hip_z_left_act_id < len(data.ctrl) - 1:
                        data.ctrl[hip_z_left_act_id] = yaw_error * return_speed * 0.7

                    # 保持基本站立姿态
                    for side in ["right", "left"]:
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                        if 0 <= hip_y_act_id < len(data.ctrl) - 1:
                            data.ctrl[hip_y_act_id] = -0.7
                        if 0 <= knee_act_id < len(data.ctrl) - 1:
                            data.ctrl[knee_act_id] = 1.0
                        if 0 <= ankle_y_act_id < len(data.ctrl) - 1:
                            data.ctrl[ankle_y_act_id] = 0.3

                elif avoid_obstacle:
                    # 避障模式：转向绕开最近障碍
                    avoid_phase = (time.time() - obstacle_avoidance_time) / obstacle_avoidance_duration
                    turn_speed = 1.4 * np.sin(avoid_phase * np.pi)  # 转向力度

                    # 转向控制
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")

                    if 0 <= hip_z_right_act_id < len(data.ctrl) - 1:
                        data.ctrl[hip_z_right_act_id] = turn_direction * turn_speed * 1.0
                    if 0 <= hip_z_left_act_id < len(data.ctrl) - 1:
                        data.ctrl[hip_z_left_act_id] = -turn_direction * turn_speed * 1.0
                    if 0 <= abdomen_z_act_id < len(data.ctrl) - 1:
                        data.ctrl[abdomen_z_act_id] = turn_direction * turn_speed * 1.8

                    # 保持平衡
                    for side in ["right", "left"]:
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")
                        if 0 <= hip_y_act_id < len(data.ctrl) - 1:
                            data.ctrl[hip_y_act_id] = -1.0
                        if 0 <= knee_act_id < len(data.ctrl) - 1:
                            data.ctrl[knee_act_id] = 1.2
                        if 0 <= ankle_x_act_id < len(data.ctrl) - 1:
                            data.ctrl[ankle_x_act_id] = 0.2

                else:
                    # 正常导航模式：向目标点移动
                    torso_quat = data.xquat[torso_id]
                    robot_yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                                           torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 + torso_quat[
                                               3] ** 2)
                    target_vector = TARGET_POS - data.xpos[torso_id][:2]
                    target_yaw = np.arctan2(target_vector[1], target_vector[0])
                    yaw_error = target_yaw - robot_yaw
                    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                    # 朝向目标点的转向控制
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    if 0 <= abdomen_z_act_id < len(data.ctrl) - 1:
                        data.ctrl[abdomen_z_act_id] = heading_kp * yaw_error * 0.1

                    # 腿部步态控制（向目标点前进）
                    for side, sign in [("right", 1), ("left", -1)]:
                        swing_phase = (phase + 0.5 * sign) % 1.0

                        # 查询电机ID
                        hip_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_x_{side}")
                        hip_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_z_{side}")
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                        ankle_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")

                        # 腿部关节控制（适配导航）
                        if 0 <= hip_x_act_id < len(data.ctrl) - 1:
                            data.ctrl[hip_x_act_id] = swing_gain * np.sin(2 * np.pi * swing_phase) * forward_speed
                        if 0 <= hip_z_act_id < len(data.ctrl) - 1:
                            data.ctrl[hip_z_act_id] = stance_gain * np.cos(
                                2 * np.pi * swing_phase) * 0.2 + yaw_error * 0.1
                        if 0 <= hip_y_act_id < len(data.ctrl) - 1:
                            data.ctrl[hip_y_act_id] = -0.9 * np.sin(2 * np.pi * swing_phase) - 0.4
                        if 0 <= knee_act_id < len(data.ctrl) - 1:
                            data.ctrl[knee_act_id] = 1.2 * np.sin(2 * np.pi * swing_phase) + 0.7
                        if 0 <= ankle_y_act_id < len(data.ctrl) - 1:
                            data.ctrl[ankle_y_act_id] = 0.3 * np.cos(2 * np.pi * swing_phase)
                        if 0 <= ankle_x_act_id < len(data.ctrl) - 1:
                            data.ctrl[ankle_x_act_id] = 0.15 * np.sin(2 * np.pi * swing_phase)

                    # 躯干稳定控制
                    abdomen_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    abdomen_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_y")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_x")

                    pitch = 2 * (torso_quat[1] * torso_quat[3] - torso_quat[0] * torso_quat[2])
                    roll = 2 * (torso_quat[0] * torso_quat[1] + torso_quat[2] * torso_quat[3])

                    # 平衡补偿
                    if 0 <= abdomen_x_act_id < len(data.ctrl) - 1:
                        data.ctrl[abdomen_x_act_id] = balance_kp * (torso_roll_target - roll) - balance_kd * \
                                                      data.qvel[abdomen_x_act_id]
                    if 0 <= abdomen_y_act_id < len(data.ctrl) - 1:
                        data.ctrl[abdomen_y_act_id] = balance_kp * (torso_pitch_target - pitch) - balance_kd * \
                                                      data.qvel[abdomen_y_act_id]

                    # 手臂自然摆动
                    for side, sign in [("right", 1), ("left", -1)]:
                        shoulder1_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder1_{side}")
                        shoulder2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder2_{side}")
                        elbow_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"elbow_{side}")

                        shoulder1_cmd = 0.2 * np.sin(2 * np.pi * (phase + 0.5 * sign))
                        shoulder2_cmd = 0.15 * np.cos(2 * np.pi * (phase + 0.5 * sign))
                        elbow_cmd = -0.4 * np.sin(2 * np.pi * (phase + 0.5 * sign)) - 0.25

                        if 0 <= shoulder1_act_id < len(data.ctrl) - 1:
                            data.ctrl[shoulder1_act_id] = shoulder1_cmd
                        if 0 <= shoulder2_act_id < len(data.ctrl) - 1:
                            data.ctrl[shoulder2_act_id] = shoulder2_cmd
                        if 0 <= elbow_act_id < len(data.ctrl) - 1:
                            data.ctrl[elbow_act_id] = elbow_cmd

                # -------------------------- 7. 仿真推进 --------------------------
                mujoco.mj_step(model, data)
                viewer_instance.sync()

                # 实时状态输出（含导航信息）
                if torso_id != -1 and int(elapsed_time * 2) % 2 == 0:
                    if target_reached:
                        status = "✅ 到达目标点"
                        nav_info = "—"
                        dist_info = "—"
                        wall2_pos_info = "—"
                    elif stop_walk:
                        status = "已停止"
                        nav_info = "—"
                        dist_info = "—"
                        wall2_pos_info = "—"
                    else:
                        if return_to_path:
                            status = "回归导航路径中"
                        elif avoid_obstacle:
                            status = f"避障中（{dir_name}）"
                        else:
                            status = "向目标点移动中"

                        # 导航信息
                        distance_to_target = np.linalg.norm(TARGET_POS - data.xpos[torso_id][:2])
                        nav_info = f"剩余{distance_to_target:.2f}m | 朝向误差{np.degrees(yaw_error):.1f}°"

                        # 障碍信息
                        dist_info = f"{distance_to_closest_wall:.2f}m（{closest_wall_name}）"

                        # 动态障碍位置信息
                        if wall2_body_id != -1:
                            wall2_current_pos = data.xpos[wall2_body_id]
                            wall2_pos_info = f"wall2(Y):{wall2_current_pos[1]:.2f}m"
                        else:
                            wall2_pos_info = "wall2:未知"

                    torso_pos = data.xpos[torso_id]
                    print(
                        f"\r时间：{elapsed_time:.1f}s | 位置：x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f} | 导航：{nav_info} | 障碍：{dist_info} | 动态障碍：{wall2_pos_info} | 状态：{status}",
                        end="")

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