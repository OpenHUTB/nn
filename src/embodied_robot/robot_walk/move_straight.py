import mujoco
from mujoco import viewer
import time
import numpy as np
import random
from collections import deque
import os
import sys

# ====================== 通用版：禁用所有日志输出（适配所有Mujoco版本） ======================
# 方法1：设置环境变量禁用Mujoco输出
os.environ['MUJOCO_QUIET'] = '1'  # 静默模式
os.environ['MUJOCO_GL'] = 'egl'  # 无窗口渲染（可选）


# 方法2：重定向标准输出/错误（彻底关闭所有控制台输出）
class QuietStream:
    def write(self, msg):
        pass

    def flush(self):
        pass


# 取消注释以下两行可完全静默运行（无任何控制台输出）
# sys.stdout = QuietStream()
# sys.stderr = QuietStream()

# 设置随机种子保证可复现
np.random.seed(42)
random.seed(42)


def control_robot(model_path):
    """
    控制DeepMind Humanoid模型：复杂动态环境下的多目标点巡逻导航
    特性：
    1. 多动态障碍（正弦组合/随机游走/圆周运动）+ 多障碍优先级避障
    2. 5个固定巡逻目标点，按顺序导航，完成后循环巡逻
    3. 无日志文件生成，精简控制台输出
    """
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # -------------------------- 多目标点巡逻配置 --------------------------
    # 定义巡逻目标点（名称、坐标、显示名称）
    PATROL_POINTS = [
        {"name": "patrol_target_1", "pos": np.array([0.0, 0.0]), "label": "起点"},
        {"name": "patrol_target_2", "pos": np.array([4.0, -2.0]), "label": "巡逻点1（西南）"},
        {"name": "patrol_target_3", "pos": np.array([8.0, 2.0]), "label": "巡逻点2（东北）"},
        {"name": "patrol_target_4", "pos": np.array([10.0, -1.0]), "label": "巡逻点3（西北）"},
        {"name": "patrol_target_5", "pos": np.array([12.0, 0.0]), "label": "终点"}
    ]
    target_reached_threshold = 0.8  # 到达目标点的判定阈值
    current_target_index = 0  # 当前导航的目标点索引
    patrol_cycles = 0  # 完成的巡逻循环次数
    patrol_completed = False  # 是否完成一轮巡逻
    target_switch_cooldown = 2.0  # 切换目标点的冷却时间
    last_target_switch_time = 0  # 上次切换目标点的时间

    # 初始化目标点ID（精简输出）
    patrol_point_ids = {}
    for point in PATROL_POINTS:
        point_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, point["name"])
        patrol_point_ids[point["name"]] = point_id

    # -------------------------- 多动态障碍初始化 --------------------------
    # 障碍1（wall2）：正弦组合运动（Y+Z轴）
    wall2_joint_ids = {
        "y": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wall2_slide_y"),
        "z": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wall2_slide_z")
    }
    wall2_motor_ids = {
        "y": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall2_motor_y"),
        "z": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall2_motor_z")
    }
    wall2_params = {
        "y_amp": 2.5, "y_freq": 0.7, "y_phase": random.uniform(0, 2 * np.pi),
        "z_amp": 0.4, "z_freq": 0.3, "z_phase": random.uniform(0, 2 * np.pi)
    }

    # 障碍2（wall3）：随机游走运动（X+Y轴）
    wall3_joint_ids = {
        "x": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wall3_slide_x"),
        "y": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wall3_slide_y")
    }
    wall3_motor_ids = {
        "x": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall3_motor_x"),
        "y": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall3_motor_y")
    }
    wall3_params = {
        "x_base": 6.0, "x_range": 1.0, "x_speed": 0.4,
        "y_base": 0.0, "y_range": 2.0, "y_speed": 0.6,
        "x_dir": random.choice([-1, 1]), "y_dir": random.choice([-1, 1]),
        "x_switch": random.uniform(2.0, 4.0), "y_switch": random.uniform(1.5, 3.5)
    }
    wall3_last_switch = {"x": 0.0, "y": 0.0}

    # 障碍3（wall4）：圆周运动（旋转+径向）
    wall4_joint_ids = {
        "rot": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wall4_rotate"),
        "rad": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wall4_radial")
    }
    wall4_motor_ids = {
        "rot": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall4_motor_rot"),
        "rad": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall4_motor_rad")
    }
    wall4_params = {
        "rot_speed": 0.5, "rot_dir": random.choice([-1, 1]),
        "rad_amp": 0.8, "rad_freq": 0.2, "rad_phase": random.uniform(0, 2 * np.pi),
        "rad_base": 1.2
    }

    # -------------------------- 障碍检测初始化 --------------------------
    valid_wall_names = ["wall1", "wall2", "wall3", "wall4"]
    wall_ids = []
    wall_names = []
    wall_types = {}

    for wall_name in valid_wall_names:
        wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wall_name)
        if wall_id != -1:
            wall_ids.append(wall_id)
            wall_names.append(wall_name)
            if wall_name == "wall1":
                wall_types[wall_name] = "fixed"
            elif wall_name == "wall2":
                wall_types[wall_name] = "dynamic1"
            elif wall_name == "wall3":
                wall_types[wall_name] = "dynamic2"
            elif wall_name == "wall4":
                wall_types[wall_name] = "dynamic3"

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # -------------------------- 核心参数配置 --------------------------
    # 多障碍避障参数
    avoid_obstacle = False
    obstacle_distance_threshold = 2.0
    obstacle_avoidance_time = 0
    obstacle_avoidance_duration = 5.0
    turn_direction = 0
    return_to_path = False
    return_time = 0
    return_duration = 4.0
    stop_walk = False
    closest_wall_id = -1
    closest_wall_type = ""
    closest_wall_name = ""
    dir_name = ""
    wall_pos_history = {name: deque(maxlen=10) for name in wall_names}

    # 步态控制参数
    gait_period = 2.2
    swing_gain = 0.8
    stance_gain = 0.75
    forward_speed = 0.3
    heading_kp = 90.0

    # 姿态稳定参数
    torso_pitch_target = 0.0
    torso_roll_target = 0.0
    balance_kp = 110.0
    balance_kd = 18.0

    # 启动可视化器（精简初始化输出）
    with viewer.launch_passive(model, data) as viewer_instance:
        # 精简启动提示
        print("🤖 多目标点巡逻仿真启动（无日志模式）")

        start_time = time.time()
        last_print_time = 0  # 控制状态输出频率，减少控制台信息

        try:
            while True:
                if not viewer_instance.is_running():
                    break

                elapsed_time = time.time() - start_time
                current_target = PATROL_POINTS[current_target_index]

                # -------------------------- 1. 多动态障碍运动控制 --------------------------
                # 障碍1（wall2）：Y+Z轴正弦组合运动
                if all(id != -1 for id in wall2_motor_ids.values()):
                    wall2_y_target = wall2_params["y_amp"] * np.sin(
                        wall2_params["y_freq"] * elapsed_time + wall2_params["y_phase"])
                    wall2_z_target = wall2_params["z_amp"] * np.sin(
                        wall2_params["z_freq"] * elapsed_time + wall2_params["z_phase"]) + 0.75
                    data.ctrl[wall2_motor_ids["y"]] = (wall2_y_target - data.qpos[wall2_joint_ids["y"]]) * 2.5
                    data.ctrl[wall2_motor_ids["z"]] = (wall2_z_target - data.qpos[wall2_joint_ids["z"]]) * 1.8

                # 障碍2（wall3）：随机游走运动
                if all(id != -1 for id in wall3_motor_ids.values()):
                    if elapsed_time - wall3_last_switch["x"] > wall3_params["x_switch"]:
                        wall3_params["x_dir"] *= -1
                        wall3_params["x_switch"] = random.uniform(2.0, 4.0)
                        wall3_last_switch["x"] = elapsed_time
                    if elapsed_time - wall3_last_switch["y"] > wall3_params["y_switch"]:
                        wall3_params["y_dir"] *= -1
                        wall3_params["y_switch"] = random.uniform(1.5, 3.5)
                        wall3_last_switch["y"] = elapsed_time

                    wall3_x_target = wall3_params["x_base"] + wall3_params["x_dir"] * wall3_params["x_speed"] * (
                                elapsed_time % 5)
                    wall3_y_target = wall3_params["y_base"] + wall3_params["y_dir"] * wall3_params["y_speed"] * (
                                elapsed_time % 4)
                    wall3_x_target = np.clip(wall3_x_target, 5.0, 7.0)
                    wall3_y_target = np.clip(wall3_y_target, -2.5, 2.5)
                    data.ctrl[wall3_motor_ids["x"]] = (wall3_x_target - data.qpos[wall3_joint_ids["x"]]) * 2.2
                    data.ctrl[wall3_motor_ids["y"]] = (wall3_y_target - data.qpos[wall3_joint_ids["y"]]) * 2.0

                # 障碍3（wall4）：圆周运动
                if all(id != -1 for id in wall4_motor_ids.values()):
                    wall4_rot_target = wall4_params["rot_dir"] * wall4_params["rot_speed"] * elapsed_time
                    wall4_rad_target = wall4_params["rad_base"] + wall4_params["rad_amp"] * np.sin(
                        wall4_params["rad_freq"] * elapsed_time + wall4_params["rad_phase"])
                    data.ctrl[wall4_motor_ids["rot"]] = (wall4_rot_target - data.qpos[wall4_joint_ids["rot"]]) * 1.5
                    data.ctrl[wall4_motor_ids["rad"]] = (wall4_rad_target - data.qpos[wall4_joint_ids["rad"]]) * 2.0

                # -------------------------- 2. 多目标点导航逻辑 --------------------------
                yaw_error = 0.0
                distance_to_target = float('inf')

                if torso_id != -1 and not stop_walk:
                    torso_pos = data.xpos[torso_id]
                    robot_xy = torso_pos[:2]

                    # 计算到当前目标点的距离
                    target_vector = current_target["pos"] - robot_xy
                    distance_to_target = np.linalg.norm(target_vector)

                    # 检查是否到达当前目标点
                    if (distance_to_target < target_reached_threshold and
                            not patrol_completed and
                            elapsed_time - last_target_switch_time > target_switch_cooldown):

                        # 精简目标点到达提示（只输出关键信息）
                        print(f"\n✅ 到达：{current_target['label']} (x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f})")

                        # 更新目标点切换时间
                        last_target_switch_time = elapsed_time

                        # 切换到下一个目标点
                        if current_target_index < len(PATROL_POINTS) - 1:
                            current_target_index += 1
                            print(f"🔄 前往：{PATROL_POINTS[current_target_index]['label']}")
                        else:
                            # 完成一轮巡逻
                            patrol_completed = True
                            patrol_cycles += 1
                            print(f"\n🏁 完成第 {patrol_cycles} 轮巡逻！")

                            # 重置巡逻状态（循环巡逻）
                            time.sleep(target_switch_cooldown)
                            current_target_index = 0
                            patrol_completed = False
                            last_target_switch_time = time.time()
                            print(f"🔄 重新开始巡逻")

                    # 计算机器人朝向和目标方向
                    torso_quat = data.xquat[torso_id]
                    robot_yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                                           torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 + torso_quat[
                                               3] ** 2)
                    target_yaw = np.arctan2(target_vector[1], target_vector[0])
                    yaw_error = target_yaw - robot_yaw
                    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                # -------------------------- 3. 多障碍检测与优先级排序 --------------------------
                distance_to_closest_wall = float('inf')
                closest_wall_pos = np.zeros(2)
                wall_priority = {"dynamic1": 4, "dynamic2": 3, "dynamic3": 2, "fixed": 1}

                if wall_ids and torso_id != -1 and not stop_walk and not patrol_completed:
                    torso_pos = data.xpos[torso_id]
                    wall_distances = []

                    for idx, wall_id in enumerate(wall_ids):
                        wall_name = wall_names[idx]
                        wall_type = wall_types.get(wall_name, "fixed")

                        wall_pos = data.xpos[wall_id]
                        wall_pos_history[wall_name].append(wall_pos[:2])
                        current_distance = np.linalg.norm(torso_pos[:2] - wall_pos[:2])

                        # 动态障碍位置预判
                        predicted_distance = current_distance
                        if len(wall_pos_history[wall_name]) > 5 and wall_type != "fixed":
                            pos_history = np.array(wall_pos_history[wall_name])
                            velocity = (pos_history[-1] - pos_history[0]) / len(pos_history) * model.opt.timestep * 10
                            future_pos = wall_pos[:2] + velocity * 0.8
                            predicted_distance = np.linalg.norm(torso_pos[:2] - future_pos)

                        weighted_distance = predicted_distance / wall_priority[wall_type]
                        wall_distances.append({
                            "name": wall_name,
                            "type": wall_type,
                            "id": wall_id,
                            "pos": wall_pos[:2],
                            "current_dist": current_distance,
                            "predicted_dist": predicted_distance,
                            "weighted_dist": weighted_distance
                        })

                    wall_distances.sort(key=lambda x: x["weighted_dist"])
                    if wall_distances:
                        closest_wall = wall_distances[0]
                        closest_wall_id = closest_wall["id"]
                        closest_wall_name = closest_wall["name"]
                        closest_wall_type = closest_wall["type"]
                        closest_wall_pos = closest_wall["pos"]
                        distance_to_closest_wall = closest_wall["predicted_dist"]

                # -------------------------- 4. 多障碍避障状态切换 --------------------------
                if closest_wall_id != -1 and torso_id != -1 and not stop_walk and not patrol_completed:
                    # 触发避障
                    if (distance_to_closest_wall < obstacle_distance_threshold and
                            not avoid_obstacle and not return_to_path):
                        avoid_obstacle = True
                        obstacle_avoidance_time = time.time()
                        torso_pos = data.xpos[torso_id]
                        wall_relative = closest_wall_pos - torso_pos[:2]
                        target_relative = current_target["pos"] - torso_pos[:2]
                        cross_product = np.cross(np.append(wall_relative, 0), np.append(target_relative, 0))[2]
                        turn_direction = -1 if cross_product > 0 else 1
                        dir_name = "左转" if turn_direction == -1 else "右转"

                        # 精简避障提示
                        print(f"\n⚠️  避障：{closest_wall_name} (距离：{distance_to_closest_wall:.2f}m) {dir_name}")

                    # 避障完成，回归导航路径
                    if avoid_obstacle and (time.time() - obstacle_avoidance_time) > obstacle_avoidance_duration:
                        avoid_obstacle = False
                        return_to_path = True
                        return_time = time.time()
                        print(f"✅ 避障完成，回归路径")

                    # 回归完成，继续向目标点移动
                    if return_to_path and (time.time() - return_time) > return_duration:
                        return_to_path = False
                        print(f"✅ 回归路径完成")

                # -------------------------- 5. 步态周期计算 --------------------------
                cycle = elapsed_time % gait_period
                phase = cycle / gait_period

                # -------------------------- 6. 关节控制核心逻辑 --------------------------
                data.ctrl[:model.nu - 6] = 0.0  # 重置机器人控制指令

                if stop_walk or patrol_completed:
                    continue

                elif return_to_path:
                    # 回归导航路径：精准朝向当前目标点
                    return_phase = (time.time() - return_time) / return_duration
                    return_speed = 1.5 * np.cos(return_phase * np.pi)

                    if torso_id != -1:
                        torso_quat = data.xquat[torso_id]
                        robot_yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                                               torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 +
                                               torso_quat[3] ** 2)
                        target_vector = current_target["pos"] - data.xpos[torso_id][:2]
                        target_yaw = np.arctan2(target_vector[1], target_vector[0])
                        yaw_error = target_yaw - robot_yaw
                        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                    # 转向控制
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")

                    if 0 <= abdomen_z_act_id < model.nu - 6:
                        data.ctrl[abdomen_z_act_id] = heading_kp * yaw_error * return_speed
                    if 0 <= hip_z_right_act_id < model.nu - 6:
                        data.ctrl[hip_z_right_act_id] = -yaw_error * return_speed * 0.8
                    if 0 <= hip_z_left_act_id < model.nu - 6:
                        data.ctrl[hip_z_left_act_id] = yaw_error * return_speed * 0.8

                    # 稳定站立
                    for side in ["right", "left"]:
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                        if 0 <= hip_y_act_id < model.nu - 6:
                            data.ctrl[hip_y_act_id] = -0.8
                        if 0 <= knee_act_id < model.nu - 6:
                            data.ctrl[knee_act_id] = 1.1
                        if 0 <= ankle_y_act_id < model.nu - 6:
                            data.ctrl[ankle_y_act_id] = 0.4

                elif avoid_obstacle:
                    # 多障碍避障模式：增强转向稳定性
                    avoid_phase = (time.time() - obstacle_avoidance_time) / obstacle_avoidance_duration
                    turn_speed = 1.6 * np.sin(avoid_phase * np.pi)

                    # 转向控制
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")

                    if 0 <= hip_z_right_act_id < model.nu - 6:
                        data.ctrl[hip_z_right_act_id] = turn_direction * turn_speed * 1.2
                    if 0 <= hip_z_left_act_id < model.nu - 6:
                        data.ctrl[hip_z_left_act_id] = -turn_direction * turn_speed * 1.2
                    if 0 <= abdomen_z_act_id < model.nu - 6:
                        data.ctrl[abdomen_z_act_id] = turn_direction * turn_speed * 2.0

                    # 增强平衡控制
                    for side in ["right", "left"]:
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")
                        if 0 <= hip_y_act_id < model.nu - 6:
                            data.ctrl[hip_y_act_id] = -1.1
                        if 0 <= knee_act_id < model.nu - 6:
                            data.ctrl[knee_act_id] = 1.3
                        if 0 <= ankle_x_act_id < model.nu - 6:
                            data.ctrl[ankle_x_act_id] = 0.3

                else:
                    # 正常导航模式：向当前目标点移动
                    if torso_id != -1:
                        torso_quat = data.xquat[torso_id]
                        robot_yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                                               torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 +
                                               torso_quat[3] ** 2)
                        target_vector = current_target["pos"] - data.xpos[torso_id][:2]
                        target_yaw = np.arctan2(target_vector[1], target_vector[0])
                        yaw_error = target_yaw - robot_yaw
                        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                        # 朝向当前目标点的转向控制
                        abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                        if 0 <= abdomen_z_act_id < model.nu - 6:
                            data.ctrl[abdomen_z_act_id] = heading_kp * yaw_error * 0.12

                    # 腿部步态控制
                    for side, sign in [("right", 1), ("left", -1)]:
                        swing_phase = (phase + 0.5 * sign) % 1.0

                        # 查询电机ID
                        hip_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_x_{side}")
                        hip_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_z_{side}")
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                        ankle_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")

                        # 腿部关节控制
                        if 0 <= hip_x_act_id < model.nu - 6:
                            data.ctrl[hip_x_act_id] = swing_gain * np.sin(2 * np.pi * swing_phase) * forward_speed
                        if 0 <= hip_z_act_id < model.nu - 6:
                            data.ctrl[hip_z_act_id] = stance_gain * np.cos(
                                2 * np.pi * swing_phase) * 0.2 + yaw_error * 0.12
                        if 0 <= hip_y_act_id < model.nu - 6:
                            data.ctrl[hip_y_act_id] = -0.95 * np.sin(2 * np.pi * swing_phase) - 0.45
                        if 0 <= knee_act_id < model.nu - 6:
                            data.ctrl[knee_act_id] = 1.25 * np.sin(2 * np.pi * swing_phase) + 0.75
                        if 0 <= ankle_y_act_id < model.nu - 6:
                            data.ctrl[ankle_y_act_id] = 0.35 * np.cos(2 * np.pi * swing_phase)
                        if 0 <= ankle_x_act_id < model.nu - 6:
                            data.ctrl[ankle_x_act_id] = 0.18 * np.sin(2 * np.pi * swing_phase)

                    # 躯干稳定控制
                    abdomen_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    abdomen_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_y")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_x")

                    if torso_id != -1:
                        pitch = 2 * (torso_quat[1] * torso_quat[3] - torso_quat[0] * torso_quat[2])
                        roll = 2 * (torso_quat[0] * torso_quat[1] + torso_quat[2] * torso_quat[3])

                        if 0 <= abdomen_x_act_id < model.nu - 6:
                            data.ctrl[abdomen_x_act_id] = balance_kp * (torso_roll_target - roll) - balance_kd * \
                                                          data.qvel[abdomen_x_act_id]
                        if 0 <= abdomen_y_act_id < model.nu - 6:
                            data.ctrl[abdomen_y_act_id] = balance_kp * (torso_pitch_target - pitch) - balance_kd * \
                                                          data.qvel[abdomen_y_act_id]

                    # 手臂自然摆动
                    for side, sign in [("right", 1), ("left", -1)]:
                        shoulder1_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder1_{side}")
                        shoulder2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder2_{side}")
                        elbow_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"elbow_{side}")

                        shoulder1_cmd = 0.22 * np.sin(2 * np.pi * (phase + 0.5 * sign))
                        shoulder2_cmd = 0.17 * np.cos(2 * np.pi * (phase + 0.5 * sign))
                        elbow_cmd = -0.42 * np.sin(2 * np.pi * (phase + 0.5 * sign)) - 0.27

                        if 0 <= shoulder1_act_id < model.nu - 6:
                            data.ctrl[shoulder1_act_id] = shoulder1_cmd
                        if 0 <= shoulder2_act_id < model.nu - 6:
                            data.ctrl[shoulder2_act_id] = shoulder2_cmd
                        if 0 <= elbow_act_id < model.nu - 6:
                            data.ctrl[elbow_act_id] = elbow_cmd

                # -------------------------- 7. 仿真推进 --------------------------
                mujoco.mj_step(model, data)
                viewer_instance.sync()

                # -------------------------- 8. 精简状态输出（降低频率） --------------------------
                if torso_id != -1 and (elapsed_time - last_print_time) > 2.0:  # 每2秒输出一次状态
                    last_print_time = elapsed_time

                    if patrol_completed:
                        status = f"完成巡逻！循环：{patrol_cycles}"
                        nav_info = "等待重启"
                        obstacle_info = "—"
                    elif stop_walk:
                        status = "已停止"
                        nav_info = "—"
                        obstacle_info = "—"
                    else:
                        if return_to_path:
                            status = "回归路径中"
                        elif avoid_obstacle:
                            status = f"避障中（{dir_name}）"
                        else:
                            status = f"向{current_target['label']}移动"

                        # 导航信息
                        nav_progress = f"{current_target_index + 1}/{len(PATROL_POINTS)}"
                        nav_info = f"剩余{distance_to_target:.2f}m | 进度{nav_progress}"

                        # 障碍信息
                        if closest_wall_name:
                            obstacle_info = f"{closest_wall_name}：{distance_to_closest_wall:.2f}m"
                        else:
                            obstacle_info = "无"

                    torso_pos = data.xpos[torso_id]
                    print(
                        f"\r🕒 {elapsed_time:.1f}s | 📍 x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f} | 🗺️ {nav_info} | 🛡️ {obstacle_info} | 📊 {status}",
                        end="")

                time.sleep(model.opt.timestep * 2)

        except KeyboardInterrupt:
            print("\n\n🛑 仿真被用户中断")
        except Exception as e:
            print(f"\n\n❌ 运行错误：{e}")
            import traceback
            traceback.print_exc()
        finally:
            # 精简结束统计
            if torso_id != -1:
                print(f"\n\n📋 仿真结束：")
                print(f"   总时间：{elapsed_time:.1f}秒 | 完成循环：{patrol_cycles}")
                print(f"   最后位置：x={data.xpos[torso_id][0]:.2f}, y={data.xpos[torso_id][1]:.2f}")


if __name__ == "__main__":
    model_file = "Robot_move_straight.xml"

    # 确保不生成任何临时文件（通用版，适配所有Mujoco版本）
    control_robot(model_file)