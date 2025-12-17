#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepMind Humanoid Robot Patrol Simulation
Multi-target patrol with dynamic obstacle avoidance
UTF-8 encoded, no Chinese characters, compatible with GitHub
"""

import mujoco
from mujoco import viewer
import time
import numpy as np
import random
from collections import deque
import os
import sys

# ====================== Disable all log output (cross-version compatible) ======================
os.environ['MUJOCO_QUIET'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

# Optional: Full silent mode (uncomment to disable all console output)
# class QuietStream:
#     def write(self, msg):
#         pass
#     def flush(self):
#         pass
# sys.stdout = QuietStream()
# sys.stderr = QuietStream()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def control_robot(model_path):
    """
    Control DeepMind Humanoid model: Multi-target patrol navigation in dynamic environments
    Features:
    1. Dynamic obstacles (sinusoidal/random walk/circular motion) + priority-based avoidance
    2. 5 fixed patrol points, sequential navigation with loop
    3. No log files, minimal console output
    """
    # Load model and data
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # -------------------------- Multi-target patrol configuration --------------------------
    PATROL_POINTS = [
        {"name": "patrol_target_1", "pos": np.array([0.0, 0.0]), "label": "Start Point"},
        {"name": "patrol_target_2", "pos": np.array([4.0, -2.0]), "label": "Patrol Point 1 (SW)"},
        {"name": "patrol_target_3", "pos": np.array([8.0, 2.0]), "label": "Patrol Point 2 (NE)"},
        {"name": "patrol_target_4", "pos": np.array([10.0, -1.0]), "label": "Patrol Point 3 (NW)"},
        {"name": "patrol_target_5", "pos": np.array([12.0, 0.0]), "label": "Final Point"}
    ]
    target_reached_threshold = 0.8
    current_target_index = 0
    patrol_cycles = 0
    patrol_completed = False
    target_switch_cooldown = 2.0
    last_target_switch_time = 0

    # Initialize patrol point IDs
    patrol_point_ids = {}
    for point in PATROL_POINTS:
        point_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, point["name"])
        patrol_point_ids[point["name"]] = point_id

    # -------------------------- Dynamic obstacle initialization --------------------------
    # Obstacle 1 (wall2): Sinusoidal motion (Y+Z axis)
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

    # Obstacle 2 (wall3): Random walk (X+Y axis)
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

    # Obstacle 3 (wall4): Circular motion (rotation + radial)
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

    # -------------------------- Obstacle detection initialization --------------------------
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

    # -------------------------- Core parameters --------------------------
    # Obstacle avoidance parameters
    avoid_obstacle = False
    obstacle_distance_threshold = 2.0
    obstacle_avoidance_time = 0
    obstacle_avoidance_duration = 5.0
    turn_direction = 0  # -1 = left, 1 = right
    return_to_path = False
    return_time = 0
    return_duration = 4.0
    stop_walk = False
    closest_wall_id = -1
    closest_wall_type = ""
    closest_wall_name = ""
    turn_dir_label = ""  # "Left" / "Right"
    wall_pos_history = {name: deque(maxlen=10) for name in wall_names}

    # Gait control parameters
    gait_period = 2.2
    swing_gain = 0.8
    stance_gain = 0.75
    forward_speed = 0.3
    heading_kp = 90.0

    # Balance control parameters
    torso_pitch_target = 0.0
    torso_roll_target = 0.0
    balance_kp = 110.0
    balance_kd = 18.0

    # Launch viewer
    with viewer.launch_passive(model, data) as viewer_instance:
        print("🤖 Robot multi-target patrol simulation started (no log mode)")

        start_time = time.time()
        last_print_time = 0

        try:
            while True:
                if not viewer_instance.is_running():
                    break

                elapsed_time = time.time() - start_time
                current_target = PATROL_POINTS[current_target_index]

                # -------------------------- 1. Dynamic obstacle control --------------------------
                # Obstacle 1 (wall2): Sinusoidal motion
                if all(id != -1 for id in wall2_motor_ids.values()):
                    wall2_y_target = wall2_params["y_amp"] * np.sin(
                        wall2_params["y_freq"] * elapsed_time + wall2_params["y_phase"])
                    wall2_z_target = wall2_params["z_amp"] * np.sin(
                        wall2_params["z_freq"] * elapsed_time + wall2_params["z_phase"]) + 0.75
                    data.ctrl[wall2_motor_ids["y"]] = (wall2_y_target - data.qpos[wall2_joint_ids["y"]]) * 2.5
                    data.ctrl[wall2_motor_ids["z"]] = (wall2_z_target - data.qpos[wall2_joint_ids["z"]]) * 1.8

                # Obstacle 2 (wall3): Random walk
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

                # Obstacle 3 (wall4): Circular motion
                if all(id != -1 for id in wall4_motor_ids.values()):
                    wall4_rot_target = wall4_params["rot_dir"] * wall4_params["rot_speed"] * elapsed_time
                    wall4_rad_target = wall4_params["rad_base"] + wall4_params["rad_amp"] * np.sin(
                        wall4_params["rad_freq"] * elapsed_time + wall4_params["rad_phase"])
                    data.ctrl[wall4_motor_ids["rot"]] = (wall4_rot_target - data.qpos[wall4_joint_ids["rot"]]) * 1.5
                    data.ctrl[wall4_motor_ids["rad"]] = (wall4_rad_target - data.qpos[wall4_joint_ids["rad"]]) * 2.0

                # -------------------------- 2. Multi-target navigation logic --------------------------
                yaw_error = 0.0
                distance_to_target = float('inf')

                if torso_id != -1 and not stop_walk:
                    torso_pos = data.xpos[torso_id]
                    robot_xy = torso_pos[:2]

                    # Calculate distance to current target
                    target_vector = current_target["pos"] - robot_xy
                    distance_to_target = np.linalg.norm(target_vector)

                    # Check if target is reached
                    if (distance_to_target < target_reached_threshold and
                            not patrol_completed and
                            elapsed_time - last_target_switch_time > target_switch_cooldown):

                        print(f"\n✅ Reached: {current_target['label']} (x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f})")
                        last_target_switch_time = elapsed_time

                        # Switch to next target
                        if current_target_index < len(PATROL_POINTS) - 1:
                            current_target_index += 1
                            print(f"🔄 Navigating to: {PATROL_POINTS[current_target_index]['label']}")
                        else:
                            # Complete one patrol cycle
                            patrol_completed = True
                            patrol_cycles += 1
                            print(f"\n🏁 Completed patrol cycle {patrol_cycles}!")

                            # Reset for next cycle
                            time.sleep(target_switch_cooldown)
                            current_target_index = 0
                            patrol_completed = False
                            last_target_switch_time = time.time()
                            print(f"🔄 Restarting patrol")

                    # Calculate heading error
                    torso_quat = data.xquat[torso_id]
                    robot_yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                                           torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 + torso_quat[
                                               3] ** 2)
                    target_yaw = np.arctan2(target_vector[1], target_vector[0])
                    yaw_error = target_yaw - robot_yaw
                    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                # -------------------------- 3. Obstacle detection & priority sorting --------------------------
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

                        # Predict dynamic obstacle position
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

                # -------------------------- 4. Obstacle avoidance state control --------------------------
                if closest_wall_id != -1 and torso_id != -1 and not stop_walk and not patrol_completed:
                    # Trigger obstacle avoidance
                    if (distance_to_closest_wall < obstacle_distance_threshold and
                            not avoid_obstacle and not return_to_path):
                        avoid_obstacle = True
                        obstacle_avoidance_time = time.time()
                        torso_pos = data.xpos[torso_id]
                        wall_relative = closest_wall_pos - torso_pos[:2]
                        target_relative = current_target["pos"] - torso_pos[:2]
                        cross_product = np.cross(np.append(wall_relative, 0), np.append(target_relative, 0))[2]
                        turn_direction = -1 if cross_product > 0 else 1
                        turn_dir_label = "Left" if turn_direction == -1 else "Right"

                        print(
                            f"\n⚠️  Obstacle avoidance: {closest_wall_name} (distance: {distance_to_closest_wall:.2f}m) Turn {turn_dir_label}")

                    # Complete avoidance, return to path
                    if avoid_obstacle and (time.time() - obstacle_avoidance_time) > obstacle_avoidance_duration:
                        avoid_obstacle = False
                        return_to_path = True
                        return_time = time.time()
                        print(f"✅ Obstacle avoidance completed, returning to path")

                    # Return completed
                    if return_to_path and (time.time() - return_time) > return_duration:
                        return_to_path = False
                        print(f"✅ Path return completed")

                # -------------------------- 5. Gait cycle calculation --------------------------
                cycle = elapsed_time % gait_period
                phase = cycle / gait_period

                # -------------------------- 6. Joint control logic --------------------------
                data.ctrl[:model.nu - 6] = 0.0  # Reset control commands

                if stop_walk or patrol_completed:
                    continue

                elif return_to_path:
                    # Return to navigation path
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

                    # Heading control
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")

                    if 0 <= abdomen_z_act_id < model.nu - 6:
                        data.ctrl[abdomen_z_act_id] = heading_kp * yaw_error * return_speed
                    if 0 <= hip_z_right_act_id < model.nu - 6:
                        data.ctrl[hip_z_right_act_id] = -yaw_error * return_speed * 0.8
                    if 0 <= hip_z_left_act_id < model.nu - 6:
                        data.ctrl[hip_z_left_act_id] = yaw_error * return_speed * 0.8

                    # Stabilize stance
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
                    # Obstacle avoidance mode
                    avoid_phase = (time.time() - obstacle_avoidance_time) / obstacle_avoidance_duration
                    turn_speed = 1.6 * np.sin(avoid_phase * np.pi)

                    # Turn control
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")

                    if 0 <= hip_z_right_act_id < model.nu - 6:
                        data.ctrl[hip_z_right_act_id] = turn_direction * turn_speed * 1.2
                    if 0 <= hip_z_left_act_id < model.nu - 6:
                        data.ctrl[hip_z_left_act_id] = -turn_direction * turn_speed * 1.2
                    if 0 <= abdomen_z_act_id < model.nu - 6:
                        data.ctrl[abdomen_z_act_id] = turn_direction * turn_speed * 2.0

                    # Enhanced balance control
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
                    # Normal navigation mode
                    if torso_id != -1:
                        torso_quat = data.xquat[torso_id]
                        robot_yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                                               torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 +
                                               torso_quat[3] ** 2)
                        target_vector = current_target["pos"] - data.xpos[torso_id][:2]
                        target_yaw = np.arctan2(target_vector[1], target_vector[0])
                        yaw_error = target_yaw - robot_yaw
                        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                        # Heading control
                        abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                        if 0 <= abdomen_z_act_id < model.nu - 6:
                            data.ctrl[abdomen_z_act_id] = heading_kp * yaw_error * 0.12

                    # Leg gait control
                    for side, sign in [("right", 1), ("left", -1)]:
                        swing_phase = (phase + 0.5 * sign) % 1.0

                        # Get actuator IDs
                        hip_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_x_{side}")
                        hip_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_z_{side}")
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                        ankle_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")

                        # Joint control commands
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

                    # Torso balance control
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

                    # Arm swing control
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

                # -------------------------- 7. Simulation step --------------------------
                mujoco.mj_step(model, data)
                viewer_instance.sync()

                # -------------------------- 8. Status output (reduced frequency) --------------------------
                if torso_id != -1 and (elapsed_time - last_print_time) > 2.0:
                    last_print_time = elapsed_time

                    if patrol_completed:
                        status = f"Patrol completed! Cycles: {patrol_cycles}"
                        nav_info = "Waiting to restart"
                        obstacle_info = "—"
                    elif stop_walk:
                        status = "Stopped"
                        nav_info = "—"
                        obstacle_info = "—"
                    else:
                        if return_to_path:
                            status = "Returning to path"
                        elif avoid_obstacle:
                            status = f"Avoiding obstacle (Turn {turn_dir_label})"
                        else:
                            status = f"Moving to {current_target['label']}"

                        # Navigation info
                        nav_progress = f"{current_target_index + 1}/{len(PATROL_POINTS)}"
                        nav_info = f"Remaining: {distance_to_target:.2f}m | Progress: {nav_progress}"

                        # Obstacle info
                        if closest_wall_name:
                            obstacle_info = f"{closest_wall_name}: {distance_to_closest_wall:.2f}m"
                        else:
                            obstacle_info = "None"

                    torso_pos = data.xpos[torso_id]
                    print(
                        f"\r🕒 {elapsed_time:.1f}s | 📍 x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f} | 🗺️ {nav_info} | 🛡️ {obstacle_info} | 📊 {status}",
                        end="")

                time.sleep(model.opt.timestep * 2)

        except KeyboardInterrupt:
            print("\n\n🛑 Simulation interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Final statistics
            if torso_id != -1:
                print(f"\n\n📋 Simulation ended:")
                print(f"   Total time: {elapsed_time:.1f}s | Cycles completed: {patrol_cycles}")
                print(f"   Final position: x={data.xpos[torso_id][0]:.2f}, y={data.xpos[torso_id][1]:.2f}")


if __name__ == "__main__":
    model_file = "Robot_move_straight.xml"
    control_robot(model_file)