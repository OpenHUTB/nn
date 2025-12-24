#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepMind Humanoid Robot Simulation
Dynamic Obstacle Avoidance + Distance-Priority Multi-Target Tracking
Final Stable Version: Completely fixed robot disappearing issue in MuJoCo viewer
GitHub compatible (UTF-8, no log files generated)
"""

import mujoco
from mujoco import viewer
import time
import numpy as np
import random
from collections import deque
import os
import sys

# ====================== Global Configuration ======================
# Suppress MuJoCo logs and set rendering backend
os.environ['MUJOCO_QUIET'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class StablePatrolController:
    def __init__(self, model_path):
        """Initialize robot controller with enhanced balance control and distance-priority target selection"""
        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # -------------------------- Core Simulation Parameters --------------------------
        self.sim_start_time = 0.0
        self.last_print_time = 0.0
        self.target_switch_cooldown = 1.5  # Avoid frequent target switching
        self.last_target_switch_time = 0.0
        self.stabilization_phase = 2.0  # Initial stabilization (no movement for first 2s)

        # -------------------------- Patrol Target Configuration --------------------------
        self.patrol_points = [
            {"name": "patrol_target_1", "pos": np.array([0.0, 0.0]), "label": "Start Point", "update_interval": 10.0},
            {"name": "patrol_target_2", "pos": np.array([2.0, -1.0]), "label": "Patrol Point 1 (SW)",
             "update_interval": 12.0},
            {"name": "patrol_target_3", "pos": np.array([4.0, 1.0]), "label": "Patrol Point 2 (NE)",
             "update_interval": 14.0},
            {"name": "patrol_target_4", "pos": np.array([6.0, -0.5]), "label": "Patrol Point 3 (NW)",
             "update_interval": 11.0},
            {"name": "patrol_target_5", "pos": np.array([8.0, 0.0]), "label": "Final Point", "update_interval": 13.0}
        ]
        self.current_target_idx = 0  # Current tracked target index
        self.patrol_cycles = 0
        self.patrol_completed = False
        self.target_reached_threshold = 1.0  # Threshold to judge target arrival

        # Dynamic target movement parameters
        self.patrol_motor_ids = {}
        self.patrol_joint_ids = {}
        self.patrol_body_ids = {}
        self.last_target_update = {i: 0.0 for i in range(len(self.patrol_points))}
        self.target_movement_range = {"x": [-1.0, 9.0], "y": [-3.0, 3.0]}
        self.target_move_speed = 0.2  # Slow target movement for stability

        # -------------------------- Obstacle Avoidance Parameters --------------------------
        self.valid_wall_names = ["wall1", "wall2", "wall3", "wall4"]
        self.wall_ids = []
        self.wall_names = []
        self.wall_types = {}
        self.wall_pos_history = {}
        self.obstacle_distance_threshold = 2.5
        self.obstacle_avoidance_duration = 6.0
        self.return_to_path_duration = 5.0
        self.wall_priority = {"dynamic1": 4, "dynamic2": 3, "dynamic3": 2, "fixed": 1}

        # Avoidance state variables
        self.avoid_obstacle = False
        self.return_to_path = False
        self.obstacle_avoidance_start = 0.0
        self.return_to_path_start = 0.0
        self.turn_direction = 0  # -1 = left, 1 = right
        self.closest_wall_info = {"name": "", "distance": float('inf'), "type": ""}
        self.turn_dir_label = ""

        # -------------------------- Robot Stability Control Parameters --------------------------
        self.gait_period = 3.0  # Longer gait period for stability
        self.swing_gain = 0.3
        self.stance_gain = 0.4
        self.forward_speed = 0.05  # Reduced speed to prevent flying (fix disappearing issue)
        self.heading_kp = 40.0
        self.balance_kp = 60.0
        self.balance_kd = 30.0
        self.torso_pitch_target = 0.05  # Slight forward tilt for balance
        self.torso_roll_target = 0.0
        self.max_joint_velocity = 0.5  # Limit joint speed to prevent falling
        self.max_ctrl_amplitude = 0.8  # Max control command amplitude (core fix for disappearing)

        # Balance assist parameters
        self.center_of_mass_target = np.array([0.0, 0.0, 0.8])
        self.com_kp = 15.0
        self.foot_height_target = 0.1
        self.foot_height_kp = 20.0

        # -------------------------- Initialize Components --------------------------
        self._init_component_ids()
        self._init_obstacle_history()
        self._set_initial_pose()  # Fixed initial position here

    def _init_component_ids(self):
        """Initialize all MuJoCo component IDs (joints, motors, bodies)"""
        # Torso ID (core robot body)
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if self.torso_id == -1:
            self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "waist_lower")

        # Dynamic patrol target IDs
        for idx, point in enumerate(self.patrol_points):
            self.patrol_motor_ids[idx] = {
                "x": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"patrol{idx + 1}_motor_x"),
                "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"patrol{idx + 1}_motor_y")
            }
            self.patrol_joint_ids[idx] = {
                "x": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"patrol{idx + 1}_slide_x"),
                "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"patrol{idx + 1}_slide_y")
            }
            self.patrol_body_ids[idx] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, point["name"])

        # Dynamic obstacle 2 (wall2) IDs and parameters
        self.wall2_joint_ids = {
            "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall2_slide_y"),
            "z": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall2_slide_z")
        }
        self.wall2_motor_ids = {
            "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall2_motor_y"),
            "z": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall2_motor_z")
        }
        self.wall2_params = {
            "y_amp": 1.5, "y_freq": 0.3, "y_phase": random.uniform(0, 2 * np.pi),
            "z_amp": 0.2, "z_freq": 0.2, "z_phase": random.uniform(0, 2 * np.pi)
        }

        # Dynamic obstacle 3 (wall3) IDs and parameters
        self.wall3_joint_ids = {
            "x": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall3_slide_x"),
            "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall3_slide_y")
        }
        self.wall3_motor_ids = {
            "x": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall3_motor_x"),
            "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall3_motor_y")
        }
        self.wall3_params = {
            "x_base": 4.0, "x_range": 0.5, "x_speed": 0.2,
            "y_base": 0.0, "y_range": 1.0, "y_speed": 0.3,
            "x_dir": random.choice([-1, 1]), "y_dir": random.choice([-1, 1]),
            "x_switch": random.uniform(3.0, 5.0), "y_switch": random.uniform(2.0, 4.0)
        }
        self.wall3_last_switch = {"x": 0.0, "y": 0.0}

        # Dynamic obstacle 4 (wall4) IDs and parameters
        self.wall4_joint_ids = {
            "rot": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall4_rotate"),
            "rad": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall4_radial")
        }
        self.wall4_motor_ids = {
            "rot": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall4_motor_rot"),
            "rad": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall4_motor_rad")
        }
        self.wall4_params = {
            "rot_speed": 0.2, "rot_dir": random.choice([-1, 1]),
            "rad_amp": 0.4, "rad_freq": 0.1, "rad_phase": random.uniform(0, 2 * np.pi),
            "rad_base": 1.0
        }

        # Wall IDs for obstacle detection
        for wall_name in self.valid_wall_names:
            wall_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, wall_name)
            if wall_id != -1:
                self.wall_ids.append(wall_id)
                self.wall_names.append(wall_name)
                if wall_name == "wall1":
                    self.wall_types[wall_name] = "fixed"
                elif wall_name == "wall2":
                    self.wall_types[wall_name] = "dynamic1"
                elif wall_name == "wall3":
                    self.wall_types[wall_name] = "dynamic2"
                elif wall_name == "wall4":
                    self.wall_types[wall_name] = "dynamic3"

        # Joint name mapping
        self.joint_name_mapping = {
            "abdomen_x": "abdomen_x",
            "abdomen_y": "abdomen_y",
            "abdomen_z": "abdomen_z",
            "hip_x_right": "hip_x_right",
            "hip_y_right": "hip_y_right",
            "hip_z_right": "hip_z_right",
            "knee_right": "knee_right",
            "ankle_x_right": "ankle_x_right",
            "ankle_y_right": "ankle_y_right",
            "hip_x_left": "hip_x_left",
            "hip_y_left": "hip_y_left",
            "hip_z_left": "hip_z_left",
            "knee_left": "knee_left",
            "ankle_x_left": "ankle_x_left",
            "ankle_y_left": "ankle_y_left",
            "shoulder1_right": "shoulder1_right",
            "shoulder2_right": "shoulder2_right",
            "elbow_right": "elbow_right",
            "shoulder1_left": "shoulder1_left",
            "shoulder2_left": "shoulder2_left",
            "elbow_left": "elbow_left"
        }

    def _init_obstacle_history(self):
        """Initialize obstacle position history for future position prediction"""
        for wall_name in self.wall_names:
            self.wall_pos_history[wall_name] = deque(maxlen=10)

    def _set_initial_pose(self):
        """强制固定机器人初始位置到视野中心，彻底解决消失问题（核心最终修复）"""
        # 1. 重置仿真数据（清空所有异常状态）
        mujoco.mj_resetData(self.model, self.data)

        # 2. 重置所有控制指令
        self.data.ctrl[:] = 0.0

        # 3. 强制设置机器人根关节初始世界坐标（不依赖关节ID，直接操作qpos前6位）
        # qpos前3位：x/y/z 位置（固定在视野中心）；后3位：roll/pitch/yaw 姿态（无倾斜旋转）
        self.data.qpos[0] = 0.0  # x坐标：视野中心
        self.data.qpos[1] = 0.0  # y坐标：视野中心
        self.data.qpos[2] = 0.8  # z坐标：站立高度，确保在地面上方
        self.data.qpos[3] = 0.0  # roll：无左右倾斜
        self.data.qpos[4] = 0.0  # pitch：无前后倾斜
        self.data.qpos[5] = 0.0  # yaw：无旋转

        # 4. 强制设置机器人关节初始位置（稳定站立姿态，避免倾倒）
        joint_positions = {
            "abdomen_x": 0.0,
            "abdomen_y": 0.0,
            "abdomen_z": 0.0,
            "hip_x_right": 0.0,
            "hip_y_right": -0.2,
            "hip_z_right": 0.0,
            "knee_right": 0.2,
            "ankle_x_right": 0.0,
            "ankle_y_right": 0.1,
            "hip_x_left": 0.0,
            "hip_y_left": -0.2,
            "hip_z_left": 0.0,
            "knee_left": 0.2,
            "ankle_x_left": 0.0,
            "ankle_y_left": 0.1,
            "shoulder1_right": 0.0,
            "shoulder2_right": 0.0,
            "elbow_right": -0.1,
            "shoulder1_left": 0.0,
            "shoulder2_left": 0.0,
            "elbow_left": -0.1
        }
        for joint_name, target_pos in joint_positions.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                # 关节ID对应的qpos索引：根关节占前6位，关节从第7位开始（索引6）
                self.data.qpos[joint_id + 6] = target_pos

        # 5. 强制设置机器人初始速度为0（无移动、无旋转，彻底杜绝初始漂移）
        self.data.qvel[:] = 0.0

    def _clip_control_command(self, cmd):
        """Clip control command to max amplitude to prevent joint over-drive (core fix)"""
        return np.clip(cmd, -self.max_ctrl_amplitude, self.max_ctrl_amplitude)

    def _select_closest_target(self, elapsed_time):
        """Core decision function: select the closest target based on Euclidean distance"""
        # Skip target selection if conditions are not met
        if (elapsed_time < self.stabilization_phase or
                self.avoid_obstacle or
                self.return_to_path or
                (elapsed_time - self.last_target_switch_time < self.target_switch_cooldown)):
            return self.current_target_idx

        # Get robot torso position (XY plane)
        if self.torso_id == -1:
            return self.current_target_idx
        torso_pos = self.data.xpos[self.torso_id][:2]

        # Real-time sync all target positions from MuJoCo simulation
        for idx in range(len(self.patrol_points)):
            if self.patrol_body_ids[idx] != -1:
                target_body_pos = self.data.xpos[self.patrol_body_ids[idx]]
                self.patrol_points[idx]["pos"] = np.array([target_body_pos[0], target_body_pos[1]])

        # Calculate distance to each target
        target_distances = []
        for idx, point in enumerate(self.patrol_points):
            target_pos = point["pos"]
            distance = np.linalg.norm(torso_pos - target_pos)
            target_distances.append((idx, distance, point["label"]))

        # Sort targets by distance (ascending: closest first)
        target_distances.sort(key=lambda x: x[1])
        closest_idx, closest_dist, closest_label = target_distances[0]

        # Switch target if closest target is not current target
        if closest_idx != self.current_target_idx:
            self.last_target_switch_time = elapsed_time
            prev_label = self.patrol_points[self.current_target_idx]["label"]
            self.current_target_idx = closest_idx
            # Print switch info (console only, no log file)
            print(
                f"\n🔀 Target switched (distance priority): {prev_label} → {closest_label} (distance: {closest_dist:.2f}m)")

        return self.current_target_idx

    def _update_dynamic_targets(self, elapsed_time):
        """Update dynamic patrol target positions at specified intervals"""
        if elapsed_time < self.stabilization_phase:
            return

        # Real-time sync with target body positions
        for idx in self.patrol_body_ids:
            if self.patrol_body_ids[idx] != -1:
                target_body_pos = self.data.xpos[self.patrol_body_ids[idx]]
                self.patrol_points[idx]["pos"] = np.array([target_body_pos[0], target_body_pos[1]])

        # Randomly move targets at their update intervals
        for idx, point in enumerate(self.patrol_points):
            if (elapsed_time - self.last_target_update[idx] > point["update_interval"] and
                    not self.avoid_obstacle and not self.return_to_path):

                # Generate new random position within range
                new_x = random.uniform(self.target_movement_range["x"][0], self.target_movement_range["x"][1])
                new_y = random.uniform(self.target_movement_range["y"][0], self.target_movement_range["y"][1])

                # Avoid positions too close to robot
                if self.torso_id != -1:
                    torso_pos = self.data.xpos[self.torso_id][:2]
                    while np.linalg.norm(np.array([new_x, new_y]) - torso_pos) < 4.0:
                        new_x = random.uniform(self.target_movement_range["x"][0], self.target_movement_range["x"][1])
                        new_y = random.uniform(self.target_movement_range["y"][0], self.target_movement_range["y"][1])

                # Control target movement (smooth and slow) with command clipping
                if (self.patrol_motor_ids[idx]["x"] != -1 and
                        self.patrol_motor_ids[idx]["y"] != -1 and
                        self.patrol_joint_ids[idx]["x"] != -1 and
                        self.patrol_joint_ids[idx]["y"] != -1):
                    current_x = self.data.qpos[self.patrol_joint_ids[idx]["x"]]
                    current_y = self.data.qpos[self.patrol_joint_ids[idx]["y"]]

                    x_cmd = (new_x - current_x) * self.target_move_speed * 0.5
                    y_cmd = (new_y - current_y) * self.target_move_speed * 0.5
                    self.data.ctrl[self.patrol_motor_ids[idx]["x"]] = self._clip_control_command(x_cmd)
                    self.data.ctrl[self.patrol_motor_ids[idx]["y"]] = self._clip_control_command(y_cmd)

                # Update target info and timestamp
                self.patrol_points[idx]["pos"] = np.array([new_x, new_y])
                self.last_target_update[idx] = elapsed_time

                if idx == self.current_target_idx:
                    print(f"\n🔄 Target updated: {point['label']} moved to ({new_x:.2f}, {new_y:.2f})")

    def _control_dynamic_obstacles(self, elapsed_time):
        """Control movement of dynamic obstacles (wall2, wall3, wall4) with clipped commands"""
        if elapsed_time < self.stabilization_phase:
            return

        # Wall2: Sinusoidal Y+Z motion
        if all(id != -1 for id in self.wall2_motor_ids.values()):
            wall2_y_target = self.wall2_params["y_amp"] * np.sin(
                self.wall2_params["y_freq"] * elapsed_time + self.wall2_params["y_phase"])
            wall2_z_target = self.wall2_params["z_amp"] * np.sin(
                self.wall2_params["z_freq"] * elapsed_time + self.wall2_params["z_phase"]) + 0.75
            y_cmd = (wall2_y_target - self.data.qpos[self.wall2_joint_ids["y"]]) * 1.0
            z_cmd = (wall2_z_target - self.data.qpos[self.wall2_joint_ids["z"]]) * 0.8
            self.data.ctrl[self.wall2_motor_ids["y"]] = self._clip_control_command(y_cmd)
            self.data.ctrl[self.wall2_motor_ids["z"]] = self._clip_control_command(z_cmd)

        # Wall3: Random walk X+Y
        if all(id != -1 for id in self.wall3_motor_ids.values()):
            if elapsed_time - self.wall3_last_switch["x"] > self.wall3_params["x_switch"]:
                self.wall3_params["x_dir"] *= -1
                self.wall3_params["x_switch"] = random.uniform(3.0, 5.0)
                self.wall3_last_switch["x"] = elapsed_time

            if elapsed_time - self.wall3_last_switch["y"] > self.wall3_params["y_switch"]:
                self.wall3_params["y_dir"] *= -1
                self.wall3_params["y_switch"] = random.uniform(2.0, 4.0)
                self.wall3_last_switch["y"] = elapsed_time

            wall3_x_target = self.wall3_params["x_base"] + self.wall3_params["x_dir"] * self.wall3_params["x_speed"] * (
                        elapsed_time % 8)
            wall3_y_target = self.wall3_params["y_base"] + self.wall3_params["y_dir"] * self.wall3_params["y_speed"] * (
                        elapsed_time % 6)
            wall3_x_target = np.clip(wall3_x_target, 3.5, 4.5)
            wall3_y_target = np.clip(wall3_y_target, -1.0, 1.0)

            x_cmd = (wall3_x_target - self.data.qpos[self.wall3_joint_ids["x"]]) * 1.0
            y_cmd = (wall3_y_target - self.data.qpos[self.wall3_joint_ids["y"]]) * 0.8
            self.data.ctrl[self.wall3_motor_ids["x"]] = self._clip_control_command(x_cmd)
            self.data.ctrl[self.wall3_motor_ids["y"]] = self._clip_control_command(y_cmd)

        # Wall4: Circular motion (rotation + radial slide)
        if all(id != -1 for id in self.wall4_motor_ids.values()):
            wall4_rot_target = self.wall4_params["rot_dir"] * self.wall4_params["rot_speed"] * elapsed_time
            wall4_rad_target = self.wall4_params["rad_base"] + self.wall4_params["rad_amp"] * np.sin(
                self.wall4_params["rad_freq"] * elapsed_time + self.wall4_params["rad_phase"])

            rot_cmd = (wall4_rot_target - self.data.qpos[self.wall4_joint_ids["rot"]]) * 0.8
            rad_cmd = (wall4_rad_target - self.data.qpos[self.wall4_joint_ids["rad"]]) * 0.8
            self.data.ctrl[self.wall4_motor_ids["rot"]] = self._clip_control_command(rot_cmd)
            self.data.ctrl[self.wall4_motor_ids["rad"]] = self._clip_control_command(rad_cmd)

    def _detect_obstacles(self, elapsed_time):
        """Detect obstacles and trigger avoidance if necessary"""
        if elapsed_time < self.stabilization_phase or not self.wall_ids or self.torso_id == -1 or self.patrol_completed:
            return

        torso_pos = self.data.xpos[self.torso_id][:2]
        wall_distances = []

        # Calculate distance to each obstacle (with future position prediction)
        for idx, wall_id in enumerate(self.wall_ids):
            wall_name = self.wall_names[idx]
            wall_type = self.wall_types.get(wall_name, "fixed")

            wall_pos = self.data.xpos[wall_id][:2]
            self.wall_pos_history[wall_name].append(wall_pos)
            current_distance = np.linalg.norm(torso_pos - wall_pos)

            # Predict future obstacle position (for dynamic obstacles)
            predicted_distance = current_distance
            if len(self.wall_pos_history[wall_name]) > 5 and wall_type != "fixed":
                pos_history = np.array(self.wall_pos_history[wall_name])
                velocity = (pos_history[-1] - pos_history[0]) / len(pos_history) * self.model.opt.timestep * 10
                future_pos = wall_pos + velocity * 1.0
                predicted_distance = np.linalg.norm(torso_pos - future_pos)

            # Weight distance by obstacle priority
            weighted_distance = predicted_distance / self.wall_priority[wall_type]
            wall_distances.append({
                "name": wall_name,
                "type": wall_type,
                "id": wall_id,
                "pos": wall_pos,
                "current_dist": current_distance,
                "predicted_dist": predicted_distance,
                "weighted_dist": weighted_distance
            })

        # Find closest obstacle
        if wall_distances:
            wall_distances.sort(key=lambda x: x["weighted_dist"])
            closest_wall = wall_distances[0]

            self.closest_wall_info = {
                "name": closest_wall["name"],
                "distance": closest_wall["predicted_dist"],
                "type": closest_wall["type"],
                "pos": closest_wall["pos"]
            }

            # Trigger obstacle avoidance
            if (closest_wall["predicted_dist"] < self.obstacle_distance_threshold and
                    not self.avoid_obstacle and not self.return_to_path):
                self.avoid_obstacle = True
                self.obstacle_avoidance_start = elapsed_time

                # Determine turn direction (left/right)
                wall_relative = closest_wall["pos"] - torso_pos
                target_vector = self.patrol_points[self.current_target_idx]["pos"] - torso_pos
                cross_product = np.cross(np.append(wall_relative, 0), np.append(target_vector, 0))[2]

                self.turn_direction = -1 if cross_product > 0 else 1
                self.turn_dir_label = "Left" if self.turn_direction == -1 else "Right"

                print(
                    f"\n⚠️  Obstacle detected: {closest_wall['name']} (distance: {closest_wall['predicted_dist']:.2f}m) - Turning {self.turn_dir_label}")

            # Complete obstacle avoidance phase
            if self.avoid_obstacle and (
                    elapsed_time - self.obstacle_avoidance_start) > self.obstacle_avoidance_duration:
                self.avoid_obstacle = False
                self.return_to_path = True
                self.return_to_path_start = elapsed_time
                print(f"✅ Obstacle avoidance completed - returning to path")

            # Complete return to path phase
            if self.return_to_path and (elapsed_time - self.return_to_path_start) > self.return_to_path_duration:
                self.return_to_path = False
                print(
                    f"✅ Back to patrol path - tracking target: {self.patrol_points[self.current_target_idx]['label']}")

    def _get_joint_id(self, joint_name):
        """Get actuator ID for a given joint name"""
        mapped_name = self.joint_name_mapping.get(joint_name, joint_name)
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, mapped_name)

    def _get_joint_vel_id(self, joint_name):
        """Get joint velocity ID for a given joint name"""
        mapped_name = self.joint_name_mapping.get(joint_name, joint_name)
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mapped_name)

    def _limit_joint_velocity(self, joint_name, cmd):
        """Limit joint velocity to prevent unstable movement"""
        vel_id = self._get_joint_vel_id(joint_name)
        if vel_id == -1:
            return cmd

        current_vel = self.data.qvel[vel_id]
        if abs(current_vel) > self.max_joint_velocity:
            cmd = cmd * (self.max_joint_velocity / abs(current_vel)) * 0.5

        return cmd

    def _compute_center_of_mass(self):
        """Calculate robot's center of mass (simplified for stability control)"""
        if self.torso_id == -1:
            return np.array([0.0, 0.0, 0.8])

        # Torso position
        torso_pos = self.data.xpos[self.torso_id]

        # Foot positions
        foot_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_right")
        foot_left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_left")

        foot_right_pos = self.data.xpos[foot_right_id] if foot_right_id != -1 else torso_pos
        foot_left_pos = self.data.xpos[foot_left_id] if foot_left_id != -1 else torso_pos

        # Weighted COM calculation (torso + feet)
        com = torso_pos * 0.7 + (foot_right_pos + foot_left_pos) * 0.15

        return com

    def _maintain_balance(self, elapsed_time):
        """Enhanced balance control to prevent falling with clipped commands"""
        # Torso attitude control (PD control for roll/pitch)
        abdomen_x_id = self._get_joint_id("abdomen_x")
        abdomen_y_id = self._get_joint_id("abdomen_y")
        torso_quat = self.data.xquat[self.torso_id]

        # Calculate roll and pitch from quaternion
        roll = 2 * (torso_quat[0] * torso_quat[1] + torso_quat[2] * torso_quat[3])
        pitch = 2 * (torso_quat[1] * torso_quat[3] - torso_quat[0] * torso_quat[2])

        # Get joint velocity IDs
        abdomen_x_vel_id = self._get_joint_vel_id("abdomen_x")
        abdomen_y_vel_id = self._get_joint_vel_id("abdomen_y")

        # PD control for roll (abdomen_x) with clipping
        if 0 <= abdomen_x_id < self.model.nu and abdomen_x_vel_id != -1:
            roll_error = self.torso_roll_target - roll
            roll_vel = self.data.qvel[abdomen_x_vel_id]
            cmd = self.balance_kp * roll_error * 0.05 - self.balance_kd * roll_vel * 0.1
            cmd = self._limit_joint_velocity("abdomen_x", cmd)
            self.data.ctrl[abdomen_x_id] = self._clip_control_command(cmd)

        # PD control for pitch (abdomen_y) with clipping
        if 0 <= abdomen_y_id < self.model.nu and abdomen_y_vel_id != -1:
            pitch_error = self.torso_pitch_target - pitch
            pitch_vel = self.data.qvel[abdomen_y_vel_id]
            cmd = self.balance_kp * pitch_error * 0.05 - self.balance_kd * pitch_vel * 0.1
            cmd = self._limit_joint_velocity("abdomen_y", cmd)
            self.data.ctrl[abdomen_y_id] = self._clip_control_command(cmd)

        # Center of mass (COM) control
        com = self._compute_center_of_mass()
        com_error = self.center_of_mass_target - com

        # Adjust torso to correct COM error with clipping
        if 0 <= abdomen_x_id < self.model.nu:
            current_cmd = self.data.ctrl[abdomen_x_id]
            cmd = current_cmd + com_error[1] * self.com_kp * 0.01
            self.data.ctrl[abdomen_x_id] = self._clip_control_command(cmd)

        if 0 <= abdomen_y_id < self.model.nu:
            current_cmd = self.data.ctrl[abdomen_y_id]
            cmd = current_cmd + com_error[0] * self.com_kp * 0.01
            self.data.ctrl[abdomen_y_id] = self._clip_control_command(cmd)

        # Leg stability control (slight joint bending for better support) with clipping
        for side in ["right", "left"]:
            hip_y_id = self._get_joint_id(f"hip_y_{side}")
            knee_id = self._get_joint_id(f"knee_{side}")
            ankle_y_id = self._get_joint_id(f"ankle_y_{side}")

            if 0 <= hip_y_id < self.model.nu:
                current_cmd = self.data.ctrl[hip_y_id]
                cmd = current_cmd - 0.1
                self.data.ctrl[hip_y_id] = self._clip_control_command(cmd)

            if 0 <= knee_id < self.model.nu:
                current_cmd = self.data.ctrl[knee_id]
                cmd = current_cmd + 0.1
                self.data.ctrl[knee_id] = self._clip_control_command(cmd)

            if 0 <= ankle_y_id < self.model.nu:
                current_cmd = self.data.ctrl[ankle_y_id]
                cmd = current_cmd + 0.05
                self.data.ctrl[ankle_y_id] = self._clip_control_command(cmd)

    def _control_robot_gait(self, elapsed_time):
        """Control robot gait with distance-priority target tracking and stability (fixed disappearing issue)"""
        if self.torso_id == -1 or self.patrol_completed:
            return

        # Step 1: Select closest target first
        self._select_closest_target(elapsed_time)
        current_target = self.patrol_points[self.current_target_idx]
        torso_pos = self.data.xpos[self.torso_id][:2]
        target_vector = current_target["pos"] - torso_pos
        distance_to_target = np.linalg.norm(target_vector)

        # Step 2: Handle target arrival (no fixed order, auto-select next closest target)
        if (distance_to_target < self.target_reached_threshold and
                not self.patrol_completed and
                elapsed_time - self.last_target_switch_time > self.target_switch_cooldown):
            print(f"\n✅ Reached target: {current_target['label']} (x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f})")
            self.last_target_switch_time = elapsed_time
            print(f"🔍 Scanning for closest next target...")

        # Step 3: Calculate heading error (yaw) with limit to prevent over-rotation
        torso_quat = self.data.xquat[self.torso_id]
        siny_cosp = 2 * (torso_quat[3] * torso_quat[2] + torso_quat[0] * torso_quat[1])
        cosy_cosp = 1 - 2 * (torso_quat[1] ** 2 + torso_quat[2] ** 2)
        robot_yaw = np.arctan2(siny_cosp, cosy_cosp)

        target_yaw = np.arctan2(target_vector[1], target_vector[0])
        yaw_error = target_yaw - robot_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Normalize to [-pi, pi]
        yaw_error = np.clip(yaw_error, -np.pi / 4, np.pi / 4)  # Limit max yaw error to 45 degrees (core fix)

        # Step 4: Reset control commands
        self.data.ctrl[:self.model.nu] = 0.0

        # Step 5: Initial stabilization phase (no movement, only balance)
        if elapsed_time < self.stabilization_phase:
            self._maintain_balance(elapsed_time)
            # Clip all control commands during stabilization
            for i in range(self.model.nu):
                self.data.ctrl[i] = self._clip_control_command(self.data.ctrl[i])
            return

        # Step 6: Return to path mode
        if self.return_to_path:
            return_phase = (elapsed_time - self.return_to_path_start) / self.return_to_path_duration
            return_speed = 0.8 * np.cos(return_phase * np.pi)

            # Heading control with clipping
            abdomen_z_id = self._get_joint_id("abdomen_z")
            hip_z_right_id = self._get_joint_id("hip_z_right")
            hip_z_left_id = self._get_joint_id("hip_z_left")

            if 0 <= abdomen_z_id < self.model.nu:
                cmd = self.heading_kp * yaw_error * return_speed * 0.05
                cmd = self._limit_joint_velocity("abdomen_z", cmd)
                self.data.ctrl[abdomen_z_id] = self._clip_control_command(cmd)
            if 0 <= hip_z_right_id < self.model.nu:
                cmd = -yaw_error * return_speed * 0.3
                cmd = self._limit_joint_velocity("hip_z_right", cmd)
                self.data.ctrl[hip_z_right_id] = self._clip_control_command(cmd)
            if 0 <= hip_z_left_id < self.model.nu:
                cmd = yaw_error * return_speed * 0.3
                cmd = self._limit_joint_velocity("hip_z_left", cmd)
                self.data.ctrl[hip_z_left_id] = self._clip_control_command(cmd)

            self._maintain_balance(elapsed_time)
            # Clip all control commands in return mode
            for i in range(self.model.nu):
                self.data.ctrl[i] = self._clip_control_command(self.data.ctrl[i])
            return

        # Step 7: Obstacle avoidance mode
        if self.avoid_obstacle:
            avoid_phase = (elapsed_time - self.obstacle_avoidance_start) / self.obstacle_avoidance_duration
            turn_speed = 0.8 * np.sin(avoid_phase * np.pi)

            # Turn control with clipping
            hip_z_right_id = self._get_joint_id("hip_z_right")
            hip_z_left_id = self._get_joint_id("hip_z_left")
            abdomen_z_id = self._get_joint_id("abdomen_z")

            if 0 <= hip_z_right_id < self.model.nu:
                cmd = self.turn_direction * turn_speed * 0.5
                cmd = self._limit_joint_velocity("hip_z_right", cmd)
                self.data.ctrl[hip_z_right_id] = self._clip_control_command(cmd)
            if 0 <= hip_z_left_id < self.model.nu:
                cmd = -self.turn_direction * turn_speed * 0.5
                cmd = self._limit_joint_velocity("hip_z_left", cmd)
                self.data.ctrl[hip_z_left_id] = self._clip_control_command(cmd)
            if 0 <= abdomen_z_id < self.model.nu:
                cmd = self.turn_direction * turn_speed * 0.8
                cmd = self._limit_joint_velocity("abdomen_z", cmd)
                self.data.ctrl[abdomen_z_id] = self._clip_control_command(cmd)

            self._maintain_balance(elapsed_time)
            # Clip all control commands in avoidance mode
            for i in range(self.model.nu):
                self.data.ctrl[i] = self._clip_control_command(self.data.ctrl[i])
            return

        # Step 8: Normal patrol mode (closest target tracking)
        # Heading control with clipping
        abdomen_z_id = self._get_joint_id("abdomen_z")
        if 0 <= abdomen_z_id < self.model.nu:
            cmd = self.heading_kp * yaw_error * 0.05
            cmd = self._limit_joint_velocity("abdomen_z", cmd)
            self.data.ctrl[abdomen_z_id] = self._clip_control_command(cmd)

        # Leg gait control with clipping
        cycle = elapsed_time % self.gait_period
        phase = cycle / self.gait_period

        for side, sign in [("right", 1), ("left", -1)]:
            swing_phase = (phase + 0.5 * sign) % 1.0

            # Joint IDs
            hip_x_id = self._get_joint_id(f"hip_x_{side}")
            hip_z_id = self._get_joint_id(f"hip_z_{side}")
            hip_y_id = self._get_joint_id(f"hip_y_{side}")
            knee_id = self._get_joint_id(f"knee_{side}")
            ankle_y_id = self._get_joint_id(f"ankle_y_{side}")
            ankle_x_id = self._get_joint_id(f"ankle_x_{side}")

            # Hip X control with clipping
            if 0 <= hip_x_id < self.model.nu:
                cmd = self.swing_gain * np.sin(2 * np.pi * swing_phase) * self.forward_speed * 0.5
                cmd = self._limit_joint_velocity(f"hip_x_{side}", cmd)
                self.data.ctrl[hip_x_id] = self._clip_control_command(cmd)
            # Hip Z control with clipping
            if 0 <= hip_z_id < self.model.nu:
                cmd = self.stance_gain * np.cos(2 * np.pi * swing_phase) * 0.08 + yaw_error * 0.05
                cmd = self._limit_joint_velocity(f"hip_z_{side}", cmd)
                self.data.ctrl[hip_z_id] = self._clip_control_command(cmd)
            # Hip Y control with clipping
            if 0 <= hip_y_id < self.model.nu:
                cmd = -0.4 * np.sin(2 * np.pi * swing_phase) - 0.2
                cmd = self._limit_joint_velocity(f"hip_y_{side}", cmd)
                self.data.ctrl[hip_y_id] = self._clip_control_command(cmd)
            # Knee control with clipping
            if 0 <= knee_id < self.model.nu:
                cmd = 0.6 * np.sin(2 * np.pi * swing_phase) + 0.4
                cmd = self._limit_joint_velocity(f"knee_{side}", cmd)
                self.data.ctrl[knee_id] = self._clip_control_command(cmd)
            # Ankle Y control with clipping
            if 0 <= ankle_y_id < self.model.nu:
                cmd = 0.15 * np.cos(2 * np.pi * swing_phase)
                cmd = self._limit_joint_velocity(f"ankle_y_{side}", cmd)
                self.data.ctrl[ankle_y_id] = self._clip_control_command(cmd)
            # Ankle X control with clipping
            if 0 <= ankle_x_id < self.model.nu:
                cmd = 0.08 * np.sin(2 * np.pi * swing_phase)
                cmd = self._limit_joint_velocity(f"ankle_x_{side}", cmd)
                self.data.ctrl[ankle_x_id] = self._clip_control_command(cmd)

        # Arm swing for balance with clipping
        for side, sign in [("right", 1), ("left", -1)]:
            shoulder1_id = self._get_joint_id(f"shoulder1_{side}")
            shoulder2_id = self._get_joint_id(f"shoulder2_{side}")
            elbow_id = self._get_joint_id(f"elbow_{side}")

            shoulder1_cmd = 0.08 * np.sin(2 * np.pi * (phase + 0.5 * sign))
            shoulder2_cmd = 0.06 * np.cos(2 * np.pi * (phase + 0.5 * sign))
            elbow_cmd = -0.15 * np.sin(2 * np.pi * (phase + 0.5 * sign)) - 0.1

            if 0 <= shoulder1_id < self.model.nu:
                cmd = self._limit_joint_velocity(f"shoulder1_{side}", shoulder1_cmd)
                self.data.ctrl[shoulder1_id] = self._clip_control_command(cmd)
            if 0 <= shoulder2_id < self.model.nu:
                cmd = self._limit_joint_velocity(f"shoulder2_{side}", shoulder2_cmd)
                self.data.ctrl[shoulder2_id] = self._clip_control_command(cmd)
            if 0 <= elbow_id < self.model.nu:
                cmd = self._limit_joint_velocity(f"elbow_{side}", elbow_cmd)
                self.data.ctrl[elbow_id] = self._clip_control_command(cmd)

        # Maintain balance
        self._maintain_balance(elapsed_time)
        # Final clip of all control commands to ensure no over-drive
        for i in range(self.model.nu):
            self.data.ctrl[i] = self._clip_control_command(self.data.ctrl[i])

    def _print_status(self, elapsed_time):
        """Print real-time robot status (console only, no log file)"""
        if (elapsed_time - self.last_print_time) < 2.0 or self.torso_id == -1:
            return

        self.last_print_time = elapsed_time
        current_target = self.patrol_points[self.current_target_idx]
        torso_pos = self.data.xpos[self.torso_id]
        distance_to_target = np.linalg.norm(current_target["pos"] - torso_pos[:2])

        # Determine current status
        if elapsed_time < self.stabilization_phase:
            status = "Stabilizing (initial phase)"
            nav_info = "Building balance..."
        elif self.patrol_completed:
            status = f"Patrol completed! Cycles: {self.patrol_cycles}"
            nav_info = "Waiting to restart"
        elif self.avoid_obstacle:
            status = f"Avoiding obstacle (Turn {self.turn_dir_label})"
            nav_info = f"Target: {current_target['label']} (Distance: {distance_to_target:.2f}m)"
        elif self.return_to_path:
            status = "Returning to patrol path"
            nav_info = f"Target: {current_target['label']} (Distance: {distance_to_target:.2f}m)"
        else:
            status = f"Tracking {current_target['label']} (distance priority)"
            nav_info = f"Progress: {self.current_target_idx + 1}/{len(self.patrol_points)} | Distance: {distance_to_target:.2f}m"

        # Obstacle info
        obstacle_info = f"{self.closest_wall_info['name']}: {self.closest_wall_info['distance']:.2f}m" if \
            self.closest_wall_info["name"] else "None"

        # COM info
        com = self._compute_center_of_mass()
        com_info = f"COM: z={com[2]:.2f}m"

        # Print status (single line refresh)
        print(
            f"\r🕒 {elapsed_time:.1f}s | 📍 x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f} | {com_info} | 🗺️ {nav_info} | 🛡️ {obstacle_info} | 📊 {status}",
            end=""
        )

    def run_simulation(self):
        """Main simulation loop with optimized camera tracking (彻底解决机器人消失问题)"""
        print("🤖 DeepMind Humanoid Simulation Started (Distance-Priority Target Tracking)")
        print("📌 Features: Enhanced Balance + Dynamic Obstacle Avoidance + Closest Target Selection")
        print("🔍 Press Ctrl+C to stop simulation\n")

        with viewer.launch_passive(self.model, self.data) as viewer_instance:
            self.sim_start_time = time.time()

            # 优化相机参数：增大距离+调整视角，确保机器人始终在视野内
            if self.torso_id != -1:
                viewer_instance.cam.trackbodyid = self.torso_id
                viewer_instance.cam.distance = 10.0  # 增大相机距离，避免模型超出视野
                viewer_instance.cam.elevation = -20  # 增大俯视角，清晰看到机器人
                viewer_instance.cam.azimuth = 90  # 正前方视角，符合直观观察
                viewer_instance.cam.fixedcamid = -1  # 使用跟踪相机，不使用固定相机

            try:
                while viewer_instance.is_running():
                    elapsed_time = time.time() - self.sim_start_time

                    # Core control pipeline
                    self._control_dynamic_obstacles(elapsed_time)
                    self._update_dynamic_targets(elapsed_time)
                    self._detect_obstacles(elapsed_time)
                    self._control_robot_gait(elapsed_time)
                    self._print_status(elapsed_time)

                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    viewer_instance.sync()

                    # 稳定休眠时间，解决渲染不同步
                    time.sleep(0.01)

            except KeyboardInterrupt:
                print("\n\n🛑 Simulation interrupted by user")
            except Exception as e:
                print(f"\n\n❌ Simulation error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if self.torso_id != -1:
                    elapsed_time = time.time() - self.sim_start_time
                    torso_pos = self.data.xpos[self.torso_id]
                    com = self._compute_center_of_mass()
                    print(f"\n\n📋 Simulation Summary:")
                    print(f"   Total runtime: {elapsed_time:.1f} seconds")
                    print(f"   Patrol cycles completed: {self.patrol_cycles}")
                    print(f"   Final position: x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f}")
                    print(f"   Final COM height: {com[2]:.2f}m (stable if > 0.7m)")
                    print("🤖 Simulation ended successfully")


if __name__ == "__main__":
    # Default model path
    model_file = "Robot_move_straight.xml"

    # Override model path with command line argument
    if len(sys.argv) > 1:
        model_file = sys.argv[1]

    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        print(f"ℹ️  Current working directory: {os.getcwd()}")
        sys.exit(1)

    # Run simulation
    try:
        controller = StablePatrolController(model_file)
        controller.run_simulation()
    except Exception as e:
        print(f"\n❌ Failed to start simulation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)