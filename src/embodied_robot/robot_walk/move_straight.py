#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepMind Humanoid Robot Simulation
Dynamic Obstacle Avoidance + Moving Target Tracking
UTF-8 encoded, GitHub compatible
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
# Disable all log output (cross-version compatible)
os.environ['MUJOCO_QUIET'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class DynamicPatrolController:
    def __init__(self, model_path):
        """Initialize robot controller with dynamic target and obstacle support"""
        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # -------------------------- Core Parameters --------------------------
        # Simulation settings
        self.sim_start_time = 0.0
        self.last_print_time = 0.0
        self.target_switch_cooldown = 2.0
        self.last_target_switch_time = 0.0

        # Patrol target configuration (dynamic updatable)
        self.patrol_points = [
            {"name": "patrol_target_1", "pos": np.array([0.0, 0.0]), "label": "Start Point", "update_interval": 8.0},
            {"name": "patrol_target_2", "pos": np.array([4.0, -2.0]), "label": "Patrol Point 1 (SW)",
             "update_interval": 10.0},
            {"name": "patrol_target_3", "pos": np.array([8.0, 2.0]), "label": "Patrol Point 2 (NE)",
             "update_interval": 12.0},
            {"name": "patrol_target_4", "pos": np.array([10.0, -1.0]), "label": "Patrol Point 3 (NW)",
             "update_interval": 9.0},
            {"name": "patrol_target_5", "pos": np.array([12.0, 0.0]), "label": "Final Point", "update_interval": 11.0}
        ]
        self.current_target_idx = 0
        self.patrol_cycles = 0
        self.patrol_completed = False
        self.target_reached_threshold = 0.8

        # Dynamic target control parameters
        self.patrol_motor_ids = {}
        self.patrol_joint_ids = {}
        self.patrol_body_ids = {}
        self.last_target_update = {i: 0.0 for i in range(len(self.patrol_points))}
        self.target_movement_range = {"x": [-2.0, 14.0], "y": [-5.0, 5.0]}
        self.target_move_speed = 0.5

        # Obstacle detection & avoidance parameters
        self.valid_wall_names = ["wall1", "wall2", "wall3", "wall4"]
        self.wall_ids = []
        self.wall_names = []
        self.wall_types = {}
        self.wall_pos_history = {}
        self.obstacle_distance_threshold = 2.0
        self.obstacle_avoidance_duration = 5.0
        self.return_to_path_duration = 4.0
        self.wall_priority = {"dynamic1": 4, "dynamic2": 3, "dynamic3": 2, "fixed": 1}

        # Avoidance state variables
        self.avoid_obstacle = False
        self.return_to_path = False
        self.obstacle_avoidance_start = 0.0
        self.return_to_path_start = 0.0
        self.turn_direction = 0  # -1 = left, 1 = right
        self.closest_wall_info = {"name": "", "distance": float('inf'), "type": ""}
        self.turn_dir_label = ""

        # Robot control parameters
        self.gait_period = 2.2
        self.swing_gain = 0.8
        self.stance_gain = 0.75
        self.forward_speed = 0.3
        self.heading_kp = 90.0
        self.balance_kp = 110.0
        self.balance_kd = 18.0
        self.torso_pitch_target = 0.0
        self.torso_roll_target = 0.0

        # Initialize component IDs
        self._init_component_ids()
        # Initialize obstacle position history
        self._init_obstacle_history()

    def _init_component_ids(self):
        """Initialize all mujoco component IDs (joints, actuators, bodies)"""
        # Torso ID (core robot body)
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        # Dynamic patrol target IDs
        for idx, point in enumerate(self.patrol_points):
            # Motor (actuator) IDs
            self.patrol_motor_ids[idx] = {
                "x": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"patrol{idx + 1}_motor_x"),
                "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"patrol{idx + 1}_motor_y")
            }
            # Joint IDs (slide joints for movement)
            self.patrol_joint_ids[idx] = {
                "x": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"patrol{idx + 1}_slide_x"),
                "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"patrol{idx + 1}_slide_y")
            }
            # Body IDs (target markers)
            self.patrol_body_ids[idx] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, point["name"])

        # Dynamic obstacle IDs
        # Wall2 (sinusoidal motion)
        self.wall2_joint_ids = {
            "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall2_slide_y"),
            "z": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall2_slide_z")
        }
        self.wall2_motor_ids = {
            "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall2_motor_y"),
            "z": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall2_motor_z")
        }
        self.wall2_params = {
            "y_amp": 2.5, "y_freq": 0.7, "y_phase": random.uniform(0, 2 * np.pi),
            "z_amp": 0.4, "z_freq": 0.3, "z_phase": random.uniform(0, 2 * np.pi)
        }

        # Wall3 (random walk)
        self.wall3_joint_ids = {
            "x": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall3_slide_x"),
            "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall3_slide_y")
        }
        self.wall3_motor_ids = {
            "x": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall3_motor_x"),
            "y": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall3_motor_y")
        }
        self.wall3_params = {
            "x_base": 6.0, "x_range": 1.0, "x_speed": 0.4,
            "y_base": 0.0, "y_range": 2.0, "y_speed": 0.6,
            "x_dir": random.choice([-1, 1]), "y_dir": random.choice([-1, 1]),
            "x_switch": random.uniform(2.0, 4.0), "y_switch": random.uniform(1.5, 3.5)
        }
        self.wall3_last_switch = {"x": 0.0, "y": 0.0}

        # Wall4 (circular motion)
        self.wall4_joint_ids = {
            "rot": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall4_rotate"),
            "rad": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wall4_radial")
        }
        self.wall4_motor_ids = {
            "rot": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall4_motor_rot"),
            "rad": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wall4_motor_rad")
        }
        self.wall4_params = {
            "rot_speed": 0.5, "rot_dir": random.choice([-1, 1]),
            "rad_amp": 0.8, "rad_freq": 0.2, "rad_phase": random.uniform(0, 2 * np.pi),
            "rad_base": 1.2
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

    def _init_obstacle_history(self):
        """Initialize obstacle position history for prediction"""
        for wall_name in self.wall_names:
            self.wall_pos_history[wall_name] = deque(maxlen=10)

    def _update_dynamic_targets(self, elapsed_time):
        """Update dynamic patrol target positions in real-time"""
        current_target = self.patrol_points[self.current_target_idx]

        # 1. Real-time sync with target body positions (from XML)
        for idx in self.patrol_body_ids:
            if self.patrol_body_ids[idx] != -1:
                target_body_pos = self.data.xpos[self.patrol_body_ids[idx]]
                self.patrol_points[idx]["pos"] = np.array([target_body_pos[0], target_body_pos[1]])

        # 2. Randomly move targets at specified intervals
        for idx, point in enumerate(self.patrol_points):
            if (elapsed_time - self.last_target_update[idx] > point["update_interval"] and
                    not self.avoid_obstacle and not self.return_to_path):

                # Generate new random position (within safe range)
                new_x = random.uniform(self.target_movement_range["x"][0], self.target_movement_range["x"][1])
                new_y = random.uniform(self.target_movement_range["y"][0], self.target_movement_range["y"][1])

                # Avoid obstacle areas
                if self.torso_id != -1:
                    torso_pos = self.data.xpos[self.torso_id][:2]
                    while np.linalg.norm(np.array([new_x, new_y]) - torso_pos) < 3.0:
                        new_x = random.uniform(self.target_movement_range["x"][0], self.target_movement_range["x"][1])
                        new_y = random.uniform(self.target_movement_range["y"][0], self.target_movement_range["y"][1])

                # Control target movement via actuators
                if (self.patrol_motor_ids[idx]["x"] != -1 and
                        self.patrol_motor_ids[idx]["y"] != -1 and
                        self.patrol_joint_ids[idx]["x"] != -1 and
                        self.patrol_joint_ids[idx]["y"] != -1):
                    # Smooth movement control
                    current_x = self.data.qpos[self.patrol_joint_ids[idx]["x"]]
                    current_y = self.data.qpos[self.patrol_joint_ids[idx]["y"]]

                    self.data.ctrl[self.patrol_motor_ids[idx]["x"]] = (new_x - current_x) * self.target_move_speed
                    self.data.ctrl[self.patrol_motor_ids[idx]["y"]] = (new_y - current_y) * self.target_move_speed

                # Update target position and timestamp
                self.patrol_points[idx]["pos"] = np.array([new_x, new_y])
                self.last_target_update[idx] = elapsed_time

                if idx == self.current_target_idx:
                    print(f"\n🔄 Target updated: {point['label']} moved to ({new_x:.2f}, {new_y:.2f})")

    def _control_dynamic_obstacles(self, elapsed_time):
        """Control dynamic obstacle movements"""
        # Wall2: Sinusoidal Y+Z motion
        if all(id != -1 for id in self.wall2_motor_ids.values()):
            wall2_y_target = self.wall2_params["y_amp"] * np.sin(
                self.wall2_params["y_freq"] * elapsed_time + self.wall2_params["y_phase"])
            wall2_z_target = self.wall2_params["z_amp"] * np.sin(
                self.wall2_params["z_freq"] * elapsed_time + self.wall2_params["z_phase"]) + 0.75
            self.data.ctrl[self.wall2_motor_ids["y"]] = (wall2_y_target - self.data.qpos[
                self.wall2_joint_ids["y"]]) * 2.5
            self.data.ctrl[self.wall2_motor_ids["z"]] = (wall2_z_target - self.data.qpos[
                self.wall2_joint_ids["z"]]) * 1.8

        # Wall3: Random walk X+Y
        if all(id != -1 for id in self.wall3_motor_ids.values()):
            if elapsed_time - self.wall3_last_switch["x"] > self.wall3_params["x_switch"]:
                self.wall3_params["x_dir"] *= -1
                self.wall3_params["x_switch"] = random.uniform(2.0, 4.0)
                self.wall3_last_switch["x"] = elapsed_time

            if elapsed_time - self.wall3_last_switch["y"] > self.wall3_params["y_switch"]:
                self.wall3_params["y_dir"] *= -1
                self.wall3_params["y_switch"] = random.uniform(1.5, 3.5)
                self.wall3_last_switch["y"] = elapsed_time

            wall3_x_target = self.wall3_params["x_base"] + self.wall3_params["x_dir"] * self.wall3_params["x_speed"] * (
                        elapsed_time % 5)
            wall3_y_target = self.wall3_params["y_base"] + self.wall3_params["y_dir"] * self.wall3_params["y_speed"] * (
                        elapsed_time % 4)
            wall3_x_target = np.clip(wall3_x_target, 5.0, 7.0)
            wall3_y_target = np.clip(wall3_y_target, -2.5, 2.5)

            self.data.ctrl[self.wall3_motor_ids["x"]] = (wall3_x_target - self.data.qpos[
                self.wall3_joint_ids["x"]]) * 2.2
            self.data.ctrl[self.wall3_motor_ids["y"]] = (wall3_y_target - self.data.qpos[
                self.wall3_joint_ids["y"]]) * 2.0

        # Wall4: Circular motion
        if all(id != -1 for id in self.wall4_motor_ids.values()):
            wall4_rot_target = self.wall4_params["rot_dir"] * self.wall4_params["rot_speed"] * elapsed_time
            wall4_rad_target = self.wall4_params["rad_base"] + self.wall4_params["rad_amp"] * np.sin(
                self.wall4_params["rad_freq"] * elapsed_time + self.wall4_params["rad_phase"])

            self.data.ctrl[self.wall4_motor_ids["rot"]] = (wall4_rot_target - self.data.qpos[
                self.wall4_joint_ids["rot"]]) * 1.5
            self.data.ctrl[self.wall4_motor_ids["rad"]] = (wall4_rad_target - self.data.qpos[
                self.wall4_joint_ids["rad"]]) * 2.0

    def _detect_obstacles(self, elapsed_time):
        """Detect obstacles and determine avoidance strategy"""
        if not self.wall_ids or self.torso_id == -1 or self.patrol_completed:
            return

        torso_pos = self.data.xpos[self.torso_id][:2]
        wall_distances = []

        # Calculate obstacle distances (with prediction for dynamic obstacles)
        for idx, wall_id in enumerate(self.wall_ids):
            wall_name = self.wall_names[idx]
            wall_type = self.wall_types.get(wall_name, "fixed")

            wall_pos = self.data.xpos[wall_id][:2]
            self.wall_pos_history[wall_name].append(wall_pos)
            current_distance = np.linalg.norm(torso_pos - wall_pos)

            # Predict future position for dynamic obstacles
            predicted_distance = current_distance
            if len(self.wall_pos_history[wall_name]) > 5 and wall_type != "fixed":
                pos_history = np.array(self.wall_pos_history[wall_name])
                velocity = (pos_history[-1] - pos_history[0]) / len(pos_history) * self.model.opt.timestep * 10
                future_pos = wall_pos + velocity * 0.8
                predicted_distance = np.linalg.norm(torso_pos - future_pos)

            # Weight distance by obstacle priority (dynamic > fixed)
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

        # Find closest obstacle (priority-based)
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

                # Determine turn direction (away from obstacle)
                wall_relative = closest_wall["pos"] - torso_pos
                target_vector = self.patrol_points[self.current_target_idx]["pos"] - torso_pos
                cross_product = np.cross(np.append(wall_relative, 0), np.append(target_vector, 0))[2]

                self.turn_direction = -1 if cross_product > 0 else 1
                self.turn_dir_label = "Left" if self.turn_direction == -1 else "Right"

                print(
                    f"\n⚠️  Obstacle detected: {closest_wall['name']} (distance: {closest_wall['predicted_dist']:.2f}m) - Turning {self.turn_dir_label}")

            # Complete avoidance: return to path
            if self.avoid_obstacle and (
                    elapsed_time - self.obstacle_avoidance_start) > self.obstacle_avoidance_duration:
                self.avoid_obstacle = False
                self.return_to_path = True
                self.return_to_path_start = elapsed_time
                print(f"✅ Obstacle avoidance completed - returning to path")

            # Return to path completed
            if self.return_to_path and (elapsed_time - self.return_to_path_start) > self.return_to_path_duration:
                self.return_to_path = False
                print(
                    f"✅ Back to patrol path - tracking target: {self.patrol_points[self.current_target_idx]['label']}")

    def _control_robot_gait(self, elapsed_time):
        """Control robot gait and movement"""
        if self.torso_id == -1 or self.patrol_completed:
            return

        current_target = self.patrol_points[self.current_target_idx]
        torso_pos = self.data.xpos[self.torso_id][:2]
        target_vector = current_target["pos"] - torso_pos
        distance_to_target = np.linalg.norm(target_vector)

        # Check if target is reached
        if (distance_to_target < self.target_reached_threshold and
                not self.patrol_completed and
                elapsed_time - self.last_target_switch_time > self.target_switch_cooldown):

            print(f"\n✅ Reached target: {current_target['label']} (x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f})")
            self.last_target_switch_time = elapsed_time

            # Switch to next target
            if self.current_target_idx < len(self.patrol_points) - 1:
                self.current_target_idx += 1
                print(f"🔄 Now tracking: {self.patrol_points[self.current_target_idx]['label']}")
            else:
                # Complete patrol cycle
                self.patrol_completed = True
                self.patrol_cycles += 1
                print(f"\n🏁 Completed patrol cycle {self.patrol_cycles}!")

                # Reset for next cycle
                time.sleep(self.target_switch_cooldown)
                self.current_target_idx = 0
                self.patrol_completed = False
                self.last_target_switch_time = time.time()
                print(f"🔄 Restarting patrol - tracking start point")

        # Calculate heading error
        torso_quat = self.data.xquat[self.torso_id]
        robot_yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                               torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 + torso_quat[3] ** 2)
        target_yaw = np.arctan2(target_vector[1], target_vector[0])
        yaw_error = target_yaw - robot_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # Reset control commands
        self.data.ctrl[:self.model.nu - 6] = 0.0

        # -------------------------- Movement Control --------------------------
        cycle = elapsed_time % self.gait_period
        phase = cycle / self.gait_period

        if self.return_to_path:
            # Return to path mode
            return_phase = (elapsed_time - self.return_to_path_start) / self.return_to_path_duration
            return_speed = 1.5 * np.cos(return_phase * np.pi)

            # Heading control
            abdomen_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
            hip_z_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
            hip_z_left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")

            if 0 <= abdomen_z_id < self.model.nu - 6:
                self.data.ctrl[abdomen_z_id] = self.heading_kp * yaw_error * return_speed
            if 0 <= hip_z_right_id < self.model.nu - 6:
                self.data.ctrl[hip_z_right_id] = -yaw_error * return_speed * 0.8
            if 0 <= hip_z_left_id < self.model.nu - 6:
                self.data.ctrl[hip_z_left_id] = yaw_error * return_speed * 0.8

            # Stabilize stance
            for side in ["right", "left"]:
                hip_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                ankle_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")

                if 0 <= hip_y_id < self.model.nu - 6:
                    self.data.ctrl[hip_y_id] = -0.8
                if 0 <= knee_id < self.model.nu - 6:
                    self.data.ctrl[knee_id] = 1.1
                if 0 <= ankle_y_id < self.model.nu - 6:
                    self.data.ctrl[ankle_y_id] = 0.4

        elif self.avoid_obstacle:
            # Obstacle avoidance mode
            avoid_phase = (elapsed_time - self.obstacle_avoidance_start) / self.obstacle_avoidance_duration
            turn_speed = 1.6 * np.sin(avoid_phase * np.pi)

            # Turn control
            hip_z_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
            hip_z_left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")
            abdomen_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")

            if 0 <= hip_z_right_id < self.model.nu - 6:
                self.data.ctrl[hip_z_right_id] = self.turn_direction * turn_speed * 1.2
            if 0 <= hip_z_left_id < self.model.nu - 6:
                self.data.ctrl[hip_z_left_id] = -self.turn_direction * turn_speed * 1.2
            if 0 <= abdomen_z_id < self.model.nu - 6:
                self.data.ctrl[abdomen_z_id] = self.turn_direction * turn_speed * 2.0

            # Enhanced balance
            for side in ["right", "left"]:
                hip_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                ankle_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")

                if 0 <= hip_y_id < self.model.nu - 6:
                    self.data.ctrl[hip_y_id] = -1.1
                if 0 <= knee_id < self.model.nu - 6:
                    self.data.ctrl[knee_id] = 1.3
                if 0 <= ankle_x_id < self.model.nu - 6:
                    self.data.ctrl[ankle_x_id] = 0.3

        else:
            # Normal patrol mode
            # Heading control
            abdomen_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
            if 0 <= abdomen_z_id < self.model.nu - 6:
                self.data.ctrl[abdomen_z_id] = self.heading_kp * yaw_error * 0.12

            # Leg gait control
            for side, sign in [("right", 1), ("left", -1)]:
                swing_phase = (phase + 0.5 * sign) % 1.0

                # Joint IDs
                hip_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_x_{side}")
                hip_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_z_{side}")
                hip_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                ankle_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                ankle_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")

                # Gait commands
                if 0 <= hip_x_id < self.model.nu - 6:
                    self.data.ctrl[hip_x_id] = self.swing_gain * np.sin(2 * np.pi * swing_phase) * self.forward_speed
                if 0 <= hip_z_id < self.model.nu - 6:
                    self.data.ctrl[hip_z_id] = self.stance_gain * np.cos(
                        2 * np.pi * swing_phase) * 0.2 + yaw_error * 0.12
                if 0 <= hip_y_id < self.model.nu - 6:
                    self.data.ctrl[hip_y_id] = -0.95 * np.sin(2 * np.pi * swing_phase) - 0.45
                if 0 <= knee_id < self.model.nu - 6:
                    self.data.ctrl[knee_id] = 1.25 * np.sin(2 * np.pi * swing_phase) + 0.75
                if 0 <= ankle_y_id < self.model.nu - 6:
                    self.data.ctrl[ankle_y_id] = 0.35 * np.cos(2 * np.pi * swing_phase)
                if 0 <= ankle_x_id < self.model.nu - 6:
                    self.data.ctrl[ankle_x_id] = 0.18 * np.sin(2 * np.pi * swing_phase)

            # Torso balance control
            abdomen_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_x")
            abdomen_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_y")

            pitch = 2 * (torso_quat[1] * torso_quat[3] - torso_quat[0] * torso_quat[2])
            roll = 2 * (torso_quat[0] * torso_quat[1] + torso_quat[2] * torso_quat[3])

            if 0 <= abdomen_x_id < self.model.nu - 6:
                self.data.ctrl[abdomen_x_id] = self.balance_kp * (self.torso_roll_target - roll) - self.balance_kd * \
                                               self.data.qvel[abdomen_x_id]
            if 0 <= abdomen_y_id < self.model.nu - 6:
                self.data.ctrl[abdomen_y_id] = self.balance_kp * (self.torso_pitch_target - pitch) - self.balance_kd * \
                                               self.data.qvel[abdomen_y_id]

            # Arm swing for balance
            for side, sign in [("right", 1), ("left", -1)]:
                shoulder1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder1_{side}")
                shoulder2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder2_{side}")
                elbow_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"elbow_{side}")

                shoulder1_cmd = 0.22 * np.sin(2 * np.pi * (phase + 0.5 * sign))
                shoulder2_cmd = 0.17 * np.cos(2 * np.pi * (phase + 0.5 * sign))
                elbow_cmd = -0.42 * np.sin(2 * np.pi * (phase + 0.5 * sign)) - 0.27

                if 0 <= shoulder1_id < self.model.nu - 6:
                    self.data.ctrl[shoulder1_id] = shoulder1_cmd
                if 0 <= shoulder2_id < self.model.nu - 6:
                    self.data.ctrl[shoulder2_id] = shoulder2_cmd
                if 0 <= elbow_id < self.model.nu - 6:
                    self.data.ctrl[elbow_id] = elbow_cmd

    def _print_status(self, elapsed_time):
        """Print robot status (reduced frequency to avoid clutter)"""
        if (elapsed_time - self.last_print_time) < 2.0 or self.torso_id == -1:
            return

        self.last_print_time = elapsed_time
        current_target = self.patrol_points[self.current_target_idx]
        torso_pos = self.data.xpos[self.torso_id]
        distance_to_target = np.linalg.norm(current_target["pos"] - torso_pos[:2])

        # Status text
        if self.patrol_completed:
            status = f"Patrol completed! Cycles: {self.patrol_cycles}"
            nav_info = "Waiting to restart"
        elif self.avoid_obstacle:
            status = f"Avoiding obstacle (Turn {self.turn_dir_label})"
            nav_info = f"Target: {current_target['label']} (Distance: {distance_to_target:.2f}m)"
        elif self.return_to_path:
            status = "Returning to patrol path"
            nav_info = f"Target: {current_target['label']} (Distance: {distance_to_target:.2f}m)"
        else:
            status = f"Tracking {current_target['label']}"
            nav_info = f"Progress: {self.current_target_idx + 1}/{len(self.patrol_points)} | Distance: {distance_to_target:.2f}m"

        # Obstacle info
        obstacle_info = f"{self.closest_wall_info['name']}: {self.closest_wall_info['distance']:.2f}m" if \
        self.closest_wall_info["name"] else "None"

        # Print status line
        print(
            f"\r🕒 {elapsed_time:.1f}s | 📍 x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f} | 🗺️ {nav_info} | 🛡️ {obstacle_info} | 📊 {status}",
            end=""
        )

    def run_simulation(self):
        """Main simulation loop"""
        print("🤖 DeepMind Humanoid Simulation Started")
        print("📌 Features: Dynamic Obstacle Avoidance + Moving Target Tracking")
        print("🔍 Press Ctrl+C to stop simulation\n")

        with viewer.launch_passive(self.model, self.data) as viewer_instance:
            self.sim_start_time = time.time()

            try:
                while viewer_instance.is_running():
                    elapsed_time = time.time() - self.sim_start_time

                    # Core control sequence
                    self._control_dynamic_obstacles(elapsed_time)  # Step 1: Move obstacles
                    self._update_dynamic_targets(elapsed_time)  # Step 2: Update targets
                    self._detect_obstacles(elapsed_time)  # Step 3: Detect obstacles
                    self._control_robot_gait(elapsed_time)  # Step 4: Control robot
                    self._print_status(elapsed_time)  # Step 5: Print status

                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    viewer_instance.sync()
                    time.sleep(self.model.opt.timestep * 2)

            except KeyboardInterrupt:
                print("\n\n🛑 Simulation interrupted by user")
            except Exception as e:
                print(f"\n\n❌ Simulation error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Final statistics
                if self.torso_id != -1:
                    elapsed_time = time.time() - self.sim_start_time
                    torso_pos = self.data.xpos[self.torso_id]
                    print(f"\n\n📋 Simulation Summary:")
                    print(f"   Total runtime: {elapsed_time:.1f} seconds")
                    print(f"   Patrol cycles completed: {self.patrol_cycles}")
                    print(f"   Final position: x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f}")
                    print("🤖 Simulation ended successfully")


if __name__ == "__main__":
    # Set default model path (relative path for GitHub compatibility)
    model_file = "Robot_move_straight.xml"

    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        sys.exit(1)

    # Initialize and run controller
    controller = DynamicPatrolController(model_file)
    controller.run_simulation()