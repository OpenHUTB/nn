#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
规律行驶的无人小车仿真 - 高速版 (目标速度30m/s)
- 移除不合法的 inertia 属性，适配标准 MuJoCo
- 通过物理参数（质量、大阻尼）近似实现平面运动
- 相机跟踪小车，全程可见
- 增强避障算法
- 目标速度调整为 30 m/s，油门范围扩大，PID参数优化
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import math
from collections import deque


class RegularCarSimulation:
    def __init__(self, model_path=None):
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            print(f"从文件加载模型: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(model_path)
        else:
            print("使用高速版内置模型")
            self.model = mujoco.MjModel.from_xml_string(self.get_regular_model_xml())

        self.data = mujoco.MjData(self.model)

        # 控制参数
        self.target_velocity = 30.0  # 目标速度提高到 30 m/s
        self.steering_gain = 0.08
        self.obstacle_threshold = 2.5
        self.side_threshold = 1.5
        self.tight_space_threshold = 1.2

        self.collision_count = 0
        self.max_collisions = 12
        self.simulation_time = 0
        self.total_distance = 0
        self.last_position = None
        self.last_time = time.time()

        self._init_actuator_ids()
        self._init_sensor_ids()

        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()

        self.stats = {
            'total_steps': 0,
            'collisions': 0,
            'resets': 0,
            'avg_speed': 0,
            'max_speed': 0,
            'trajectory_points': [],
            'straight_line_ratio': 0
        }

        self.velocity_history = deque(maxlen=50)
        self.steering_history = deque(maxlen=25)
        self.position_history = deque(maxlen=100)

        self.obstacle_detected = False
        self.last_obstacle_time = 0
        self.obstacle_avoidance_mode = False
        self.obstacle_steering_smoother = deque(maxlen=12)
        self.last_steer_direction = 0
        self.direction_memory_duration = 30
        self.direction_memory_counter = 0

        self.straight_line_target = None
        self.curvature_limit = 0.02
        self.path_following_mode = True
        self.path_points = []

        # 调整PID参数以适应更高速度
        self.velocity_pid = PIDController(kp=0.8, ki=0.02, kd=0.2)   # 原 kp=0.6, kd=0.15
        self.steering_pid = PIDController(kp=0.3, ki=0.005, kd=0.1)  # 转向PID保持不变

        self.is_moving_straight = True
        self.last_turn_time = 0
        self.straight_line_duration = 0
        self.tilt_angle = 0.0

        print(f"模型加载成功，执行器: {self.model.nu}, 传感器: {self.model.nsensor}")
        self._init_path_points()

    def get_regular_model_xml(self):
        """返回最终兼容版XML - 移除不合法的 inertia 属性"""
        return """
<mujoco>
  <compiler angle="radian" inertiafromgeom="true" coordinate="local"/>
  <option timestep="0.002" integrator="RK4" iterations="200" tolerance="1e-10"/>

  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.9 0.9 0.9" ambient="0.3 0.3 0.3"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.2 0.3 0.5" width="512" height="512"/>
    <texture name="grass" type="2d" builtin="checker" rgb1="0.3 0.6 0.3" rgb2="0.2 0.5 0.2" width="512" height="512"/>
    <material name="grass_mat" texture="grass" texrepeat="30 30" reflectance="0.1"/>
    <material name="asphalt" rgba="0.2 0.2 0.2 1" specular="0.4"/>
    <material name="obstacle_red" rgba="0.9 0.1 0.1 1"/>
    <material name="car_body" rgba="0.1 0.4 0.8 1" specular="0.8"/>
    <material name="tire" rgba="0.1 0.1 0.1 1"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="50 50 0.1" material="grass_mat" friction="2.0 0.5 0.2" condim="3"/>
    <geom name="road" type="box" size="25 25 0.08" pos="0 0 0.08" material="asphalt" friction="2.5 0.8 0.3"/>

    <!-- 障碍物 -->
    <body name="obs1" pos="12 0 0.8"><geom type="cylinder" size="0.8 0.8" material="obstacle_red"/></body>
    <body name="obs2" pos="-12 0 0.8"><geom type="cylinder" size="0.8 0.8" material="obstacle_red"/></body>
    <body name="obs3" pos="0 12 1.0"><geom type="box" size="1.0 1.0 1.0" material="obstacle_red"/></body>
    <body name="obs4" pos="0 -12 1.0"><geom type="box" size="1.0 1.0 1.0" material="obstacle_red"/></body>
    <body name="obs5" pos="8 8 0.7"><geom type="cylinder" size="0.7 0.7" material="obstacle_red"/></body>
    <body name="obs6" pos="-8 -8 0.7"><geom type="cylinder" size="0.7 0.7" material="obstacle_red"/></body>
    <body name="obs7" pos="5 0 0.2"><geom type="cylinder" size="0.3 0.2" material="obstacle_red"/></body>
    <body name="obs8" pos="-5 0 0.2"><geom type="cylinder" size="0.3 0.2" material="obstacle_red"/></body>

    <!-- 小车 - 优化物理参数，通过质量和大阻尼抑制倾斜 -->
    <body name="car" pos="0 0 0.3">
      <freejoint name="car_free"/>
      <!-- 车身: 质量20kg，让 MuJoCo 自动计算惯性 -->
      <geom name="car_body" type="box" size="0.5 0.3 0.1" mass="20.0" material="car_body"
            friction="2.0 0.5 0.2" pos="0 0 0"/>

      <!-- 前左轮 -->
      <body name="front_left_wheel" pos="0.3 0.2 -0.1">
        <joint name="fl_steer" type="hinge" axis="0 0 1" limited="true" range="-0.3 0.3"
               damping="0.8" armature="0.1" stiffness="2.0"/>
        <joint name="fl_spin" type="hinge" axis="0 1 0" damping="0.3" armature="0.1"/>
        <geom name="fl_geom" type="cylinder" size="0.12 0.06" euler="1.57 0 0"
              material="tire" friction="3.0 1.5 0.4" mass="1.0"/>
      </body>

      <!-- 前右轮 -->
      <body name="front_right_wheel" pos="0.3 -0.2 -0.1">
        <joint name="fr_steer" type="hinge" axis="0 0 1" limited="true" range="-0.3 0.3"
               damping="0.8" armature="0.1" stiffness="2.0"/>
        <joint name="fr_spin" type="hinge" axis="0 1 0" damping="0.3" armature="0.1"/>
        <geom name="fr_geom" type="cylinder" size="0.12 0.06" euler="1.57 0 0"
              material="tire" friction="3.0 1.5 0.4" mass="1.0"/>
      </body>

      <!-- 后左轮 -->
      <body name="rear_left_wheel" pos="-0.3 0.2 -0.1">
        <joint name="rl_spin" type="hinge" axis="0 1 0" damping="0.5" armature="0.1"/>
        <geom name="rl_geom" type="cylinder" size="0.12 0.06" euler="1.57 0 0"
              material="tire" friction="3.2 1.6 0.4" mass="1.0"/>
      </body>

      <!-- 后右轮 -->
      <body name="rear_right_wheel" pos="-0.3 -0.2 -0.1">
        <joint name="rr_spin" type="hinge" axis="0 1 0" damping="0.5" armature="0.1"/>
        <geom name="rr_geom" type="cylinder" size="0.12 0.06" euler="1.57 0 0"
              material="tire" friction="3.2 1.6 0.4" mass="1.0"/>
      </body>

      <!-- 传感器 -->
      <site name="front_sensor" pos="0.6 0 0.1" size="0.05" rgba="0 1 0 0.6" group="3"/>
      <site name="front_left_sensor" pos="0.5 0.25 0.1" size="0.05" rgba="0 1 0 0.6" group="3"/>
      <site name="front_right_sensor" pos="0.5 -0.25 0.1" size="0.05" rgba="0 1 0 0.6" group="3"/>
      <site name="left_sensor" pos="0.1 0.35 0.1" size="0.05" rgba="0 1 0 0.6" group="3"/>
      <site name="right_sensor" pos="0.1 -0.35 0.1" size="0.05" rgba="0 1 0 0.6" group="3"/>
      <site name="trail_marker" pos="0 0 0.2" size="0.06" rgba="1 1 0 0.8" group="2"/>
    </body>

    <!-- 边界墙 -->
    <geom name="wall_north" type="box" size="40 0.5 1.5" pos="0 22 1.5" rgba="0.6 0.6 0.6 0.8"/>
    <geom name="wall_south" type="box" size="40 0.5 1.5" pos="0 -22 1.5" rgba="0.6 0.6 0.6 0.8"/>
    <geom name="wall_east" type="box" size="0.5 40 1.5" pos="22 0 1.5" rgba="0.6 0.6 0.6 0.8"/>
    <geom name="wall_west" type="box" size="0.5 40 1.5" pos="-22 0 1.5" rgba="0.6 0.6 0.6 0.8"/>
  </worldbody>

  <actuator>
    <motor name="throttle_left" joint="rl_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="throttle_right" joint="rr_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="steer_left" joint="fl_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.3 0.3"/>
    <motor name="steer_right" joint="fr_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.3 0.3"/>
  </actuator>

  <sensor>
    <rangefinder name="front_sensor" site="front_sensor"/>
    <rangefinder name="front_left_sensor" site="front_left_sensor"/>
    <rangefinder name="front_right_sensor" site="front_right_sensor"/>
    <rangefinder name="left_sensor" site="left_sensor"/>
    <rangefinder name="right_sensor" site="right_sensor"/>
    <velocimeter name="car_velocity" site="trail_marker"/>
    <framepos name="car_position" objtype="site" objname="trail_marker"/>
    <accelerometer name="car_accel" site="trail_marker"/>
  </sensor>
</mujoco>
"""

    def _init_path_points(self):
        self.path_points = [
            [0,0], [9,0], [9,9], [0,9], [-9,9], [-9,0], [-9,-9], [0,-9], [9,-9], [9,0], [0,0]
        ]

    def _init_actuator_ids(self):
        self.actuator_ids = {}
        names = ["throttle_left", "throttle_right", "steer_left", "steer_right"]
        for name in names:
            id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if id != -1:
                self.actuator_ids[name] = id
                print(f"执行器 {name} ID: {id}")

    def _init_sensor_ids(self):
        self.sensor_ids = {}
        names = ["front_sensor", "front_left_sensor", "front_right_sensor", "left_sensor", "right_sensor",
                 "car_velocity", "car_position", "car_accel"]
        for name in names:
            id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if id != -1:
                self.sensor_ids[name] = id
        print(f"找到 {len(self.sensor_ids)} 个传感器")

    def get_sensor_data(self):
        data = {}
        for name, id in self.sensor_ids.items():
            if name.endswith("_sensor"):
                data[name] = self.data.sensordata[id]
            elif "velocity" in name:
                data[name] = self.data.sensordata[id:id+3]
            elif "position" in name:
                data[name] = self.data.sensordata[id:id+3]
            elif "accel" in name:
                data[name] = self.data.sensordata[id:id+3]
        return data

    def check_collision(self):
        thresh = 5.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if np.linalg.norm(force[:3]) < thresh:
                continue
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            car_geoms = ["car_body", "fl_geom", "fr_geom", "rl_geom", "rr_geom"]
            if (g1 in car_geoms and g2 not in ["ground","road"]) or (g2 in car_geoms and g1 not in ["ground","road"]):
                return True
        return False

    def get_car_heading(self):
        quat = self.data.qpos[3:7]
        return math.atan2(2.0*(quat[0]*quat[1]+quat[2]*quat[3]),
                          1.0-2.0*(quat[1]*quat[1]+quat[2]*quat[2]))

    def check_tilt_angle(self):
        if 'car_accel' in self.sensor_ids:
            acc = self.data.sensordata[self.sensor_ids['car_accel']:self.sensor_ids['car_accel']+3]
            pitch = math.atan2(acc[0], math.sqrt(acc[1]**2+acc[2]**2))
            roll = math.atan2(acc[1], math.sqrt(acc[0]**2+acc[2]**2))
            self.tilt_angle = max(abs(pitch), abs(roll))
            return self.tilt_angle > math.radians(10)
        return False

    def regular_obstacle_avoidance(self, sensor_data):
        front = sensor_data.get('front_sensor', 10.0)
        front_left = sensor_data.get('front_left_sensor', 10.0)
        front_right = sensor_data.get('front_right_sensor', 10.0)
        left = sensor_data.get('left_sensor', 10.0)
        right = sensor_data.get('right_sensor', 10.0)

        left_space = min(left, front_left)
        right_space = min(right, front_right)
        steer_factor = 0.0
        self.obstacle_detected = False

        if front < self.obstacle_threshold:
            self.obstacle_detected = True
            self.last_obstacle_time = time.time()
            self.obstacle_avoidance_mode = True
            if left_space > self.tight_space_threshold and right_space > self.tight_space_threshold:
                steer_factor = 0.35 if left_space > right_space else -0.35
                self.last_steer_direction = -1 if left_space > right_space else 1
            elif left_space > self.tight_space_threshold:
                steer_factor = 0.35
                self.last_steer_direction = -1
            elif right_space > self.tight_space_threshold:
                steer_factor = -0.35
                self.last_steer_direction = 1
            else:
                steer_factor = 0.0
                self.target_velocity = -0.2
            self.direction_memory_counter = self.direction_memory_duration

        elif left < self.side_threshold or right < self.side_threshold:
            if left < self.side_threshold and right >= self.side_threshold:
                steer_factor = -0.08
                self.last_steer_direction = 1
            elif right < self.side_threshold and left >= self.side_threshold:
                steer_factor = 0.08
                self.last_steer_direction = -1
            else:
                if self.last_steer_direction != 0 and self.direction_memory_counter > 0:
                    steer_factor = 0.05 * self.last_steer_direction
            self.obstacle_detected = True
            self.last_obstacle_time = time.time()
            self.obstacle_avoidance_mode = True
            self.direction_memory_counter = self.direction_memory_duration // 2
        else:
            if self.obstacle_detected and time.time() - self.last_obstacle_time < 1.5:
                if self.last_steer_direction != 0:
                    steer_factor = -0.05 * self.last_steer_direction
            else:
                self.obstacle_detected = False
                self.obstacle_avoidance_mode = False
                self.last_steer_direction = 0
            self.direction_memory_counter = max(0, self.direction_memory_counter - 1)

        base = 0.0
        if self.path_following_mode and self.path_points:
            base = self.path_following_control()
        else:
            base = self.straight_line_control()

        total = base + steer_factor * self.steering_gain
        self.obstacle_steering_smoother.append(total)
        if len(self.obstacle_steering_smoother) > 3:
            total = np.mean(self.obstacle_steering_smoother)
        if len(self.steering_history) > 0:
            total = 0.8 * total + 0.2 * self.steering_history[-1]
        total = np.clip(total, -0.25, 0.25)
        self.steering_history.append(total)
        return total

    def path_following_control(self):
        if not self.path_points:
            return 0.0
        if 'car_position' in self.sensor_ids:
            pos = self.data.sensordata[self.sensor_ids['car_position']:self.sensor_ids['car_position']+3]
            cx, cy = pos[0], pos[1]
        else:
            cx, cy = self.data.qpos[0], self.data.qpos[1]

        dists = [math.hypot(p[0]-cx, p[1]-cy) for p in self.path_points]
        closest = np.argmin(dists)
        lookahead = 4.0
        target = closest
        for i in range(closest, min(closest+15, len(self.path_points))):
            d = math.hypot(self.path_points[i][0]-cx, self.path_points[i][1]-cy)
            if d >= lookahead:
                target = i
                break
        if target == closest:
            target = min(closest+1, len(self.path_points)-1)
        tx, ty = self.path_points[target]
        heading = self.get_car_heading()
        target_heading = math.atan2(ty-cy, tx-cx)
        diff = target_heading - heading
        while diff > math.pi: diff -= 2*math.pi
        while diff < -math.pi: diff += 2*math.pi
        return diff * 0.35

    def straight_line_control(self):
        if self.straight_line_target is None:
            self.straight_line_target = self.get_car_heading()
            self.straight_line_duration = 0
        if self.straight_line_duration > 15.0 and np.random.random() < 0.003:
            self.straight_line_target += np.random.uniform(-0.2, 0.2)
            self.straight_line_duration = 0
        heading = self.get_car_heading()
        diff = self.straight_line_target - heading
        while diff > math.pi: diff -= 2*math.pi
        while diff < -math.pi: diff += 2*math.pi
        steer = diff * 0.25
        if self.get_car_velocity() < 0.5:
            steer *= 0.6
        if abs(steer) < 0.02:
            self.straight_line_duration += self.model.opt.timestep
        else:
            self.straight_line_duration = 0
        return steer

    def regular_velocity_control(self, current_vel, steering):
        base = self.target_velocity
        factor = 1.0 - min(abs(steering)*0.5, 0.2)
        target = base * factor
        if self.obstacle_avoidance_mode:
            if self.target_velocity < 0:
                target = self.target_velocity
            else:
                target *= 0.6
        self.velocity_pid.setpoint = target
        throttle = self.velocity_pid.update(current_vel, dt=self.model.opt.timestep)
        if len(self.velocity_history) > 0:
            last = self.velocity_history[-1]
            max_accel = 2.0
            accel = (target - last) / self.model.opt.timestep
            if abs(accel) > max_accel:
                throttle = np.clip(throttle, -max_accel*8, max_accel*8)
        throttle = np.clip(throttle, -8.0, 25.0)
        self.velocity_history.append(current_vel)
        return throttle

    def get_car_velocity(self):
        if 'car_velocity' in self.sensor_ids:
            v = self.data.sensordata[self.sensor_ids['car_velocity']:self.sensor_ids['car_velocity']+3]
            speed = np.linalg.norm(v[:2])
            if self.velocity_history:
                speed = 0.8*speed + 0.2*self.velocity_history[-1]
            return speed
        if self.last_position is not None:
            dt = time.time() - self.last_time
            if dt > 0:
                return np.linalg.norm(self.data.qpos[:2] - self.last_position) / dt
        return 0.0

    def update_position_history(self):
        if 'car_position' in self.sensor_ids:
            p = self.data.sensordata[self.sensor_ids['car_position']:self.sensor_ids['car_position']+3]
            if self.stats['total_steps'] % 30 == 0:
                self.stats['trajectory_points'].append([p[0], p[1]])
                if len(self.stats['trajectory_points']) > 200:
                    self.stats['trajectory_points'].pop(0)
            self.position_history.append([p[0], p[1]])
        cur = self.data.qpos[:2]
        if self.last_position is not None:
            dist = np.linalg.norm(cur - self.last_position)
            self.total_distance += dist
            if self.stats['total_steps'] > 0:
                self.stats['avg_speed'] = (self.stats['avg_speed']*(self.stats['total_steps']-1) +
                                           dist/self.model.opt.timestep) / self.stats['total_steps']
            if len(self.position_history) >= 3:
                p1 = self.position_history[-3]
                p2 = self.position_history[-2]
                p3 = [cur[0], cur[1]]
                v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
                v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
                if np.linalg.norm(v1) > 0.01 and np.linalg.norm(v2) > 0.01:
                    dot = np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2))
                    angle = math.acos(max(min(dot, 1.0), -1.0))
                    self.is_moving_straight = angle < 0.03
        self.last_position = cur.copy()
        self.last_time = time.time()

    def apply_regular_controls(self, throttle, steering):
        if 'throttle_left' in self.actuator_ids:
            self.data.ctrl[self.actuator_ids['throttle_left']] = throttle
        if 'throttle_right' in self.actuator_ids:
            self.data.ctrl[self.actuator_ids['throttle_right']] = throttle
        if 'steer_left' in self.actuator_ids:
            self.steering_pid.setpoint = steering
            cur = self.data.ctrl[self.actuator_ids['steer_left']]
            self.data.ctrl[self.actuator_ids['steer_left']] = self.steering_pid.update(cur, dt=self.model.opt.timestep)
        if 'steer_right' in self.actuator_ids:
            self.data.ctrl[self.actuator_ids['steer_right']] = self.data.ctrl[self.actuator_ids['steer_left']]

    def reset_simulation(self, reason="碰撞"):
        print(f"\n{'='*60}\n重置仿真! 原因: {reason}\n碰撞: {self.collision_count}, 距离: {self.total_distance:.2f}m, 时间: {self.simulation_time:.2f}s\n{'='*60}\n")
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = 0.0
        self.collision_count = 0
        self.simulation_time = 0
        self.total_distance = 0
        self.tilt_angle = 0.0
        self.last_position = None
        self.last_time = time.time()
        self.velocity_history.clear()
        self.steering_history.clear()
        self.position_history.clear()
        self.obstacle_steering_smoother.clear()
        self.obstacle_detected = False
        self.obstacle_avoidance_mode = False
        self.straight_line_target = None
        self.straight_line_duration = 0
        self.is_moving_straight = True
        self.velocity_pid.reset()
        self.steering_pid.reset()
        if np.random.random() > 0.5:
            angle = np.random.uniform(-math.pi/6, math.pi/6)
            self.data.qpos[3:7] = [math.cos(angle/2), 0, 0, math.sin(angle/2)]
        self.stats['resets'] += 1
        self.stats['trajectory_points'] = []
        mujoco.mj_forward(self.model, self.data)

    def check_stagnation(self):
        if len(self.velocity_history) < 15:
            return False
        return np.mean(list(self.velocity_history)[-15:]) < 0.1

    def check_ground_contact(self):
        return True

    def run_regular_simulation_step(self):
        sensor = self.get_sensor_data()
        vel = self.get_car_velocity()
        self.stats['max_speed'] = max(self.stats['max_speed'], vel)
        if self.check_tilt_angle():
            print(f"倾斜 {math.degrees(self.tilt_angle):.1f}°，减速")
            self.target_velocity = 0.3
        else:
            self.target_velocity = 30.0  # 正常状态目标速度 30 m/s
        steer = self.regular_obstacle_avoidance(sensor)
        throttle = self.regular_velocity_control(vel, steer)
        self.apply_regular_controls(throttle, steer)
        self.update_position_history()
        if self.check_collision():
            self.collision_count += 1
            self.stats['collisions'] += 1
            print(f"碰撞! 总数: {self.collision_count}")
            if self.collision_count >= self.max_collisions:
                self.reset_simulation("达到最大碰撞次数")
                return vel, steer
        if self.check_stagnation() and self.simulation_time > 8.0:
            self.reset_simulation("运动停滞")
            return vel, steer
        mujoco.mj_step(self.model, self.data)
        self.simulation_time += self.model.opt.timestep
        self.stats['total_steps'] += 1
        return vel, steer

    def display_regular_stats(self):
        v = self.get_car_velocity()
        s = "直行" if self.is_moving_straight else "转弯"
        print(f"\n{'='*70}")
        print(f"状态: {s} | 速度: {v:.2f} | 目标: {self.target_velocity:.2f}")
        print(f"步数: {self.stats['total_steps']} | 时间: {self.simulation_time:.2f}s | 距离: {self.total_distance:.2f}m")
        print(f"平均速: {self.stats['avg_speed']:.2f} | 最大速: {self.stats['max_speed']:.2f}")
        print(f"碰撞: {self.stats['collisions']} | 重置: {self.stats['resets']} | 倾斜: {math.degrees(self.tilt_angle):.1f}°")
        print(f"{'='*70}\n")

    def run_regular(self, max_simulation_time=240.0, realtime_factor=1.0):
        print("="*80)
        print("高速版 - 小车仿真 (目标速度30m/s)")
        print("="*80)
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 获取小车 body id 用于跟踪
                car_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "car")
                if car_body_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = car_body_id
                    viewer.cam.distance = 8.0
                    viewer.cam.elevation = -30
                    viewer.cam.azimuth = 90
                else:
                    print("警告: 未找到 car body，使用固定视角")
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = -45
                    viewer.cam.distance = 25.0
                    viewer.cam.lookat[:] = [0,0,1]

                last_reset = time.time()
                last_display = time.time()
                step = 0

                while viewer.is_running() and self.simulation_time < max_simulation_time:
                    start = time.time()
                    vel, steer = self.run_regular_simulation_step()

                    if step % 200 == 0 and 'car_position' in self.sensor_ids:
                        pos = self.data.sensordata[self.sensor_ids['car_position']:self.sensor_ids['car_position']+3]
                        status = "直行" if self.is_moving_straight else "转弯"
                        print(f"步数 {step:6d} | 位置 [{pos[0]:6.1f},{pos[1]:6.1f}] | 速度 {vel:4.1f} | 转向 {steer:5.2f} | {status}")

                    if time.time() - last_reset > 70.0:
                        self.reset_simulation("定期重置")
                        last_reset = time.time()

                    if time.time() - last_display > 15.0:
                        self.display_regular_stats()
                        last_display = time.time()

                    viewer.sync()
                    elapsed = time.time() - start
                    sleep = self.model.opt.timestep / realtime_factor - elapsed
                    if sleep > 0:
                        time.sleep(min(sleep, 0.01))
                    step += 1

                print("\n仿真结束")
                self.display_regular_stats()
                if self.stats['trajectory_points']:
                    pts = self.stats['trajectory_points']
                    print(f"\n轨迹点数: {len(pts)}")
                    if len(pts) > 1:
                        dist = math.hypot(pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1])
                        print(f"起点终点直线距离: {dist:.2f}m")
        except KeyboardInterrupt:
            print("\n用户中断")
            self.display_regular_stats()
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            self.display_regular_stats()


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = 5.0

    def update(self, current, dt=None):
        if dt is None:
            dt = 0.002
        error = self.setpoint - current
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        deriv = (error - self.prev_error) / dt if dt > 0 else 0.0
        out = self.kp*error + self.ki*self.integral + self.kd*deriv
        self.prev_error = error
        return out

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


def main():
    sim = RegularCarSimulation()
    sim.run_regular()


if __name__ == "__main__":
    try:
        import mujoco
        import mujoco.viewer
        print("MuJoCo 已正确安装")
    except ImportError:
        print("错误: MuJoCo 未安装，请运行: pip install mujoco mujoco-viewer")
        sys.exit(1)
    main()
