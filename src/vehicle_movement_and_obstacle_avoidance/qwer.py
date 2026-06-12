#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多车协同避让仿真（道路场景）- 右侧车道行驶版（优化版）
- 道路场景不变，两车目标点位于右侧车道 (2.5, ±20)
- 安全距离缩小，速度降至1.5m/s
- 车辆相遇时依靠人工势场法避让（已移除两车之间的斥力）
- 修正车辆初始朝向，优化速度获取，平滑油门控制
- 【修复】初始化时正确设置航向角，避免启动时误转向
"""

import mujoco
import mujoco.viewer
import argparse
import csv
import numpy as np
import threading
import time
import math
from collections import deque
from pathlib import Path

# ==================== 共享数据结构 ====================
car_states = {}
state_lock = threading.Lock()
simulation_running = True
step_barrier = None


class SimulationMetrics:
    """Collect per-step vehicle metrics and export a CSV report."""

    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.rows = []
        self.min_distance = float("inf")
        self.emergency_events = {1: 0, 2: 0}
        self.last_emergency = {1: False, 2: False}

    def record(self, step, sim_time, car1_ctrl, car2_ctrl):
        with state_lock:
            state1 = car_states.get(1, {}).copy()
            state2 = car_states.get(2, {}).copy()

        pos1 = state1.get("position", (0.0, 0.0))
        pos2 = state2.get("position", (0.0, 0.0))
        distance = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
        self.min_distance = min(self.min_distance, distance)

        for ctrl in (car1_ctrl, car2_ctrl):
            if ctrl.emergency_brake and not self.last_emergency[ctrl.car_id]:
                self.emergency_events[ctrl.car_id] += 1
            self.last_emergency[ctrl.car_id] = ctrl.emergency_brake

        self.rows.append({
            "step": step,
            "time_sec": f"{sim_time:.3f}",
            "car1_x": f"{pos1[0]:.4f}",
            "car1_y": f"{pos1[1]:.4f}",
            "car1_velocity": f"{state1.get('velocity', 0.0):.4f}",
            "car1_throttle": f"{car1_ctrl.last_throttle:.4f}",
            "car1_steer": f"{car1_ctrl.last_steer:.4f}",
            "car1_yield": int(car1_ctrl.yield_flag),
            "car1_emergency": int(car1_ctrl.emergency_brake),
            "car1_avoidance": int(car1_ctrl.avoidance_active),
            "car1_nearest_obstacle": f"{car1_ctrl.min_obstacle_distance:.4f}",
            "car2_x": f"{pos2[0]:.4f}",
            "car2_y": f"{pos2[1]:.4f}",
            "car2_velocity": f"{state2.get('velocity', 0.0):.4f}",
            "car2_throttle": f"{car2_ctrl.last_throttle:.4f}",
            "car2_steer": f"{car2_ctrl.last_steer:.4f}",
            "car2_yield": int(car2_ctrl.yield_flag),
            "car2_emergency": int(car2_ctrl.emergency_brake),
            "car2_avoidance": int(car2_ctrl.avoidance_active),
            "car2_nearest_obstacle": f"{car2_ctrl.min_obstacle_distance:.4f}",
            "distance_between_cars": f"{distance:.4f}",
        })

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "simulation_metrics.csv"
        if not self.rows:
            return csv_path

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        return csv_path

    def summary(self, car1_ctrl, car2_ctrl):
        if not self.rows:
            return {
                "steps": 0,
                "min_distance": 0.0,
                "car1_avg_velocity": 0.0,
                "car2_avg_velocity": 0.0,
                "car1_emergency_events": 0,
                "car2_emergency_events": 0,
            }

        car1_avg = sum(float(row["car1_velocity"]) for row in self.rows) / len(self.rows)
        car2_avg = sum(float(row["car2_velocity"]) for row in self.rows) / len(self.rows)
        return {
            "steps": len(self.rows),
            "min_distance": self.min_distance,
            "car1_avg_velocity": car1_avg,
            "car2_avg_velocity": car2_avg,
            "car1_emergency_events": self.emergency_events[1],
            "car2_emergency_events": self.emergency_events[2],
            "car1_goal_distance": car1_ctrl.distance_to_goal(),
            "car2_goal_distance": car2_ctrl.distance_to_goal(),
        }


# ==================== PID控制器（用于速度） ====================
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
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        self.prev_error = error
        return out

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


# ==================== 小车控制器类（人工势场法） ====================
class CarController:
    def __init__(self, car_id, model, data, actuator_ids, sensor_ids, goal_pos):
        self.car_id = car_id
        self.model = model
        self.data = data
        self.actuator_ids = actuator_ids
        self.sensor_ids = sensor_ids
        self.goal_pos = goal_pos  # 固定目标点 (x, y)
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"car{car_id}")

        # 控制参数 - 速度减慢
        self.target_velocity = 1.5

        # 人工势场法参数 - 优化后的值
        self.att_gain = 1.0
        self.rep_car_gain = 12.0         # 动态车辆斥力增益
        self.rep_wall_gain = 1.0          # 减小边界斥力强度
        self.rep_distance_threshold = 0.8 # 缩小作用范围
        self.safe_distance = 3.0          # 车辆间安全距离
        self.emergency_distance = 0.45
        self.avoidance_distance = 3.5
        self.max_steer = 0.4

        # 传感器方向（用于调试，未实际使用）
        self.sensor_angles = {
            'front': 0.0,
            'front_left': math.radians(45),
            'front_right': -math.radians(45),
            'left': math.radians(90),
            'right': -math.radians(90)
        }

        # 道路边界
        self.road_left = -4.0
        self.road_right = 4.0
        self.road_bottom = -20.0
        self.road_top = 20.0

        # 防侧翻参数
        self.MAX_LATERAL_ACC = 5.0
        self.WHEELBASE = 0.6
        self.MAX_STEER_CHANGE = 0.05
        self.MAX_THROTTLE_CHANGE = 0.5   # 油门变化率限制

        # PID 速度控制器
        self.velocity_pid = PIDController(kp=0.5, ki=0.01, kd=0.1)

        # 历史数据
        self.velocity_history = deque(maxlen=50)
        self.steering_history = deque(maxlen=25)
        self.last_velocity_computed = 0.0

        # 协商状态
        self.yield_flag = False
        self.near_car = False
        self.emergency_brake = False
        self.avoidance_active = False
        self.min_obstacle_distance = 10.0

        # 当前控制输出
        self.last_throttle = 0.0
        self.last_steer = 0.0

        # 速度索引（用于直接从qvel获取全局速度）
        # freejoint 速度维度：前3为线速度(x,y,z)，后3为角速度
        self.vel_start = 0 if car_id == 1 else 7  # car1从0开始，car2从7开始

    def distance_to_goal(self):
        state = car_states.get(self.car_id, {})
        pos = state.get("position", (0.0, 0.0))
        return math.hypot(self.goal_pos[0] - pos[0], self.goal_pos[1] - pos[1])

    def get_sensor_data(self):
        """读取传感器数据，返回字典，优先使用全局速度"""
        data = {}
        # 读取距离传感器
        for key in ['front', 'front_left', 'front_right', 'left', 'right']:
            if key in self.sensor_ids:
                sensor_val = self.data.sensordata[self.sensor_ids[key]]
                # rangefinder 返回单个距离值
                if isinstance(sensor_val, (list, tuple, np.ndarray)) and len(sensor_val) > 0:
                    dist = float(sensor_val[0])
                    # 过滤无效值（MuJoCo 中 rangefinder 可能返回 -1 表示无检测）
                    data[key] = dist if dist > 0 else 10.0
                elif isinstance(sensor_val, (int, float)):
                    dist = float(sensor_val)
                    data[key] = dist if dist > 0 else 10.0
                else:
                    data[key] = 10.0

        # 直接从 qvel 获取全局速度（更准确）
        vx = self.data.qvel[self.vel_start]
        vy = self.data.qvel[self.vel_start + 1]
        self.last_velocity_computed = math.hypot(vx, vy)

        data['velocity'] = self.last_velocity_computed
        return data

    def update_state(self):
        """更新本车状态到共享字典"""
        start = 0 if self.car_id == 1 else 7
        pos_xy = (self.data.xpos[self.body_id][0], self.data.xpos[self.body_id][1])
        quat = self.data.xquat[self.body_id]
        # 计算航向角（绕z轴）
        heading = math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                             1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]))
        with state_lock:
            car_states[self.car_id] = {
                'position': pos_xy,
                'heading': heading,
                'velocity': self.get_sensor_data().get('velocity', 0),
                'timestamp': time.time()
            }

    def compute_apf_steering(self, my_sensor, other_cars):
        """
        人工势场法计算期望转向角，融合道路边界和动态车辆斥力。
        """
        my_state = car_states.get(self.car_id, {})
        my_pos = my_state.get('position', (0, 0))
        my_heading = my_state.get('heading', 0)

        # 引力（指向固定目标点）
        F_att = np.array([self.goal_pos[0] - my_pos[0], self.goal_pos[1] - my_pos[1]])
        F_att = self.att_gain * F_att

        # 斥力
        F_rep_total = np.array([0.0, 0.0])

        self.min_obstacle_distance = 10.0
        self.avoidance_active = False

        for other_id, state in other_cars.items():
            if other_id == self.car_id:
                continue

            other_pos = state.get('position', (0, 0))
            dx = other_pos[0] - my_pos[0]
            dy = other_pos[1] - my_pos[1]
            dist = math.hypot(dx, dy)
            self.min_obstacle_distance = min(self.min_obstacle_distance, dist)

            if 0.1 < dist < self.avoidance_distance:
                dir_vec = np.array([-dx, -dy]) / dist
                closing_speed = max(0.0, my_sensor.get('velocity', 0.0) - state.get('velocity', 0.0))
                distance_scale = (self.avoidance_distance - dist) / self.avoidance_distance
                mag = self.rep_car_gain * distance_scale * (1.0 + 0.35 * closing_speed) / max(dist, 0.2)
                F_rep_total += mag * dir_vec
                self.avoidance_active = True

        # 道路边界斥力
        # 左边界
        if my_pos[0] - self.road_left < self.rep_distance_threshold:
            dist = my_pos[0] - self.road_left
            if dist > 0:
                mag = self.rep_wall_gain / (dist * dist)
                F_rep_total += mag * np.array([1.0, 0.0])
        # 右边界
        if self.road_right - my_pos[0] < self.rep_distance_threshold:
            dist = self.road_right - my_pos[0]
            if dist > 0:
                mag = self.rep_wall_gain / (dist * dist)
                F_rep_total += mag * np.array([-1.0, 0.0])
        # 下边界
        if my_pos[1] - self.road_bottom < self.rep_distance_threshold:
            dist = my_pos[1] - self.road_bottom
            if dist > 0:
                mag = self.rep_wall_gain / (dist * dist)
                F_rep_total += mag * np.array([0.0, 1.0])
        # 上边界
        if self.road_top - my_pos[1] < self.rep_distance_threshold:
            dist = self.road_top - my_pos[1]
            if dist > 0:
                mag = self.rep_wall_gain / (dist * dist)
                F_rep_total += mag * np.array([0.0, -1.0])

        # 合力
        F_total = F_att + F_rep_total

        if np.linalg.norm(F_total) < 1e-6:
            desired_angle = my_heading
        else:
            desired_angle = math.atan2(F_total[1], F_total[0])

        angle_diff = desired_angle - my_heading
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        steer = np.clip(angle_diff * 0.8, -self.max_steer, self.max_steer)

        # 更新协商标志
        self.near_car = any(
            other_id != self.car_id and
            math.hypot(other_cars[other_id]['position'][0] - my_pos[0],
                       other_cars[other_id]['position'][1] - my_pos[1]) < self.safe_distance
            for other_id in other_cars
        )
        # 简单的让行规则：ID大的让行
        self.yield_flag = self.near_car and (self.car_id > min(other_cars.keys()))

        # 紧急刹车 - 缩小距离阈值
        front_dist = my_sensor.get('front', 10.0)
        self.emergency_brake = front_dist < self.emergency_distance or any(
            other_id != self.car_id and
            math.hypot(other_cars[other_id]['position'][0] - my_pos[0],
                       other_cars[other_id]['position'][1] - my_pos[1]) < self.emergency_distance
            for other_id in other_cars
        )

        return steer

    def compute_throttle(self, current_vel, steering):
        target = self.target_velocity
        factor = 1.0 - min(abs(steering) * 0.8, 0.4)
        target *= factor
        if self.yield_flag:
            target *= 0.2
        if self.emergency_brake:
            target = 0.0
        if abs(steering) > 1e-6:
            v_safe = math.sqrt(self.MAX_LATERAL_ACC * self.WHEELBASE / abs(math.tan(steering)))
            target = min(target, v_safe)
        if target < 0.5 and not self.emergency_brake:
            target = 0.5

        self.velocity_pid.setpoint = target
        throttle = self.velocity_pid.update(current_vel, dt=self.model.opt.timestep)

        # 限制油门变化率，防止急加速/减速
        throttle = np.clip(throttle,
                           self.last_throttle - self.MAX_THROTTLE_CHANGE,
                           self.last_throttle + self.MAX_THROTTLE_CHANGE)
        throttle = np.clip(throttle, -1.0, 1.0)

        self.velocity_history.append(current_vel)
        return throttle

    def control_step(self):
        my_sensor = self.get_sensor_data()
        with state_lock:
            other_cars = car_states.copy()
        steer = self.compute_apf_steering(my_sensor, other_cars)
        if len(self.steering_history) > 0:
            prev = self.steering_history[-1]
            steer = np.clip(steer, prev - self.MAX_STEER_CHANGE, prev + self.MAX_STEER_CHANGE)
        self.steering_history.append(steer)

        vel = my_sensor.get('velocity', 0)
        throttle = self.compute_throttle(vel, steer)

        self.last_throttle = throttle
        self.last_steer = steer
        self.update_state()


# ==================== 构建道路场景XML（修正车辆朝向） ====================
def create_road_xml():
    return """
<mujoco>
  <compiler angle="radian" inertiafromgeom="true" coordinate="local"/>
  <option timestep="0.002" integrator="RK4"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.2 0.3 0.5" width="512" height="512"/>
    <texture name="road" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="64" height="64"/>
    <material name="road_mat" texture="road" texrepeat="1 40" reflectance="0.2"/>
    <material name="lane_line" rgba="1 1 0 1"/>
    <material name="car1_mat" rgba="0.1 0.4 0.8 1" specular="0.8"/>
    <material name="car2_mat" rgba="0.8 0.2 0.2 1" specular="0.8"/>
    <material name="tire" rgba="0.1 0.1 0.1 1"/>
  </asset>

  <worldbody>
    <!-- 路面 -->
    <geom name="road" type="box" size="4 20 0.05" pos="0 0 -0.05" material="road_mat"/>

    <!-- 车道分隔线（中间虚线） -->
    <geom name="center_line1" type="box" size="0.1 2 0.01" pos="0 -5 0.05" material="lane_line"/>
    <geom name="center_line2" type="box" size="0.1 2 0.01" pos="0 5 0.05" material="lane_line"/>
    <geom name="center_line3" type="box" size="0.1 2 0.01" pos="0 15 0.05" material="lane_line"/>
    <geom name="center_line4" type="box" size="0.1 2 0.01" pos="0 -15 0.05" material="lane_line"/>

    <!-- 路沿（边界墙） -->
    <geom name="left_curb" type="box" size="0.2 20 0.3" pos="-4 0 0.25" rgba="0.5 0.5 0.5 1"/>
    <geom name="right_curb" type="box" size="0.2 20 0.3" pos="4 0 0.25" rgba="0.5 0.5 0.5 1"/>
    <geom name="bottom_curb" type="box" size="4 0.2 0.3" pos="0 -20 0.25" rgba="0.5 0.5 0.5 1"/>
    <geom name="top_curb" type="box" size="4 0.2 0.3" pos="0 20 0.25" rgba="0.5 0.5 0.5 1"/>

    <!-- ========== 小车1（蓝色，朝北）右侧车道 ========== -->
    <body name="car1" pos="2.5 -5 0.3" euler="0 0 1.5708">  <!-- 旋转+90°，使车头朝北 -->
      <freejoint name="car1_free"/>
      <geom name="car1_body" type="box" size="0.5 0.3 0.1" mass="20.0" material="car1_mat" pos="0 0 -0.1"/>

      <body name="car1_fl" pos="0.3 0.3 -0.1">
        <joint name="car1_fl_steer" type="hinge" axis="0 0 1" limited="true" range="-0.4 0.4" damping="0.8"/>
        <joint name="car1_fl_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>
      <body name="car1_fr" pos="0.3 -0.3 -0.1">
        <joint name="car1_fr_steer" type="hinge" axis="0 0 1" limited="true" range="-0.4 0.4" damping="0.8"/>
        <joint name="car1_fr_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>
      <body name="car1_rl" pos="-0.3 0.3 -0.1">
        <joint name="car1_rl_spin" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>
      <body name="car1_rr" pos="-0.3 -0.3 -0.1">
        <joint name="car1_rr_spin" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>

      <site name="car1_front" pos="0.6 0 0.1" size="0.05" rgba="1 0 0 1" group="3"/>
      <site name="car1_front_left" pos="0.5 0.25 0.1" size="0.05" group="3"/>
      <site name="car1_front_right" pos="0.5 -0.25 0.1" size="0.05" group="3"/>
      <site name="car1_left" pos="0.1 0.35 0.1" size="0.05" group="3"/>
      <site name="car1_right" pos="0.1 -0.35 0.1" size="0.05" group="3"/>
      <site name="car1_trail" pos="0 0 0.2" size="0.06" rgba="1 1 0 1" group="2"/>
    </body>

    <!-- ========== 小车2（红色，朝南）右侧车道 ========== -->
    <body name="car2" pos="2.5 5 0.3" euler="0 0 -1.5708"> <!-- 旋转-90°，使车头朝南 -->
      <freejoint name="car2_free"/>
      <geom name="car2_body" type="box" size="0.5 0.3 0.1" mass="20.0" material="car2_mat" pos="0 0 -0.1"/>

      <body name="car2_fl" pos="0.3 0.3 -0.1">
        <joint name="car2_fl_steer" type="hinge" axis="0 0 1" limited="true" range="-0.4 0.4" damping="0.8"/>
        <joint name="car2_fl_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>
      <body name="car2_fr" pos="0.3 -0.3 -0.1">
        <joint name="car2_fr_steer" type="hinge" axis="0 0 1" limited="true" range="-0.4 0.4" damping="0.8"/>
        <joint name="car2_fr_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>
      <body name="car2_rl" pos="-0.3 0.3 -0.1">
        <joint name="car2_rl_spin" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>
      <body name="car2_rr" pos="-0.3 -0.3 -0.1">
        <joint name="car2_rr_spin" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>

      <site name="car2_front" pos="0.6 0 0.1" size="0.05" rgba="0 1 0 1" group="3"/>
      <site name="car2_front_left" pos="0.5 0.25 0.1" size="0.05" group="3"/>
      <site name="car2_front_right" pos="0.5 -0.25 0.1" size="0.05" group="3"/>
      <site name="car2_left" pos="0.1 0.35 0.1" size="0.05" group="3"/>
      <site name="car2_right" pos="0.1 -0.35 0.1" size="0.05" group="3"/>
      <site name="car2_trail" pos="0 0 0.2" size="0.06" rgba="1 1 0 1" group="2"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="car1_throttle_left" joint="car1_rl_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="car1_throttle_right" joint="car1_rr_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="car1_steer_left" joint="car1_fl_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.4 0.4"/>
    <motor name="car1_steer_right" joint="car1_fr_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.4 0.4"/>

    <motor name="car2_throttle_left" joint="car2_rl_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="car2_throttle_right" joint="car2_rr_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="car2_steer_left" joint="car2_fl_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.4 0.4"/>
    <motor name="car2_steer_right" joint="car2_fr_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.4 0.4"/>
  </actuator>

  <sensor>
    <!-- 小车1传感器 -->
    <rangefinder name="car1_front" site="car1_front"/>
    <rangefinder name="car1_front_left" site="car1_front_left"/>
    <rangefinder name="car1_front_right" site="car1_front_right"/>
    <rangefinder name="car1_left" site="car1_left"/>
    <rangefinder name="car1_right" site="car1_right"/>
    <velocimeter name="car1_vel" site="car1_trail"/>
    <framepos name="car1_pos" objtype="site" objname="car1_trail"/>
    <accelerometer name="car1_acc" site="car1_trail"/>

    <!-- 小车2传感器 -->
    <rangefinder name="car2_front" site="car2_front"/>
    <rangefinder name="car2_front_left" site="car2_front_left"/>
    <rangefinder name="car2_front_right" site="car2_front_right"/>
    <rangefinder name="car2_left" site="car2_left"/>
    <rangefinder name="car2_right" site="car2_right"/>
    <velocimeter name="car2_vel" site="car2_trail"/>
    <framepos name="car2_pos" objtype="site" objname="car2_trail"/>
    <accelerometer name="car2_acc" site="car2_trail"/>
  </sensor>
</mujoco>
"""


# ==================== 主仿真函数 ====================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="MuJoCo vehicle movement and obstacle avoidance simulation")
    parser.add_argument("--headless", action="store_true", help="run without MuJoCo viewer")
    parser.add_argument("--duration", type=float, default=12.0, help="simulation duration in seconds")
    parser.add_argument("--output-dir", default="output", help="directory for CSV metric reports")
    parser.add_argument("--log-interval", type=float, default=2.0, help="console log interval in seconds")
    return parser


def apply_control_outputs(data, actuator_ids, car1_ctrl, car2_ctrl):
    data.ctrl[actuator_ids[1]['throttle_left']] = car1_ctrl.last_throttle
    data.ctrl[actuator_ids[1]['throttle_right']] = car1_ctrl.last_throttle
    data.ctrl[actuator_ids[1]['steer_left']] = car1_ctrl.last_steer
    data.ctrl[actuator_ids[1]['steer_right']] = car1_ctrl.last_steer

    data.ctrl[actuator_ids[2]['throttle_left']] = car2_ctrl.last_throttle
    data.ctrl[actuator_ids[2]['throttle_right']] = car2_ctrl.last_throttle
    data.ctrl[actuator_ids[2]['steer_left']] = car2_ctrl.last_steer
    data.ctrl[actuator_ids[2]['steer_right']] = car2_ctrl.last_steer


def update_shared_positions(model, data):
    car1_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car1")
    car2_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car2")
    with state_lock:
        car_states[1]['position'] = (data.xpos[car1_body][0], data.xpos[car1_body][1])
        quat1 = data.xquat[car1_body]
        car_states[1]['heading'] = math.atan2(
            2.0 * (quat1[0] * quat1[3] + quat1[1] * quat1[2]),
            1.0 - 2.0 * (quat1[2] * quat1[2] + quat1[3] * quat1[3])
        )
        car_states[2]['position'] = (data.xpos[car2_body][0], data.xpos[car2_body][1])
        quat2 = data.xquat[car2_body]
        car_states[2]['heading'] = math.atan2(
            2.0 * (quat2[0] * quat2[3] + quat2[1] * quat2[2]),
            1.0 - 2.0 * (quat2[2] * quat2[2] + quat2[3] * quat2[3])
        )


def print_runtime_status(step, car1_ctrl, car2_ctrl):
    with state_lock:
        v1 = car_states[1]['velocity']
        v2 = car_states[2]['velocity']
    print(
        f"Step {step}: "
        f"Car1 vel={v1:.2f} yield={car1_ctrl.yield_flag} avoid={car1_ctrl.avoidance_active} emerg={car1_ctrl.emergency_brake}, "
        f"Car2 vel={v2:.2f} yield={car2_ctrl.yield_flag} avoid={car2_ctrl.avoidance_active} emerg={car2_ctrl.emergency_brake}"
    )


def print_metric_summary(summary, csv_path):
    print("\n=== Simulation Metrics Summary ===")
    print(f"CSV report: {csv_path}")
    print(f"Steps recorded: {summary['steps']}")
    print(f"Minimum distance between cars: {summary['min_distance']:.3f} m")
    print(f"Car1 average velocity: {summary['car1_avg_velocity']:.3f} m/s")
    print(f"Car2 average velocity: {summary['car2_avg_velocity']:.3f} m/s")
    print(f"Car1 emergency brake events: {summary['car1_emergency_events']}")
    print(f"Car2 emergency brake events: {summary['car2_emergency_events']}")
    print(f"Car1 distance to goal: {summary['car1_goal_distance']:.3f} m")
    print(f"Car2 distance to goal: {summary['car2_goal_distance']:.3f} m")


def multi_car_simulation(args=None):
    global simulation_running, step_barrier
    args = args or build_arg_parser().parse_args()
    simulation_running = True
    car_states.clear()
    xml = create_road_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    max_steps = max(1, int(args.duration / model.opt.timestep))
    metrics = SimulationMetrics(args.output_dir)

    actuator_ids = {1: {}, 2: {}}
    sensor_ids = {1: {}, 2: {}}

    for name in ["throttle_left", "throttle_right", "steer_left", "steer_right"]:
        for car in [1, 2]:
            full_name = f"car{car}_{name}"
            id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)
            if id != -1:
                actuator_ids[car][name] = id
                print(f"Car{car} {name} ID: {id}")

    sensor_prefix = {1: "car1_", 2: "car2_"}
    sensor_types = ["front", "front_left", "front_right", "left", "right", "vel", "pos", "acc"]
    for car in [1, 2]:
        for stype in sensor_types:
            full_name = sensor_prefix[car] + stype
            id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, full_name)
            if id != -1:
                if stype == "vel":
                    sensor_ids[car]['car_velocity'] = id
                elif stype == "pos":
                    sensor_ids[car]['car_position'] = id
                elif stype == "acc":
                    sensor_ids[car]['car_accel'] = id
                else:
                    sensor_ids[car][stype] = id
        print(f"Car{car} 找到 {len(sensor_ids[car])} 个传感器")

    # 目标点改为右侧车道 (x=2.5)
    car1_ctrl = CarController(1, model, data, actuator_ids[1], sensor_ids[1], goal_pos=(2.5, 20))
    car2_ctrl = CarController(2, model, data, actuator_ids[2], sensor_ids[2], goal_pos=(2.5, -20))

    # 正确初始化共享状态中的航向角（关键修复）
    with state_lock:
        car_states[1] = {'position': (data.xpos[car1_ctrl.body_id][0], data.xpos[car1_ctrl.body_id][1]), 'heading': 0, 'velocity': 0}
        car_states[2] = {'position': (data.xpos[car2_ctrl.body_id][0], data.xpos[car2_ctrl.body_id][1]), 'heading': 0, 'velocity': 0}
    car1_ctrl.update_state()
    car2_ctrl.update_state()

    global step_barrier
    step_barrier = threading.Barrier(3)

    def controller_loop(ctrl):
        global simulation_running
        while simulation_running:
            try:
                step_barrier.wait()
                if not simulation_running:
                    break
                ctrl.control_step()
                step_barrier.wait()
            except threading.BrokenBarrierError:
                break
            except Exception as e:
                print(f"控制器线程 {ctrl.car_id} 发生异常: {e}")
                import traceback
                traceback.print_exc()
                simulation_running = False
                break

    t1 = threading.Thread(target=controller_loop, args=(car1_ctrl,))
    t2 = threading.Thread(target=controller_loop, args=(car2_ctrl,))
    t1.start()
    t2.start()

    try:
        step = 0
        last_display = time.time()
        if args.headless:
            while simulation_running and step < max_steps:
                step_barrier.wait()
                apply_control_outputs(data, actuator_ids, car1_ctrl, car2_ctrl)
                mujoco.mj_step(model, data)
                update_shared_positions(model, data)
                step_barrier.wait()
                step += 1
                metrics.record(step, step * model.opt.timestep, car1_ctrl, car2_ctrl)

                if time.time() - last_display > args.log_interval:
                    print_runtime_status(step, car1_ctrl, car2_ctrl)
                    last_display = time.time()
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car1")
                viewer.cam.distance = 10.0
                viewer.cam.elevation = -30
                viewer.cam.azimuth = 90

                while viewer.is_running() and simulation_running and step < max_steps:
                    step_barrier.wait()
                    apply_control_outputs(data, actuator_ids, car1_ctrl, car2_ctrl)
                    mujoco.mj_step(model, data)
                    update_shared_positions(model, data)
                    step_barrier.wait()
                    viewer.sync()
                    step += 1
                    metrics.record(step, step * model.opt.timestep, car1_ctrl, car2_ctrl)

                    if time.time() - last_display > args.log_interval:
                        print_runtime_status(step, car1_ctrl, car2_ctrl)
                        last_display = time.time()

    except KeyboardInterrupt:
        print("用户中断")
    finally:
        simulation_running = False
        if step_barrier:
            step_barrier.abort()
        try:
            t1.join()
            t2.join()
        except:
            pass
        csv_path = metrics.save()
        print_metric_summary(metrics.summary(car1_ctrl, car2_ctrl), csv_path)
        print("仿真结束")


if __name__ == "__main__":
    multi_car_simulation()
