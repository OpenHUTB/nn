#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多车协同避让仿真（道路场景）- 右侧车道行驶版（优化版）
- 道路场景不变，两车目标点位于右侧车道 (2.5, ±20)
- 安全距离缩小，速度降至1.5m/s
- 车辆相遇时依靠人工势场法避让
- 修正车辆初始朝向，优化速度获取，平滑油门控制
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import math
from collections import deque

# ==================== 共享数据结构 ====================
car_states = {}
state_lock = threading.Lock()
simulation_running = True
step_barrier = None


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

        # 控制参数 - 速度减慢
        self.target_velocity = 1.5

        # 人工势场法参数 - 优化后的值
        self.att_gain = 1.0
        self.rep_car_gain = 8.0          # 增加车辆斥力增益
        self.rep_wall_gain = 5.0
        self.rep_distance_threshold = 2.0 # 边界斥力作用距离
        self.safe_distance = 3.0          # 车辆间安全距离
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

        # 当前控制输出
        self.last_throttle = 0.0
        self.last_steer = 0.0

        # 速度索引（用于直接从qvel获取全局速度）
        # freejoint 速度维度：前3为线速度(x,y,z)，后3为角速度
        self.vel_start = 0 if car_id == 1 else 7  # car1从0开始，car2从7开始

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
        pos_xy = (self.data.qpos[start], self.data.qpos[start + 1])
        quat = self.data.qpos[start + 3:start + 7]
        # 计算航向角（绕z轴）
        heading = math.atan2(2.0 * (quat[0] * quat[1] + quat[2] * quat[3]),
                             1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]))
        with state_lock:
            car_states[self.car_id] = {
                'position': pos_xy,
                'heading': heading,
                'velocity': self.get_sensor_data().get('velocity', 0),
                'timestamp': time.time()
            }

    def compute_apf_steering(self, my_sensor, other_cars):
        """
        人工势场法计算期望转向角
        """
        my_state = car_states.get(self.car_id, {})
        my_pos = my_state.get('position', (0, 0))
        my_heading = my_state.get('heading', 0)

        # 引力（指向固定目标点）
        F_att = np.array([self.goal_pos[0] - my_pos[0], self.goal_pos[1] - my_pos[1]])
        F_att = self.att_gain * F_att

        # 斥力
        F_rep_total = np.array([0.0, 0.0])

        # 动态车辆斥力
        for other_id, state in other_cars.items():
            if other_id == self.car_id:
                continue
            other_pos = state.get('position', (0, 0))
            dx = other_pos[0] - my_pos[0]
            dy = other_pos[1] - my_pos[1]
            dist = math.hypot(dx, dy)
            if dist < self.safe_distance and dist > 0.1:
                dir_vec = np.array([-dx, -dy]) / dist
                mag = self.rep_car_gain / (dist * dist)
                F_rep_total += mag * dir_vec

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
                       other_cars[other_id]['position'][1] - my_pos[1]) < 3.0
            for other_id in other_cars
        )
        # 简单的让行规则：ID大的让行
        self.yield_flag = self.near_car and (self.car_id > min(other_cars.keys()))

        # 紧急刹车 - 缩小距离阈值
        front_dist = my_sensor.get('front', 10.0)
        self.emergency_brake = front_dist < 0.5 or any(
            other_id != self.car_id and
            math.hypot(other_cars[other_id]['position'][0] - my_pos[0],
                       other_cars[other_id]['position'][1] - my_pos[1]) < 0.5
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
def multi_car_simulation():
    global simulation_running, step_barrier
    xml = create_road_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

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

    with state_lock:
        car_states[1] = {'position': (data.qpos[0], data.qpos[1]), 'heading': 0, 'velocity': 0}
        car_states[2] = {'position': (data.qpos[7], data.qpos[8]), 'heading': 0, 'velocity': 0}

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
            except Exception as e:
                print(f"控制器线程 {ctrl.car_id} 发生异常: {e}")
                import traceback
                traceback.print_exc()
                simulation_running = False
                break
            except threading.BrokenBarrierError:
                break

    t1 = threading.Thread(target=controller_loop, args=(car1_ctrl,))
    t2 = threading.Thread(target=controller_loop, args=(car2_ctrl,))
    t1.start()
    t2.start()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car1")
            viewer.cam.distance = 10.0
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 90

            step = 0
            last_display = time.time()
            while viewer.is_running() and simulation_running:
                step_barrier.wait()
                # 应用控制
                data.ctrl[actuator_ids[1]['throttle_left']] = car1_ctrl.last_throttle
                data.ctrl[actuator_ids[1]['throttle_right']] = car1_ctrl.last_throttle
                data.ctrl[actuator_ids[1]['steer_left']] = car1_ctrl.last_steer
                data.ctrl[actuator_ids[1]['steer_right']] = car1_ctrl.last_steer

                data.ctrl[actuator_ids[2]['throttle_left']] = car2_ctrl.last_throttle
                data.ctrl[actuator_ids[2]['throttle_right']] = car2_ctrl.last_throttle
                data.ctrl[actuator_ids[2]['steer_left']] = car2_ctrl.last_steer
                data.ctrl[actuator_ids[2]['steer_right']] = car2_ctrl.last_steer

                mujoco.mj_step(model, data)

                # 更新共享状态（航向角）
                with state_lock:
                    car_states[1]['position'] = (data.qpos[0], data.qpos[1])
                    quat1 = data.qpos[3:7]
                    car_states[1]['heading'] = math.atan2(2.0 * (quat1[0] * quat1[1] + quat1[2] * quat1[3]),
                                                          1.0 - 2.0 * (quat1[1] * quat1[1] + quat1[2] * quat1[2]))
                    car_states[2]['position'] = (data.qpos[7], data.qpos[8])
                    quat2 = data.qpos[10:14]
                    car_states[2]['heading'] = math.atan2(2.0 * (quat2[0] * quat2[1] + quat2[2] * quat2[3]),
                                                          1.0 - 2.0 * (quat2[1] * quat2[1] + quat2[2] * quat2[2]))

                step_barrier.wait()
                viewer.sync()
                step += 1

                if time.time() - last_display > 2.0:
                    v1 = car_states[1]['velocity']
                    v2 = car_states[2]['velocity']
                    y1 = car1_ctrl.yield_flag
                    y2 = car2_ctrl.yield_flag
                    e1 = car1_ctrl.emergency_brake
                    e2 = car2_ctrl.emergency_brake
                    print(
                        f"Step {step}: Car1 vel={v1:.2f} yield={y1} emerg={e1}, Car2 vel={v2:.2f} yield={y2} emerg={e2}")
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
        print("仿真结束")


if __name__ == "__main__":
    multi_car_simulation()