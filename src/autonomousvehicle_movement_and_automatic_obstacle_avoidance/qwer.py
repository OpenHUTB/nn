#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多车协同避让仿真 - 改进版
- 修复让行逻辑复位问题
- 调优PID参数，避免速度失控
- 增加调试输出
- 优化线程退出
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

# ==================== PID控制器 ====================
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

# ==================== 小车控制器类 ====================
class CarController:
    def __init__(self, car_id, model, data, actuator_ids, sensor_ids):
        self.car_id = car_id
        self.model = model
        self.data = data
        self.actuator_ids = actuator_ids
        self.sensor_ids = sensor_ids

        # 控制参数
        self.target_velocity = 15.0  # 降低目标速度便于观察
        self.steering_gain = 0.08
        self.obstacle_threshold = 2.5
        self.side_threshold = 1.5
        self.tight_space_threshold = 1.2

        # 防侧翻参数
        self.MAX_LATERAL_ACC = 5.0
        self.WHEELBASE = 0.6
        self.MAX_STEER_CHANGE = 0.05
        self.TILT_THRESHOLD = math.radians(5)

        # PID参数调低，使加速更平缓
        self.velocity_pid = PIDController(kp=0.4, ki=0.01, kd=0.1)   # 原为0.8/0.02/0.2
        self.steering_pid = PIDController(kp=0.2, ki=0.002, kd=0.05) # 原为0.3/0.005/0.1

        # 历史数据
        self.velocity_history = deque(maxlen=50)
        self.steering_history = deque(maxlen=25)
        self.last_position = None
        self.last_time = time.time()

        # 协商状态
        self.yield_flag = False
        self.near_car = False   # 是否检测到其他车

        # 当前控制输出
        self.last_throttle = 0.0
        self.last_steer = 0.0

    def get_sensor_data(self):
        data = {}
        if 'front' in self.sensor_ids:
            data['front'] = self.data.sensordata[self.sensor_ids['front']]
        if 'front_left' in self.sensor_ids:
            data['front_left'] = self.data.sensordata[self.sensor_ids['front_left']]
        if 'front_right' in self.sensor_ids:
            data['front_right'] = self.data.sensordata[self.sensor_ids['front_right']]
        if 'left' in self.sensor_ids:
            data['left'] = self.data.sensordata[self.sensor_ids['left']]
        if 'right' in self.sensor_ids:
            data['right'] = self.data.sensordata[self.sensor_ids['right']]
        if 'car_velocity' in self.sensor_ids:
            v = self.data.sensordata[self.sensor_ids['car_velocity']:self.sensor_ids['car_velocity']+3]
            data['velocity'] = np.linalg.norm(v[:2])
        if 'car_position' in self.sensor_ids:
            pos = self.data.sensordata[self.sensor_ids['car_position']:self.sensor_ids['car_position']+3]
            data['position'] = pos[:2]
        return data

    def update_state(self):
        start = 0 if self.car_id == 1 else 7
        pos_xy = (self.data.qpos[start], self.data.qpos[start+1])
        quat = self.data.qpos[start+3:start+7]
        heading = math.atan2(2.0*(quat[0]*quat[1]+quat[2]*quat[3]),
                              1.0-2.0*(quat[1]*quat[1]+quat[2]*quat[2]))
        with state_lock:
            car_states[self.car_id] = {
                'position': pos_xy,
                'heading': heading,
                'velocity': self.get_sensor_data().get('velocity', 0),
                'timestamp': time.time()
            }

    def compute_steering(self, my_sensor, other_cars):
        front = my_sensor.get('front', 10.0)
        front_left = my_sensor.get('front_left', 10.0)
        front_right = my_sensor.get('front_right', 10.0)
        left = my_sensor.get('left', 10.0)
        right = my_sensor.get('right', 10.0)

        left_space = min(left, front_left)
        right_space = min(right, front_right)
        steer_factor = 0.0

        if front < self.obstacle_threshold:
            if left_space > self.tight_space_threshold and right_space > self.tight_space_threshold:
                steer_factor = 0.35 if left_space > right_space else -0.35
            elif left_space > self.tight_space_threshold:
                steer_factor = 0.35
            elif right_space > self.tight_space_threshold:
                steer_factor = -0.35
            else:
                steer_factor = 0.0
        elif left < self.side_threshold or right < self.side_threshold:
            if left < self.side_threshold and right >= self.side_threshold:
                steer_factor = -0.08
            elif right < self.side_threshold and left >= self.side_threshold:
                steer_factor = 0.08
            else:
                steer_factor = 0.0

        base_steer = steer_factor * self.steering_gain

        # ---------- 车车协商规则 ----------
        my_pos = car_states.get(self.car_id, {}).get('position', (0,0))
        my_heading = car_states.get(self.car_id, {}).get('heading', 0)

        self.near_car = False
        for other_id, state in other_cars.items():
            if other_id == self.car_id:
                continue
            other_pos = state.get('position', (0,0))
            dx = other_pos[0] - my_pos[0]
            dy = other_pos[1] - my_pos[1]
            distance = math.hypot(dx, dy)
            if distance < 6.0:   # 检测范围扩大
                self.near_car = True
                # 相对角度
                angle_to_other = math.atan2(dy, dx)
                angle_diff = angle_to_other - my_heading
                while angle_diff > math.pi: angle_diff -= 2*math.pi
                while angle_diff < -math.pi: angle_diff += 2*math.pi

                if abs(angle_diff) < math.radians(60):
                    if angle_diff > 0:
                        base_steer -= 0.1   # 右转
                    else:
                        base_steer += 0.1

                    # 优先级让行：ID大的让行
                    if self.car_id > other_id:
                        self.yield_flag = True
                    else:
                        self.yield_flag = False

        # 如果没有检测到其他车，清除让行标志
        if not self.near_car:
            self.yield_flag = False

        base_steer = np.clip(base_steer, -0.25, 0.25)
        return base_steer

    def compute_throttle(self, current_vel, steering):
        target = self.target_velocity
        # 转向减速
        factor = 1.0 - min(abs(steering)*0.5, 0.2)
        target *= factor
        # 让行减速
        if self.yield_flag:
            target *= 0.3   # 减速更明显
        # 防侧翻速度限制
        if abs(steering) > 1e-6:
            v_safe = math.sqrt(self.MAX_LATERAL_ACC * self.WHEELBASE / abs(math.tan(steering)))
            target = min(target, v_safe)

        # 增加最小速度，避免PID积分过小导致停滞
        if target < 0.5 and self.near_car:
            target = 0.5   # 即使让行也保持最低速度

        self.velocity_pid.setpoint = target
        throttle = self.velocity_pid.update(current_vel, dt=self.model.opt.timestep)

        # 加速度限制
        if len(self.velocity_history) > 0:
            last = self.velocity_history[-1]
            max_accel = 2.0
            accel = (target - last) / self.model.opt.timestep
            if abs(accel) > max_accel:
                throttle = np.clip(throttle, -max_accel*8, max_accel*8)

        throttle = np.clip(throttle, -8.0, 25.0)
        self.velocity_history.append(current_vel)
        return throttle

    def control_step(self):
        my_sensor = self.get_sensor_data()
        with state_lock:
            other_cars = car_states.copy()
        steer = self.compute_steering(my_sensor, other_cars)
        if len(self.steering_history) > 0:
            prev = self.steering_history[-1]
            steer = np.clip(steer, prev - self.MAX_STEER_CHANGE, prev + self.MAX_STEER_CHANGE)
        self.steering_history.append(steer)
        vel = my_sensor.get('velocity', 0)
        throttle = self.compute_throttle(vel, steer)
        self.last_throttle = throttle
        self.last_steer = steer
        self.update_state()

# ==================== XML模型（与之前相同，略）====================
def create_multi_car_xml():
    return """
<mujoco>
  <compiler angle="radian" inertiafromgeom="true" coordinate="local"/>
  <option timestep="0.002" integrator="RK4"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.2 0.3 0.5" width="512" height="512"/>
    <texture name="grass" type="2d" builtin="checker" rgb1="0.3 0.6 0.3" rgb2="0.2 0.5 0.2" width="512" height="512"/>
    <material name="grass_mat" texture="grass" texrepeat="30 30" reflectance="0.1"/>
    <material name="car1_mat" rgba="0.1 0.4 0.8 1" specular="0.8"/>
    <material name="car2_mat" rgba="0.8 0.2 0.2 1" specular="0.8"/>
    <material name="tire" rgba="0.1 0.1 0.1 1"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="50 50 0.1" material="grass_mat" friction="2.0"/>

    <!-- ========== 小车1 ========== -->
    <body name="car1" pos="-3 0 0.3">
      <freejoint name="car1_free"/>
      <geom name="car1_body" type="box" size="0.5 0.3 0.1" mass="20.0" material="car1_mat" pos="0 0 -0.1"/>

      <body name="car1_fl" pos="0.3 0.3 -0.1">
        <joint name="car1_fl_steer" type="hinge" axis="0 0 1" limited="true" range="-0.3 0.3" damping="0.8"/>
        <joint name="car1_fl_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>
      <body name="car1_fr" pos="0.3 -0.3 -0.1">
        <joint name="car1_fr_steer" type="hinge" axis="0 0 1" limited="true" range="-0.3 0.3" damping="0.8"/>
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

    <!-- ========== 小车2 ========== -->
    <body name="car2" pos="3 0 0.3">
      <freejoint name="car2_free"/>
      <geom name="car2_body" type="box" size="0.5 0.3 0.1" mass="20.0" material="car2_mat" pos="0 0 -0.1"/>

      <body name="car2_fl" pos="0.3 0.3 -0.1">
        <joint name="car2_fl_steer" type="hinge" axis="0 0 1" limited="true" range="-0.3 0.3" damping="0.8"/>
        <joint name="car2_fl_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom type="cylinder" size="0.12 0.06" euler="1.57 0 0" material="tire" friction="3.0" mass="1.0"/>
      </body>
      <body name="car2_fr" pos="0.3 -0.3 -0.1">
        <joint name="car2_fr_steer" type="hinge" axis="0 0 1" limited="true" range="-0.3 0.3" damping="0.8"/>
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

    <geom name="wall_north" type="box" size="40 0.5 1.5" pos="0 22 1.5" rgba="0.6 0.6 0.6 0.8"/>
    <geom name="wall_south" type="box" size="40 0.5 1.5" pos="0 -22 1.5" rgba="0.6 0.6 0.6 0.8"/>
    <geom name="wall_east" type="box" size="0.5 40 1.5" pos="22 0 1.5" rgba="0.6 0.6 0.6 0.8"/>
    <geom name="wall_west" type="box" size="0.5 40 1.5" pos="-22 0 1.5" rgba="0.6 0.6 0.6 0.8"/>
  </worldbody>

  <actuator>
    <motor name="car1_throttle_left" joint="car1_rl_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="car1_throttle_right" joint="car1_rr_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="car1_steer_left" joint="car1_fl_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.3 0.3"/>
    <motor name="car1_steer_right" joint="car1_fr_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.3 0.3"/>

    <motor name="car2_throttle_left" joint="car2_rl_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="car2_throttle_right" joint="car2_rr_spin" gear="12" ctrllimited="true" ctrlrange="-6 6"/>
    <motor name="car2_steer_left" joint="car2_fl_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.3 0.3"/>
    <motor name="car2_steer_right" joint="car2_fr_steer" gear="2.5" ctrllimited="true" ctrlrange="-0.3 0.3"/>
  </actuator>

  <sensor>
    <rangefinder name="car1_front" site="car1_front"/>
    <rangefinder name="car1_front_left" site="car1_front_left"/>
    <rangefinder name="car1_front_right" site="car1_front_right"/>
    <rangefinder name="car1_left" site="car1_left"/>
    <rangefinder name="car1_right" site="car1_right"/>
    <velocimeter name="car1_vel" site="car1_trail"/>
    <framepos name="car1_pos" objtype="site" objname="car1_trail"/>
    <accelerometer name="car1_acc" site="car1_trail"/>

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
    xml = create_multi_car_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    actuator_ids = {1: {}, 2: {}}
    sensor_ids = {1: {}, 2: {}}

    for name in ["throttle_left", "throttle_right", "steer_left", "steer_right"]:
        for car in [1,2]:
            full_name = f"car{car}_{name}"
            id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)
            if id != -1:
                actuator_ids[car][name] = id
                print(f"Car{car} {name} ID: {id}")

    sensor_prefix = {1: "car1_", 2: "car2_"}
    sensor_types = ["front", "front_left", "front_right", "left", "right", "vel", "pos", "acc"]
    for car in [1,2]:
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

    car1_ctrl = CarController(1, model, data, actuator_ids[1], sensor_ids[1])
    car2_ctrl = CarController(2, model, data, actuator_ids[2], sensor_ids[2])

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
            viewer.cam.distance = 8.0
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 90

            step = 0
            last_display = time.time()
            while viewer.is_running():
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

                with state_lock:
                    car_states[1]['position'] = (data.qpos[0], data.qpos[1])
                    quat1 = data.qpos[3:7]
                    car_states[1]['heading'] = math.atan2(2.0*(quat1[0]*quat1[1]+quat1[2]*quat1[3]),
                                                           1.0-2.0*(quat1[1]*quat1[1]+quat1[2]*quat1[2]))
                    car_states[2]['position'] = (data.qpos[7], data.qpos[8])
                    quat2 = data.qpos[10:14]
                    car_states[2]['heading'] = math.atan2(2.0*(quat2[0]*quat2[1]+quat2[2]*quat2[3]),
                                                           1.0-2.0*(quat2[1]*quat2[1]+quat2[2]*quat2[2]))

                step_barrier.wait()
                viewer.sync()
                step += 1

                if time.time() - last_display > 2.0:
                    v1 = car_states[1]['velocity']
                    v2 = car_states[2]['velocity']
                    y1 = car1_ctrl.yield_flag
                    y2 = car2_ctrl.yield_flag
                    print(f"Step {step}: Car1 vel={v1:.2f} yield={y1}, Car2 vel={v2:.2f} yield={y2}")
                    last_display = time.time()

    except KeyboardInterrupt:
        print("中断")
    finally:
        global simulation_running

        step_barrier.abort()
        t1.join()
        t2.join()
        print("仿真结束")

if __name__ == "__main__":
    multi_car_simulation()