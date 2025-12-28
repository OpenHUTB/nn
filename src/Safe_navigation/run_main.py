#!/usr/bin/env python3
"""
AirSimNH 无人车仿真控制脚本 - 丁字路口通过版本 V2.0
智能防碰撞 + 路口导航 + 动态路径调整 + 改进左转策略
"""

import airsim
import time
import numpy as np
import cv2
import json
import os
from datetime import datetime
from collections import deque, defaultdict
import math
import random
import threading


class AirSimNHCarSimulator:
    """AirSim无人车仿真主类 - V2.0 改进版"""

    def __init__(self, ip="127.0.0.1", port=41451, vehicle_name="PhysXCar"):
        self.ip = ip
        self.port = port
        self.vehicle_name = vehicle_name
        self.client = None
        self.is_connected = False
        self.is_api_control_enabled = False

        # 车辆状态跟踪
        self.initial_position = None
        self.initial_yaw = None
        self.path_history = []
        self.velocity_history = deque(maxlen=100)

        # 碰撞计数器
        self.collision_count = 0
        self.last_collision_state = False
        self.last_collision_time = 0
        self.collision_recovery_mode = False
        self.recovery_start_time = 0

        # 路口导航参数 - 改进版
        self.intersection_detected = False
        self.approaching_intersection = False
        self.in_intersection = False
        self.intersection_passed = False
        self.intersection_decision_made = False
        self.intersection_turn_direction = None  # 'left', 'right', 'straight'
        self.intersection_approach_distance = 20.0  # 增加检测距离
        self.intersection_pass_distance = 12.0  # 增加通过距离

        # 路口检测状态
        self.intersection_check_counter = 0
        self.straight_line_counter = 0
        self.side_obstacle_detected = False

        # 左转避障参数
        self.left_turn_obstacle_avoidance = False
        self.left_turn_safe_distance = 3.5  # 安全距离阈值
        self.left_turn_adjustment_count = 0
        self.max_left_adjustments = 3

        # 路径记忆和学习
        self.position_history = deque(maxlen=500)
        self.yaw_history = deque(maxlen=500)
        self.successful_paths = []
        self.failed_paths = []

        # 动态调整参数
        self.aggressiveness = 0.4  # 降低侵略性，更谨慎
        self.steering_adjustment_rate = 0.08  # 降低转向调整速率
        self.last_steering = 0
        self.target_speed = 18  # km/h，降低目标速度

        # 环境感知参数
        self.right_obstacle_distance = float('inf')
        self.left_obstacle_distance = float('inf')
        self.front_obstacle_distance = float('inf')

        # 改进的偏移控制
        self.offset_history = deque(maxlen=20)
        self.target_offset_range = (-0.05, 0.05)  # 目标偏移范围
        self.offset_adjustment_gain = 0.3  # 偏移调整增益

        # 创建数据保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = f"simulation_data_{timestamp}"
        os.makedirs(self.data_dir, exist_ok=True)

        # 日志文件
        self.log_file = None
        self._init_log_file()

        # 状态监视器
        self.monitor_running = False
        self.monitor_thread = None

        print(f"数据保存目录: {self.data_dir}")

    def _init_log_file(self):
        """初始化日志文件"""
        try:
            self.log_file = open(f"{self.data_dir}/simulation_log.txt", "w")
            self.log_file.write(f"AirSimNH仿真日志 - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 50 + "\n")
        except Exception as e:
            print(f"无法创建日志文件: {e}")
            self.log_file = None

    def log_message(self, message):
        """记录日志信息"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        if self.log_file and not self.log_file.closed:
            try:
                self.log_file.write(log_entry + "\n")
                self.log_file.flush()
            except Exception as e:
                print(f"写入日志失败: {e}")

    def connect(self):
        """连接到AirSim仿真器"""
        try:
            self.log_message(f"正在连接到AirSim仿真器 {self.ip}:{self.port}...")
            self.client = airsim.CarClient(ip=self.ip, port=self.port)
            self.client.confirmConnection()

            vehicles = self.client.listVehicles()
            if self.vehicle_name not in vehicles:
                self.log_message(f"警告: 车辆 '{self.vehicle_name}' 未找到，可用车辆: {vehicles}")
                if vehicles:
                    self.vehicle_name = vehicles[0]
                    self.log_message(f"使用车辆: {self.vehicle_name}")

            self.is_connected = True
            self.log_message("✓ 成功连接到AirSim仿真器！")

            self.initial_position = self.get_position()
            self.initial_yaw = self.get_yaw()
            self.log_message(
                f"初始位置: x={self.initial_position['x']:.3f}, y={self.initial_position['y']:.3f}, z={self.initial_position['z']:.3f}")
            self.log_message(f"初始偏航角: {self.initial_yaw:.2f}°")

            return True

        except Exception as e:
            self.log_message(f"✗ 连接失败: {e}")
            self.log_message("请确保AirSimNH环境正在运行")
            return False

    def get_position(self):
        """获取车辆位置"""
        try:
            kinematics = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            return {
                "x": kinematics.position.x_val,
                "y": kinematics.position.y_val,
                "z": kinematics.position.z_val
            }
        except:
            return {"x": 0, "y": 0, "z": 0}

    def get_yaw(self):
        """获取车辆偏航角"""
        try:
            kinematics = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            orientation = kinematics.orientation

            q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
            siny_cosp = 2 * (q0 * q3 + q1 * q2)
            cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            yaw_deg = math.degrees(yaw)

            if yaw_deg < 0:
                yaw_deg += 360

            return yaw_deg
        except:
            return 0.0

    def get_velocity(self):
        """获取车辆速度向量"""
        try:
            state = self.client.getCarState(vehicle_name=self.vehicle_name)
            return {
                "x": state.kinematics_estimated.linear_velocity.x_val,
                "y": state.kinematics_estimated.linear_velocity.y_val,
                "z": state.kinematics_estimated.linear_velocity.z_val
            }
        except:
            return {"x": 0, "y": 0, "z": 0}

    def get_speed_kmh(self):
        """获取速度(km/h)"""
        try:
            state = self.client.getCarState(vehicle_name=self.vehicle_name)
            return state.speed
        except:
            return 0.0

    def enable_api_control(self, enable=True):
        """启用/禁用API控制"""
        try:
            self.client.enableApiControl(enable, vehicle_name=self.vehicle_name)
            self.is_api_control_enabled = enable

            if enable:
                self.log_message("✓ API控制已启用")
                controls = airsim.CarControls()
                controls.throttle = 0
                controls.steering = 0
                controls.brake = 0
                self.client.setCarControls(controls, vehicle_name=self.vehicle_name)
            else:
                self.log_message("✓ API控制已禁用")

            return True
        except Exception as e:
            self.log_message(f"✗ API控制设置失败: {e}")
            return False

    def detect_intersection_improved(self, current_position, current_yaw, speed):
        """改进的路口检测算法"""
        if self.intersection_passed or self.in_intersection:
            return False

        # 记录位置历史
        self.position_history.append(current_position)
        self.yaw_history.append(current_yaw)

        if len(self.position_history) < 30:
            return False

        # 分析行驶特征
        recent_positions = list(self.position_history)[-30:]
        recent_yaws = list(self.yaw_history)[-30:]

        # 计算方向稳定性
        yaw_variance = np.var(recent_yaws)
        if yaw_variance < 3.0:
            self.straight_line_counter += 1
        else:
            self.straight_line_counter = max(0, self.straight_line_counter - 2)

        # 计算行驶距离
        distance_traveled = self.calculate_distance_traveled()

        # 路口检测条件：直行一定距离后
        if (distance_traveled > self.intersection_approach_distance and
                self.straight_line_counter > 25 and
                not self.intersection_detected):

            self.approaching_intersection = True
            self.intersection_detected = True

            # 改进的转向决策逻辑
            if len(recent_positions) > 40:
                # 分析路径趋势
                start_idx = max(0, len(recent_positions) - 40)
                end_idx = len(recent_positions) - 1

                start_pos = recent_positions[start_idx]
                end_pos = recent_positions[end_idx]

                # 计算总体偏移
                y_offset = end_pos['y'] - start_pos['y']
                x_offset = end_pos['x'] - start_pos['x']

                self.log_message(
                    f"路口检测: 行驶距离{distance_traveled:.1f}m, Y偏移{y_offset:.2f}m, X偏移{x_offset:.2f}m")

                # 基于历史偏移的智能决策
                if y_offset > 1.5:  # 明显右偏
                    self.intersection_turn_direction = 'left'
                    self.log_message("检测到右偏趋势，路口左转")
                elif y_offset < -1.5:  # 明显左偏
                    self.intersection_turn_direction = 'right'
                    self.log_message("检测到左偏趋势，路口右转")
                else:
                    # 基于相对位置决策
                    if current_position['x'] < 0:  # 在左侧区域
                        self.intersection_turn_direction = 'right'
                        self.log_message("位于左侧区域，路口右转")
                    else:  # 在右侧区域
                        self.intersection_turn_direction = 'left'
                        self.log_message("位于右侧区域，路口左转")

                # 记录路口信息
                self.intersection_entry_position = current_position.copy()
                self.intersection_entry_yaw = current_yaw

                return True

        return False

    def calculate_distance_traveled(self):
        """计算行驶距离"""
        if len(self.position_history) < 2:
            return 0

        total_distance = 0
        positions = list(self.position_history)

        for i in range(1, len(positions)):
            p1 = positions[i - 1]
            p2 = positions[i]
            dx = p2['x'] - p1['x']
            dy = p2['y'] - p1['y']
            dz = p2['z'] - p1['z']
            segment_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            total_distance += segment_distance

        return total_distance

    def get_vehicle_state(self):
        """获取车辆状态 - 增强版"""
        try:
            state = self.client.getCarState(vehicle_name=self.vehicle_name)
            kinematics = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            yaw = self.get_yaw()
            velocity = self.get_velocity()

            current_position = {
                "x": kinematics.position.x_val,
                "y": kinematics.position.y_val,
                "z": kinematics.position.z_val
            }

            # 检查碰撞状态并更新计数器
            current_time = time.time()
            current_collision = state.collision.has_collided

            if current_collision and not self.last_collision_state:
                self.collision_count += 1
                self.last_collision_time = current_time

                # 进入碰撞恢复模式
                self.collision_recovery_mode = True
                self.recovery_start_time = current_time

                self.log_message(f"!!! 检测到碰撞！碰撞次数: {self.collision_count}")

                # 记录失败路径
                if len(self.position_history) > 10:
                    self.failed_paths.append(list(self.position_history)[-50:])

            elif not current_collision and self.last_collision_state:
                self.log_message("✓ 碰撞状态解除")

            self.last_collision_state = current_collision

            # 退出碰撞恢复模式
            if self.collision_recovery_mode and current_time - self.recovery_start_time > 3.0:
                self.collision_recovery_mode = False
                self.log_message("退出碰撞恢复模式")

            # 记录路径
            path_point = {
                "timestamp": current_time,
                "position": current_position.copy(),
                "yaw": yaw,
                "speed": state.speed,
                "velocity": velocity,
                "collision": current_collision
            }

            self.path_history.append(path_point)
            self.velocity_history.append(velocity)

            # 限制历史记录长度
            if len(self.path_history) > 1000:
                # 改为删除前100个元素，提高效率
                del self.path_history[:100]

            state_info = {
                "timestamp": current_time,
                "speed_kmh": state.speed,
                "speed_ms": state.speed / 3.6,
                "position": current_position,
                "yaw": yaw,
                "velocity": velocity,
                "rpm": state.rpm,
                "max_rpm": state.maxrpm,
                "gear": state.gear,
                "handbrake": state.handbrake,
                "collision": current_collision,
                "collision_count": self.collision_count,
                "collision_recovery_mode": self.collision_recovery_mode,
                "intersection_detected": self.intersection_detected,
                "intersection_turn_direction": self.intersection_turn_direction,
                "in_intersection": self.in_intersection
            }

            return state_info
        except Exception as e:
            self.log_message(f"获取车辆状态失败: {e}")
            return None

    def calculate_lateral_offset(self, current_position):
        """计算横向偏移（改进版本）"""
        if self.initial_position is None:
            return 0.0

        # 计算绝对偏移
        absolute_offset = current_position["y"] - self.initial_position["y"]

        # 记录偏移历史
        self.offset_history.append(absolute_offset)

        return absolute_offset

    def navigate_intersection_improved(self, current_state, elapsed_time):
        """改进的路口导航算法"""
        if not self.intersection_detected or self.intersection_passed:
            return None

        current_position = current_state['position']
        current_yaw = current_state['yaw']
        current_speed = current_state['speed_kmh']

        # 首次进入路口
        if not self.in_intersection:
            self.in_intersection = True
            self.intersection_entry_position = current_position.copy()
            self.intersection_entry_yaw = current_yaw
            self.left_turn_adjustment_count = 0
            self.log_message(f"进入丁字路口，决策: {self.intersection_turn_direction}")

        # 计算在路口内的距离
        distance_in_intersection = self.calculate_distance_from_point(
            current_position, self.intersection_entry_position)

        # 检查是否通过路口
        if distance_in_intersection > self.intersection_pass_distance:
            self.intersection_passed = True
            self.in_intersection = False
            self.left_turn_obstacle_avoidance = False
            self.log_message("✓ 成功通过丁字路口！")
            return None

        # 路口控制逻辑
        controls = airsim.CarControls()

        # 根据转向决策调整控制
        if self.intersection_turn_direction == 'left':
            # 改进的左转策略 - 防止撞上左侧车辆
            self.navigate_left_turn_improved(controls, distance_in_intersection, current_speed)

        elif self.intersection_turn_direction == 'right':
            # 右转策略
            self.navigate_right_turn(controls, distance_in_intersection, current_speed)
        else:
            # 直行策略
            controls.throttle = 0.4
            controls.brake = 0
            controls.steering = -0.05  # 轻微左倾

        return controls

    def navigate_left_turn_improved(self, controls, distance_in_intersection, current_speed):
        """改进的左转导航策略"""
        # 阶段1: 进入路口，减速观察 (0-4米)
        if distance_in_intersection < 4.0:
            controls.throttle = 0.25
            controls.brake = 0.05
            controls.steering = -0.05  # 轻微左转准备

        # 阶段2: 开始左转，增加观察 (4-6米)
        elif distance_in_intersection < 6.0:
            # 检查是否需要避障
            if not self.left_turn_obstacle_avoidance:
                controls.throttle = 0.2
                controls.brake = 0
                controls.steering = -0.25  # 中等左转
            else:
                # 避障模式：减少左转，保持距离
                controls.throttle = 0.15
                controls.brake = 0.05
                controls.steering = -0.15

        # 阶段3: 主要左转阶段 (6-9米)
        elif distance_in_intersection < 9.0:
            # 如果之前有碰撞风险，调整策略
            if self.collision_count > 0 and self.left_turn_adjustment_count < self.max_left_adjustments:
                # 调整策略：增加转向或调整路径
                controls.throttle = 0.15
                controls.brake = 0
                controls.steering = -0.4  # 更急的左转
                self.left_turn_adjustment_count += 1
                self.log_message(f"左转避障调整 {self.left_turn_adjustment_count}/{self.max_left_adjustments}")
            else:
                controls.throttle = 0.2
                controls.brake = 0
                controls.steering = -0.3  # 标准左转

        # 阶段4: 出路口 (9-12米)
        else:
            controls.throttle = 0.3
            controls.brake = 0
            controls.steering = -0.15  # 减少转向

    def navigate_right_turn(self, controls, distance_in_intersection, current_speed):
        """右转导航策略"""
        # 阶段1: 进入路口，减速 (0-3米)
        if distance_in_intersection < 3.0:
            controls.throttle = 0.2
            controls.brake = 0.1
            controls.steering = 0.05  # 轻微右转准备

        # 阶段2: 执行右转 (3-7米)
        elif distance_in_intersection < 7.0:
            controls.throttle = 0.25
            controls.brake = 0
            controls.steering = 0.25  # 右转

        # 阶段3: 完成右转 (7-12米)
        else:
            controls.throttle = 0.35
            controls.brake = 0
            controls.steering = 0.1  # 减少右转

    def calculate_distance_from_point(self, position1, position2):
        """计算两点之间的距离"""
        dx = position1['x'] - position2['x']
        dy = position1['y'] - position2['y']
        dz = position1['z'] - position2['z']
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def execute_collision_recovery_improved(self, current_state):
        """改进的碰撞恢复程序"""
        controls = airsim.CarControls()
        current_time = time.time()
        recovery_duration = current_time - self.recovery_start_time

        # 获取当前状态信息
        current_yaw = current_state['yaw']
        current_position = current_state['position']

        # 阶段1: 紧急刹车和后退 (0-1.5秒)
        if recovery_duration < 1.5:
            controls.throttle = -0.4  # 倒车
            controls.brake = 0.6
            # 根据碰撞位置决定转向方向
            controls.steering = 0.2  # 向右转向摆脱
            self.log_message("碰撞恢复：倒车脱离")

        # 阶段2: 停车观察 (1.5-2.5秒)
        elif recovery_duration < 2.5:
            controls.throttle = 0
            controls.brake = 0.3
            controls.steering = -0.1  # 轻微左转调整
            self.log_message("碰撞恢复：停车观察")

        # 阶段3: 小幅度前进，调整方向 (2.5-3.5秒)
        elif recovery_duration < 3.5:
            # 尝试不同的方向
            if self.collision_count % 2 == 0:
                controls.steering = -0.2  # 左转尝试
            else:
                controls.steering = 0.15  # 右转尝试

            controls.throttle = 0.15
            controls.brake = 0
            self.log_message("碰撞恢复：小幅度前进")

        else:
            # 恢复完成
            self.collision_recovery_mode = False
            self.log_message("碰撞恢复完成")

            # 根据碰撞情况调整策略
            if self.collision_count > 0:
                # 降低目标速度
                self.target_speed = max(12, self.target_speed - 2)
                self.log_message(f"调整目标速度至: {self.target_speed} km/h")

            return None

        return controls

    def smart_offset_correction(self, current_offset):
        """智能偏移校正"""
        # 计算平均偏移
        if len(self.offset_history) >= 5:
            avg_offset = sum(self.offset_history) / len(self.offset_history)
        else:
            avg_offset = current_offset

        # 目标偏移控制
        target_offset = 0.0  # 目标是在中心

        # 计算偏移误差
        offset_error = current_offset - target_offset

        # 根据误差大小决定转向
        if offset_error > 0.15:  # 严重右偏
            return -0.3
        elif offset_error > 0.08:  # 中度右偏
            return -0.18
        elif offset_error > 0.03:  # 轻微右偏
            return -0.08
        elif offset_error < -0.12:  # 严重左偏
            return 0.2
        elif offset_error < -0.06:  # 中度左偏
            return 0.1
        elif offset_error < -0.02:  # 轻微左偏
            return 0.05
        else:  # 良好位置
            return -0.02  # 轻微左倾预防

    def advanced_safe_control_improved(self, duration=70):
        """
        改进的增强安全控制：防碰撞 + 智能路口导航
        参数:
            duration: 演示总时长（秒）
        """
        if not self.is_connected or not self.is_api_control_enabled:
            self.log_message("错误: 请先连接并启用API控制")
            return False

        self.log_message(f"\n开始改进版增强安全控制演示 ({duration}秒)...")
        self.log_message("策略: 智能防碰撞 + 改进路口导航 + 动态调整")

        start_time = time.time()
        controls = airsim.CarControls()

        # 控制参数
        target_speed_kmh = self.target_speed
        base_throttle = 0.45  # 降低基础油门

        # 状态跟踪
        offset_history = deque(maxlen=10)
        max_right_offset = 0
        last_good_position = self.initial_position.copy()

        # 路口相关
        intersection_navigation_started = False
        intersection_navigation_complete = False

        # 改进的碰撞预防
        collision_prevention_active = False
        collision_prevention_timer = 0

        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time

                # 获取当前状态
                state = self.get_vehicle_state()
                if not state:
                    time.sleep(0.1)
                    continue

                current_speed = state['speed_kmh']
                current_position = state['position']
                current_yaw = state['yaw']
                collision_detected = state['collision']

                # 记录偏移历史 - 确保每次循环都记录
                absolute_offset = self.calculate_lateral_offset(current_position)
                offset_history.append(absolute_offset)

                # 1. 碰撞恢复处理（最高优先级）
                if self.collision_recovery_mode:
                    recovery_controls = self.execute_collision_recovery_improved(state)
                    if recovery_controls:
                        self.client.setCarControls(recovery_controls, vehicle_name=self.vehicle_name)
                        time.sleep(0.12)  # 降低控制频率
                        continue

                # 2. 检测路口（改进版）
                if not self.intersection_passed and elapsed > 15.0:  # 行驶15秒后开始检测
                    self.detect_intersection_improved(current_position, current_yaw, current_speed)

                # 3. 路口导航处理（改进版）
                if self.intersection_detected and not self.intersection_passed:
                    intersection_controls = self.navigate_intersection_improved(state, elapsed)
                    if intersection_controls:
                        self.client.setCarControls(intersection_controls, vehicle_name=self.vehicle_name)

                        # 显示状态
                        status_line = (f"🚦路口导航 | 转向: {self.intersection_turn_direction} | "
                                       f"速度: {current_speed:5.1f} km/h | "
                                       f"转向角: {intersection_controls.steering:+.3f}")
                        print(f"\r{status_line}", end="")
                        time.sleep(0.1)  # 降低控制频率
                        continue

                # 4. 正常行驶防碰撞控制
                if absolute_offset > max_right_offset:
                    max_right_offset = absolute_offset

                # 动态调整目标速度
                if self.intersection_detected and not self.intersection_passed:
                    target_speed_kmh = 12  # 路口更慢的速度
                elif collision_detected or self.collision_count > 0:
                    target_speed_kmh = max(10, self.target_speed - 5)  # 碰撞后低速
                else:
                    target_speed_kmh = self.target_speed  # 正常速度

                # 智能转向决策（改进版）
                base_steering = self.smart_offset_correction(absolute_offset)
                collision_risk = False

                # 根据偏移历史预测碰撞风险
                if len(offset_history) >= 3:
                    recent_trend = offset_history[-1] - offset_history[-3]
                    if recent_trend > 0.05:  # 快速右偏趋势
                        collision_risk = True
                        base_steering = max(base_steering, -0.25)  # 增加左转力度

                # 油门控制（改进版）
                speed_error = target_speed_kmh - current_speed

                if collision_risk or self.collision_recovery_mode:
                    controls.throttle = base_throttle * 0.25
                    controls.brake = 0.15
                elif self.collision_count > 0:  # 之前发生过碰撞，更谨慎
                    if speed_error > 3:
                        controls.throttle = base_throttle * 0.6
                        controls.brake = 0
                    elif speed_error > -2:
                        controls.throttle = base_throttle * 0.4
                        controls.brake = 0.05
                    else:
                        controls.throttle = 0
                        controls.brake = 0.15
                else:
                    # 正常速度控制
                    if speed_error > 5:
                        controls.throttle = min(0.6, base_throttle + 0.15)
                        controls.brake = 0
                    elif speed_error > 0:
                        controls.throttle = base_throttle
                        controls.brake = 0
                    elif speed_error > -4:
                        controls.throttle = base_throttle * 0.6
                        controls.brake = 0.05
                    else:
                        controls.throttle = 0
                        controls.brake = 0.2

                # 应用转向（平滑处理）
                steering_change = base_steering - self.last_steering
                steering_change = max(-0.08, min(0.08, steering_change))  # 限制变化率
                controls.steering = self.last_steering + steering_change
                self.last_steering = controls.steering

                # 5. 阶段控制（基于时间的策略调整）
                phase_control_factor = 1.0

                if elapsed < 10.0:  # 起步阶段，更谨慎
                    phase_control_factor = 0.6
                    controls.steering = -0.03  # 轻微左倾

                elif elapsed > duration - 15.0:  # 结束前15秒
                    # 逐渐减速停车
                    stop_progress = (elapsed - (duration - 15.0)) / 15.0
                    controls.throttle *= (1.0 - stop_progress)
                    controls.brake += stop_progress * 0.4

                    # 尝试回到中心
                    if absolute_offset > 0.03:
                        controls.steering = -0.08
                    elif absolute_offset < -0.03:
                        controls.steering = 0.05

                # 应用阶段控制
                controls.throttle *= phase_control_factor

                # 6. 发送控制命令
                self.client.setCarControls(controls, vehicle_name=self.vehicle_name)

                # 7. 显示状态
                status_symbol = "✓"
                if collision_risk:
                    status_symbol = "⚠️"
                if collision_detected:
                    status_symbol = "💥"
                if self.intersection_detected:
                    status_symbol = "🚦"
                if self.collision_recovery_mode:
                    status_symbol = "🔄"

                status_line = (f"{status_symbol} 速度: {current_speed:5.1f} km/h | "
                               f"转向: {controls.steering:+.3f} | "
                               f"油门: {controls.throttle:.2f} | "
                               f"刹车: {controls.brake:.2f} | "
                               f"偏航: {current_yaw:6.1f}° | "
                               f"偏移: {absolute_offset:+.3f}m")

                if self.intersection_detected:
                    status_line += f" | 路口: {self.intersection_turn_direction}"
                    if self.in_intersection:
                        distance_in_intersection = self.calculate_distance_from_point(
                            current_position, self.intersection_entry_position)
                        status_line += f" ({distance_in_intersection:.1f}m)"

                print(f"\r{status_line}", end="")

                time.sleep(0.1)  # 10Hz控制频率，更稳定

            print("\n✓ 改进版增强安全控制演示完成")

            # 最终分析
            self.log_message(f"\n最终统计:")
            self.log_message(f"最大向右偏移: {max_right_offset:.3f}米")
            self.log_message(f"碰撞次数: {self.collision_count}")
            self.log_message(f"路径点数量: {len(self.path_history)}")
            self.log_message(f"总行驶距离: {self.calculate_distance_traveled():.1f}米")

            if self.intersection_detected:
                self.log_message(f"路口检测: {'成功' if self.intersection_passed else '失败'}")
                if self.intersection_passed:
                    self.log_message(f"路口转向: {self.intersection_turn_direction}")
                    self.log_message(f"左转调整次数: {self.left_turn_adjustment_count}")

            if self.collision_count > 0:
                self.log_message(f"⚠️  发生碰撞: {self.collision_count}次")
                if self.collision_count <= 2:
                    self.log_message("碰撞次数在可接受范围内")
                else:
                    self.log_message("碰撞次数较多，需要进一步优化")
            else:
                self.log_message("✓ 安全：无碰撞发生")

            return True

        except KeyboardInterrupt:
            self.log_message("\n\n演示被用户中断")
            return False
        except Exception as e:
            self.log_message(f"\n✗ 控制演示出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_simulation_data(self):
        """保存仿真数据 - 增强版"""
        try:
            # 保存路径历史
            if self.path_history:
                # 转换为可序列化格式
                serializable_history = []
                for point in self.path_history:
                    serializable_point = {
                        "timestamp": point["timestamp"],
                        "position": point["position"],
                        "yaw": point["yaw"],
                        "speed": point["speed"],
                        "velocity": point.get("velocity", {"x": 0, "y": 0, "z": 0}),
                        "collision": point.get("collision", False)
                    }
                    serializable_history.append(serializable_point)

                path_file = f"{self.data_dir}/path_history.json"
                with open(path_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_history, f, indent=2, ensure_ascii=False)
                self.log_message(f"✓ 路径历史已保存: {path_file}")

            # 保存路口决策数据
            intersection_data = {
                "intersection_detected": self.intersection_detected,
                "intersection_passed": self.intersection_passed,
                "turn_direction": self.intersection_turn_direction,
                "collision_count": self.collision_count,
                "left_turn_adjustments": self.left_turn_adjustment_count,
                "total_distance": self.calculate_distance_traveled() if self.position_history else 0
            }

            intersection_file = f"{self.data_dir}/intersection_data.json"
            with open(intersection_file, 'w', encoding='utf-8') as f:
                json.dump(intersection_data, f, indent=2, ensure_ascii=False)

            # 保存统计数据
            stats = {
                "timestamp": datetime.now().isoformat(),
                "vehicle_name": self.vehicle_name,
                "collision_count": self.collision_count,
                "path_history_length": len(self.path_history),
                "initial_position": self.initial_position,
                "initial_yaw": self.initial_yaw,
                "intersection_data": intersection_data,
                "target_speed": self.target_speed,
                "aggressiveness": self.aggressiveness
            }

            stats_file = f"{self.data_dir}/simulation_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            # 生成详细报告
            self.generate_detailed_report()

            return True

        except Exception as e:
            self.log_message(f"✗ 保存数据失败: {e}")
            return False

    def generate_detailed_report(self):
        """生成详细报告"""
        report_file = f"{self.data_dir}/detailed_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("AirSim无人车智能控制演示详细报告 V2.0\n")
            f.write("丁字路口通过改进版本\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"车辆名称: {self.vehicle_name}\n")
            f.write(f"碰撞次数: {self.collision_count}\n")
            f.write(f"路径点数量: {len(self.path_history)}\n")
            f.write(f"行驶距离: {self.calculate_distance_traveled():.1f}米\n")
            f.write(f"目标速度: {self.target_speed} km/h\n")
            f.write(f"驾驶侵略性: {self.aggressiveness}\n\n")

            f.write("路口导航统计:\n")
            f.write(f"  路口检测: {'是' if self.intersection_detected else '否'}\n")
            if self.intersection_detected:
                f.write(f"  路口通过: {'是' if self.intersection_passed else '否'}\n")
                f.write(f"  转向决策: {self.intersection_turn_direction}\n")
                f.write(f"  左转调整次数: {self.left_turn_adjustment_count}\n")

            f.write("\n路径分析:\n")
            if self.path_history:
                y_values = [p['position']['y'] for p in self.path_history]
                x_values = [p['position']['x'] for p in self.path_history]

                min_y = min(y_values)
                max_y = max(y_values)
                avg_y = sum(y_values) / len(y_values)

                f.write(f"  Y坐标范围: {min_y:.3f} 到 {max_y:.3f} 米\n")
                f.write(f"  平均Y坐标: {avg_y:.3f} 米\n")
                f.write(f"  初始Y坐标: {self.initial_position['y']:.3f} 米\n")
                f.write(f"  最大右偏移: {max_y - self.initial_position['y']:.3f} 米\n")

                # 分析偏移趋势
                if len(y_values) > 20:
                    start_avg = sum(y_values[:10]) / 10
                    middle_avg = sum(y_values[len(y_values) // 2 - 5:len(y_values) // 2 + 5]) / 10
                    end_avg = sum(y_values[-10:]) / 10

                    f.write(f"  起始平均Y: {start_avg:.3f} 米\n")
                    f.write(f"  中间平均Y: {middle_avg:.3f} 米\n")
                    f.write(f"  结束平均Y: {end_avg:.3f} 米\n")

                    start_to_mid = middle_avg - start_avg
                    mid_to_end = end_avg - middle_avg

                    f.write(f"  第一阶段偏移: {start_to_mid:+.3f} 米\n")
                    f.write(f"  第二阶段偏移: {mid_to_end:+.3f} 米\n")

                    if abs(start_to_mid) < 0.5 and abs(mid_to_end) < 0.5:
                        f.write("  结论: 车辆基本保持稳定\n")
                    elif start_to_mid > 0.5:
                        f.write("  结论: 第一阶段有明显右偏\n")
                    elif mid_to_end < -0.5:
                        f.write("  结论: 第二阶段有明显左偏\n")

            f.write("\n性能评估:\n")
            if self.collision_count == 0:
                f.write("  ✓ 碰撞避免: 优秀\n")
            elif self.collision_count <= 1:
                f.write("  ⚠️ 碰撞避免: 良好\n")
            elif self.collision_count <= 3:
                f.write("  ⚠️ 碰撞避免: 一般\n")
            else:
                f.write("  ✗ 碰撞避免: 需要改进\n")

            if self.intersection_passed:
                f.write("  ✓ 路口通过: 成功\n")
            elif self.intersection_detected:
                f.write("  ⚠️ 路口通过: 部分成功\n")
            else:
                f.write("  ? 路口通过: 未检测到\n")

            f.write("\n改进建议:\n")
            if self.collision_count > 0:
                f.write("  1. 进一步降低目标速度\n")
                f.write("  2. 增加左转避障检测\n")
                f.write("  3. 优化碰撞恢复策略\n")
                f.write("  4. 调整转向增益参数\n")

            if not self.intersection_passed and self.collision_count == 0:
                f.write("  1. 延长演示时间\n")
                f.write("  2. 提高路口检测灵敏度\n")
                f.write("  3. 优化转向决策逻辑\n")

            if self.left_turn_adjustment_count > 0:
                f.write("  1. 左转策略需要进一步优化\n")
                f.write("  2. 考虑增加传感器检测\n")
                f.write("  3. 调整左转阶段参数\n")

        self.log_message(f"✓ 详细报告已保存: {report_file}")

    def run_enhanced_demo(self, duration=70):
        """运行增强演示 - 支持路口导航"""
        self.log_message("=" * 70)
        self.log_message("AirSimNH 无人车智能控制演示 V2.0")
        self.log_message("丁字路口通过改进版本")
        self.log_message("=" * 70)

        # 连接仿真器
        if not self.connect():
            return False

        try:
            # 启用API控制
            if not self.enable_api_control(True):
                return False

            self.log_message("\n等待车辆稳定...")
            time.sleep(3)  # 增加等待时间

            # 运行改进版增强安全控制演示
            self.log_message("\n" + "=" * 70)
            self.log_message("开始改进版增强安全控制演示")
            self.log_message("策略: 智能防碰撞 + 改进路口导航 + 动态调整")
            self.log_message("=" * 70)

            success = self.advanced_safe_control_improved(duration)

            if success:
                self.log_message("\n" + "=" * 70)
                self.log_message("演示完成，保存数据...")
                self.log_message("=" * 70)
                self.save_simulation_data()

            return success

        except Exception as e:
            print(f"\n演示过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        try:
            # 先记录清理开始
            print("\n正在清理资源...")

            # 停止车辆
            if self.is_api_control_enabled and self.client:
                controls = airsim.CarControls()
                controls.throttle = 0
                controls.brake = 1.0
                controls.steering = 0
                controls.handbrake = True
                try:
                    self.client.setCarControls(controls, vehicle_name=self.vehicle_name)
                    time.sleep(0.5)
                except:
                    pass

                # 禁用API控制
                try:
                    if self.client:
                        self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
                except:
                    pass

            # 关闭日志文件
            if self.log_file and not self.log_file.closed:
                try:
                    # 先写入清理完成信息
                    self.log_file.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ✓ 清理完成\n")
                    self.log_file.flush()
                    self.log_file.close()
                except:
                    pass

        except Exception as e:
            print(f"清理过程中出错: {e}")
        finally:
            print("✓ 清理完成")


def main():
    """主函数"""
    simulator = AirSimNHCarSimulator(
        ip="127.0.0.1",
        port=41451,
        vehicle_name="PhysXCar"
    )

    try:
        # 运行增强演示，时间延长到70秒
        simulator.run_enhanced_demo(duration=70)

        print("\n" + "=" * 70)
        print("改进版智能控制演示完成！")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
        simulator.cleanup()
    except Exception as e:
        print(f"\n演示出错: {e}")
        import traceback
        traceback.print_exc()
        simulator.cleanup()


if __name__ == "__main__":
    main()