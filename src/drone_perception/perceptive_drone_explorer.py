"""
AirSimNH 感知驱动自主探索无人机 - 智能决策增强版（修复版）
核心：视觉感知 → 语义理解 → 智能决策 → 安全执行
集成：配置管理、日志系统、异常恢复、前视窗口显示
新增：向量场避障算法、基于网格的信息增益探索、平滑飞行控制
新增：性能监控与数据闭环系统
版本: 3.2 (完整版)
"""

import airsim
import time
import numpy as np
import cv2
import math
import json
import csv
from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import signal
import sys
from typing import Tuple, List, Optional, Dict, Set, Any
import traceback
import logging
from datetime import datetime
import random
import psutil
import os

# ============ 导入配置文件 ============
try:
    import config
    CONFIG_LOADED = True
except ImportError as e:
    print(f"❌ 无法加载配置文件 config.py: {e}")
    print("正在使用默认配置...")
    CONFIG_LOADED = False
    class DefaultConfig:
        EXPLORATION = {'TOTAL_TIME': 120, 'PREFERRED_SPEED': 2.5, 'BASE_HEIGHT': -15.0,
                      'MAX_ALTITUDE': -30.0, 'MIN_ALTITUDE': -5.0, 'TAKEOFF_HEIGHT': -10.0}
        PERCEPTION = {'DEPTH_NEAR_THRESHOLD': 5.0, 'DEPTH_SAFE_THRESHOLD': 10.0,
                     'MIN_GROUND_CLEARANCE': 2.0, 'MAX_PITCH_ANGLE_DEG': 15,
                     'SCAN_ANGLES': [-60, -45, -30, -15, 0, 15, 30, 45, 60],
                     'HEIGHT_STRATEGY': {'STEEP_SLOPE': -20.0, 'OPEN_SPACE': -12.0,
                                         'DEFAULT': -15.0, 'SLOPE_THRESHOLD': 5.0,
                                         'OPENNESS_THRESHOLD': 0.7}}
        DISPLAY = {'WINDOW_WIDTH': 640, 'WINDOW_HEIGHT': 480, 'ENABLE_SHARPENING': True,
                  'SHOW_INFO_OVERLAY': True, 'REFRESH_RATE_MS': 30}
        SYSTEM = {'LOG_LEVEL': 'INFO', 'LOG_TO_FILE': True, 'LOG_FILENAME': 'drone_log.txt',
                 'MAX_RECONNECT_ATTEMPTS': 3, 'RECONNECT_DELAY': 2.0,
                 'ENABLE_HEALTH_CHECK': True, 'HEALTH_CHECK_INTERVAL': 20}
        CAMERA = {'DEFAULT_NAME': "0"}
        MANUAL = {
            'CONTROL_SPEED': 3.0,
            'ALTITUDE_SPEED': 2.0,
            'YAW_SPEED': 30.0,
            'ENABLE_AUTO_HOVER': True,
            'DISPLAY_CONTROLS': True,
            'SAFETY_ENABLED': True,
            'MAX_MANUAL_SPEED': 5.0,
            'MIN_ALTITUDE_LIMIT': -5.0,
            'MAX_ALTITUDE_LIMIT': -30.0
        }
        # 新增：智能决策参数 - 修复键名问题
        INTELLIGENT_DECISION = {
            'VECTOR_FIELD_RADIUS': 8.0,           # 向量场影响半径
            'OBSTACLE_REPULSION_GAIN': 3.0,       # 障碍物排斥增益
            'GOAL_ATTRACTION_GAIN': 2.0,          # 目标吸引力增益
            'SMOOTHING_FACTOR': 0.3,              # 向量平滑因子
            'MIN_TURN_ANGLE_DEG': 10,             # 最小转弯角度（度）
            'MAX_TURN_ANGLE_DEG': 60,             # 最大转弯角度（度）

            'GRID_RESOLUTION': 2.0,               # 网格分辨率（米）
            'GRID_SIZE': 50,                      # 网格大小（单元格数）
            'INFORMATION_GAIN_DECAY': 0.95,       # 信息增益衰减率
            'EXPLORATION_FRONTIER_THRESHOLD': 0.3,# 探索前沿阈值

            'PID_KP': 1.5,                        # 比例系数
            'PID_KI': 0.05,                       # 积分系数
            'PID_KD': 0.2,                        # 微分系数
            'SMOOTHING_WINDOW_SIZE': 5,           # 平滑窗口大小

            'ADAPTIVE_SPEED_ENABLED': True,       # 启用自适应速度
            'MIN_SPEED_FACTOR': 0.3,              # 最小速度因子
            'MAX_SPEED_FACTOR': 1.5,              # 最大速度因子

            'MEMORY_WEIGHT': 0.7,                 # 记忆权重（避免重复访问）
            'CURIOUSITY_WEIGHT': 0.3,             # 好奇心权重（探索新区域）

            'TARGET_LIFETIME': 15.0,              # 目标有效期（秒）
            'TARGET_REACHED_DISTANCE': 3.0,       # 目标到达判定距离（米）
        }
        DEBUG = {
            'SAVE_PERCEPTION_IMAGES': False,
            'IMAGE_SAVE_INTERVAL': 50,
            'LOG_DECISION_DETAILS': False
        }
        # 新增：数据记录参数
        DATA_RECORDING = {
            'ENABLED': True,
            'RECORD_INTERVAL': 0.2,               # 记录间隔（秒）
            'SAVE_TO_CSV': True,
            'SAVE_TO_JSON': True,
            'CSV_FILENAME': 'flight_data.csv',
            'JSON_FILENAME': 'flight_data.json',
            'PERFORMANCE_MONITORING': True,       # 性能监控
            'SYSTEM_METRICS_INTERVAL': 5.0,       # 系统指标记录间隔
        }
        # 新增：性能监控参数
        PERFORMANCE = {
            'ENABLE_REALTIME_METRICS': True,
            'CPU_WARNING_THRESHOLD': 80.0,        # CPU使用率警告阈值
            'MEMORY_WARNING_THRESHOLD': 80.0,     # 内存使用率警告阈值
            'LOOP_TIME_WARNING_THRESHOLD': 0.2,   # 循环时间警告阈值（秒）
            'SAVE_PERFORMANCE_REPORT': True,
            'REPORT_INTERVAL': 30.0,              # 性能报告间隔（秒）
        }
    config = DefaultConfig()


class FlightState(Enum):
    """无人机飞行状态枚举"""
    TAKEOFF = "起飞"
    HOVERING = "悬停观测"
    EXPLORING = "主动探索"
    AVOIDING = "避障机动"
    RETURNING = "返航中"
    LANDING = "降落"
    EMERGENCY = "紧急状态"
    MANUAL = "手动控制"
    PLANNING = "路径规划"


class Vector2D:
    """二维向量类"""
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vector2D(self.x / scalar, self.y / scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector2D(self.x / mag, self.y / mag)
        return Vector2D()

    def rotate(self, angle):
        """旋转向量"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

    def to_tuple(self):
        return (self.x, self.y)

    @staticmethod
    def from_angle(angle, magnitude=1.0):
        return Vector2D(magnitude * math.cos(angle), magnitude * math.sin(angle))


class PIDController:
    """PID控制器类"""
    def __init__(self, kp, ki, kd, integral_limit=5.0, output_limit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit

        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = time.time()

    def update(self, error, dt=None):
        if dt is None:
            current_time = time.time()
            dt = current_time - self.previous_time
            self.previous_time = current_time

        # 积分项
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        # 微分项
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        self.previous_error = error

        # 计算输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


class ExplorationGrid:
    """探索网格地图类"""
    def __init__(self, resolution=2.0, grid_size=50):
        self.resolution = resolution
        self.grid_size = grid_size
        self.half_size = grid_size // 2

        # 初始化网格：0=未知，1=已探索，0.x=部分探索
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        # 信息增益缓存
        self.information_gain = np.zeros((grid_size, grid_size), dtype=np.float32)

        # 障碍物标记
        self.obstacle_grid = np.zeros((grid_size, grid_size), dtype=bool)

        # 访问时间记录
        self.visit_time = np.zeros((grid_size, grid_size), dtype=np.float32)

        # 当前位置索引
        self.current_idx = (self.half_size, self.half_size)

        # 探索前沿
        self.frontier_cells = set()

        print(f"🗺️ 初始化探索网格: {grid_size}x{grid_size}, 分辨率: {resolution}m")

    def world_to_grid(self, world_x, world_y):
        """世界坐标转网格索引"""
        grid_x = int(world_x / self.resolution) + self.half_size
        grid_y = int(world_y / self.resolution) + self.half_size

        # 边界检查
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))

        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """网格索引转世界坐标"""
        world_x = (grid_x - self.half_size) * self.resolution
        world_y = (grid_y - self.half_size) * self.resolution
        return (world_x, world_y)

    def update_position(self, world_x, world_y):
        """更新当前位置"""
        self.current_idx = self.world_to_grid(world_x, world_y)

        # 标记当前位置为已探索
        x, y = self.current_idx
        radius = 3  # 探索半径（网格单元）

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    distance = math.sqrt(dx**2 + dy**2)
                    exploration_value = max(0, 1.0 - distance / radius)
                    self.grid[nx, ny] = max(self.grid[nx, ny], exploration_value)
                    self.visit_time[nx, ny] = time.time()

        # 更新探索前沿
        self._update_frontiers()

    def _update_frontiers(self):
        """更新探索前沿"""
        self.frontier_cells.clear()

        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                # 如果当前单元格已探索，检查邻居是否有未探索的
                if self.grid[x, y] > 0.7:  # 足够探索
                    neighbors = [
                        (x-1, y), (x+1, y), (x, y-1), (x, y+1),
                        (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)
                    ]

                    for nx, ny in neighbors:
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if self.grid[nx, ny] < 0.3 and not self.obstacle_grid[nx, ny]:
                                # 计算信息增益：基于未探索邻居数量
                                unexplored_neighbors = 0
                                for nnx in range(nx-1, nx+2):
                                    for nny in range(ny-1, ny+2):
                                        if 0 <= nnx < self.grid_size and 0 <= nny < self.grid_size:
                                            if self.grid[nnx, nny] < 0.3:
                                                unexplored_neighbors += 1

                                self.information_gain[nx, ny] = unexplored_neighbors / 9.0
                                self.frontier_cells.add((nx, ny))

    def update_obstacles(self, obstacles_world):
        """更新障碍物信息"""
        for obs_x, obs_y in obstacles_world:
            grid_x, grid_y = self.world_to_grid(obs_x, obs_y)

            # 标记障碍物及其周围区域
            radius = 2
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.obstacle_grid[nx, ny] = True
                        self.grid[nx, ny] = 0.0  # 障碍物区域不可探索

    def get_best_exploration_target(self, current_pos):
        """获取最佳探索目标"""
        if not self.frontier_cells:
            # 如果没有前沿，返回随机方向
            angle = random.uniform(0, 2 * math.pi)
            distance = 10.0  # 10米外
            return (
                current_pos[0] + distance * math.cos(angle),
                current_pos[1] + distance * math.sin(angle)
            )

        # 计算每个前沿单元格的得分
        best_score = -1
        best_target = None
        current_x, current_y = current_pos

        for fx, fy in self.frontier_cells:
            # 计算信息增益
            info_gain = self.information_gain[fx, fy]

            # 计算距离成本
            world_x, world_y = self.grid_to_world(fx, fy)
            distance = math.sqrt((world_x - current_x)**2 + (world_y - current_y)**2)
            distance_cost = min(1.0, distance / 30.0)  # 归一化

            # 计算最近访问时间（避免重复）
            time_since_visit = time.time() - self.visit_time[fx, fy]
            time_factor = min(1.0, time_since_visit / 60.0)  # 1分钟内衰减

            # 综合得分
            score = (
                config.INTELLIGENT_DECISION['CURIOUSITY_WEIGHT'] * info_gain +
                (1 - config.INTELLIGENT_DECISION['MEMORY_WEIGHT'] * time_factor) -
                distance_cost * 0.3
            )

            if score > best_score:
                best_score = score
                best_target = (world_x, world_y)

        return best_target

    def visualize_grid(self, size=300):
        """可视化网格"""
        if self.grid.size == 0:
            return None

        img_size = min(size, self.grid_size * 5)
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        cell_size = img_size // self.grid_size

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                color = (0, 0, 0)

                if (x, y) == self.current_idx:
                    color = (0, 255, 0)  # 当前位置：绿色
                elif self.obstacle_grid[x, y]:
                    color = (0, 0, 255)  # 障碍物：红色
                elif self.grid[x, y] > 0.7:
                    color = (200, 200, 200)  # 已探索：灰色
                elif self.grid[x, y] > 0.3:
                    color = (100, 100, 100)  # 部分探索：深灰色
                elif (x, y) in self.frontier_cells:
                    # 前沿单元格：根据信息增益着色
                    gain = self.information_gain[x, y]
                    color = (0, int(255 * gain), int(255 * (1 - gain)))  # 绿到黄

                # 绘制单元格
                x1 = x * cell_size
                y1 = y * cell_size
                x2 = (x + 1) * cell_size
                y2 = (y + 1) * cell_size

                cv2.rectangle(img, (y1, x1), (y2, x2), color, -1)

        return img


class DataLogger:
    """数据记录器类 - 用于记录飞行数据"""

    def __init__(self, enable_csv=True, enable_json=True, csv_filename=None, json_filename=None):
        self.enable_csv = enable_csv
        self.enable_json = enable_json

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 设置文件名
        if csv_filename:
            self.csv_filename = csv_filename
        else:
            self.csv_filename = f"flight_data_{timestamp}.csv"

        if json_filename:
            self.json_filename = json_filename
        else:
            self.json_filename = f"flight_data_{timestamp}.json"

        # 数据缓存
        self.data_buffer = []
        self.json_data = {
            "flight_info": {
                "start_time": datetime.now().isoformat(),
                "config_loaded": CONFIG_LOADED,
                "system": config.SYSTEM,
                "exploration": config.EXPLORATION,
                "perception": config.PERCEPTION,
                "intelligent_decision": config.INTELLIGENT_DECISION,
                "performance": config.PERFORMANCE
            },
            "flight_data": []
        }

        # 性能指标
        self.performance_metrics = {
            "start_time": time.time(),
            "cpu_usage": [],
            "memory_usage": [],
            "loop_times": [],
            "data_points": 0
        }

        # 定义CSV列名
        self.csv_columns = [
            'timestamp', 'loop_count', 'state', 'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z', 'yaw', 'pitch', 'roll',
            'obstacle_distance', 'open_space_score', 'terrain_slope',
            'has_obstacle', 'obstacle_direction', 'recommended_height',
            'target_x', 'target_y', 'target_z', 'velocity_command_x',
            'velocity_command_y', 'velocity_command_z', 'yaw_command',
            'battery_level', 'cpu_usage', 'memory_usage', 'loop_time',
            'grid_frontiers', 'grid_explored', 'vector_field_magnitude',
            'adaptive_speed_factor', 'decision_making_time', 'perception_time'
        ]

        # 初始化CSV文件
        if self.enable_csv:
            self._init_csv_file()

        print(f"📊 数据记录器初始化完成")
        print(f"  CSV文件: {self.csv_filename}")
        print(f"  JSON文件: {self.json_filename}")

    def _init_csv_file(self):
        """初始化CSV文件，写入列名"""
        try:
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()
        except Exception as e:
            print(f"❌ 无法初始化CSV文件: {e}")
            self.enable_csv = False

    def record_flight_data(self, data_dict):
        """记录飞行数据"""
        if not config.DATA_RECORDING['ENABLED']:
            return

        try:
            # 添加时间戳
            data_dict['timestamp'] = datetime.now().isoformat()

            # 记录到CSV
            if self.enable_csv:
                with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_columns)

                    # 确保所有字段都存在
                    row = {col: data_dict.get(col, '') for col in self.csv_columns}
                    writer.writerow(row)

            # 记录到JSON缓冲区
            if self.enable_json:
                self.json_data['flight_data'].append(data_dict)

            # 更新性能指标
            self.performance_metrics['data_points'] += 1

            # 收集系统性能数据
            if self.performance_metrics['data_points'] % 10 == 0:
                self._collect_system_metrics()

        except Exception as e:
            print(f"⚠️ 记录飞行数据时出错: {e}")

    def _collect_system_metrics(self):
        """收集系统性能指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.performance_metrics['cpu_usage'].append(cpu_percent)

            # 内存使用率
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            self.performance_metrics['memory_usage'].append(memory_percent)

            # 限制数据长度
            max_length = 1000
            if len(self.performance_metrics['cpu_usage']) > max_length:
                self.performance_metrics['cpu_usage'] = self.performance_metrics['cpu_usage'][-max_length:]
            if len(self.performance_metrics['memory_usage']) > max_length:
                self.performance_metrics['memory_usage'] = self.performance_metrics['memory_usage'][-max_length:]

        except Exception as e:
            print(f"⚠️ 收集系统指标时出错: {e}")

    def record_loop_time(self, loop_time):
        """记录循环时间"""
        self.performance_metrics['loop_times'].append(loop_time)

        # 限制数据长度
        max_length = 1000
        if len(self.performance_metrics['loop_times']) > max_length:
            self.performance_metrics['loop_times'] = self.performance_metrics['loop_times'][-max_length:]

    def record_event(self, event_type, event_data):
        """记录事件（状态改变、异常等）"""
        try:
            event_record = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'event_data': event_data
            }

            if 'events' not in self.json_data:
                self.json_data['events'] = []

            self.json_data['events'].append(event_record)

        except Exception as e:
            print(f"⚠️ 记录事件时出错: {e}")

    def save_json_data(self):
        """保存JSON数据到文件"""
        if not self.enable_json:
            return

        try:
            # 计算性能统计
            self._calculate_performance_stats()

            # 保存到文件
            with open(self.json_filename, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)

            print(f"✅ JSON数据已保存: {self.json_filename}")

        except Exception as e:
            print(f"❌ 保存JSON数据时出错: {e}")

    def _calculate_performance_stats(self):
        """计算性能统计数据"""
        if not self.performance_metrics['cpu_usage']:
            return

        # 计算CPU统计
        cpu_avg = np.mean(self.performance_metrics['cpu_usage'])
        cpu_max = np.max(self.performance_metrics['cpu_usage'])
        cpu_min = np.min(self.performance_metrics['cpu_usage'])

        # 计算内存统计
        mem_avg = np.mean(self.performance_metrics['memory_usage'])
        mem_max = np.max(self.performance_metrics['memory_usage'])
        mem_min = np.min(self.performance_metrics['memory_usage'])

        # 计算循环时间统计
        if self.performance_metrics['loop_times']:
            loop_avg = np.mean(self.performance_metrics['loop_times'])
            loop_max = np.max(self.performance_metrics['loop_times'])
            loop_min = np.min(self.performance_metrics['loop_times'])
        else:
            loop_avg = loop_max = loop_min = 0

        # 添加到JSON数据
        self.json_data['performance_summary'] = {
            'total_data_points': self.performance_metrics['data_points'],
            'total_time_seconds': time.time() - self.performance_metrics['start_time'],
            'cpu_usage': {
                'average': float(cpu_avg),
                'maximum': float(cpu_max),
                'minimum': float(cpu_min)
            },
            'memory_usage': {
                'average': float(mem_avg),
                'maximum': float(mem_max),
                'minimum': float(mem_min)
            },
            'loop_times': {
                'average_seconds': float(loop_avg),
                'maximum_seconds': float(loop_max),
                'minimum_seconds': float(loop_min)
            }
        }

    def generate_performance_report(self):
        """生成性能报告"""
        try:
            if not self.performance_metrics['cpu_usage']:
                return "无性能数据可用"

            # 计算统计
            self._calculate_performance_stats()

            report = "\n" + "="*60 + "\n"
            report += "📊 系统性能报告\n"
            report += "="*60 + "\n"

            report += f"总数据点数: {self.performance_metrics['data_points']}\n"
            report += f"运行时间: {time.time() - self.performance_metrics['start_time']:.1f}秒\n"

            if self.performance_metrics['cpu_usage']:
                cpu_avg = np.mean(self.performance_metrics['cpu_usage'])
                cpu_max = np.max(self.performance_metrics['cpu_usage'])
                report += f"CPU使用率: 平均{cpu_avg:.1f}%, 最大{cpu_max:.1f}%\n"

            if self.performance_metrics['memory_usage']:
                mem_avg = np.mean(self.performance_metrics['memory_usage'])
                mem_max = np.max(self.performance_metrics['memory_usage'])
                report += f"内存使用率: 平均{mem_avg:.1f}%, 最大{mem_max:.1f}%\n"

            if self.performance_metrics['loop_times']:
                loop_avg = np.mean(self.performance_metrics['loop_times'])
                loop_max = np.max(self.performance_metrics['loop_times'])
                report += f"循环时间: 平均{loop_avg*1000:.1f}ms, 最大{loop_max*1000:.1f}ms\n"

            report += "="*60 + "\n"

            # 检查警告
            warnings = []
            if cpu_avg > config.PERFORMANCE['CPU_WARNING_THRESHOLD']:
                warnings.append(f"⚠️ CPU使用率过高: {cpu_avg:.1f}%")

            if mem_avg > config.PERFORMANCE['MEMORY_WARNING_THRESHOLD']:
                warnings.append(f"⚠️ 内存使用率过高: {mem_avg:.1f}%")

            if loop_avg > config.PERFORMANCE['LOOP_TIME_WARNING_THRESHOLD']:
                warnings.append(f"⚠️ 循环时间过长: {loop_avg*1000:.1f}ms")

            if warnings:
                report += "\n⚠️ 性能警告:\n"
                for warning in warnings:
                    report += f"  {warning}\n"

            return report

        except Exception as e:
            return f"生成性能报告时出错: {e}"


@dataclass
class PerceptionResult:
    """感知结果数据结构"""
    has_obstacle: bool = False
    obstacle_distance: float = 100.0
    obstacle_direction: float = 0.0
    terrain_slope: float = 0.0
    open_space_score: float = 0.0
    recommended_height: float = config.PERCEPTION['HEIGHT_STRATEGY']['DEFAULT']
    safe_directions: List[float] = None
    front_image: Optional[np.ndarray] = None
    obstacle_positions: List[Tuple[float, float]] = None  # 新增：障碍物位置列表

    def __post_init__(self):
        if self.safe_directions is None:
            self.safe_directions = []
        if self.obstacle_positions is None:
            self.obstacle_positions = []


class VectorFieldPlanner:
    """向量场规划器"""
    def __init__(self):
        self.repulsion_gain = config.INTELLIGENT_DECISION['OBSTACLE_REPULSION_GAIN']
        self.attraction_gain = config.INTELLIGENT_DECISION['GOAL_ATTRACTION_GAIN']
        self.field_radius = config.INTELLIGENT_DECISION['VECTOR_FIELD_RADIUS']
        self.smoothing_factor = config.INTELLIGENT_DECISION['SMOOTHING_FACTOR']

        # 修复：使用正确的配置键名，将角度转换为弧度
        self.min_turn_angle = math.radians(config.INTELLIGENT_DECISION['MIN_TURN_ANGLE_DEG'])
        self.max_turn_angle = math.radians(config.INTELLIGENT_DECISION['MAX_TURN_ANGLE_DEG'])

        # 历史向量用于平滑
        self.vector_history = deque(maxlen=config.INTELLIGENT_DECISION['SMOOTHING_WINDOW_SIZE'])
        self.current_vector = Vector2D()

    def compute_vector(self, current_pos, goal_pos, obstacles):
        """计算合成向量"""
        # 目标吸引力
        attraction_vector = self._compute_attraction(current_pos, goal_pos)

        # 障碍物排斥力
        repulsion_vector = self._compute_repulsion(current_pos, obstacles)

        # 合成向量
        combined_vector = attraction_vector + repulsion_vector

        # 向量平滑
        smoothed_vector = self._smooth_vector(combined_vector)

        # 限制转向角度
        limited_vector = self._limit_turn_angle(smoothed_vector)

        self.current_vector = limited_vector
        return limited_vector

    def _compute_attraction(self, current_pos, goal_pos):
        """计算目标吸引力"""
        if goal_pos is None:
            return Vector2D()

        # 计算朝向目标的向量
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance < 0.1:  # 已到达目标
            return Vector2D()

        # 吸引力与距离成反比（接近目标时减速）
        strength = min(self.attraction_gain, self.attraction_gain / max(1.0, distance))

        return Vector2D(dx, dy).normalize() * strength

    def _compute_repulsion(self, current_pos, obstacles):
        """计算障碍物排斥力"""
        repulsion = Vector2D()

        for obs_x, obs_y in obstacles:
            dx = current_pos[0] - obs_x
            dy = current_pos[1] - obs_y
            distance = math.sqrt(dx**2 + dy**2)

            if distance < self.field_radius and distance > 0.1:
                # 排斥力与距离平方成反比
                strength = self.repulsion_gain * (1.0 / distance**2)
                direction = Vector2D(dx, dy).normalize()
                repulsion += direction * strength

        return repulsion

    def _smooth_vector(self, new_vector):
        """平滑向量"""
        self.vector_history.append(new_vector)

        if len(self.vector_history) < 2:
            return new_vector

        # 指数加权平均
        smoothed = Vector2D()
        total_weight = 0.0

        for i, vec in enumerate(reversed(self.vector_history)):
            weight = math.exp(-i * self.smoothing_factor)
            smoothed += vec * weight
            total_weight += weight

        if total_weight > 0:
            smoothed = smoothed / total_weight

        return smoothed

    def _limit_turn_angle(self, vector):
        """限制转向角度"""
        if self.current_vector.magnitude() < 0.1:
            return vector

        current_angle = math.atan2(self.current_vector.y, self.current_vector.x)
        new_angle = math.atan2(vector.y, vector.x)

        angle_diff = new_angle - current_angle
        # 将角度差归一化到[-π, π]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # 限制角度变化率
        if abs(angle_diff) > self.max_turn_angle:
            angle_diff = math.copysign(self.max_turn_angle, angle_diff)
        elif abs(angle_diff) < self.min_turn_angle and vector.magnitude() > 0.1:
            angle_diff = math.copysign(self.min_turn_angle, angle_diff)

        # 应用限制后的角度
        magnitude = vector.magnitude()
        limited_angle = current_angle + angle_diff

        return Vector2D.from_angle(limited_angle, magnitude)


class FrontViewDisplay:
    """前视画面显示管理器"""

    def __init__(self, window_name="无人机前视画面", width=None, height=None,
                 enable_sharpening=None, show_info=None):
        self.window_name = window_name
        self.window_width = width if width is not None else config.DISPLAY['WINDOW_WIDTH']
        self.window_height = height if height is not None else config.DISPLAY['WINDOW_HEIGHT']
        self.enable_sharpening = (enable_sharpening if enable_sharpening is not None
                                 else config.DISPLAY['ENABLE_SHARPENING'])
        self.show_info = (show_info if show_info is not None
                         else config.DISPLAY['SHOW_INFO_OVERLAY'])

        # 图像队列
        self.image_queue = queue.Queue(maxsize=3)
        self.display_active = True
        self.display_thread = None
        self.paused = False

        # 手动控制状态
        self.manual_mode = False
        self.key_states = {}
        self.last_keys = {}

        # 控制退出标志
        self.exit_manual_flag = False
        self.exit_display_flag = False

        # 显示统计
        self.display_stats = {
            'fps': 0.0,
            'last_update': time.time(),
            'frame_count': 0
        }

        # 性能监控显示
        self.performance_info = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'update_time': time.time()
        }

        # 启动显示线程
        self.start()

    def start(self):
        """启动显示线程"""
        if self.display_thread and self.display_thread.is_alive():
            return

        self.display_active = True
        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="FrontViewDisplay"
        )
        self.display_thread.start()

    def stop(self):
        """停止显示线程"""
        self.display_active = False
        self.exit_display_flag = True
        if self.display_thread:
            self.display_thread.join(timeout=2.0)

    def update_image(self, image_data: np.ndarray, info: Optional[Dict] = None,
                     manual_info: Optional[List[str]] = None,
                     additional_images: Optional[Dict] = None):
        """更新要显示的图像"""
        if not self.display_active or self.paused or image_data is None:
            return

        try:
            # 图像增强处理
            if self.enable_sharpening and image_data is not None and image_data.size > 0:
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
                image_data = cv2.filter2D(image_data, -1, kernel)

            # 如果队列已满，丢弃最旧的一帧
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass

            display_packet = {
                'image': image_data.copy(),
                'info': info.copy() if info else {},
                'manual_info': manual_info.copy() if manual_info else [],
                'additional_images': additional_images.copy() if additional_images else {},
                'timestamp': time.time()
            }

            self.image_queue.put_nowait(display_packet)

        except Exception as e:
            print(f"⚠️ 更新图像时出错: {e}")

    def update_performance_info(self, cpu_usage, memory_usage):
        """更新性能信息"""
        self.performance_info['cpu_usage'] = cpu_usage
        self.performance_info['memory_usage'] = memory_usage
        self.performance_info['update_time'] = time.time()

    def set_manual_mode(self, manual_mode):
        """设置手动模式状态"""
        self.manual_mode = manual_mode
        self.exit_manual_flag = False
        self.key_states = {}
        self.last_keys = {}
        print(f"🔄 {'进入' if manual_mode else '退出'}手动控制模式")

    def get_control_inputs(self):
        """获取当前控制输入"""
        return self.key_states.copy()

    def should_exit_manual(self):
        """检查是否应该退出手动模式"""
        return self.exit_manual_flag

    def _display_loop(self):
        """显示线程主循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        # 初始显示等待画面
        wait_img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(wait_img, "等待无人机图像...", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(self.window_name, wait_img)
        cv2.waitKey(100)

        print("💡 前视窗口控制:")
        print("   - 通用控制: P=暂停/继续, I=信息显示, H=锐化效果")
        print("   - 非手动模式: Q=关闭窗口, S=保存截图")
        print("   - 手动模式: ESC=退出手动模式")
        print("\n🎮 手动控制键位:")
        print("   - W/S: 前进/后退, A/D: 左移/右移")
        print("   - Q/E: 上升/下降, Z/X: 左转/右转")
        print("   - 空格: 悬停, ESC: 退出手动模式")

        while self.display_active and not self.exit_display_flag:
            display_image = None
            info = {}
            manual_info = []
            additional_images = {}

            try:
                # 获取最新图像
                if not self.image_queue.empty():
                    packet = self.image_queue.get_nowait()
                    display_image = packet['image']
                    info = packet['info']
                    manual_info = packet['manual_info']
                    additional_images = packet.get('additional_images', {})

                    # 更新统计
                    self._update_stats()

                    # 清空队列中的旧帧
                    while not self.image_queue.empty():
                        try:
                            self.image_queue.get_nowait()
                        except queue.Empty:
                            break
            except queue.Empty:
                pass

            # 显示图像
            if display_image is not None:
                # 添加信息叠加
                if self.show_info:
                    display_image = self._add_info_overlay(display_image, info, manual_info, additional_images)

                cv2.imshow(self.window_name, display_image)
            elif self.paused:
                # 暂停时显示提示
                blank = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(blank, "PAUSED", (120, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow(self.window_name, blank)

            # 处理键盘输入（非阻塞）
            key = cv2.waitKey(config.DISPLAY.get('REFRESH_RATE_MS', 30)) & 0xFF

            # 记录当前按键
            current_keys = {}
            if key != 255:  # 有按键按下
                current_keys[key] = True

                # 根据模式处理按键
                if self.manual_mode:
                    self._handle_manual_mode_key(key)
                else:
                    self._handle_window_control_key(key, display_image)

            # 更新按键状态
            self._update_key_states(current_keys)

            # 检查窗口是否被关闭
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("🔄 用户关闭了前视窗口")
                    self.display_active = False
                    break
            except:
                self.display_active = False
                break

        # 清理窗口
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
        cv2.waitKey(1)

    def _handle_manual_mode_key(self, key):
        """处理手动控制模式下的按键"""
        key_char = chr(key).lower() if 0 <= key <= 255 else ''

        # ESC键：退出手动模式
        if key == 27:  # ESC
            print("收到退出手动模式指令")
            self.exit_manual_flag = True
            return

        # 记录手动控制按键
        self.key_states[key] = True

        # 特别处理空格键（悬停）
        if key == 32:  # 空格
            print("⏸️ 悬停指令")

    def _handle_window_control_key(self, key, display_image):
        """处理通用窗口控制按键"""
        key_char = chr(key).lower() if 0 <= key <= 255 else ''

        if key_char == 'q':
            print("🔄 用户关闭显示窗口")
            self.display_active = False
        elif key_char == 's' and display_image is not None:
            self._save_screenshot(display_image)
        elif key_char == 'p':
            self.paused = not self.paused
            status = "已暂停" if self.paused else "已恢复"
            print(f"⏸️ 视频流{status}")
        elif key_char == 'i':
            self.show_info = not self.show_info
            status = "开启" if self.show_info else "关闭"
            print(f"📊 信息叠加层{status}")
        elif key_char == 'h':
            self.enable_sharpening = not self.enable_sharpening
            status = "开启" if self.enable_sharpening else "关闭"
            print(f"🔍 图像锐化{status}")

    def _update_key_states(self, current_keys):
        """更新按键状态，检测按键释放"""
        # 找出被释放的键
        released_keys = []
        for key in list(self.key_states.keys()):
            if key not in current_keys:
                released_keys.append(key)

        # 移除已释放的键
        for key in released_keys:
            del self.key_states[key]

        # 保存当前按键状态
        self.last_keys = current_keys.copy()

    def _update_stats(self):
        """更新显示统计信息"""
        now = time.time()
        self.display_stats['frame_count'] += 1

        if now - self.display_stats['last_update'] >= 1.0:
            self.display_stats['fps'] = self.display_stats['frame_count'] / (now - self.display_stats['last_update'])
            self.display_stats['frame_count'] = 0
            self.display_stats['last_update'] = now

    def _add_info_overlay(self, image: np.ndarray, info: Dict, manual_info: List[str] = None,
                         additional_images: Dict = None) -> np.ndarray:
        """在图像上叠加状态信息"""
        if image is None or image.size == 0:
            return image

        try:
            overlay = image.copy()
            height, width = image.shape[:2]

            # 判断是否为手动模式
            is_manual = info.get('state', '') == "手动控制"

            # 如果有附加图像（如网格图），调整信息栏高度
            grid_img = additional_images.get('grid') if additional_images else None
            info_height = 180 if (is_manual and manual_info) or grid_img is not None else 100

            # 创建半透明信息栏
            cv2.rectangle(overlay, (0, 0), (width, info_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # 飞行状态
            state = info.get('state', 'UNKNOWN')
            state_color = (0, 255, 0) if '探索' in state else (0, 255, 255) if '悬停' in state else (255, 255, 0) if '手动' in state else (0, 0, 255)
            cv2.putText(image, f"状态: {state}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

            # 位置信息
            pos = info.get('position', (0, 0, 0))
            cv2.putText(image, f"位置: ({pos[0]:.1f}, {pos[1]:.1f}, {-pos[2]:.1f}m)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 智能决策信息
            decision_info = info.get('decision_info', {})
            if decision_info:
                y_pos = 90
                for key, value in decision_info.items():
                    if key == 'vector_angle':
                        cv2.putText(image, f"方向: {math.degrees(value):.0f}°", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                        y_pos += 20
                    elif key == 'grid_score':
                        cv2.putText(image, f"探索得分: {value:.2f}", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
                        y_pos += 20

            # 手动控制信息
            if is_manual and manual_info:
                y_start = 130
                for i, line in enumerate(manual_info):
                    y_pos = y_start + i * 20
                    cv2.putText(image, line, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

                cv2.putText(image, "手动控制中...", (width - 150, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            elif not is_manual:
                # 障碍物信息
                obs_dist = info.get('obstacle_distance', 0.0)
                obs_color = (0, 0, 255) if obs_dist < 5.0 else (0, 165, 255) if obs_dist < 10.0 else (0, 255, 0)
                cv2.putText(image, f"障碍: {obs_dist:.1f}m", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, obs_color, 2)

            # 显示统计信息
            fps_text = f"FPS: {self.display_stats['fps']:.1f}"
            cv2.putText(image, fps_text, (width - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            # 性能信息
            if self.performance_info['cpu_usage'] > 0:
                cpu_text = f"CPU: {self.performance_info['cpu_usage']:.1f}%"
                mem_text = f"MEM: {self.performance_info['memory_usage']:.1f}%"

                cpu_color = (0, 200, 255) if self.performance_info['cpu_usage'] > 80 else (0, 255, 0)
                mem_color = (0, 200, 255) if self.performance_info['memory_usage'] > 80 else (0, 255, 0)

                cv2.putText(image, cpu_text, (width - 120, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, cpu_color, 1)
                cv2.putText(image, mem_text, (width - 120, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, mem_color, 1)

            # 如果有网格图，显示在右上角
            if grid_img is not None and grid_img.size > 0:
                grid_size = 150
                grid_resized = cv2.resize(grid_img, (grid_size, grid_size))

                # 在图像右上角放置网格图
                x_offset = width - grid_size - 10
                y_offset = info_height + 10

                if y_offset + grid_size < height:
                    # 创建网格图的背景
                    cv2.rectangle(image, (x_offset-2, y_offset-2),
                                 (x_offset+grid_size+2, y_offset+grid_size+2),
                                 (255, 255, 255), 1)

                    # 将网格图放入指定位置
                    image[y_offset:y_offset+grid_size, x_offset:x_offset+grid_size] = grid_resized

                    cv2.putText(image, "探索网格", (x_offset, y_offset-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return image
        except Exception as e:
            print(f"⚠️ 添加信息叠加层出错: {e}")
            return image

    def _save_screenshot(self, image: Optional[np.ndarray]):
        """保存当前画面为截图"""
        if image is not None and image.size > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"drone_snapshot_{timestamp}.png"
            cv2.imwrite(filename, image)
            print(f"📸 截图已保存: {filename}")
        else:
            print("⚠️ 无法保存截图：无有效图像数据")


class PerceptiveExplorer:
    """基于感知的自主探索无人机 - 智能决策增强版（修复版）"""

    def __init__(self, drone_name=""):
        # 初始化日志系统
        self._setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("AirSimNH 感知驱动自主探索系统 - 智能决策增强版（修复版）")
        self.logger.info("=" * 60)

        # 初始化AirSim连接
        self.client = None
        self.drone_name = drone_name
        self._connect_to_airsim()

        # 启用API控制
        try:
            self.client.enableApiControl(True, vehicle_name=drone_name)
            self.client.armDisarm(True, vehicle_name=drone_name)
            self.logger.info("✅ API控制已启用")
        except Exception as e:
            self.logger.error(f"❌ 启用API控制失败: {e}")
            raise

        # 状态管理
        self.state = FlightState.TAKEOFF
        self.state_history = deque(maxlen=20)
        self.emergency_flag = False

        # 从配置文件读取参数
        self.depth_threshold_near = config.PERCEPTION['DEPTH_NEAR_THRESHOLD']
        self.depth_threshold_safe = config.PERCEPTION['DEPTH_SAFE_THRESHOLD']
        self.min_ground_clearance = config.PERCEPTION['MIN_GROUND_CLEARANCE']
        self.max_pitch_angle = math.radians(config.PERCEPTION['MAX_PITCH_ANGLE_DEG'])
        self.scan_angles = config.PERCEPTION['SCAN_ANGLES']

        # 探索参数
        self.exploration_time = config.EXPLORATION['TOTAL_TIME']
        self.preferred_speed = config.EXPLORATION['PREFERRED_SPEED']
        self.max_altitude = config.EXPLORATION['MAX_ALTITUDE']
        self.min_altitude = config.EXPLORATION['MIN_ALTITUDE']
        self.base_height = config.EXPLORATION['BASE_HEIGHT']
        self.takeoff_height = config.EXPLORATION['TAKEOFF_HEIGHT']

        # 智能决策组件
        self.vector_planner = VectorFieldPlanner()
        self.exploration_grid = ExplorationGrid(
            resolution=config.INTELLIGENT_DECISION['GRID_RESOLUTION'],
            grid_size=config.INTELLIGENT_DECISION['GRID_SIZE']
        )

        # PID控制器
        self.velocity_pid = PIDController(
            config.INTELLIGENT_DECISION['PID_KP'],
            config.INTELLIGENT_DECISION['PID_KI'],
            config.INTELLIGENT_DECISION['PID_KD']
        )
        self.height_pid = PIDController(1.0, 0.1, 0.3)

        # 探索目标
        self.exploration_target = None
        self.target_update_time = 0
        # 修复：使用配置中的目标有效期
        self.target_lifetime = config.INTELLIGENT_DECISION.get('TARGET_LIFETIME', 15.0)
        self.target_reached_distance = config.INTELLIGENT_DECISION.get('TARGET_REACHED_DISTANCE', 3.0)

        # 记忆系统
        self.visited_positions = deque(maxlen=100)

        # 性能监控与健康检查
        self.loop_count = 0
        self.start_time = time.time()
        self.last_health_check = 0
        self.reconnect_attempts = 0
        self.last_successful_loop = time.time()

        # 数据记录器
        self.data_logger = None
        self.last_data_record_time = 0
        self.data_record_interval = config.DATA_RECORDING.get('RECORD_INTERVAL', 0.2)
        if config.DATA_RECORDING['ENABLED']:
            self._setup_data_logger()

        # 性能监控
        self.last_performance_report = time.time()
        self.performance_report_interval = config.PERFORMANCE.get('REPORT_INTERVAL', 30.0)

        # 运行统计
        self.stats = {
            'perception_cycles': 0,
            'decision_cycles': 0,
            'exceptions_caught': 0,
            'obstacles_detected': 0,
            'state_changes': 0,
            'front_image_updates': 0,
            'manual_control_time': 0.0,
            'vector_field_updates': 0,
            'grid_updates': 0,
            'data_points_recorded': 0,
            'average_loop_time': 0.0,
            'max_loop_time': 0.0,
            'min_loop_time': 100.0,
        }

        # 前视窗口
        self.front_display = None
        self._setup_front_display()

        # 手动控制状态
        self.manual_control_start = 0
        self.control_keys = {}

        self.logger.info("✅ 系统初始化完成")
        self.logger.info(f"   开始时间: {datetime.now().strftime('%H:%M:%S')}")
        self.logger.info(f"   预计探索时长: {self.exploration_time}秒")
        self.logger.info(f"   智能决策: 向量场避障 + 网格探索")
        if config.DATA_RECORDING['ENABLED']:
            self.logger.info(f"   数据记录: CSV + JSON 格式")

    def _setup_logging(self):
        """配置日志系统"""
        self.logger = logging.getLogger('DroneExplorer')
        self.logger.setLevel(getattr(logging, config.SYSTEM['LOG_LEVEL']))

        # 清除已有的处理器，避免重复
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # 文件处理器（如果需要）
        if config.SYSTEM['LOG_TO_FILE']:
            try:
                file_handler = logging.FileHandler(config.SYSTEM['LOG_FILENAME'], encoding='utf-8')
                file_format = logging.Formatter('%(asctime)s | %(name)s | %(levelname)-8s | %(message)s')
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)
                self.logger.info(f"📝 日志将保存至: {config.SYSTEM['LOG_FILENAME']}")
            except Exception as e:
                print(f"⚠️ 无法创建日志文件: {e}")

    def _setup_data_logger(self):
        """初始化数据记录器"""
        try:
            self.data_logger = DataLogger(
                enable_csv=config.DATA_RECORDING['SAVE_TO_CSV'],
                enable_json=config.DATA_RECORDING['SAVE_TO_JSON'],
                csv_filename=config.DATA_RECORDING.get('CSV_FILENAME'),
                json_filename=config.DATA_RECORDING.get('JSON_FILENAME')
            )
            self.logger.info("📊 数据记录器初始化完成")
        except Exception as e:
            self.logger.error(f"❌ 数据记录器初始化失败: {e}")
            self.data_logger = None

    def _connect_to_airsim(self):
        """连接到AirSim，支持重试机制"""
        max_attempts = config.SYSTEM['MAX_RECONNECT_ATTEMPTS']
        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"🔄 尝试连接到AirSim (第{attempt}次)...")
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                self.logger.info("✅ 成功连接到AirSim")
                self.reconnect_attempts = 0
                return
            except ConnectionRefusedError:
                self.logger.warning(f"❌ 连接被拒绝，请确保AirSim正在运行")
            except Exception as e:
                self.logger.warning(f"❌ 连接失败: {e}")

            if attempt < max_attempts:
                self.logger.info(f"⏳ {config.SYSTEM['RECONNECT_DELAY']}秒后重试...")
                time.sleep(config.SYSTEM['RECONNECT_DELAY'])

        self.logger.error(f"❌ 经过{max_attempts}次尝试后仍无法连接到AirSim")
        self.logger.error("请检查：1. AirSim是否启动 2. 网络设置 3. 防火墙")
        sys.exit(1)

    def _check_connection_health(self):
        """检查连接健康状态 - 新增方法"""
        try:
            # 简单的心跳检查
            self.client.ping()
            self.logger.debug("✅ 连接健康检查通过")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ 连接健康检查失败: {e}")
            # 尝试重新连接
            try:
                self._connect_to_airsim()
                return True
            except:
                return False

    def _setup_front_display(self):
        """初始化前视显示窗口"""
        try:
            self.front_display = FrontViewDisplay(
                window_name=f"无人机前视 - {self.drone_name or 'AirSimNH'}",
                width=config.DISPLAY['WINDOW_WIDTH'],
                height=config.DISPLAY['WINDOW_HEIGHT'],
                enable_sharpening=config.DISPLAY['ENABLE_SHARPENING'],
                show_info=config.DISPLAY['SHOW_INFO_OVERLAY']
            )
            self.logger.info("🎥 前视窗口已初始化")
        except Exception as e:
            self.logger.error(f"❌ 前视窗口初始化失败: {e}")
            self.front_display = None

    def get_depth_perception(self) -> PerceptionResult:
        """获取并分析深度图像，理解环境 - 增强版（修复健康检查）"""
        result = PerceptionResult()
        self.stats['perception_cycles'] += 1

        try:
            # 健康检查 - 修复：使用新方法
            if config.SYSTEM['ENABLE_HEALTH_CHECK']:
                current_time = time.time()
                if current_time - self.last_successful_loop > 10.0:
                    self.logger.warning("⚠️ 感知循环长时间无响应，尝试恢复...")
                    self._check_connection_health()

            # 请求深度图像和前视图像
            camera_name = config.CAMERA['DEFAULT_NAME']
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    camera_name,
                    airsim.ImageType.DepthPlanar,
                    pixels_as_float=True,
                    compress=False
                ),
                airsim.ImageRequest(
                    camera_name,
                    airsim.ImageType.Scene,
                    False,
                    False
                )
            ])

            if not responses or len(responses) < 2:
                self.logger.warning("⚠️ 图像获取失败：响应为空或数量不足")
                return result

            # 处理深度图像
            depth_img = responses[0]
            if depth_img and hasattr(depth_img, 'image_data_float'):
                try:
                    depth_array = np.array(depth_img.image_data_float, dtype=np.float32)
                    depth_array = depth_array.reshape(depth_img.height, depth_img.width)

                    # 分析深度图像的不同区域
                    h, w = depth_array.shape

                    # 前方近距离区域（紧急避障）
                    front_near = depth_array[h // 2:, w // 3:2 * w // 3]
                    min_front_distance = np.min(front_near) if front_near.size > 0 else 100

                    # 多方向扇形扫描
                    directions = []
                    for angle_deg in self.scan_angles:
                        angle_rad = math.radians(angle_deg)
                        col = int(w / 2 + (w / 2) * math.tan(angle_rad) * 0.5)
                        col = max(0, min(w - 1, col))

                        col_data = depth_array[h // 2:, col]
                        if col_data.size > 0:
                            dir_distance = np.percentile(col_data, 25)
                            directions.append((angle_rad, dir_distance))

                            if dir_distance > self.depth_threshold_safe:
                                result.safe_directions.append(angle_rad)

                    # 提取障碍物位置（用于向量场）
                    result.obstacle_positions = self._extract_obstacle_positions(depth_array, h, w)

                    # 地形分析
                    ground_region = depth_array[3 * h // 4:, :]
                    if ground_region.size > 10:
                        row_variances = np.var(ground_region, axis=1)
                        result.terrain_slope = np.mean(row_variances) * 100

                    # 开阔度评分
                    open_pixels = np.sum(depth_array[h // 2:, :] > self.depth_threshold_safe)
                    total_pixels = depth_array[h // 2:, :].size
                    result.open_space_score = open_pixels / total_pixels if total_pixels > 0 else 0

                    # 整合感知结果
                    result.has_obstacle = min_front_distance < self.depth_threshold_near
                    result.obstacle_distance = min_front_distance
                    if result.has_obstacle:
                        self.stats['obstacles_detected'] += 1

                    if directions:
                        closest_dir = min(directions, key=lambda x: x[1])
                        result.obstacle_direction = closest_dir[0]

                    # 根据感知动态调整推荐高度
                    if result.terrain_slope > config.PERCEPTION['HEIGHT_STRATEGY']['SLOPE_THRESHOLD']:
                        result.recommended_height = config.PERCEPTION['HEIGHT_STRATEGY']['STEEP_SLOPE']
                    elif result.open_space_score > config.PERCEPTION['HEIGHT_STRATEGY']['OPENNESS_THRESHOLD']:
                        result.recommended_height = config.PERCEPTION['HEIGHT_STRATEGY']['OPEN_SPACE']

                except ValueError as e:
                    self.logger.error(f"❌ 深度图像数据转换错误: {e}")
                    return result
                except Exception as e:
                    self.logger.error(f"❌ 深度图像处理异常: {e}")
                    return result

            # 处理前视图像
            front_response = responses[1]
            if front_response and hasattr(front_response, 'image_data_uint8'):
                try:
                    # 转换为OpenCV格式
                    img_array = np.frombuffer(front_response.image_data_uint8, dtype=np.uint8)

                    if len(img_array) > 0:
                        img_rgb = img_array.reshape(front_response.height, front_response.width, 3)
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        result.front_image = img_bgr

                        # 准备显示信息
                        display_info = self._prepare_display_info(result)

                        # 更新探索网格
                        self._update_exploration_grid(result)

                        # 记录数据
                        self._record_flight_data(result)

                        # 更新前视窗口
                        if self.front_display:
                            manual_info = None
                            if self.state == FlightState.MANUAL:
                                manual_info = self._get_manual_control_info()

                            # 获取网格可视化图像
                            grid_img = self.exploration_grid.visualize_grid(size=150)
                            additional_images = {'grid': grid_img} if grid_img is not None else {}

                            # 更新性能信息
                            if config.PERFORMANCE['ENABLE_REALTIME_METRICS']:
                                cpu_usage = psutil.cpu_percent(interval=0)
                                memory_usage = psutil.virtual_memory().percent
                                self.front_display.update_performance_info(cpu_usage, memory_usage)

                            self.front_display.update_image(img_bgr, display_info, manual_info, additional_images)
                            self.stats['front_image_updates'] += 1

                except Exception as e:
                    self.logger.warning(f"⚠️ 前视图像处理异常: {e}")

            self.last_successful_loop = time.time()

            # 详细日志
            if self.loop_count % 50 == 0 and config.DEBUG.get('LOG_DECISION_DETAILS', False):
                self.logger.debug(f"感知结果: 障碍={result.has_obstacle}, 距离={result.obstacle_distance:.1f}m, "
                                f"开阔度={result.open_space_score:.2f}, 障碍物数={len(result.obstacle_positions)}")

        except Exception as e:  # 修复：捕获通用异常
            if "ClientException" in str(type(e)) or "Connection" in str(e):
                self.logger.error(f"❌ AirSim客户端异常: {e}")
                self.stats['exceptions_caught'] += 1
                # 记录事件
                if self.data_logger:
                    self.data_logger.record_event('airsim_exception', {'error': str(e)})
                # 尝试重新连接
                self._check_connection_health()
            else:
                self.logger.error(f"❌ 感知过程中发生未知异常: {e}")
                self.logger.debug(f"异常详情: {traceback.format_exc()}")
                self.stats['exceptions_caught'] += 1
                if self.data_logger:
                    self.data_logger.record_event('perception_exception', {'error': str(e)})

        return result

    def _record_flight_data(self, perception: PerceptionResult):
        """记录飞行数据"""
        if not config.DATA_RECORDING['ENABLED'] or not self.data_logger:
            return

        current_time = time.time()
        if current_time - self.last_data_record_time < self.data_record_interval:
            return

        try:
            # 获取无人机状态
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            orientation = state.kinematics_estimated.orientation

            # 转换为欧拉角
            roll, pitch, yaw = airsim.to_eularian_angles(orientation)

            # 收集系统性能数据
            cpu_usage = psutil.cpu_percent(interval=0) if config.PERFORMANCE['ENABLE_REALTIME_METRICS'] else 0.0
            memory_usage = psutil.virtual_memory().percent if config.PERFORMANCE['ENABLE_REALTIME_METRICS'] else 0.0

            # 构建数据字典
            data_dict = {
                'timestamp': datetime.now().isoformat(),
                'loop_count': self.loop_count,
                'state': self.state.value,
                'pos_x': pos.x_val,
                'pos_y': pos.y_val,
                'pos_z': pos.z_val,
                'vel_x': vel.x_val,
                'vel_y': vel.y_val,
                'vel_z': vel.z_val,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'obstacle_distance': perception.obstacle_distance,
                'open_space_score': perception.open_space_score,
                'terrain_slope': perception.terrain_slope,
                'has_obstacle': perception.has_obstacle,
                'obstacle_direction': perception.obstacle_direction,
                'recommended_height': perception.recommended_height,
                'target_x': self.exploration_target[0] if self.exploration_target else 0.0,
                'target_y': self.exploration_target[1] if self.exploration_target else 0.0,
                'target_z': perception.recommended_height,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'grid_frontiers': len(self.exploration_grid.frontier_cells),
                'grid_explored': np.sum(self.exploration_grid.grid > 0.7),
                'adaptive_speed_factor': self._calculate_adaptive_speed(perception, 0) if hasattr(self, '_calculate_adaptive_speed') else 1.0,
            }

            # 记录数据
            self.data_logger.record_flight_data(data_dict)
            self.stats['data_points_recorded'] += 1
            self.last_data_record_time = current_time

        except Exception as e:
            self.logger.warning(f"⚠️ 记录飞行数据时出错: {e}")

    def _extract_obstacle_positions(self, depth_array, height, width):
        """从深度图像中提取障碍物位置"""
        obstacles = []

        try:
            # 获取当前无人机位置和朝向
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            roll, pitch, yaw = airsim.to_eularian_angles(orientation)

            # 只处理近距离障碍物
            near_mask = depth_array < self.depth_threshold_near * 1.5

            # 采样障碍点
            step = 4  # 采样步长
            for i in range(0, height, step):
                for j in range(0, width, step):
                    if near_mask[i, j]:
                        distance = depth_array[i, j]

                        # 将像素坐标转换为相对于相机的3D坐标
                        # 简化模型：假设相机水平视角为90度
                        fov_h = math.radians(90)
                        pixel_angle_x = (j - width/2) / (width/2) * (fov_h/2)
                        pixel_angle_y = (i - height/2) / (height/2) * (fov_h/2)

                        # 计算相对位置
                        z = distance  # 深度方向
                        x = z * math.tan(pixel_angle_x)
                        y = z * math.tan(pixel_angle_y)

                        # 旋转到世界坐标系（考虑无人机偏航角）
                        world_x = x * math.cos(yaw) - y * math.sin(yaw) + pos.x_val
                        world_y = x * math.sin(yaw) + y * math.cos(yaw) + pos.y_val

                        obstacles.append((world_x, world_y))

            # 限制障碍物数量
            max_obstacles = 20
            if len(obstacles) > max_obstacles:
                obstacles = random.sample(obstacles, max_obstacles)

        except Exception as e:
            self.logger.warning(f"⚠️ 提取障碍物位置失败: {e}")

        return obstacles

    def _update_exploration_grid(self, perception: PerceptionResult):
        """更新探索网格"""
        try:
            # 获取当前位置
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position

            # 更新网格位置
            self.exploration_grid.update_position(pos.x_val, pos.y_val)

            # 更新障碍物
            if perception.obstacle_positions:
                self.exploration_grid.update_obstacles(perception.obstacle_positions)

            self.stats['grid_updates'] += 1

        except Exception as e:
            self.logger.warning(f"⚠️ 更新探索网格失败: {e}")

    def _prepare_display_info(self, perception: PerceptionResult) -> Dict:
        """准备显示信息"""
        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position

            info = {
                'state': self.state.value,
                'obstacle_distance': perception.obstacle_distance,
                'position': (pos.x_val, pos.y_val, pos.z_val),
                'loop_count': self.loop_count,
            }

            # 添加决策信息
            if hasattr(self, 'last_decision_info'):
                info['decision_info'] = self.last_decision_info

            # 添加数据记录信息
            if config.DATA_RECORDING['ENABLED']:
                info['data_points'] = self.stats['data_points_recorded']

            return info
        except:
            return {}

    def _get_manual_control_info(self):
        """获取手动控制信息"""
        info_lines = []

        # 控制状态
        if self.control_keys:
            key_names = []
            for key in self.control_keys:
                if key == ord('w'):
                    key_names.append("前进")
                elif key == ord('s'):
                    key_names.append("后退")
                elif key == ord('a'):
                    key_names.append("左移")
                elif key == ord('d'):
                    key_names.append("右移")
                elif key == ord('q'):
                    key_names.append("上升")
                elif key == ord('e'):
                    key_names.append("下降")
                elif key == ord('z'):
                    key_names.append("左转")
                elif key == ord('x'):
                    key_names.append("右转")
                elif key == 32:  # 空格
                    key_names.append("悬停")

            if key_names:
                info_lines.append(f"控制: {', '.join(key_names)}")
        else:
            info_lines.append("控制: 悬停")

        # 时间信息
        if self.manual_control_start > 0:
            elapsed = time.time() - self.manual_control_start
            info_lines.append(f"手动模式: {elapsed:.1f}秒")

        # 提示信息
        info_lines.append("ESC: 退出手动模式")

        return info_lines

    def apply_manual_control(self):
        """应用手动控制指令"""
        if self.state != FlightState.MANUAL:
            return

        try:
            # 获取当前无人机状态
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation

            # 计算偏航角
            _, _, yaw = airsim.to_eularian_angles(orientation)

            # 初始化控制向量
            vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0

            # 处理控制键
            for key in list(self.control_keys.keys()):
                key_char = chr(key).lower() if 0 <= key <= 255 else ''

                # 前后移动
                if key_char == 'w':
                    vx += config.MANUAL['CONTROL_SPEED'] * math.cos(yaw)
                    vy += config.MANUAL['CONTROL_SPEED'] * math.sin(yaw)
                elif key_char == 's':
                    vx -= config.MANUAL['CONTROL_SPEED'] * math.cos(yaw)
                    vy -= config.MANUAL['CONTROL_SPEED'] * math.sin(yaw)

                # 左右移动
                if key_char == 'a':
                    vx += config.MANUAL['CONTROL_SPEED'] * math.cos(yaw + math.pi/2)
                    vy += config.MANUAL['CONTROL_SPEED'] * math.sin(yaw + math.pi/2)
                elif key_char == 'd':
                    vx += config.MANUAL['CONTROL_SPEED'] * math.cos(yaw - math.pi/2)
                    vy += config.MANUAL['CONTROL_SPEED'] * math.sin(yaw - math.pi/2)

                # 垂直移动
                if key_char == 'q':
                    vz = -config.MANUAL['ALTITUDE_SPEED']  # AirSim中Z轴向下为正
                elif key_char == 'e':
                    vz = config.MANUAL['ALTITUDE_SPEED']

                # 偏航控制
                if key_char == 'z':
                    yaw_rate = -math.radians(config.MANUAL['YAW_SPEED'])
                elif key_char == 'x':
                    yaw_rate = math.radians(config.MANUAL['YAW_SPEED'])

                # 悬停
                if key == 32:  # 空格
                    self.client.hoverAsync(vehicle_name=self.drone_name)
                    self.control_keys = {}  # 清空控制键
                    return

            # 安全限制
            if config.MANUAL['SAFETY_ENABLED']:
                # 限制速度
                speed = math.sqrt(vx**2 + vy**2)
                if speed > config.MANUAL['MAX_MANUAL_SPEED']:
                    scale = config.MANUAL['MAX_MANUAL_SPEED'] / speed
                    vx *= scale
                    vy *= scale

                # 限制高度
                target_z = pos.z_val + vz * 0.1
                if target_z > config.MANUAL['MIN_ALTITUDE_LIMIT']:
                    vz = max(vz, (config.MANUAL['MIN_ALTITUDE_LIMIT'] - pos.z_val) * 10)
                if target_z < config.MANUAL['MAX_ALTITUDE_LIMIT']:
                    vz = min(vz, (config.MANUAL['MAX_ALTITUDE_LIMIT'] - pos.z_val) * 10)

            # 应用控制
            if vx != 0.0 or vy != 0.0 or vz != 0.0:
                self.client.moveByVelocityAsync(vx, vy, vz, 0.1, vehicle_name=self.drone_name)
            elif yaw_rate != 0.0:
                self.client.rotateByYawRateAsync(yaw_rate, 0.1, vehicle_name=self.drone_name)
            elif config.MANUAL['ENABLE_AUTO_HOVER'] and not self.control_keys:
                # 没有按键时自动悬停
                self.client.hoverAsync(vehicle_name=self.drone_name)

        except Exception as e:
            self.logger.warning(f"⚠️ 手动控制应用失败: {e}")

    def change_state(self, new_state: FlightState):
        """状态转换"""
        if self.state != new_state:
            old_state = self.state.value
            self.logger.info(f"🔄 状态转换: {old_state} → {new_state.value}")
            self.state = new_state
            self.state_history.append((time.time(), new_state))
            self.stats['state_changes'] += 1

            # 记录状态改变事件
            if self.data_logger:
                event_data = {
                    'old_state': old_state,
                    'new_state': new_state.value,
                    'loop_count': self.loop_count
                }
                self.data_logger.record_event('state_change', event_data)

    def run_manual_control(self):
        """手动控制模式"""
        self.logger.info("=" * 60)
        self.logger.info("启动手动控制模式")
        self.logger.info("=" * 60)

        if not self.front_display:
            self.logger.error("❌ 前视窗口未初始化")
            return

        try:
            # 切换到手动控制状态
            self.change_state(FlightState.MANUAL)
            self.manual_control_start = time.time()

            # 设置前视窗口为手动模式
            self.front_display.set_manual_mode(True)

            self.logger.info("🕹️ 进入手动控制模式")
            print("\n" + "="*60)
            print("🎮 手动控制模式已启动")
            print("="*60)
            print("控制键位:")
            print("  W: 前进 | S: 后退 | A: 左移 | D: 右移")
            print("  Q: 上升 | E: 下降 | Z: 左转 | X: 右转")
            print("  空格: 悬停 | ESC: 退出手动模式")
            print("="*60)
            print("💡 提示: 按键时控制持续生效，松开自动停止")
            print("        请在无人机前视窗口操作")
            print("="*60)

            # 清空控制键
            self.control_keys = {}

            # 手动控制主循环
            manual_active = True
            last_control_time = time.time()
            last_image_time = time.time()

            while manual_active and not self.emergency_flag:
                try:
                    # 检查是否应该退出
                    if self.front_display.should_exit_manual():
                        self.logger.info("收到退出手动模式指令")
                        manual_active = False
                        break

                    # 获取前视窗口的按键状态
                    if self.front_display:
                        window_keys = self.front_display.get_control_inputs()
                        self.control_keys = window_keys.copy()

                    # 检查前视窗口是否还在运行
                    if not self.front_display.display_active:
                        self.logger.info("前视窗口已关闭，退出手动模式")
                        manual_active = False
                        break

                    # 应用手动控制（限制频率）
                    current_time = time.time()
                    if current_time - last_control_time >= 0.05:  # 20Hz
                        self.apply_manual_control()
                        last_control_time = current_time

                    # 定期获取并显示图像
                    if current_time - last_image_time >= 0.1:  # 10Hz
                        try:
                            camera_name = config.CAMERA['DEFAULT_NAME']
                            responses = self.client.simGetImages([
                                airsim.ImageRequest(
                                    camera_name,
                                    airsim.ImageType.Scene,
                                    False,
                                    False
                                )
                            ])

                            if responses and responses[0] and hasattr(responses[0], 'image_data_uint8'):
                                img_array = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                                if len(img_array) > 0:
                                    img_rgb = img_array.reshape(responses[0].height, responses[0].width, 3)
                                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                                    # 准备显示信息
                                    try:
                                        state = self.client.getMultirotorState(vehicle_name=self.drone_name)
                                        pos = state.kinematics_estimated.position
                                        display_info = {
                                            'state': self.state.value,
                                            'position': (pos.x_val, pos.y_val, pos.z_val),
                                            'loop_count': self.loop_count,
                                        }
                                    except:
                                        display_info = {}

                                    # 更新前视窗口
                                    if self.front_display:
                                        manual_info = self._get_manual_control_info()
                                        self.front_display.update_image(img_bgr, display_info, manual_info)
                                        last_image_time = current_time
                        except Exception as img_error:
                            pass

                    # 短暂休眠
                    time.sleep(0.01)

                except KeyboardInterrupt:
                    self.logger.warning("⏹️ 用户中断手动控制")
                    manual_active = False
                    break
                except Exception as e:
                    self.logger.error(f"❌ 手动控制循环异常: {e}")
                    time.sleep(0.1)

            # 记录手动控制时间
            manual_time = time.time() - self.manual_control_start
            self.stats['manual_control_time'] = manual_time

            # 重置状态
            self.manual_control_start = 0
            self.control_keys = {}
            if self.front_display:
                self.front_display.set_manual_mode(False)

            # 停止运动，悬停
            try:
                self.client.hoverAsync(vehicle_name=self.drone_name).join()
            except:
                pass

            self.logger.info(f"⏱️  手动控制结束，持续时间: {manual_time:.1f}秒")

            # 回到悬停状态
            self.change_state(FlightState.HOVERING)

            # 询问用户下一步操作
            print("\n" + "="*60)
            print("手动控制模式已结束")
            print(f"控制时间: {manual_time:.1f}秒")
            print("="*60)
            print("请选择下一步操作:")
            print("  1. 继续自动探索")
            print("  2. 再次进入手动模式")
            print("  3. 降落并结束任务")
            print("="*60)

            choice = input("请输入选择 (1/2/3): ").strip()

            if choice == '1':
                self.logger.info("🔄 返回自动探索模式")
                remaining_time = self.exploration_time - (time.time() - self.start_time)
                if remaining_time > 10:
                    self.exploration_time = remaining_time
                    self.run_perception_loop()
                else:
                    self.logger.info("⏰ 剩余探索时间不足，开始返航")
                    self._finish_mission()
            elif choice == '2':
                self.logger.info("🔄 重新进入手动控制模式")
                self.run_manual_control()
            else:
                self.logger.info("🛬 用户选择结束任务")
                self._finish_mission()

        except Exception as e:
            self.logger.error(f"❌ 手动控制模式发生异常: {e}")
            self.logger.debug(f"异常堆栈: {traceback.format_exc()}")
            self.emergency_stop()

    def run_perception_loop(self):
        """主感知-决策-控制循环"""
        self.logger.info("=" * 60)
        self.logger.info("启动感知-决策-控制主循环")
        self.logger.info("=" * 60)

        try:
            # 起飞
            self.logger.info("🚀 起飞中...")
            self.client.takeoffAsync(vehicle_name=self.drone_name).join()
            time.sleep(2)

            # 上升到目标高度
            self.client.moveToZAsync(self.takeoff_height, 3, vehicle_name=self.drone_name).join()
            self.change_state(FlightState.HOVERING)
            time.sleep(2)

            # 主循环
            exploration_start = time.time()

            while (time.time() - exploration_start < self.exploration_time and
                   not self.emergency_flag):

                self.loop_count += 1
                loop_start = time.time()

                # 1. 感知阶段
                perception = self.get_depth_perception()

                # 2. 智能决策阶段
                decision = self.make_intelligent_decision(perception)

                # 3. 控制执行阶段
                self._execute_control_decision(decision)

                # 记录循环时间
                loop_time = time.time() - loop_start
                self.stats['average_loop_time'] = (self.stats['average_loop_time'] * (self.loop_count-1) + loop_time) / self.loop_count
                self.stats['max_loop_time'] = max(self.stats['max_loop_time'], loop_time)
                self.stats['min_loop_time'] = min(self.stats['min_loop_time'], loop_time)

                # 记录循环时间
                if self.data_logger:
                    self.data_logger.record_loop_time(loop_time)

                # 定期性能报告
                current_time = time.time()
                if current_time - self.last_performance_report >= self.performance_report_interval:
                    self._generate_performance_report()
                    self.last_performance_report = current_time

                # 定期状态报告
                if self.loop_count % config.SYSTEM.get('HEALTH_CHECK_INTERVAL', 20) == 0:
                    self._report_status(exploration_start, perception)

                # 循环频率控制
                loop_time = time.time() - loop_start
                if loop_time < 0.1:
                    time.sleep(0.1 - loop_time)

            # 正常结束
            self.logger.info("⏰ 探索时间到，开始返航")
            self._finish_mission()

        except KeyboardInterrupt:
            self.logger.warning("⏹️ 用户中断探索")
            self.emergency_stop()
        except Exception as e:
            self.logger.error(f"❌ 主循环发生异常: {e}")
            self.logger.debug(f"异常堆栈: {traceback.format_exc()}")
            self.emergency_stop()

    def _generate_performance_report(self):
        """生成性能报告"""
        try:
            if not config.PERFORMANCE['ENABLE_REALTIME_METRICS']:
                return

            # 获取当前性能数据
            cpu_usage = psutil.cpu_percent(interval=0)
            memory_usage = psutil.virtual_memory().percent

            # 检查警告
            warnings = []
            if cpu_usage > config.PERFORMANCE['CPU_WARNING_THRESHOLD']:
                warnings.append(f"⚠️ CPU使用率过高: {cpu_usage:.1f}%")

            if memory_usage > config.PERFORMANCE['MEMORY_WARNING_THRESHOLD']:
                warnings.append(f"⚠️ 内存使用率过高: {memory_usage:.1f}%")

            avg_loop_time = self.stats.get('average_loop_time', 0)
            if avg_loop_time > config.PERFORMANCE['LOOP_TIME_WARNING_THRESHOLD']:
                warnings.append(f"⚠️ 平均循环时间过长: {avg_loop_time*1000:.1f}ms")

            if warnings:
                self.logger.warning("📊 性能警告:")
                for warning in warnings:
                    self.logger.warning(f"  {warning}")

            # 记录性能数据
            if self.data_logger:
                performance_data = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'average_loop_time': avg_loop_time,
                    'max_loop_time': self.stats.get('max_loop_time', 0),
                    'min_loop_time': self.stats.get('min_loop_time', 0),
                    'warnings': warnings
                }
                self.data_logger.record_event('performance_report', performance_data)

        except Exception as e:
            self.logger.warning(f"⚠️ 生成性能报告时出错: {e}")

    def make_intelligent_decision(self, perception: PerceptionResult) -> Tuple[float, float, float, float]:
        """基于感知结果做出智能决策 - 增强版（修复配置键名）"""
        self.stats['decision_cycles'] += 1
        decision_start = time.time()

        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity

            target_vx, target_vy, target_z, target_yaw = 0.0, 0.0, perception.recommended_height, 0.0

            if self.state == FlightState.TAKEOFF:
                target_z = self.takeoff_height
                if pos.z_val < self.takeoff_height + 0.5:
                    self.change_state(FlightState.HOVERING)

            elif self.state == FlightState.HOVERING:
                # 扫描环境，选择探索目标
                target_yaw = (time.time() % 10) * 0.2

                # 检查是否需要更新探索目标
                current_time = time.time()
                if (self.exploration_target is None or
                    current_time - self.target_update_time > self.target_lifetime):

                    self.exploration_target = self.exploration_grid.get_best_exploration_target((pos.x_val, pos.y_val))
                    self.target_update_time = current_time

                    if self.exploration_target:
                        self.logger.info(f"🎯 新探索目标: {self.exploration_target[0]:.1f}, {self.exploration_target[1]:.1f}")

                if self.exploration_target:
                    self.change_state(FlightState.EXPLORING)

            elif self.state == FlightState.EXPLORING:
                if perception.has_obstacle:
                    self.change_state(FlightState.AVOIDING)
                    # 紧急避障：后退
                    target_vx, target_vy = -vel.x_val * 2, -vel.y_val * 2
                else:
                    # 使用向量场算法计算最佳方向
                    current_pos = (pos.x_val, pos.y_val)

                    # 获取探索目标
                    if self.exploration_target is None:
                        self.exploration_target = self.exploration_grid.get_best_exploration_target(current_pos)
                        self.target_update_time = time.time()

                    # 计算向量场
                    vector = self.vector_planner.compute_vector(
                        current_pos,
                        self.exploration_target,
                        perception.obstacle_positions
                    )

                    # 自适应速度调整
                    speed_factor = self._calculate_adaptive_speed(perception, vector.magnitude())

                    # 应用PID控制平滑速度
                    target_speed = self.preferred_speed * speed_factor
                    current_speed = math.sqrt(vel.x_val**2 + vel.y_val**2)
                    speed_error = target_speed - current_speed
                    speed_adjustment = self.velocity_pid.update(speed_error)

                    # 计算最终速度向量
                    final_vector = vector.normalize() * (target_speed + speed_adjustment)
                    target_vx = final_vector.x
                    target_vy = final_vector.y

                    self.stats['vector_field_updates'] += 1

                    # 保存决策信息用于显示
                    self.last_decision_info = {
                        'vector_angle': math.atan2(vector.y, vector.x),
                        'vector_magnitude': vector.magnitude(),
                        'grid_score': len(self.exploration_grid.frontier_cells) / 100.0,
                        'speed_factor': speed_factor,
                        'decision_time': time.time() - decision_start
                    }

                    # 检查是否到达目标附近
                    if self.exploration_target:
                        distance_to_target = math.sqrt(
                            (self.exploration_target[0] - current_pos[0])**2 +
                            (self.exploration_target[1] - current_pos[1])**2
                        )
                        if distance_to_target < self.target_reached_distance:  # 使用配置的距离
                            self.exploration_target = None
                            self.change_state(FlightState.HOVERING)
                            self.logger.info("✅ 到达探索目标")

            elif self.state == FlightState.AVOIDING:
                if perception.has_obstacle:
                    # 使用向量场进行避障
                    current_pos = (pos.x_val, pos.y_val)

                    # 计算远离障碍物的方向
                    avoid_vector = self.vector_planner.compute_vector(
                        current_pos,
                        None,  # 没有目标，只有排斥力
                        perception.obstacle_positions
                    )

                    if avoid_vector.magnitude() > 0.1:
                        avoid_vector = avoid_vector.normalize() * 1.5  # 避障速度
                        target_vx = avoid_vector.x
                        target_vy = avoid_vector.y

                    # 尝试改变高度
                    target_z = pos.z_val - 3
                else:
                    # 障碍物清除，回到探索状态
                    self.change_state(FlightState.EXPLORING)
                    time.sleep(1)

            elif self.state == FlightState.EMERGENCY:
                target_vx, target_vy, target_yaw = 0, 0, 0
                target_z = max(pos.z_val, -20)

            elif self.state == FlightState.PLANNING:
                # 路径规划状态（预留）
                target_vx, target_vy = 0, 0
                target_z = perception.recommended_height

            # 高度PID控制
            height_error = target_z - pos.z_val
            height_adjustment = self.height_pid.update(height_error)
            target_z += height_adjustment

            # 高度安全限制
            target_z = max(self.max_altitude, min(self.min_altitude, target_z))

            # 记录决策时间
            decision_time = time.time() - decision_start
            self.last_decision_info['total_decision_time'] = decision_time

            return target_vx, target_vy, target_z, target_yaw

        except Exception as e:
            self.logger.error(f"❌ 决策过程异常: {e}")
            # 记录异常事件
            if self.data_logger:
                self.data_logger.record_event('decision_exception', {'error': str(e)})
            return 0.0, 0.0, self.base_height, 0.0

    def _calculate_adaptive_speed(self, perception: PerceptionResult, vector_magnitude: float) -> float:
        """计算自适应速度因子"""
        if not config.INTELLIGENT_DECISION['ADAPTIVE_SPEED_ENABLED']:
            return 1.0

        # 基于开阔度调整速度
        open_factor = min(1.0, perception.open_space_score * 1.2)

        # 基于障碍物距离调整速度
        if perception.obstacle_distance < self.depth_threshold_near * 2:
            obs_factor = max(0.3, perception.obstacle_distance / (self.depth_threshold_near * 2))
        else:
            obs_factor = 1.0

        # 基于向量场稳定性调整速度
        vector_factor = min(1.0, vector_magnitude * 2)

        # 综合速度因子
        speed_factor = open_factor * obs_factor * vector_factor * 0.7

        # 限制在允许范围内
        speed_factor = max(
            config.INTELLIGENT_DECISION['MIN_SPEED_FACTOR'],
            min(config.INTELLIGENT_DECISION['MAX_SPEED_FACTOR'], speed_factor)
        )

        return speed_factor

    def _execute_control_decision(self, decision):
        """执行控制决策，增强平滑性"""
        try:
            target_vx, target_vy, target_z, target_yaw = decision

            if self.state in [FlightState.EXPLORING, FlightState.AVOIDING, FlightState.PLANNING]:
                # 使用平滑的速度控制
                self.client.moveByVelocityZAsync(
                    target_vx, target_vy, target_z, 0.5,  # 增加持续时间以平滑
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=target_yaw),
                    vehicle_name=self.drone_name
                )
            else:
                self.client.moveToPositionAsync(
                    0, 0, target_z, 2,
                    vehicle_name=self.drone_name
                )

            # 记录位置
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            self.visited_positions.append((pos.x_val, pos.y_val, pos.z_val))

        except Exception as e:
            self.logger.warning(f"⚠️ 控制指令执行失败: {e}")
            # 记录异常事件
            if self.data_logger:
                self.data_logger.record_event('control_exception', {'error': str(e)})
            try:
                self.client.hoverAsync(vehicle_name=self.drone_name).join()
            except:
                pass

    def _report_status(self, exploration_start, perception):
        """报告系统状态"""
        elapsed = time.time() - exploration_start
        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position

            self.logger.info(f"\n📊 系统状态报告 [循环#{self.loop_count}]")
            self.logger.info(f"   运行时间: {elapsed:.1f}s / {self.exploration_time}s")
            self.logger.info(f"   飞行状态: {self.state.value}")
            self.logger.info(f"   当前位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}m)")
            self.logger.info(f"   环境感知: 障碍{'有' if perception.has_obstacle else '无'} "
                            f"| 距离={perception.obstacle_distance:.1f}m "
                            f"| 开阔度={perception.open_space_score:.2f}")
            self.logger.info(f"   智能决策: 向量场{self.stats['vector_field_updates']}次 "
                            f"| 网格更新{self.stats['grid_updates']}次")
            self.logger.info(f"   探索网格: 前沿{len(self.exploration_grid.frontier_cells)}个")
            self.logger.info(f"   系统统计: 异常{self.stats['exceptions_caught']}次 "
                            f"| 状态切换{self.stats['state_changes']}次")
            self.logger.info(f"   数据记录: {self.stats['data_points_recorded']}个数据点")
            self.logger.info(f"   性能统计: 平均循环{self.stats['average_loop_time']*1000:.1f}ms "
                            f"| 最大{self.stats['max_loop_time']*1000:.1f}ms "
                            f"| 最小{self.stats['min_loop_time']*1000:.1f}ms")
            if self.stats['manual_control_time'] > 0:
                self.logger.info(f"   手动控制: {self.stats['manual_control_time']:.1f}秒")
        except:
            self.logger.info("状态报告: 无法获取无人机状态")

    def _finish_mission(self):
        """完成任务并生成总结报告"""
        self.logger.info("=" * 60)
        self.logger.info("探索任务完成，开始返航程序")
        self.logger.info("=" * 60)

        self.change_state(FlightState.RETURNING)

        try:
            # 返航
            self.logger.info("↩️ 返回起始区域...")
            self.client.moveToPositionAsync(0, 0, -10, 4, vehicle_name=self.drone_name).join()
            time.sleep(2)

            # 降落
            self.logger.info("🛬 降落中...")
            self.change_state(FlightState.LANDING)
            self.client.landAsync(vehicle_name=self.drone_name).join()
            time.sleep(3)

        except Exception as e:
            self.logger.error(f"❌ 降落过程中出现异常: {e}")

        finally:
            # 无论成功与否，都执行清理
            self._cleanup_system()

            # 生成最终报告
            self._generate_summary_report()

    def _cleanup_system(self):
        """清理系统资源"""
        self.logger.info("🧹 清理系统资源...")

        try:
            self.client.armDisarm(False, vehicle_name=self.drone_name)
            self.client.enableApiControl(False, vehicle_name=self.drone_name)
            self.logger.info("✅ 无人机控制已安全释放")
        except:
            self.logger.warning("⚠️ 释放控制时出现异常")

        # 关闭前视窗口
        if self.front_display:
            self.front_display.stop()
            self.logger.info("✅ 前视窗口已关闭")

        # 保存数据
        if self.data_logger:
            self.logger.info("💾 正在保存飞行数据...")
            self.data_logger.save_json_data()

            # 生成性能报告
            if config.PERFORMANCE['SAVE_PERFORMANCE_REPORT']:
                performance_report = self.data_logger.generate_performance_report()
                self.logger.info(performance_report)

                # 保存性能报告到文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"performance_report_{timestamp}.txt"
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(performance_report)
                self.logger.info(f"📄 性能报告已保存至: {report_filename}")

    def _generate_summary_report(self):
        """生成运行总结报告"""
        total_time = time.time() - self.start_time

        self.logger.info("\n" + "=" * 60)
        self.logger.info("🏁 任务总结报告")
        self.logger.info("=" * 60)
        self.logger.info(f"   总运行时间: {total_time:.1f}秒")
        self.logger.info(f"   总循环次数: {self.loop_count}")
        if total_time > 0:
            self.logger.info(f"   平均循环频率: {self.loop_count/total_time:.1f} Hz")
        self.logger.info(f"   探索航点数量: {len(self.visited_positions)}")
        self.logger.info(f"   状态切换次数: {self.stats['state_changes']}")
        self.logger.info(f"   检测到障碍次数: {self.stats['obstacles_detected']}")
        self.logger.info(f"   向量场计算次数: {self.stats['vector_field_updates']}")
        self.logger.info(f"   网格更新次数: {self.stats['grid_updates']}")
        self.logger.info(f"   探索前沿数量: {len(self.exploration_grid.frontier_cells)}")
        self.logger.info(f"   前视图像更新次数: {self.stats['front_image_updates']}")
        self.logger.info(f"   数据记录点数: {self.stats['data_points_recorded']}")
        self.logger.info(f"   手动控制时间: {self.stats['manual_control_time']:.1f}秒")
        self.logger.info(f"   捕获的异常数: {self.stats['exceptions_caught']}")
        self.logger.info(f"   重连尝试次数: {self.reconnect_attempts}")
        self.logger.info(f"   平均循环时间: {self.stats['average_loop_time']*1000:.1f}ms")
        self.logger.info(f"   最大循环时间: {self.stats['max_loop_time']*1000:.1f}ms")
        self.logger.info(f"   最小循环时间: {self.stats['min_loop_time']*1000:.1f}ms")

        # 保存报告到文件
        try:
            report_filename = f"mission_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("AirSimNH 无人机任务报告 (智能决策增强版 - 数据闭环版)\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总运行时间: {total_time:.1f}秒\n")
                f.write(f"总循环次数: {self.loop_count}\n")
                f.write(f"探索航点数量: {len(self.visited_positions)}\n")
                f.write(f"状态切换次数: {self.stats['state_changes']}\n")
                f.write(f"向量场计算次数: {self.stats['vector_field_updates']}\n")
                f.write(f"网格更新次数: {self.stats['grid_updates']}\n")
                f.write(f"探索前沿数量: {len(self.exploration_grid.frontier_cells)}\n")
                f.write(f"数据记录点数: {self.stats['data_points_recorded']}\n")
                f.write(f"手动控制时间: {self.stats['manual_control_time']:.1f}秒\n")
                f.write(f"异常捕获次数: {self.stats['exceptions_caught']}\n")
                f.write(f"前视图像更新次数: {self.stats['front_image_updates']}\n")
                f.write(f"平均循环时间: {self.stats['average_loop_time']*1000:.1f}ms\n")
                f.write(f"最大循环时间: {self.stats['max_loop_time']*1000:.1f}ms\n")
                f.write(f"最小循环时间: {self.stats['min_loop_time']*1000:.1f}ms\n")
                f.write("=" * 50 + "\n")
                f.write("智能决策配置:\n")
                for key, value in config.INTELLIGENT_DECISION.items():
                    f.write(f"  {key}: {value}\n")
                f.write("=" * 50 + "\n")
                f.write("飞行航点记录:\n")
                for i, pos in enumerate(self.visited_positions[:20]):
                    f.write(f"  航点{i+1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n")
                if len(self.visited_positions) > 20:
                    f.write(f"  ... 还有{len(self.visited_positions)-20}个航点\n")
                f.write("=" * 50 + "\n")
                f.write("数据记录信息:\n")
                if self.data_logger and config.DATA_RECORDING['ENABLED']:
                    f.write(f"  CSV文件: {self.data_logger.csv_filename}\n")
                    f.write(f"  JSON文件: {self.data_logger.json_filename}\n")
                else:
                    f.write("  数据记录未启用\n")
            self.logger.info(f"📄 详细报告已保存至: {report_filename}")
        except Exception as e:
            self.logger.warning(f"⚠️ 无法保存报告文件: {e}")

    def emergency_stop(self):
        """紧急停止"""
        if self.emergency_flag:
            return

        self.logger.error("\n🆘 紧急停止程序启动!")
        self.emergency_flag = True

        # 切换到紧急状态
        self.change_state(FlightState.EMERGENCY)

        try:
            # 停止运动，悬停
            self.client.hoverAsync(vehicle_name=self.drone_name).join()
            time.sleep(1)
            self.client.landAsync(vehicle_name=self.drone_name).join()
            time.sleep(2)
            self.logger.info("✅ 紧急降落指令已发送")
        except Exception as e:
            self.logger.error(f"⚠️ 紧急降落异常: {e}")

        # 关闭前视窗口
        if self.front_display:
            self.front_display.stop()

        self._cleanup_system()


# ==================== 主程序入口 ====================

def main():
    """主程序入口"""
    # 显示启动信息
    print("=" * 70)
    print("AirSimNH 无人机感知探索系统 - 智能决策增强版（数据闭环版）")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置状态: {'已加载' if CONFIG_LOADED else '使用默认配置'}")
    print(f"日志级别: {config.SYSTEM['LOG_LEVEL']}")
    print(f"探索时间: {config.EXPLORATION['TOTAL_TIME']}秒")
    print("=" * 70)
    print("智能决策特性:")
    print("  • 向量场避障算法 (VFH)")
    print("  • 基于网格的信息增益探索")
    print("  • PID平滑飞行控制")
    print("  • 自适应速度调整")
    print("  • 性能监控与数据闭环")
    print("=" * 70)
    print("数据记录:")
    print(f"  • CSV格式: {config.DATA_RECORDING.get('SAVE_TO_CSV', False)}")
    print(f"  • JSON格式: {config.DATA_RECORDING.get('SAVE_TO_JSON', False)}")
    print(f"  • 性能监控: {config.DATA_RECORDING.get('PERFORMANCE_MONITORING', False)}")
    print("=" * 70)

    # 用户选择模式
    print("\n请选择运行模式:")
    print("  1. 智能探索模式 (AI自主决策)")
    print("  2. 手动控制模式 (键盘控制)")
    print("  3. 混合模式 (先自动探索，后可切换)")
    print("=" * 50)

    mode_choice = input("请输入选择 (1/2/3): ").strip()

    explorer = None
    try:
        # 创建感知探索器
        explorer = PerceptiveExplorer(drone_name="")

        # 设置键盘中断处理
        def signal_handler(sig, frame):
            print("\n⚠️ 用户中断，正在安全停止...")
            if explorer:
                explorer.emergency_stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # 根据选择运行相应模式
        if mode_choice == '1':
            # 自动探索模式
            print("\n" + "="*50)
            print("启动智能探索模式")
            print("="*50)
            explorer.run_perception_loop()

        elif mode_choice == '2':
            # 手动控制模式
            print("\n" + "="*50)
            print("启动手动控制模式")
            print("="*50)

            # 先起飞到安全高度
            print("正在起飞...")
            explorer.client.takeoffAsync(vehicle_name="").join()
            time.sleep(2)
            explorer.client.moveToZAsync(-10, 3, vehicle_name="").join()
            time.sleep(2)
            print("起飞完成，可以开始手动控制")
            print("请切换到无人机前视窗口，使用WSAD键控制")

            # 进入手动控制
            explorer.run_manual_control()

        elif mode_choice == '3':
            # 混合模式：先自动探索，后询问是否切换手动
            print("\n" + "="*50)
            print("启动混合模式")
            print("="*50)

            # 先运行一段时间的自动探索
            explorer.logger.info("🔍 开始智能探索...")
            original_time = config.EXPLORATION['TOTAL_TIME']
            explorer.exploration_time = min(60, original_time)

            # 运行自动探索
            explorer.run_perception_loop()

            # 如果自动探索正常结束
            if not explorer.emergency_flag:
                print("\n" + "="*50)
                print("智能探索阶段结束")
                print("请选择下一步:")
                print("  1. 进入手动控制模式")
                print("  2. 继续智能探索")
                print("  3. 结束任务返航")
                print("="*50)

                next_choice = input("请输入选择 (1/2/3): ").strip()

                if next_choice == '1':
                    explorer.run_manual_control()
                elif next_choice == '2':
                    explorer.exploration_time = original_time - 60
                    if explorer.exploration_time > 10:
                        explorer.run_perception_loop()
                    else:
                        explorer.logger.info("⏰ 剩余时间不足，开始返航")
                        explorer._finish_mission()
                else:
                    explorer._finish_mission()

        else:
            print("❌ 无效的选择，程序退出")
            if explorer:
                explorer._cleanup_system()

    except Exception as e:
        print(f"\n❌ 程序启动异常: {e}")
        traceback.print_exc()

        # 尝试安全降落
        try:
            if explorer and explorer.client:
                explorer.client.landAsync().join()
                explorer.client.armDisarm(False)
                explorer.client.enableApiControl(False)
        except:
            pass


if __name__ == "__main__":
    main()