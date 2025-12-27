# -*- coding: utf-8 -*-
"""
AirSimNH 感知驱动自主探索无人机 - 智能决策增强版（红色、蓝色与黑色物体检测版）
核心：视觉感知 → 语义理解 → 智能决策 → 安全执行
集成：配置管理、日志系统、异常恢复、前视窗口显示
新增：向量场避障算法、基于网格的信息增益探索、平滑飞行控制
新增：性能监控与数据闭环系统、红色、蓝色与黑色物体检测与记录
新增：信息显示窗口，分离前视画面与系统信息
版本: 3.6 (双窗口三色物体检测版)
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
import gc
import platform

# 导入PIL用于中文文本绘制
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL/Pillow未安装，中文显示可能不正常。请运行: pip install Pillow")

# 全局字体缓存
_chinese_font_cache = {}
# 全局标志：是否禁用中文显示（如果字体加载失败）
_chinese_font_disabled = False

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
                                         'OPENNESS_THRESHOLD': 0.7},
                     'RED_OBJECT_DETECTION': {'ENABLED': True, 'MIN_AREA': 50,
                                            'MAX_AREA': 10000, 'UPDATE_INTERVAL': 1.0,
                                            'MEMORY_TIME': 5.0},
                     'BLUE_OBJECT_DETECTION': {'ENABLED': True, 'MIN_AREA': 50,
                                              'MAX_AREA': 10000, 'UPDATE_INTERVAL': 1.0,
                                              'MEMORY_TIME': 5.0},
                     'BLACK_OBJECT_DETECTION': {'ENABLED': True, 'MIN_AREA': 50,
                                               'MAX_AREA': 10000, 'UPDATE_INTERVAL': 1.0,
                                               'MEMORY_TIME': 5.0}}
        DISPLAY = {'FRONT_VIEW_WINDOW': {'NAME': "无人机前视画面", 'WIDTH': 640, 'HEIGHT': 480,
                                        'ENABLE_SHARPENING': True, 'SHOW_INFO_OVERLAY': True,
                                        'REFRESH_RATE_MS': 30, 'SHOW_RED_OBJECTS': True,
                                        'SHOW_BLUE_OBJECTS': True, 'SHOW_BLACK_OBJECTS': True},
                   'INFO_WINDOW': {'NAME': "无人机信息面板", 'WIDTH': 800, 'HEIGHT': 600,
                                  'BACKGROUND_COLOR': (20, 20, 30), 'TEXT_COLOR': (220, 220, 255),
                                  'HIGHLIGHT_COLOR': (0, 200, 255), 'WARNING_COLOR': (0, 100, 255),
                                  'SUCCESS_COLOR': (0, 255, 150), 'REFRESH_RATE_MS': 100,
                                  'SHOW_GRID': True, 'GRID_SIZE': 300,
                                  'SHOW_OBJECTS_STATS': True, 'SHOW_SYSTEM_STATS': True,
                                  'SHOW_PERFORMANCE': True, 'SHOW_TRAJECTORY': True,
                                  'TRAJECTORY_SIZE': 280, 'TRAJECTORY_MAX_POINTS': 1000,
                                  'TRAJECTORY_LINE_COLOR': (0, 255, 255),
                                  'TRAJECTORY_CURRENT_COLOR': (0, 255, 0),
                                  'TRAJECTORY_START_COLOR': (255, 255, 0)}}
        SYSTEM = {'LOG_LEVEL': 'INFO', 'LOG_TO_FILE': True, 'LOG_FILENAME': 'drone_log.txt',
                 'MAX_RECONNECT_ATTEMPTS': 3, 'RECONNECT_DELAY': 2.0,
                 'ENABLE_HEALTH_CHECK': True, 'HEALTH_CHECK_INTERVAL': 20}
        CAMERA = {'DEFAULT_NAME': "0",
                 'RED_COLOR_RANGE': {'LOWER1': [0, 120, 70], 'UPPER1': [10, 255, 255],
                                    'LOWER2': [170, 120, 70], 'UPPER2': [180, 255, 255]},
                 'BLUE_COLOR_RANGE': {'LOWER': [100, 150, 50], 'UPPER': [130, 255, 255]},
                 'BLACK_COLOR_RANGE': {'LOWER': [0, 0, 0], 'UPPER': [180, 255, 50]}}
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
        INTELLIGENT_DECISION = {
            'VECTOR_FIELD_RADIUS': 8.0,
            'OBSTACLE_REPULSION_GAIN': 3.0,
            'GOAL_ATTRACTION_GAIN': 2.0,
            'SMOOTHING_FACTOR': 0.3,
            'MIN_TURN_ANGLE_DEG': 10,
            'MAX_TURN_ANGLE_DEG': 60,
            'GRID_RESOLUTION': 2.0,
            'GRID_SIZE': 50,
            'INFORMATION_GAIN_DECAY': 0.95,
            'EXPLORATION_FRONTIER_THRESHOLD': 0.3,
            'PID_KP': 1.5,
            'PID_KI': 0.05,
            'PID_KD': 0.2,
            'SMOOTHING_WINDOW_SIZE': 5,
            'ADAPTIVE_SPEED_ENABLED': True,
            'MIN_SPEED_FACTOR': 0.3,
            'MAX_SPEED_FACTOR': 1.5,
            'MEMORY_WEIGHT': 0.7,
            'CURIOUSITY_WEIGHT': 0.3,
            'TARGET_LIFETIME': 15.0,
            'TARGET_REACHED_DISTANCE': 3.0,
            'RED_OBJECT_EXPLORATION': {'ATTRACTION_GAIN': 1.5, 'DETECTION_RADIUS': 10.0,
                                      'MIN_DISTANCE': 2.0, 'EXPLORATION_BONUS': 0.5},
            'BLUE_OBJECT_EXPLORATION': {'ATTRACTION_GAIN': 1.2, 'DETECTION_RADIUS': 8.0,
                                       'MIN_DISTANCE': 2.0, 'EXPLORATION_BONUS': 0.3},
            'BLACK_OBJECT_EXPLORATION': {'ATTRACTION_GAIN': 1.0, 'DETECTION_RADIUS': 8.0,
                                         'MIN_DISTANCE': 2.0, 'EXPLORATION_BONUS': 0.2}
        }
        DEBUG = {
            'SAVE_PERCEPTION_IMAGES': False,
            'IMAGE_SAVE_INTERVAL': 50,
            'LOG_DECISION_DETAILS': False,
            'SAVE_RED_OBJECT_IMAGES': False,
            'SAVE_BLUE_OBJECT_IMAGES': False,
            'SAVE_BLACK_OBJECT_IMAGES': False
        }
        DATA_RECORDING = {
            'ENABLED': True,
            'RECORD_INTERVAL': 0.2,
            'SAVE_TO_CSV': True,
            'SAVE_TO_JSON': True,
            'CSV_FILENAME': 'flight_data.csv',
            'JSON_FILENAME': 'flight_data.json',
            'PERFORMANCE_MONITORING': True,
            'SYSTEM_METRICS_INTERVAL': 5.0,
            'RECORD_RED_OBJECTS': True,
            'RECORD_BLUE_OBJECTS': True,
            'RECORD_BLACK_OBJECTS': True
        }
        PERFORMANCE = {
            'ENABLE_REALTIME_METRICS': True,
            'CPU_WARNING_THRESHOLD': 80.0,
            'MEMORY_WARNING_THRESHOLD': 80.0,
            'LOOP_TIME_WARNING_THRESHOLD': 0.2,
            'SAVE_PERFORMANCE_REPORT': True,
            'REPORT_INTERVAL': 30.0,
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
    RED_OBJECT_INSPECTION = "红色物体检查"
    BLUE_OBJECT_INSPECTION = "蓝色物体检查"
    BLACK_OBJECT_INSPECTION = "黑色物体检查"


@dataclass
class RedObject:
    """红色物体数据结构"""
    id: int
    position: Tuple[float, float, float]
    pixel_position: Tuple[int, int]
    size: float
    confidence: float
    timestamp: float
    last_seen: float
    visited: bool = False


@dataclass
class BlueObject:
    """蓝色物体数据结构"""
    id: int
    position: Tuple[float, float, float]
    pixel_position: Tuple[int, int]
    size: float
    confidence: float
    timestamp: float
    last_seen: float
    visited: bool = False


@dataclass
class BlackObject:
    """黑色物体数据结构"""
    id: int
    position: Tuple[float, float, float]
    pixel_position: Tuple[int, int]
    size: float
    confidence: float
    timestamp: float
    last_seen: float
    visited: bool = False


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

        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        self.previous_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


class ExplorationGrid:
    """探索网格地图类"""
    def __init__(self, resolution=2.0, grid_size=50):
        self.resolution = resolution
        self.grid_size = grid_size
        self.half_size = grid_size // 2

        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.information_gain = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.obstacle_grid = np.zeros((grid_size, grid_size), dtype=bool)
        self.visit_time = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.red_object_grid = np.zeros((grid_size, grid_size), dtype=bool)
        self.blue_object_grid = np.zeros((grid_size, grid_size), dtype=bool)
        self.black_object_grid = np.zeros((grid_size, grid_size), dtype=bool)
        self.current_idx = (self.half_size, self.half_size)
        self.frontier_cells = set()

        print(f"🗺️ 初始化探索网格: {grid_size}x{grid_size}, 分辨率: {resolution}m")

    def world_to_grid(self, world_x, world_y):
        grid_x = int(world_x / self.resolution) + self.half_size
        grid_y = int(world_y / self.resolution) + self.half_size

        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))

        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        world_x = (grid_x - self.half_size) * self.resolution
        world_y = (grid_y - self.half_size) * self.resolution
        return (world_x, world_y)

    def update_position(self, world_x, world_y):
        self.current_idx = self.world_to_grid(world_x, world_y)

        x, y = self.current_idx
        radius = 3

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    distance = math.sqrt(dx**2 + dy**2)
                    exploration_value = max(0, 1.0 - distance / radius)
                    self.grid[nx, ny] = max(self.grid[nx, ny], exploration_value)
                    self.visit_time[nx, ny] = time.time()

        self._update_frontiers()

    def _update_frontiers(self):
        self.frontier_cells.clear()

        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                if self.grid[x, y] > 0.7:
                    neighbors = [
                        (x-1, y), (x+1, y), (x, y-1), (x, y+1),
                        (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)
                    ]

                    for nx, ny in neighbors:
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if self.grid[nx, ny] < 0.3 and not self.obstacle_grid[nx, ny]:
                                unexplored_neighbors = 0
                                for nnx in range(nx-1, nx+2):
                                    for nny in range(ny-1, ny+2):
                                        if 0 <= nnx < self.grid_size and 0 <= nny < self.grid_size:
                                            if self.grid[nnx, nny] < 0.3:
                                                unexplored_neighbors += 1

                                self.information_gain[nx, ny] = unexplored_neighbors / 9.0
                                self.frontier_cells.add((nx, ny))

    def update_obstacles(self, obstacles_world):
        for obs_x, obs_y in obstacles_world:
            grid_x, grid_y = self.world_to_grid(obs_x, obs_y)

            radius = 2
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.obstacle_grid[nx, ny] = True
                        self.grid[nx, ny] = 0.0

    def update_red_objects(self, red_objects):
        self.red_object_grid.fill(False)

        for obj in red_objects:
            grid_x, grid_y = self.world_to_grid(obj.position[0], obj.position[1])

            radius = 1
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.red_object_grid[nx, ny] = True

    def update_blue_objects(self, blue_objects):
        self.blue_object_grid.fill(False)

        for obj in blue_objects:
            grid_x, grid_y = self.world_to_grid(obj.position[0], obj.position[1])

            radius = 1
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.blue_object_grid[nx, ny] = True

    def update_black_objects(self, black_objects):
        self.black_object_grid.fill(False)

        for obj in black_objects:
            grid_x, grid_y = self.world_to_grid(obj.position[0], obj.position[1])

            radius = 1
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.black_object_grid[nx, ny] = True

    def get_best_exploration_target(self, current_pos, red_objects=None, blue_objects=None, black_objects=None):
        # 优先检查红色物体
        if red_objects and len(red_objects) > 0:
            nearest_obj = None
            min_distance = float('inf')
            current_x, current_y = current_pos

            for obj in red_objects:
                if not obj.visited:
                    distance = math.sqrt((obj.position[0] - current_x)**2 +
                                        (obj.position[1] - current_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_obj = obj

            if nearest_obj and min_distance < 15.0:
                return (nearest_obj.position[0], nearest_obj.position[1])

        # 其次检查蓝色物体
        if blue_objects and len(blue_objects) > 0:
            nearest_obj = None
            min_distance = float('inf')
            current_x, current_y = current_pos

            for obj in blue_objects:
                if not obj.visited:
                    distance = math.sqrt((obj.position[0] - current_x)**2 +
                                        (obj.position[1] - current_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_obj = obj

            if nearest_obj and min_distance < 12.0:
                return (nearest_obj.position[0], nearest_obj.position[1])

        # 再次检查黑色物体
        if black_objects and len(black_objects) > 0:
            nearest_obj = None
            min_distance = float('inf')
            current_x, current_y = current_pos

            for obj in black_objects:
                if not obj.visited:
                    distance = math.sqrt((obj.position[0] - current_x)**2 +
                                        (obj.position[1] - current_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_obj = obj

            if nearest_obj and min_distance < 12.0:
                return (nearest_obj.position[0], nearest_obj.position[1])

        if not self.frontier_cells:
            angle = random.uniform(0, 2 * math.pi)
            distance = 10.0
            return (
                current_pos[0] + distance * math.cos(angle),
                current_pos[1] + distance * math.sin(angle)
            )

        best_score = -1
        best_target = None
        current_x, current_y = current_pos

        for fx, fy in self.frontier_cells:
            info_gain = self.information_gain[fx, fy]

            world_x, world_y = self.grid_to_world(fx, fy)
            distance = math.sqrt((world_x - current_x)**2 + (world_y - current_y)**2)
            distance_cost = min(1.0, distance / 30.0)

            time_since_visit = time.time() - self.visit_time[fx, fy]
            time_factor = min(1.0, time_since_visit / 60.0)

            red_bonus = 0.0
            if self.red_object_grid[fx, fy]:
                red_bonus = config.INTELLIGENT_DECISION['RED_OBJECT_EXPLORATION']['EXPLORATION_BONUS']

            blue_bonus = 0.0
            if self.blue_object_grid[fx, fy]:
                blue_bonus = config.INTELLIGENT_DECISION['BLUE_OBJECT_EXPLORATION']['EXPLORATION_BONUS']

            black_bonus = 0.0
            if self.black_object_grid[fx, fy]:
                black_bonus = config.INTELLIGENT_DECISION['BLACK_OBJECT_EXPLORATION']['EXPLORATION_BONUS']

            score = (
                config.INTELLIGENT_DECISION['CURIOUSITY_WEIGHT'] * info_gain +
                (1 - config.INTELLIGENT_DECISION['MEMORY_WEIGHT'] * time_factor) -
                distance_cost * 0.3 +
                red_bonus + blue_bonus + black_bonus
            )

            if score > best_score:
                best_score = score
                best_target = (world_x, world_y)

        return best_target

    def visualize_grid(self, size=300):
        if self.grid.size == 0:
            return None

        img_size = min(size, self.grid_size * 5)
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        cell_size = img_size // self.grid_size

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                color = (0, 0, 0)

                if (x, y) == self.current_idx:
                    color = (0, 255, 0)
                elif self.obstacle_grid[x, y]:
                    color = (0, 0, 255)
                elif self.red_object_grid[x, y]:
                    color = (0, 100, 255)  # 红色物体显示为橙色
                elif self.blue_object_grid[x, y]:
                    color = (255, 100, 0)  # 蓝色物体显示为青色
                elif self.black_object_grid[x, y]:
                    color = (128, 128, 128)  # 黑色物体显示为灰色
                elif self.grid[x, y] > 0.7:
                    color = (200, 200, 200)
                elif self.grid[x, y] > 0.3:
                    color = (100, 100, 100)
                elif (x, y) in self.frontier_cells:
                    gain = self.information_gain[x, y]
                    color = (0, int(255 * gain), int(255 * (1 - gain)))

                x1 = x * cell_size
                y1 = y * cell_size
                x2 = (x + 1) * cell_size
                y2 = (y + 1) * cell_size

                cv2.rectangle(img, (y1, x1), (y2, x2), color, -1)

        return img


class DataLogger:
    """数据记录器类"""

    def __init__(self, enable_csv=True, enable_json=True, csv_filename=None, json_filename=None):
        self.enable_csv = enable_csv
        self.enable_json = enable_json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if csv_filename:
            self.csv_filename = csv_filename
        else:
            self.csv_filename = f"flight_data_{timestamp}.csv"

        if json_filename:
            self.json_filename = json_filename
        else:
            self.json_filename = f"flight_data_{timestamp}.json"

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

        self.performance_metrics = {
            "start_time": time.time(),
            "cpu_usage": [],
            "memory_usage": [],
            "loop_times": [],
            "data_points": 0
        }

        self.red_objects_detected = []
        self.blue_objects_detected = []
        self.black_objects_detected = []

        # 内存优化：限制缓冲区大小
        self.max_flight_data = config.DATA_RECORDING.get('MAX_FLIGHT_DATA_BUFFER', 500)
        self.max_objects_buffer = config.DATA_RECORDING.get('MAX_OBJECTS_BUFFER', 200)
        self.max_events_buffer = config.DATA_RECORDING.get('MAX_EVENTS_BUFFER', 100)
        self.auto_save_interval = config.DATA_RECORDING.get('AUTO_SAVE_INTERVAL', 60.0)
        self.last_auto_save_time = time.time()
        self.max_metrics_buffer = config.PERFORMANCE.get('MAX_METRICS_BUFFER', 500)

        self.csv_columns = [
            'timestamp', 'loop_count', 'state', 'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z', 'yaw', 'pitch', 'roll',
            'obstacle_distance', 'open_space_score', 'terrain_slope',
            'has_obstacle', 'obstacle_direction', 'recommended_height',
            'target_x', 'target_y', 'target_z', 'velocity_command_x',
            'velocity_command_y', 'velocity_command_z', 'yaw_command',
            'battery_level', 'cpu_usage', 'memory_usage', 'loop_time',
            'grid_frontiers', 'grid_explored', 'vector_field_magnitude',
            'adaptive_speed_factor', 'decision_making_time', 'perception_time',
            'red_objects_count', 'red_objects_detected', 'red_objects_visited',
            'blue_objects_count', 'blue_objects_detected', 'blue_objects_visited',
            'black_objects_count', 'black_objects_detected', 'black_objects_visited'
        ]

        if self.enable_csv:
            self._init_csv_file()

        print(f"📊 数据记录器初始化完成")
        print(f"  CSV文件: {self.csv_filename}")
        print(f"  JSON文件: {self.json_filename}")

    def _init_csv_file(self):
        try:
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()
        except Exception as e:
            print(f"❌ 无法初始化CSV文件: {e}")
            self.enable_csv = False

    def record_flight_data(self, data_dict):
        if not config.DATA_RECORDING['ENABLED']:
            return

        try:
            data_dict['timestamp'] = datetime.now().isoformat()

            if self.enable_csv:
                with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                    row = {col: data_dict.get(col, '') for col in self.csv_columns}
                    writer.writerow(row)

            if self.enable_json:
                self.json_data['flight_data'].append(data_dict)
                # 内存优化：限制flight_data长度，超过限制时保存并清空
                if len(self.json_data['flight_data']) >= self.max_flight_data:
                    self._auto_save_and_clear()

            self.performance_metrics['data_points'] += 1

            # 内存优化：定期自动保存
            current_time = time.time()
            if current_time - self.last_auto_save_time >= self.auto_save_interval:
                self._auto_save_and_clear()
                self.last_auto_save_time = current_time

            if self.performance_metrics['data_points'] % 10 == 0:
                self._collect_system_metrics()

        except Exception as e:
            print(f"⚠️ 记录飞行数据时出错: {e}")

    def record_red_object(self, red_object):
        try:
            red_object_data = {
                'id': red_object.id,
                'position': red_object.position,
                'pixel_position': red_object.pixel_position,
                'size': red_object.size,
                'confidence': red_object.confidence,
                'timestamp': red_object.timestamp,
                'visited': red_object.visited
            }

            # 内存优化：限制物体记录列表长度
            if len(self.red_objects_detected) >= self.max_objects_buffer:
                self.red_objects_detected = self.red_objects_detected[-self.max_objects_buffer//2:]
            self.red_objects_detected.append(red_object_data)

            if 'red_objects' not in self.json_data:
                self.json_data['red_objects'] = []

            # 内存优化：限制JSON中的物体列表长度
            if len(self.json_data['red_objects']) >= self.max_objects_buffer:
                self.json_data['red_objects'] = self.json_data['red_objects'][-self.max_objects_buffer//2:]
            self.json_data['red_objects'].append(red_object_data)

        except Exception as e:
            print(f"⚠️ 记录红色物体时出错: {e}")

    def record_blue_object(self, blue_object):
        try:
            blue_object_data = {
                'id': blue_object.id,
                'position': blue_object.position,
                'pixel_position': blue_object.pixel_position,
                'size': blue_object.size,
                'confidence': blue_object.confidence,
                'timestamp': blue_object.timestamp,
                'visited': blue_object.visited
            }

            # 内存优化：限制物体记录列表长度
            if len(self.blue_objects_detected) >= self.max_objects_buffer:
                self.blue_objects_detected = self.blue_objects_detected[-self.max_objects_buffer//2:]
            self.blue_objects_detected.append(blue_object_data)

            if 'blue_objects' not in self.json_data:
                self.json_data['blue_objects'] = []

            # 内存优化：限制JSON中的物体列表长度
            if len(self.json_data['blue_objects']) >= self.max_objects_buffer:
                self.json_data['blue_objects'] = self.json_data['blue_objects'][-self.max_objects_buffer//2:]
            self.json_data['blue_objects'].append(blue_object_data)

        except Exception as e:
            print(f"⚠️ 记录蓝色物体时出错: {e}")

    def record_black_object(self, black_object):
        try:
            black_object_data = {
                'id': black_object.id,
                'position': black_object.position,
                'pixel_position': black_object.pixel_position,
                'size': black_object.size,
                'confidence': black_object.confidence,
                'timestamp': black_object.timestamp,
                'visited': black_object.visited
            }

            # 内存优化：限制物体记录列表长度
            if len(self.black_objects_detected) >= self.max_objects_buffer:
                self.black_objects_detected = self.black_objects_detected[-self.max_objects_buffer//2:]
            self.black_objects_detected.append(black_object_data)

            if 'black_objects' not in self.json_data:
                self.json_data['black_objects'] = []

            # 内存优化：限制JSON中的物体列表长度
            if len(self.json_data['black_objects']) >= self.max_objects_buffer:
                self.json_data['black_objects'] = self.json_data['black_objects'][-self.max_objects_buffer//2:]
            self.json_data['black_objects'].append(black_object_data)

        except Exception as e:
            print(f"⚠️ 记录黑色物体时出错: {e}")

    def _collect_system_metrics(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.performance_metrics['cpu_usage'].append(cpu_percent)

            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            self.performance_metrics['memory_usage'].append(memory_percent)

            # 内存优化：使用配置的最大缓冲区大小
            max_length = self.max_metrics_buffer
            if len(self.performance_metrics['cpu_usage']) > max_length:
                self.performance_metrics['cpu_usage'] = self.performance_metrics['cpu_usage'][-max_length:]
            if len(self.performance_metrics['memory_usage']) > max_length:
                self.performance_metrics['memory_usage'] = self.performance_metrics['memory_usage'][-max_length:]

        except Exception as e:
            print(f"⚠️ 收集系统指标时出错: {e}")

    def record_loop_time(self, loop_time):
        self.performance_metrics['loop_times'].append(loop_time)

        # 内存优化：使用配置的最大缓冲区大小
        max_length = self.max_metrics_buffer
        if len(self.performance_metrics['loop_times']) > max_length:
            self.performance_metrics['loop_times'] = self.performance_metrics['loop_times'][-max_length:]

    def record_event(self, event_type, event_data):
        try:
            event_record = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'event_data': event_data
            }

            if 'events' not in self.json_data:
                self.json_data['events'] = []

            # 内存优化：限制events列表长度
            if len(self.json_data['events']) >= self.max_events_buffer:
                self.json_data['events'] = self.json_data['events'][-self.max_events_buffer//2:]
            self.json_data['events'].append(event_record)

        except Exception as e:
            print(f"⚠️ 记录事件时出错: {e}")

    def _auto_save_and_clear(self):
        """自动保存数据并清空缓冲区（内存优化）"""
        if not self.enable_json or len(self.json_data['flight_data']) == 0:
            return

        try:
            # 创建临时文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = self.json_filename.replace('.json', f'_temp_{timestamp}.json')
            
            # 保存当前数据
            with open(temp_filename, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)
            
            # 清空flight_data，保留其他数据
            saved_count = len(self.json_data['flight_data'])
            self.json_data['flight_data'] = []
            
            # 强制垃圾回收
            gc.collect()
            
            print(f"💾 自动保存 {saved_count} 条数据到: {temp_filename} (已清空缓冲区)")
        except Exception as e:
            print(f"⚠️ 自动保存数据时出错: {e}")

    def save_json_data(self):
        if not self.enable_json:
            return

        try:
            self._calculate_performance_stats()

            # 红色物体统计
            if 'red_objects' in self.json_data:
                red_count = len(self.json_data['red_objects'])
                visited_count = sum(1 for obj in self.json_data['red_objects'] if obj.get('visited', False))
                self.json_data['red_objects_summary'] = {
                    'total_detected': red_count,
                    'total_visited': visited_count,
                    'visit_rate': visited_count / red_count if red_count > 0 else 0
                }

            # 蓝色物体统计
            if 'blue_objects' in self.json_data:
                blue_count = len(self.json_data['blue_objects'])
                visited_count = sum(1 for obj in self.json_data['blue_objects'] if obj.get('visited', False))
                self.json_data['blue_objects_summary'] = {
                    'total_detected': blue_count,
                    'total_visited': visited_count,
                    'visit_rate': visited_count / blue_count if blue_count > 0 else 0
                }

            # 黑色物体统计
            if 'black_objects' in self.json_data:
                black_count = len(self.json_data['black_objects'])
                visited_count = sum(1 for obj in self.json_data['black_objects'] if obj.get('visited', False))
                self.json_data['black_objects_summary'] = {
                    'total_detected': black_count,
                    'total_visited': visited_count,
                    'visit_rate': visited_count / black_count if black_count > 0 else 0
                }

            with open(self.json_filename, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)

            print(f"✅ JSON数据已保存: {self.json_filename}")

        except Exception as e:
            print(f"❌ 保存JSON数据时出错: {e}")

    def _calculate_performance_stats(self):
        if not self.performance_metrics['cpu_usage']:
            return

        cpu_avg = np.mean(self.performance_metrics['cpu_usage'])
        cpu_max = np.max(self.performance_metrics['cpu_usage'])
        cpu_min = np.min(self.performance_metrics['cpu_usage'])

        mem_avg = np.mean(self.performance_metrics['memory_usage'])
        mem_max = np.max(self.performance_metrics['memory_usage'])
        mem_min = np.min(self.performance_metrics['memory_usage'])

        if self.performance_metrics['loop_times']:
            loop_avg = np.mean(self.performance_metrics['loop_times'])
            loop_max = np.max(self.performance_metrics['loop_times'])
            loop_min = np.min(self.performance_metrics['loop_times'])
        else:
            loop_avg = loop_max = loop_min = 0

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
        try:
            if not self.performance_metrics['cpu_usage']:
                return "无性能数据可用"

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

            if 'red_objects' in self.json_data:
                red_count = len(self.json_data['red_objects'])
                visited_count = sum(1 for obj in self.json_data['red_objects'] if obj.get('visited', False))
                report += f"红色物体检测: 总数{red_count}个, 已访问{visited_count}个\n"

            if 'blue_objects' in self.json_data:
                blue_count = len(self.json_data['blue_objects'])
                visited_count = sum(1 for obj in self.json_data['blue_objects'] if obj.get('visited', False))
                report += f"蓝色物体检测: 总数{blue_count}个, 已访问{visited_count}个\n"

            if 'black_objects' in self.json_data:
                black_count = len(self.json_data['black_objects'])
                visited_count = sum(1 for obj in self.json_data['black_objects'] if obj.get('visited', False))
                report += f"黑色物体检测: 总数{black_count}个, 已访问{visited_count}个\n"

            report += "="*60 + "\n"

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
    obstacle_positions: List[Tuple[float, float]] = None
    red_objects: List[RedObject] = None
    red_objects_count: int = 0
    red_objects_image: Optional[np.ndarray] = None
    blue_objects: List[BlueObject] = None
    blue_objects_count: int = 0
    blue_objects_image: Optional[np.ndarray] = None
    black_objects: List[BlackObject] = None
    black_objects_count: int = 0
    black_objects_image: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.safe_directions is None:
            self.safe_directions = []
        if self.obstacle_positions is None:
            self.obstacle_positions = []
        if self.red_objects is None:
            self.red_objects = []
        if self.blue_objects is None:
            self.blue_objects = []
        if self.black_objects is None:
            self.black_objects = []


class VectorFieldPlanner:
    """向量场规划器"""
    def __init__(self):
        self.repulsion_gain = config.INTELLIGENT_DECISION['OBSTACLE_REPULSION_GAIN']
        self.attraction_gain = config.INTELLIGENT_DECISION['GOAL_ATTRACTION_GAIN']
        self.field_radius = config.INTELLIGENT_DECISION['VECTOR_FIELD_RADIUS']
        self.smoothing_factor = config.INTELLIGENT_DECISION['SMOOTHING_FACTOR']
        self.red_attraction_gain = config.INTELLIGENT_DECISION['RED_OBJECT_EXPLORATION']['ATTRACTION_GAIN']
        self.blue_attraction_gain = config.INTELLIGENT_DECISION['BLUE_OBJECT_EXPLORATION']['ATTRACTION_GAIN']
        self.black_attraction_gain = config.INTELLIGENT_DECISION['BLACK_OBJECT_EXPLORATION']['ATTRACTION_GAIN']

        self.min_turn_angle = math.radians(config.INTELLIGENT_DECISION['MIN_TURN_ANGLE_DEG'])
        self.max_turn_angle = math.radians(config.INTELLIGENT_DECISION['MAX_TURN_ANGLE_DEG'])

        self.vector_history = deque(maxlen=config.INTELLIGENT_DECISION['SMOOTHING_WINDOW_SIZE'])
        self.current_vector = Vector2D()

    def compute_vector(self, current_pos, goal_pos, obstacles, red_objects=None, blue_objects=None, black_objects=None):
        attraction_vector = self._compute_attraction(current_pos, goal_pos)
        repulsion_vector = self._compute_repulsion(current_pos, obstacles)
        red_attraction_vector = Vector2D()
        blue_attraction_vector = Vector2D()
        black_attraction_vector = Vector2D()

        if red_objects:
            red_attraction_vector = self._compute_red_attraction(current_pos, red_objects)

        if blue_objects:
            blue_attraction_vector = self._compute_blue_attraction(current_pos, blue_objects)

        if black_objects:
            black_attraction_vector = self._compute_black_attraction(current_pos, black_objects)

        combined_vector = attraction_vector + repulsion_vector + red_attraction_vector + blue_attraction_vector + black_attraction_vector
        smoothed_vector = self._smooth_vector(combined_vector)
        limited_vector = self._limit_turn_angle(smoothed_vector)

        self.current_vector = limited_vector
        return limited_vector

    def _compute_attraction(self, current_pos, goal_pos):
        if goal_pos is None:
            return Vector2D()

        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance < 0.1:
            return Vector2D()

        strength = min(self.attraction_gain, self.attraction_gain / max(1.0, distance))
        return Vector2D(dx, dy).normalize() * strength

    def _compute_repulsion(self, current_pos, obstacles):
        repulsion = Vector2D()

        for obs_x, obs_y in obstacles:
            dx = current_pos[0] - obs_x
            dy = current_pos[1] - obs_y
            distance = math.sqrt(dx**2 + dy**2)

            if distance < self.field_radius and distance > 0.1:
                strength = self.repulsion_gain * (1.0 / distance**2)
                direction = Vector2D(dx, dy).normalize()
                repulsion += direction * strength

        return repulsion

    def _compute_red_attraction(self, current_pos, red_objects):
        attraction = Vector2D()

        for obj in red_objects:
            if not obj.visited:
                dx = obj.position[0] - current_pos[0]
                dy = obj.position[1] - current_pos[1]
                distance = math.sqrt(dx**2 + dy**2)

                if distance < config.INTELLIGENT_DECISION['RED_OBJECT_EXPLORATION']['DETECTION_RADIUS']:
                    strength = self.red_attraction_gain / max(1.0, distance)
                    direction = Vector2D(dx, dy).normalize()
                    attraction += direction * strength

        return attraction

    def _compute_blue_attraction(self, current_pos, blue_objects):
        attraction = Vector2D()

        for obj in blue_objects:
            if not obj.visited:
                dx = obj.position[0] - current_pos[0]
                dy = obj.position[1] - current_pos[1]
                distance = math.sqrt(dx**2 + dy**2)

                if distance < config.INTELLIGENT_DECISION['BLUE_OBJECT_EXPLORATION']['DETECTION_RADIUS']:
                    strength = self.blue_attraction_gain / max(1.0, distance)
                    direction = Vector2D(dx, dy).normalize()
                    attraction += direction * strength

        return attraction

    def _compute_black_attraction(self, current_pos, black_objects):
        attraction = Vector2D()

        for obj in black_objects:
            if not obj.visited:
                dx = obj.position[0] - current_pos[0]
                dy = obj.position[1] - current_pos[1]
                distance = math.sqrt(dx**2 + dy**2)

                if distance < config.INTELLIGENT_DECISION['BLACK_OBJECT_EXPLORATION']['DETECTION_RADIUS']:
                    strength = self.black_attraction_gain / max(1.0, distance)
                    direction = Vector2D(dx, dy).normalize()
                    attraction += direction * strength

        return attraction

    def _smooth_vector(self, new_vector):
        self.vector_history.append(new_vector)

        if len(self.vector_history) < 2:
            return new_vector

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
        if self.current_vector.magnitude() < 0.1:
            return vector

        current_angle = math.atan2(self.current_vector.y, self.current_vector.x)
        new_angle = math.atan2(vector.y, vector.x)

        angle_diff = new_angle - current_angle
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) > self.max_turn_angle:
            angle_diff = math.copysign(self.max_turn_angle, angle_diff)
        elif abs(angle_diff) < self.min_turn_angle and vector.magnitude() > 0.1:
            angle_diff = math.copysign(self.min_turn_angle, angle_diff)

        magnitude = vector.magnitude()
        limited_angle = current_angle + angle_diff

        return Vector2D.from_angle(limited_angle, magnitude)


def _load_chinese_font(font_size=20):
    """加载中文字体，使用缓存避免重复加载"""
    global _chinese_font_cache
    
    # 检查缓存
    cache_key = font_size
    if cache_key in _chinese_font_cache:
        return _chinese_font_cache[cache_key]
    
    if not PIL_AVAILABLE:
        return None
    
    font = None
    font_paths = []
    
    # Windows系统字体路径
    if platform.system() == "Windows":
        windir = os.environ.get('WINDIR', 'C:\\Windows')
        font_dir = os.path.join(windir, 'Fonts')
        
        # 首先尝试已知的中文字体文件名
        known_fonts = [
            "simhei.ttf",      # 黑体
            "msyh.ttc",        # 微软雅黑
            "msyhbd.ttc",      # 微软雅黑 Bold
            "simsun.ttc",      # 宋体
            "simkai.ttf",      # 楷体
            "simli.ttf",       # 隶书
            "STHeiti.ttf",     # 华文黑体
            "STSong.ttf",      # 华文宋体
        ]
        
        for font_name in known_fonts:
            font_path = os.path.join(font_dir, font_name)
            font_paths.append(font_path)
        
        # 如果找不到，尝试扫描字体目录
        if os.path.exists(font_dir):
            try:
                for filename in os.listdir(font_dir):
                    filename_lower = filename.lower()
                    # 检查文件名是否包含中文字体关键词
                    if any(keyword in filename_lower for keyword in ['simhei', 'msyh', 'simsun', 'simkai', 'simli', 'stheit', 'stsong', 'chinese', 'cjk']):
                        font_path = os.path.join(font_dir, filename)
                        if font_path not in font_paths:
                            font_paths.append(font_path)
            except:
                pass
    
    # 也尝试常见的路径格式
    common_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simkai.ttf",
        "C:/Windows/Fonts/simli.ttf",
    ]
    font_paths.extend(common_paths)
    
    # 去重
    font_paths = list(dict.fromkeys(font_paths))
    
    # 尝试加载字体
    loaded_font_path = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                # 如果成功加载，缓存字体
                loaded_font_path = font_path
                _chinese_font_cache[cache_key] = font
                break
            except Exception as e:
                continue
    
    # 如果找到了字体，打印信息（仅第一次）
    if loaded_font_path and cache_key == 20:  # 只在第一次加载时打印
        print(f"✅ 成功加载中文字体: {os.path.basename(loaded_font_path)}")
    
    # 如果找不到字体，缓存None并打印警告（仅第一次）
    global _chinese_font_disabled
    if font is None:
        # 立即设置禁用标志，避免后续尝试
        _chinese_font_disabled = True
        if cache_key == 20:  # 只在第一次加载时打印
            print("⚠️ 未找到中文字体，将使用英文显示")
            if platform.system() == "Windows":
                windir = os.environ.get('WINDIR', 'C:\\Windows')
                font_dir = os.path.join(windir, 'Fonts')
                print(f"   字体目录: {font_dir}")
                print(f"   请确保该目录存在中文字体文件（如simhei.ttf, msyh.ttc等）")
    
    _chinese_font_cache[cache_key] = font
    return font


def _translate_to_english(text):
    """将中文文本转换为英文"""
    # 按长度从长到短排序，确保先替换长的短语，避免部分替换问题
    translations = [
        ("手动控制中...", "Manual Ctrl"),
        ("等待无人机图像...", "Waiting..."),
        ("系统正在初始化，请稍候...", "Initializing..."),
        ("按 Q 或 ESC 关闭窗口", "Press Q/ESC to close"),
        ("无人机信息面板", "Info Panel"),
        ("红色物体检查", "Red Object Inspection"),
        ("蓝色物体检查", "Blue Object Inspection"),
        ("黑色物体检查", "Black Object Inspection"),
        ("飞行状态:", "State:"),
        ("障碍距离:", "Obs:"),
        ("CPU使用率:", "CPU:"),
        ("内存使用率:", "Mem:"),
        ("循环时间:", "Loop:"),
        ("更新时间:", "Time:"),
        ("探索网格:", "Grid:"),
        ("红色物体:", "Red:"),
        ("红色物体", "Red Object"),  # 图例标签（无冒号）
        ("蓝色物体:", "Blue:"),
        ("蓝色物体", "Blue Object"),  # 图例标签（无冒号）
        ("黑色物体:", "Black:"),
        ("黑色物体", "Black Object"),  # 图例标签（无冒号）
        ("主动探索", "Exploring"),  # 完整状态值（必须在"探索"之前）
        ("探索前沿", "Frontier"),
        ("当前位置", "Current"),
        ("等待数据...", "Waiting..."),
        ("渲染错误", "Render Error"),
        ("状态:", "State:"),
        ("位置:", "Pos:"),
        ("障碍:", "Obs:"),
        ("开阔度:", "Open:"),
        ("高度:", "Height:"),
        ("图例:", "Legend:"),
        ("障碍物", "Obstacle"),
        ("已探索", "Explored"),
        ("前沿", "Frontier"),
        ("悬停", "Hovering"),
        ("手动控制", "Manual"),
        ("紧急", "Emergency"),
        ("探索", "Exploring"),  # 放在最后，避免与"主动探索"冲突
        ("主动", ""),  # 单独处理剩余的"主动"，避免显示问号
    ]
    
    result = text
    for chinese, english in translations:
        result = result.replace(chinese, english)
    
    return result


def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255), thickness=1):
    """
    在OpenCV图像上绘制中文文本
    使用PIL/Pillow来支持中文显示，如果失败则回退到英文
    """
    global _chinese_font_disabled
    
    # 如果已禁用中文显示，直接使用英文
    if _chinese_font_disabled:
        text_en = _translate_to_english(text)
        cv2.putText(img, text_en, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, thickness)
        return img
    
    # 首先检查PIL是否可用
    if not PIL_AVAILABLE:
        _chinese_font_disabled = True
        text_en = _translate_to_english(text)
        cv2.putText(img, text_en, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, thickness)
        return img
    
    # 尝试加载中文字体
    font = _load_chinese_font(font_size)
    
    # 如果字体加载失败，立即回退到英文并设置禁用标志
    if font is None:
        _chinese_font_disabled = True
        text_en = _translate_to_english(text)
        cv2.putText(img, text_en, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, thickness)
        return img
    
    # 尝试使用PIL绘制中文
    try:
        # 检查文本是否包含中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if not has_chinese:
            # 如果没有中文字符，直接使用cv2.putText
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, thickness)
            return img
        
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 绘制文本（PIL使用RGB颜色）
        color_rgb = (color[2], color[1], color[0])  # BGR转RGB
        
        # PIL的text函数位置参数是(x, y)
        x, y = position
        
        # 尝试绘制文本，如果失败则回退到英文
        try:
            draw.text((x, y), text, font=font, fill=color_rgb)
            # 转换回OpenCV格式
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as draw_error:
            # 如果绘制失败（可能是字体不支持某些字符），回退到英文
            raise draw_error
        
    except Exception as e:
        # 如果绘制失败，回退到英文显示并设置禁用标志
        _chinese_font_disabled = True
        text_en = _translate_to_english(text)
        cv2.putText(img, text_en, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, thickness)
    
    return img


class FrontViewWindow:
    """前视窗口 - 显示摄像头画面和手动控制"""

    def __init__(self, window_name=None, width=None, height=None,
                 enable_sharpening=None, show_info=None):
        # 窗口标题使用英文，避免乱码
        if window_name:
            # 如果传入的窗口名包含中文，翻译为英文
            self.window_name = _translate_to_english(window_name) if any('\u4e00' <= char <= '\u9fff' for char in window_name) else window_name
        else:
            default_name = _translate_to_english(config.DISPLAY['FRONT_VIEW_WINDOW']['NAME'])
            self.window_name = default_name
        self.window_width = width if width is not None else config.DISPLAY['FRONT_VIEW_WINDOW']['WIDTH']
        self.window_height = height if height is not None else config.DISPLAY['FRONT_VIEW_WINDOW']['HEIGHT']
        self.enable_sharpening = (enable_sharpening if enable_sharpening is not None
                                 else config.DISPLAY['FRONT_VIEW_WINDOW']['ENABLE_SHARPENING'])
        self.show_info = (show_info if show_info is not None
                         else config.DISPLAY['FRONT_VIEW_WINDOW']['SHOW_INFO_OVERLAY'])

        # 内存优化：使用配置的队列大小
        queue_maxsize = config.DISPLAY['FRONT_VIEW_WINDOW'].get('QUEUE_MAXSIZE', 2)
        self.image_queue = queue.Queue(maxsize=queue_maxsize)
        self.reduce_image_copy = config.DISPLAY['FRONT_VIEW_WINDOW'].get('REDUCE_IMAGE_COPY', True)
        self.display_active = True
        self.display_thread = None
        self.paused = False

        self.manual_mode = False
        self.key_states = {}
        self.last_keys = {}

        self.exit_manual_flag = False
        self.exit_display_flag = False

        self.display_stats = {
            'fps': 0.0,
            'last_update': time.time(),
            'frame_count': 0
        }

        self.start()

    def start(self):
        if self.display_thread and self.display_thread.is_alive():
            return

        self.display_active = True
        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="FrontViewWindow"
        )
        self.display_thread.start()

    def stop(self):
        self.display_active = False
        self.exit_display_flag = True
        if self.display_thread:
            self.display_thread.join(timeout=2.0)

    def update_image(self, image_data: np.ndarray, info: Optional[Dict] = None,
                     manual_info: Optional[List[str]] = None):
        if not self.display_active or self.paused or image_data is None:
            return

        try:
            if self.enable_sharpening and image_data is not None and image_data.size > 0:
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
                image_data = cv2.filter2D(image_data, -1, kernel)

            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass

            # 内存优化：仅在必要时复制图像
            if self.reduce_image_copy and image_data is not None:
                # 如果队列为空或只有一个元素，直接使用引用（避免复制）
                if self.image_queue.qsize() == 0:
                    display_image = image_data
                else:
                    display_image = image_data.copy()
            else:
                display_image = image_data.copy() if image_data is not None else None
            
            display_packet = {
                'image': display_image,
                'info': info.copy() if info else {},
                'manual_info': manual_info.copy() if manual_info else [],
                'timestamp': time.time()
            }

            self.image_queue.put_nowait(display_packet)

        except Exception as e:
            print(f"⚠️ 更新图像时出错: {e}")

    def set_manual_mode(self, manual_mode):
        self.manual_mode = manual_mode
        self.exit_manual_flag = False
        self.key_states = {}
        self.last_keys = {}
        print(f"🔄 {'进入' if manual_mode else '退出'}手动控制模式")

    def get_control_inputs(self):
        return self.key_states.copy()

    def should_exit_manual(self):
        return self.exit_manual_flag

    def _display_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        # 等待画面：只显示黑色背景，不显示任何文字
        wait_img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
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

            try:
                if not self.image_queue.empty():
                    packet = self.image_queue.get_nowait()
                    display_image = packet['image']
                    info = packet['info']
                    manual_info = packet['manual_info']

                    self._update_stats()

                    while not self.image_queue.empty():
                        try:
                            self.image_queue.get_nowait()
                        except queue.Empty:
                            break
            except queue.Empty:
                pass

            if display_image is not None:
                # 前视窗口只显示纯图像，不添加任何信息叠加
                cv2.imshow(self.window_name, display_image)
            elif self.paused:
                # 暂停时也显示黑色背景，不显示任何文字
                blank = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
                cv2.imshow(self.window_name, blank)

            key = cv2.waitKey(config.DISPLAY['FRONT_VIEW_WINDOW'].get('REFRESH_RATE_MS', 30)) & 0xFF

            current_keys = {}
            if key != 255:
                current_keys[key] = True

                if self.manual_mode:
                    self._handle_manual_mode_key(key)
                else:
                    self._handle_window_control_key(key, display_image)

            self._update_key_states(current_keys)

            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("🔄 用户关闭了前视窗口")
                    self.display_active = False
                    break
            except:
                self.display_active = False
                break

        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
        cv2.waitKey(1)

    def _handle_manual_mode_key(self, key):
        if key == 27:
            print("收到退出手动模式指令")
            self.exit_manual_flag = True
            return

        self.key_states[key] = True

        if key == 32:
            print("⏸️ 悬停指令")

    def _handle_window_control_key(self, key, display_image):
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
        released_keys = []
        for key in list(self.key_states.keys()):
            if key not in current_keys:
                released_keys.append(key)

        for key in released_keys:
            del self.key_states[key]

        self.last_keys = current_keys.copy()

    def _update_stats(self):
        now = time.time()
        self.display_stats['frame_count'] += 1

        if now - self.display_stats['last_update'] >= 1.0:
            self.display_stats['fps'] = self.display_stats['frame_count'] / (now - self.display_stats['last_update'])
            self.display_stats['frame_count'] = 0
            self.display_stats['last_update'] = now

    def _add_info_overlay(self, image: np.ndarray, info: Dict, manual_info: List[str] = None) -> np.ndarray:
        """前视窗口不再显示任何信息叠加，只显示纯图像"""
        # 直接返回原始图像，不添加任何文字或叠加层
        return image

    def _save_screenshot(self, image: Optional[np.ndarray]):
        if image is not None and image.size > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"drone_snapshot_{timestamp}.png"
            cv2.imwrite(filename, image)
            print(f"📸 截图已保存: {filename}")
        else:
            print("⚠️ 无法保存截图：无有效图像数据")


class InfoDisplayWindow:
    """信息显示窗口 - 显示系统状态、探索网格、物体统计等信息"""

    def __init__(self, window_name=None, width=None, height=None):
        # 窗口标题使用英文，避免乱码
        if window_name:
            # 如果传入的窗口名包含中文，翻译为英文
            self.window_name = _translate_to_english(window_name) if any('\u4e00' <= char <= '\u9fff' for char in window_name) else window_name
        else:
            default_name = _translate_to_english(config.DISPLAY['INFO_WINDOW']['NAME'])
            self.window_name = default_name
        self.window_width = width if width is not None else config.DISPLAY['INFO_WINDOW']['WIDTH']
        self.window_height = height if height is not None else config.DISPLAY['INFO_WINDOW']['HEIGHT']

        self.display_config = config.DISPLAY['INFO_WINDOW']
        self.info_queue = queue.Queue(maxsize=3)
        self.display_active = True
        self.display_thread = None
        self.last_update = time.time()

        # 轨迹记录
        self.trajectory_points = deque(maxlen=config.DISPLAY['INFO_WINDOW'].get('TRAJECTORY_MAX_POINTS', 1000))
        self.start_position = None

        self.start()

    def start(self):
        if self.display_thread and self.display_thread.is_alive():
            return

        self.display_active = True
        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="InfoDisplayWindow"
        )
        self.display_thread.start()
        print(f"📊 信息显示窗口已启动: {self.window_name}")

    def stop(self):
        self.display_active = False
        if self.display_thread:
            self.display_thread.join(timeout=2.0)

    def update_info(self, info_data: Dict):
        if not self.display_active:
            return

        try:
            # 更新轨迹数据
            if 'position' in info_data:
                pos = info_data['position']
                self.trajectory_points.append((pos[0], pos[1], pos[2]))
                
                # 记录起始位置
                if self.start_position is None:
                    self.start_position = (pos[0], pos[1], pos[2])

            if self.info_queue.full():
                try:
                    self.info_queue.get_nowait()
                except queue.Empty:
                    pass

            self.info_queue.put_nowait(info_data.copy())

        except Exception as e:
            print(f"⚠️ 更新信息数据时出错: {e}")

    def _display_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        wait_img = self._create_waiting_screen()
        cv2.imshow(self.window_name, wait_img)
        cv2.waitKey(100)

        print("📊 信息显示窗口已就绪")
        print("  显示内容: 探索网格、系统状态、物体统计、性能信息")

        last_render_time = time.time()
        info_data = {}

        while self.display_active:
            current_time = time.time()

            # 从队列获取最新信息
            try:
                while not self.info_queue.empty():
                    info_data = self.info_queue.get_nowait()
            except queue.Empty:
                pass

            # 定期刷新显示
            if current_time - last_render_time >= self.display_config['REFRESH_RATE_MS'] / 1000.0:
                display_image = self._render_info_display(info_data)
                if display_image is not None:
                    cv2.imshow(self.window_name, display_image)
                last_render_time = current_time

            # 处理窗口事件
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:  # Q或ESC关闭窗口
                print("🔄 用户关闭信息窗口")
                self.display_active = False
                break

            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("🔄 信息窗口被关闭")
                    self.display_active = False
                    break
            except:
                self.display_active = False
                break

        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
        cv2.waitKey(1)

    def _create_waiting_screen(self):
        """创建等待屏幕"""
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        bg_color = self.display_config['BACKGROUND_COLOR']
        img[:, :] = bg_color

        center_x = self.window_width // 2
        center_y = self.window_height // 2

        # 标题
        title = "无人机信息面板"
        img = put_chinese_text(img, title, (center_x - 150, center_y - 100), 36, self.display_config['HIGHLIGHT_COLOR'], 2)

        # 状态信息
        status = "等待数据..."
        img = put_chinese_text(img, status, (center_x - 80, center_y), 24, self.display_config['TEXT_COLOR'], 1)

        # 提示
        tip = "系统正在初始化，请稍候..."
        img = put_chinese_text(img, tip, (center_x - 120, center_y + 50), 18, self.display_config['TEXT_COLOR'], 1)

        return img

    def _draw_trajectory(self, trajectory_points, current_pos, start_pos, size):
        """绘制运动轨迹图"""
        try:
            # 创建轨迹图（始终返回有效图像）
            traj_img = np.zeros((size, size, 3), dtype=np.uint8)
            traj_img.fill(30)  # 深灰色背景

            # 收集所有点用于计算边界
            all_points = []
            if trajectory_points:
                try:
                    all_points.extend([(float(p[0]), float(p[1])) for p in trajectory_points if len(p) >= 2])
                except (IndexError, TypeError, ValueError):
                    pass
            if current_pos and len(current_pos) >= 2:
                try:
                    all_points.append((float(current_pos[0]), float(current_pos[1])))
                except (TypeError, ValueError, IndexError):
                    pass
            if start_pos and len(start_pos) >= 2:
                try:
                    all_points.append((float(start_pos[0]), float(start_pos[1])))
                except (TypeError, ValueError, IndexError):
                    pass

            if not all_points:
                # 没有轨迹点时，返回空白图像
                return traj_img

            # 计算边界和缩放
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # 添加边距
            margin = max((max_x - min_x) * 0.1, (max_y - min_y) * 0.1, 5.0)
            range_x = max_x - min_x + 2 * margin
            range_y = max_y - min_y + 2 * margin

            if range_x == 0:
                range_x = 10.0
            if range_y == 0:
                range_y = 10.0

            # 坐标转换函数
            def world_to_pixel(wx, wy):
                try:
                    px = int((wx - min_x + margin) / range_x * size)
                    py = int((wy - min_y + margin) / range_y * size)
                    px = max(0, min(size - 1, px))
                    py = max(0, min(size - 1, py))
                    return px, py
                except (ZeroDivisionError, ValueError, TypeError):
                    return size // 2, size // 2

            # 绘制轨迹线
            if len(trajectory_points) > 1:
                try:
                    line_color = self.display_config.get('TRAJECTORY_LINE_COLOR', (0, 255, 255))
                    for i in range(len(trajectory_points) - 1):
                        try:
                            p1 = trajectory_points[i]
                            p2 = trajectory_points[i + 1]
                            if len(p1) >= 2 and len(p2) >= 2:
                                px1, py1 = world_to_pixel(p1[0], p1[1])
                                px2, py2 = world_to_pixel(p2[0], p2[1])
                                cv2.line(traj_img, (px1, py1), (px2, py2), line_color, 2)
                        except (IndexError, TypeError, ValueError):
                            continue
                except Exception:
                    pass

            # 绘制起始点
            if start_pos and len(start_pos) >= 2:
                try:
                    start_color = self.display_config.get('TRAJECTORY_START_COLOR', (255, 255, 0))
                    sx, sy = world_to_pixel(start_pos[0], start_pos[1])
                    cv2.circle(traj_img, (sx, sy), 6, start_color, -1)
                    cv2.circle(traj_img, (sx, sy), 6, (255, 255, 255), 1)
                except (IndexError, TypeError, ValueError):
                    pass

            # 绘制当前位置
            if current_pos and len(current_pos) >= 2:
                try:
                    current_color = self.display_config.get('TRAJECTORY_CURRENT_COLOR', (0, 255, 0))
                    cx, cy = world_to_pixel(current_pos[0], current_pos[1])
                    cv2.circle(traj_img, (cx, cy), 8, current_color, -1)
                    cv2.circle(traj_img, (cx, cy), 8, (255, 255, 255), 2)
                except (IndexError, TypeError, ValueError):
                    pass

            # 绘制坐标轴
            try:
                axis_color = (100, 100, 100)
                center_x_px, center_y_px = world_to_pixel(0, 0)
                if 0 <= center_x_px < size:
                    cv2.line(traj_img, (center_x_px, 0), (center_x_px, size), axis_color, 1)
                if 0 <= center_y_px < size:
                    cv2.line(traj_img, (0, center_y_px), (size, center_y_px), axis_color, 1)
            except Exception:
                pass

            return traj_img
        except Exception as e:
            # 发生任何错误时返回空白图像
            print(f"⚠️ 绘制轨迹图内部错误: {e}")
            traj_img = np.zeros((size, size, 3), dtype=np.uint8)
            traj_img.fill(30)
            return traj_img

    def _render_info_display(self, info_data: Dict) -> np.ndarray:
        """渲染信息显示"""
        try:
            # 创建背景
            img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            bg_color = self.display_config['BACKGROUND_COLOR']
            img[:, :] = bg_color

            text_color = self.display_config['TEXT_COLOR']
            highlight_color = self.display_config['HIGHLIGHT_COLOR']
            warning_color = self.display_config['WARNING_COLOR']
            success_color = self.display_config['SUCCESS_COLOR']

            y_offset = 40
            x_offset = 20

            # 标题栏
            title = "无人机信息面板"
            img = put_chinese_text(img, title, (self.window_width // 2 - 100, 30), 30, highlight_color, 2)

            # 分隔线
            cv2.line(img, (10, 50), (self.window_width - 10, 50), text_color, 1)

            y_offset = 80

            # 1. 飞行状态信息
            if 'state' in info_data:
                state = info_data['state']
                # 状态值已经在_update_info_window中翻译过了，这里直接使用
                state_color = success_color if 'Exploring' in state else highlight_color if 'Hovering' in state else warning_color if 'Emergency' in state else text_color
                img = put_chinese_text(img, f"飞行状态: {state}", (x_offset, y_offset), 21, state_color, 2)
                y_offset += 30

            # 2. 位置信息
            if 'position' in info_data:
                pos = info_data['position']
                pos_text = f"位置: X:{pos[0]:.1f}m Y:{pos[1]:.1f}m 高度:{-pos[2]:.1f}m"
                img = put_chinese_text(img, pos_text, (x_offset, y_offset), 18, text_color, 1)
                y_offset += 25

            # 3. 环境感知信息
            if 'perception' in info_data:
                perception = info_data['perception']
                obs_text = f"障碍距离: {perception.get('obstacle_distance', 0):.1f}m"
                obs_color = warning_color if perception.get('obstacle_distance', 0) < 5.0 else text_color
                img = put_chinese_text(img, obs_text, (x_offset, y_offset), 18, obs_color, 1)
                y_offset += 25

                open_text = f"开阔度: {perception.get('open_space_score', 0):.2f}"
                img = put_chinese_text(img, open_text, (x_offset, y_offset), 18, text_color, 1)
                y_offset += 25

            # 4. 物体检测统计
            if 'objects_stats' in info_data:
                objects_stats = info_data['objects_stats']

                # 红色物体统计
                red_total = objects_stats.get('red_total', 0)
                red_visited = objects_stats.get('red_visited', 0)
                red_text = f"红色物体: {red_visited}/{red_total}"
                red_color = success_color if red_visited > 0 else text_color
                img = put_chinese_text(img, red_text, (x_offset, y_offset), 21, red_color, 1)
                y_offset += 30

                # 蓝色物体统计
                blue_total = objects_stats.get('blue_total', 0)
                blue_visited = objects_stats.get('blue_visited', 0)
                blue_text = f"蓝色物体: {blue_visited}/{blue_total}"
                blue_color = success_color if blue_visited > 0 else text_color
                img = put_chinese_text(img, blue_text, (x_offset, y_offset), 21, blue_color, 1)
                y_offset += 30
                
                # 黑色物体统计
                black_total = objects_stats.get('black_total', 0)
                black_visited = objects_stats.get('black_visited', 0)
                if black_total > 0:
                    black_text = f"黑色物体: {black_visited}/{black_total}"
                    black_color = success_color if black_visited > 0 else text_color
                    img = put_chinese_text(img, black_text, (x_offset, y_offset), 21, black_color, 1)
                    y_offset += 30

            # 5. 探索网格信息
            if 'grid_stats' in info_data:
                grid_stats = info_data['grid_stats']
                frontiers = grid_stats.get('frontiers', 0)
                explored = grid_stats.get('explored', 0)
                total = grid_stats.get('total', 1)

                grid_text = f"探索网格: {frontiers}前沿 | {explored}/{total}已探索"
                img = put_chinese_text(img, grid_text, (x_offset, y_offset), 18, text_color, 1)
                y_offset += 25

                # 探索进度条
                progress = explored / total if total > 0 else 0
                bar_width = 200
                bar_height = 15
                bar_x = x_offset
                bar_y = y_offset

                # 背景条
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                # 进度条
                progress_width = int(bar_width * progress)
                progress_color = (0, int(255 * progress), int(255 * (1 - progress)))
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), progress_color, -1)
                # 边框
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), text_color, 1)
                # 进度文本
                progress_text = f"{progress*100:.1f}%"
                cv2.putText(img, progress_text, (bar_x + bar_width + 10, bar_y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                y_offset += 35

            # 6. 系统性能信息
            if 'performance' in info_data:
                performance = info_data['performance']
                cpu_usage = performance.get('cpu_usage', 0)
                memory_usage = performance.get('memory_usage', 0)
                loop_time = performance.get('loop_time', 0) * 1000  # 转换为毫秒

                cpu_color = warning_color if cpu_usage > 80 else text_color
                mem_color = warning_color if memory_usage > 80 else text_color
                loop_color = warning_color if loop_time > 200 else text_color

                img = put_chinese_text(img, f"CPU使用率: {cpu_usage:.1f}%", (x_offset, y_offset), 18, cpu_color, 1)
                y_offset += 25

                img = put_chinese_text(img, f"内存使用率: {memory_usage:.1f}%", (x_offset, y_offset), 18, mem_color, 1)
                y_offset += 25

                img = put_chinese_text(img, f"循环时间: {loop_time:.1f}ms", (x_offset, y_offset), 18, loop_color, 1)
                y_offset += 25

            # 7. 手动控制信息（如果处于手动控制模式）
            if 'manual_info' in info_data and info_data['manual_info']:
                y_offset += 10  # 添加一些间距
                manual_title = "手动控制信息:"
                img = put_chinese_text(img, manual_title, (x_offset, y_offset), 18, highlight_color, 2)
                y_offset += 25
                
                for line in info_data['manual_info']:
                    img = put_chinese_text(img, line, (x_offset + 10, y_offset), 16, text_color, 1)
                    y_offset += 22

            # 8. 探索网格图像（右侧）
            if self.display_config['SHOW_GRID'] and 'grid_image' in info_data:
                grid_img = info_data['grid_image']
                if grid_img is not None and grid_img.size > 0:
                    grid_size = self.display_config['GRID_SIZE']
                    grid_resized = cv2.resize(grid_img, (grid_size, grid_size))

                    grid_x = self.window_width - grid_size - 20
                    grid_y = 80

                    # 添加网格标题（直接使用英文，避免问号问题）
                    grid_title = "Grid Map"  # 直接使用英文，不再翻译
                    cv2.putText(img, grid_title, (grid_x, grid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 18 / 30.0, highlight_color, 1)
                    
                    # 不再在网格上方显示状态值，避免问号问题

                    # 添加图例
                    legend_y = grid_y + grid_size + 20
                    img = put_chinese_text(img, "图例:", (grid_x, legend_y), 15, text_color, 1)
                    legend_y += 20

                    # 当前位置
                    cv2.rectangle(img, (grid_x, legend_y), (grid_x + 15, legend_y + 15), (0, 255, 0), -1)
                    img = put_chinese_text(img, "当前位置", (grid_x + 20, legend_y + 12), 12, text_color, 1)
                    legend_y += 25

                    # 障碍物
                    cv2.rectangle(img, (grid_x, legend_y), (grid_x + 15, legend_y + 15), (0, 0, 255), -1)
                    img = put_chinese_text(img, "障碍物", (grid_x + 20, legend_y + 12), 12, text_color, 1)
                    legend_y += 25

                    # 红色物体
                    cv2.rectangle(img, (grid_x, legend_y), (grid_x + 15, legend_y + 15), (0, 100, 255), -1)
                    img = put_chinese_text(img, "红色物体", (grid_x + 20, legend_y + 12), 12, text_color, 1)
                    legend_y += 25

                    # 蓝色物体
                    cv2.rectangle(img, (grid_x, legend_y), (grid_x + 15, legend_y + 15), (255, 100, 0), -1)
                    img = put_chinese_text(img, "蓝色物体", (grid_x + 20, legend_y + 12), 12, text_color, 1)
                    legend_y += 25

                    # 黑色物体
                    cv2.rectangle(img, (grid_x, legend_y), (grid_x + 15, legend_y + 15), (128, 128, 128), -1)
                    img = put_chinese_text(img, "黑色物体", (grid_x + 20, legend_y + 12), 12, text_color, 1)
                    legend_y += 25

                    # 前沿区域
                    cv2.rectangle(img, (grid_x, legend_y), (grid_x + 15, legend_y + 15), (0, 200, 0), -1)
                    img = put_chinese_text(img, "探索前沿", (grid_x + 20, legend_y + 12), 12, text_color, 1)

                    # 将网格图像放到主图像上
                    img[grid_y:grid_y+grid_size, grid_x:grid_x+grid_size] = grid_resized

            # 9. 运动轨迹图（最左下角）
            if self.display_config.get('SHOW_TRAJECTORY', True):
                try:
                    trajectory_size = self.display_config.get('TRAJECTORY_SIZE', 280)
                    current_pos = info_data.get('position')
                    
                    # 绘制轨迹图
                    trajectory_img = self._draw_trajectory(
                        list(self.trajectory_points),
                        current_pos,
                        self.start_position,
                        trajectory_size
                    )
                    
                    if trajectory_img is not None and trajectory_img.size > 0:
                        # 轨迹图放在最左下角，从底部开始
                        traj_x = 20  # 左边距
                        # 从窗口底部向上，留出底部提示文字的空间（约50像素，确保不重叠）
                        bottom_hint_height = 50
                        traj_y = self.window_height - trajectory_size - bottom_hint_height
                        
                        # 确保轨迹图不会超出窗口范围，且不与上方内容重叠（至少距离顶部80像素）
                        if traj_y >= 80 and traj_y + trajectory_size <= self.window_height - bottom_hint_height:
                            # 添加轨迹图标题（在轨迹图上方）
                            traj_title = "Trajectory"  # 直接使用英文
                            cv2.putText(img, traj_title, (traj_x, traj_y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 18 / 30.0, highlight_color, 1)
                            
                            # 确保图像尺寸匹配
                            if (traj_y + trajectory_size <= self.window_height and 
                                traj_x + trajectory_size <= self.window_width):
                                # 将轨迹图放到主图像上
                                img[traj_y:traj_y+trajectory_size, traj_x:traj_x+trajectory_size] = trajectory_img
                                
                                # 添加轨迹图图例（在轨迹图右侧，使用英文避免乱码）
                                legend_x = traj_x + trajectory_size + 15
                                legend_y = traj_y
                                
                                if legend_x + 150 < self.window_width:
                                    # 使用英文避免乱码
                                    cv2.putText(img, "Legend:", (legend_x, legend_y), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                                    legend_y += 20
                                    
                                    # 起始位置
                                    cv2.circle(img, (legend_x + 7, legend_y + 7), 6, 
                                              self.display_config.get('TRAJECTORY_START_COLOR', (255, 255, 0)), -1)
                                    cv2.circle(img, (legend_x + 7, legend_y + 7), 6, (255, 255, 255), 1)
                                    cv2.putText(img, "Start", (legend_x + 20, legend_y + 12), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                                    legend_y += 25
                                    
                                    # 当前位置
                                    cv2.circle(img, (legend_x + 7, legend_y + 7), 8, 
                                              self.display_config.get('TRAJECTORY_CURRENT_COLOR', (0, 255, 0)), -1)
                                    cv2.circle(img, (legend_x + 7, legend_y + 7), 8, (255, 255, 255), 2)
                                    cv2.putText(img, "Current", (legend_x + 20, legend_y + 12), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                                    legend_y += 25
                                    
                                    # 轨迹线
                                    cv2.line(img, (legend_x, legend_y + 7), (legend_x + 20, legend_y + 7), 
                                            self.display_config.get('TRAJECTORY_LINE_COLOR', (0, 255, 255)), 2)
                                    cv2.putText(img, "Path", (legend_x + 25, legend_y + 12), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                                    
                                    # 显示轨迹点数
                                    point_count = len(self.trajectory_points)
                                    count_text = f"Points: {point_count}"
                                    cv2.putText(img, count_text, (legend_x, legend_y + 30), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                except Exception as e:
                    # 轨迹图绘制失败时不影响其他显示
                    print(f"⚠️ 绘制轨迹图时出错: {e}")
                    import traceback
                    traceback.print_exc()

            # 10. 时间戳
            if 'timestamp' in info_data:
                timestamp = info_data['timestamp']
                time_text = f"更新时间: {timestamp}"
                img = put_chinese_text(img, time_text, (self.window_width - 200, self.window_height - 10), 15, text_color, 1)

            # 11. 底部提示
            hint_text = "按 Q 或 ESC 关闭窗口"
            img = put_chinese_text(img, hint_text, (self.window_width // 2 - 80, self.window_height - 30), 15, text_color, 1)

            return img

        except Exception as e:
            print(f"⚠️ 渲染信息显示时出错: {e}")
            error_img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            error_img[:, :] = self.display_config['BACKGROUND_COLOR']
            error_img = put_chinese_text(error_img, "渲染错误", (self.window_width // 2 - 50, self.window_height // 2), 30, warning_color, 2)
            return error_img


class PerceptiveExplorer:
    """基于感知的自主探索无人机 - 智能决策增强版（双色物体检测版）"""

    def __init__(self, drone_name=""):
        self._setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("AirSimNH 感知驱动自主探索系统 - 双窗口双色物体检测版")
        self.logger.info("=" * 60)

        self.client = None
        self.drone_name = drone_name
        self._connect_to_airsim()

        try:
            self.client.enableApiControl(True, vehicle_name=drone_name)
            self.client.armDisarm(True, vehicle_name=drone_name)
            self.logger.info("✅ API控制已启用")
        except Exception as e:
            self.logger.error(f"❌ 启用API控制失败: {e}")
            raise

        self.state = FlightState.TAKEOFF
        self.state_history = deque(maxlen=20)
        self.emergency_flag = False

        self.depth_threshold_near = config.PERCEPTION['DEPTH_NEAR_THRESHOLD']
        self.depth_threshold_safe = config.PERCEPTION['DEPTH_SAFE_THRESHOLD']
        self.min_ground_clearance = config.PERCEPTION['MIN_GROUND_CLEARANCE']
        self.max_pitch_angle = math.radians(config.PERCEPTION['MAX_PITCH_ANGLE_DEG'])
        self.scan_angles = config.PERCEPTION['SCAN_ANGLES']

        self.exploration_time = config.EXPLORATION['TOTAL_TIME']
        self.preferred_speed = config.EXPLORATION['PREFERRED_SPEED']
        self.max_altitude = config.EXPLORATION['MAX_ALTITUDE']
        self.min_altitude = config.EXPLORATION['MIN_ALTITUDE']
        self.base_height = config.EXPLORATION['BASE_HEIGHT']
        self.takeoff_height = config.EXPLORATION['TAKEOFF_HEIGHT']

        self.vector_planner = VectorFieldPlanner()
        self.exploration_grid = ExplorationGrid(
            resolution=config.INTELLIGENT_DECISION['GRID_RESOLUTION'],
            grid_size=config.INTELLIGENT_DECISION['GRID_SIZE']
        )

        self.velocity_pid = PIDController(
            config.INTELLIGENT_DECISION['PID_KP'],
            config.INTELLIGENT_DECISION['PID_KI'],
            config.INTELLIGENT_DECISION['PID_KD']
        )
        self.height_pid = PIDController(1.0, 0.1, 0.3)

        self.exploration_target = None
        self.target_update_time = 0
        self.target_lifetime = config.INTELLIGENT_DECISION.get('TARGET_LIFETIME', 15.0)
        self.target_reached_distance = config.INTELLIGENT_DECISION.get('TARGET_REACHED_DISTANCE', 3.0)

        self.red_objects = []
        self.red_object_id_counter = 0
        self.last_red_detection_time = 0
        self.red_detection_interval = config.PERCEPTION['RED_OBJECT_DETECTION']['UPDATE_INTERVAL']
        self.red_object_memory_time = config.PERCEPTION['RED_OBJECT_DETECTION']['MEMORY_TIME']

        self.blue_objects = []
        self.blue_object_id_counter = 0
        self.last_blue_detection_time = 0
        self.blue_detection_interval = config.PERCEPTION['BLUE_OBJECT_DETECTION']['UPDATE_INTERVAL']
        self.blue_object_memory_time = config.PERCEPTION['BLUE_OBJECT_DETECTION']['MEMORY_TIME']

        self.black_objects = []
        self.black_object_id_counter = 0
        self.last_black_detection_time = 0
        self.black_detection_interval = config.PERCEPTION['BLACK_OBJECT_DETECTION']['UPDATE_INTERVAL']
        self.black_object_memory_time = config.PERCEPTION['BLACK_OBJECT_DETECTION']['MEMORY_TIME']

        self.visited_positions = deque(maxlen=100)

        self.loop_count = 0
        self.start_time = time.time()
        self.last_health_check = 0
        self.reconnect_attempts = 0
        self.last_successful_loop = time.time()

        self.data_logger = None
        self.last_data_record_time = 0
        self.data_record_interval = config.DATA_RECORDING.get('RECORD_INTERVAL', 0.2)
        if config.DATA_RECORDING['ENABLED']:
            self._setup_data_logger()

        self.last_performance_report = time.time()
        self.performance_report_interval = config.PERFORMANCE.get('REPORT_INTERVAL', 30.0)

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
            'red_objects_detected': 0,
            'red_objects_visited': 0,
            'blue_objects_detected': 0,
            'blue_objects_visited': 0,
            'black_objects_detected': 0,
            'black_objects_visited': 0,
        }

        # 初始化两个窗口
        self.front_window = None
        self.info_window = None
        self._setup_windows()

        self.manual_control_start = 0
        self.control_keys = {}

        self.logger.info("✅ 系统初始化完成")
        self.logger.info(f"   开始时间: {datetime.now().strftime('%H:%M:%S')}")
        self.logger.info(f"   预计探索时长: {self.exploration_time}秒")
        self.logger.info(f"   智能决策: 向量场避障 + 网格探索 + 三色物体检测")
        self.logger.info(f"   显示系统: 双窗口模式 (前视窗口 + 信息窗口)")
        if config.DATA_RECORDING['ENABLED']:
            self.logger.info(f"   数据记录: CSV + JSON 格式")
        if config.PERCEPTION['RED_OBJECT_DETECTION']['ENABLED']:
            self.logger.info(f"   红色物体检测: 已启用")
        if config.PERCEPTION['BLUE_OBJECT_DETECTION']['ENABLED']:
            self.logger.info(f"   蓝色物体检测: 已启用")
        if config.PERCEPTION['BLACK_OBJECT_DETECTION']['ENABLED']:
            self.logger.info(f"   黑色物体检测: 已启用")

    def _setup_logging(self):
        self.logger = logging.getLogger('DroneExplorer')
        self.logger.setLevel(getattr(logging, config.SYSTEM['LOG_LEVEL']))

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

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
        try:
            self.client.ping()
            self.logger.debug("✅ 连接健康检查通过")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ 连接健康检查失败: {e}")
            try:
                self._connect_to_airsim()
                return True
            except:
                return False

    def _setup_windows(self):
        """初始化两个显示窗口"""
        try:
            # 前视窗口（窗口标题使用英文，避免乱码）
            front_window_name = _translate_to_english(config.DISPLAY['FRONT_VIEW_WINDOW']['NAME'])
            self.front_window = FrontViewWindow(
                window_name=f"{front_window_name} - {self.drone_name or 'AirSimNH'}",
                width=config.DISPLAY['FRONT_VIEW_WINDOW']['WIDTH'],
                height=config.DISPLAY['FRONT_VIEW_WINDOW']['HEIGHT'],
                enable_sharpening=config.DISPLAY['FRONT_VIEW_WINDOW']['ENABLE_SHARPENING'],
                show_info=config.DISPLAY['FRONT_VIEW_WINDOW']['SHOW_INFO_OVERLAY']
            )
            self.logger.info("🎥 前视窗口已初始化")

            # 信息显示窗口（窗口标题使用英文，避免乱码）
            info_window_name = _translate_to_english(config.DISPLAY['INFO_WINDOW']['NAME'])
            self.info_window = InfoDisplayWindow(
                window_name=f"{info_window_name} - {self.drone_name or 'AirSimNH'}",
                width=config.DISPLAY['INFO_WINDOW']['WIDTH'],
                height=config.DISPLAY['INFO_WINDOW']['HEIGHT']
            )
            self.logger.info("📊 信息显示窗口已初始化")

        except Exception as e:
            self.logger.error(f"❌ 窗口初始化失败: {e}")

    def _update_info_window(self, perception: PerceptionResult):
        """更新信息显示窗口"""
        if not self.info_window:
            return

        try:
            # 获取无人机状态
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position

            # 收集性能信息
            cpu_usage = psutil.cpu_percent(interval=0) if config.PERFORMANCE['ENABLE_REALTIME_METRICS'] else 0.0
            memory_usage = psutil.virtual_memory().percent if config.PERFORMANCE['ENABLE_REALTIME_METRICS'] else 0.0

            # 准备信息数据
            # 无论字体是否禁用，都提前翻译状态值，避免部分翻译导致问号
            state_value = self.state.value
            if any('\u4e00' <= char <= '\u9fff' for char in state_value):
                state_value = _translate_to_english(state_value)
            
            # 获取手动控制信息
            manual_info = None
            if self.state == FlightState.MANUAL:
                manual_info = self._get_manual_control_info()
            
            info_data = {
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'state': state_value,
                'position': (pos.x_val, pos.y_val, pos.z_val),
                'perception': {
                    'obstacle_distance': perception.obstacle_distance,
                    'open_space_score': perception.open_space_score,
                    'has_obstacle': perception.has_obstacle
                },
                'objects_stats': {
                    'red_total': len(self.red_objects),
                    'red_visited': sum(1 for obj in self.red_objects if obj.visited),
                    'blue_total': len(self.blue_objects),
                    'blue_visited': sum(1 for obj in self.blue_objects if obj.visited),
                    'black_total': len(self.black_objects),
                    'black_visited': sum(1 for obj in self.black_objects if obj.visited),
                    'red_in_view': perception.red_objects_count,
                    'blue_in_view': perception.blue_objects_count,
                    'black_in_view': perception.black_objects_count
                },
                'grid_stats': {
                    'frontiers': len(self.exploration_grid.frontier_cells),
                    'explored': np.sum(self.exploration_grid.grid > 0.7),
                    'total': self.exploration_grid.grid_size * self.exploration_grid.grid_size
                },
                'performance': {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'loop_time': self.stats.get('average_loop_time', 0)
                },
                'manual_info': manual_info  # 添加手动控制信息
            }

            # 添加网格图像
            if config.DISPLAY['INFO_WINDOW']['SHOW_GRID']:
                grid_img = self.exploration_grid.visualize_grid(size=config.DISPLAY['INFO_WINDOW']['GRID_SIZE'])
                info_data['grid_image'] = grid_img

            # 更新信息窗口
            self.info_window.update_info(info_data)

        except Exception as e:
            self.logger.warning(f"⚠️ 更新信息窗口时出错: {e}")

    def _detect_color_objects_generic(
        self, 
        image: np.ndarray, 
        color_type: str,
        depth_array: Optional[np.ndarray] = None
    ) -> Tuple[List[Any], np.ndarray]:
        """
        通用的颜色物体检测函数
        
        Args:
            image: 输入图像
            color_type: 颜色类型 ('red', 'blue', 'black')
            depth_array: 深度数组（可选）
        
        Returns:
            (检测到的物体列表, 标记后的图像)
        """
        # 颜色类型配置映射
        COLOR_CONFIG = {
            'red': {
                'detection_config': config.PERCEPTION['RED_OBJECT_DETECTION'],
                'color_range_config': config.CAMERA['RED_COLOR_RANGE'],
                'object_class': RedObject,
                'object_list_attr': 'red_objects',
                'id_counter_attr': 'red_object_id_counter',
                'memory_time_attr': 'red_object_memory_time',
                'stats_key': 'red_objects_detected',
                'log_emoji': '🔴',
                'log_name': '红色物体',
                'mark_color': (0, 100, 255),  # BGR格式
                'label_prefix': 'R:',
                'record_func': 'record_red_object',
                'record_enabled_config': 'RECORD_RED_OBJECTS',
                'has_dual_range': True,  # 红色需要两个颜色范围
                'is_same_func': self._is_same_object_generic,
            },
            'blue': {
                'detection_config': config.PERCEPTION['BLUE_OBJECT_DETECTION'],
                'color_range_config': config.CAMERA['BLUE_COLOR_RANGE'],
                'object_class': BlueObject,
                'object_list_attr': 'blue_objects',
                'id_counter_attr': 'blue_object_id_counter',
                'memory_time_attr': 'blue_object_memory_time',
                'stats_key': 'blue_objects_detected',
                'log_emoji': '🔵',
                'log_name': '蓝色物体',
                'mark_color': (255, 100, 0),  # BGR格式
                'label_prefix': 'B:',
                'record_func': 'record_blue_object',
                'record_enabled_config': 'RECORD_BLUE_OBJECTS',
                'has_dual_range': False,
                'is_same_func': self._is_same_object_generic,
            },
            'black': {
                'detection_config': config.PERCEPTION['BLACK_OBJECT_DETECTION'],
                'color_range_config': config.CAMERA['BLACK_COLOR_RANGE'],
                'object_class': BlackObject,
                'object_list_attr': 'black_objects',
                'id_counter_attr': 'black_object_id_counter',
                'memory_time_attr': 'black_object_memory_time',
                'stats_key': 'black_objects_detected',
                'log_emoji': '⚫',
                'log_name': '黑色物体',
                'mark_color': (128, 128, 128),  # BGR格式
                'label_prefix': 'K:',
                'record_func': 'record_black_object',
                'record_enabled_config': 'RECORD_BLACK_OBJECTS',
                'has_dual_range': False,
                'is_same_func': self._is_same_object_generic,
            }
        }
        
        if color_type not in COLOR_CONFIG:
            raise ValueError(f"不支持的颜色类型: {color_type}，支持的类型: {list(COLOR_CONFIG.keys())}")
        
        cfg = COLOR_CONFIG[color_type]
        detected_objects = []
        marked_image = image.copy() if image is not None else None
        
        # 检查是否启用检测
        if not cfg['detection_config']['ENABLED'] or image is None:
            return detected_objects, marked_image
        
        try:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 创建颜色掩码（红色需要特殊处理，有两个范围）
            if cfg['has_dual_range']:
                # 红色：两个颜色范围
                lower1 = np.array(cfg['color_range_config']['LOWER1'])
                upper1 = np.array(cfg['color_range_config']['UPPER1'])
                lower2 = np.array(cfg['color_range_config']['LOWER2'])
                upper2 = np.array(cfg['color_range_config']['UPPER2'])
                
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                color_mask = cv2.bitwise_or(mask1, mask2)
            else:
                # 其他颜色：单个颜色范围
                lower = np.array(cfg['color_range_config']['LOWER'])
                upper = np.array(cfg['color_range_config']['UPPER'])
                color_mask = cv2.inRange(hsv, lower, upper)
            
            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 获取无人机状态
            try:
                state = self.client.getMultirotorState(vehicle_name=self.drone_name)
                drone_pos = state.kinematics_estimated.position
                orientation = state.kinematics_estimated.orientation
                roll, pitch, yaw = airsim.to_eularian_angles(orientation)
            except:
                drone_pos = None
                yaw = 0.0
            
            # 获取配置参数
            min_area = cfg['detection_config']['MIN_AREA']
            max_area = cfg['detection_config']['MAX_AREA']
            existing_objects = getattr(self, cfg['object_list_attr'])
            id_counter_attr = cfg['id_counter_attr']
            
            # 处理每个轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # 计算置信度
                    aspect_ratio = w / h if h > 0 else 1.0
                    confidence = min(1.0, area / 1000.0) * (1.0 / (1.0 + abs(aspect_ratio - 1.0)))
                    
                    # 计算世界坐标位置
                    world_pos = None
                    if drone_pos is not None and depth_array is not None:
                        try:
                            if 0 <= center_y < depth_array.shape[0] and 0 <= center_x < depth_array.shape[1]:
                                distance = depth_array[center_y, center_x]
                                
                                if 0.5 < distance < 50.0:
                                    height, width = depth_array.shape
                                    fov_h = math.radians(90)
                                    
                                    pixel_angle_x = (center_x - width/2) / (width/2) * (fov_h/2)
                                    pixel_angle_y = (center_y - height/2) / (height/2) * (fov_h/2)
                                    
                                    z = distance
                                    x_rel = z * math.tan(pixel_angle_x)
                                    y_rel = z * math.tan(pixel_angle_y)
                                    
                                    world_x = x_rel * math.cos(yaw) - y_rel * math.sin(yaw) + drone_pos.x_val
                                    world_y = x_rel * math.sin(yaw) + y_rel * math.cos(yaw) + drone_pos.y_val
                                    world_z = drone_pos.z_val
                                    
                                    world_pos = (world_x, world_y, world_z)
                        except:
                            pass
                    
                    # 创建物体对象
                    current_id = getattr(self, id_counter_attr)
                    color_object = cfg['object_class'](
                        id=current_id,
                        position=world_pos if world_pos else (0.0, 0.0, 0.0),
                        pixel_position=(center_x, center_y),
                        size=area,
                        confidence=confidence,
                        timestamp=time.time(),
                        last_seen=time.time(),
                        visited=False
                    )
                    
                    # 检查是否是已存在的物体
                    is_new_object = True
                    for existing_obj in existing_objects:
                        if cfg['is_same_func'](color_object, existing_obj):
                            existing_obj.last_seen = time.time()
                            existing_obj.pixel_position = color_object.pixel_position
                            existing_obj.confidence = max(existing_obj.confidence, confidence)
                            if world_pos:
                                existing_obj.position = world_pos
                            color_object = existing_obj
                            is_new_object = False
                            break
                    
                    # 如果是新物体，增加ID计数器和统计
                    if is_new_object:
                        setattr(self, id_counter_attr, current_id + 1)
                        detected_objects.append(color_object)
                        self.stats[cfg['stats_key']] += 1
                        self.logger.info(f"{cfg['log_emoji']} 检测到{cfg['log_name']} #{color_object.id} (置信度: {confidence:.2f})")
                        
                        # 记录到数据日志
                        if self.data_logger and config.DATA_RECORDING.get(cfg['record_enabled_config'], False):
                            record_func = getattr(self.data_logger, cfg['record_func'], None)
                            if record_func:
                                record_func(color_object)
                    else:
                        detected_objects.append(color_object)
                    
                    # 在图像上标记物体
                    if marked_image is not None:
                        mark_color = cfg['mark_color']
                        if color_object.visited:
                            mark_color = (0, 200, 0)  # 已访问的物体显示为绿色
                        
                        cv2.rectangle(marked_image, (x, y), (x+w, y+h), mark_color, 2)
                        cv2.circle(marked_image, (center_x, center_y), 5, mark_color, -1)
                        
                        label = f"{cfg['label_prefix']}{color_object.id} ({confidence:.2f})"
                        cv2.putText(marked_image, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mark_color, 1)
            
            # 清理过期的物体记忆
            current_time = time.time()
            memory_time = getattr(self, cfg['memory_time_attr'])
            filtered_objects = [obj for obj in existing_objects 
                              if current_time - obj.last_seen < memory_time]
            setattr(self, cfg['object_list_attr'], filtered_objects)
            
            # 更新统计信息
            visited_count = sum(1 for obj in filtered_objects if obj.visited)
            if len(detected_objects) > 0:
                self.logger.debug(f"{cfg['log_emoji']} 当前{cfg['log_name']}: {len(filtered_objects)}个, 已访问: {visited_count}个")
        
        except Exception as e:
            cfg = COLOR_CONFIG[color_type]
            self.logger.warning(f"⚠️ {cfg['log_name']}检测失败: {e}")
        
        return detected_objects, marked_image

    def _detect_red_objects(self, image: np.ndarray, depth_array: Optional[np.ndarray] = None) -> Tuple[List[RedObject], np.ndarray]:
        """检测红色物体 - 使用通用检测函数"""
        objects, marked_image = self._detect_color_objects_generic(image, 'red', depth_array)
        return objects, marked_image

    def _detect_blue_objects(self, image: np.ndarray, depth_array: Optional[np.ndarray] = None) -> Tuple[List[BlueObject], np.ndarray]:
        """检测蓝色物体 - 使用通用检测函数"""
        objects, marked_image = self._detect_color_objects_generic(image, 'blue', depth_array)
        return objects, marked_image

    def _detect_black_objects(self, image: np.ndarray, depth_array: Optional[np.ndarray] = None) -> Tuple[List[BlackObject], np.ndarray]:
        """检测黑色物体 - 使用通用检测函数"""
        objects, marked_image = self._detect_color_objects_generic(image, 'black', depth_array)
        return objects, marked_image

    def _is_same_object_generic(self, obj1: Any, obj2: Any, distance_threshold=2.0) -> bool:
        """
        通用的物体相似度判断函数
        适用于RedObject、BlueObject、BlackObject等具有相同结构的对象
        
        Args:
            obj1: 第一个物体对象
            obj2: 第二个物体对象
            distance_threshold: 距离阈值（米），默认2.0米
        
        Returns:
            bool: 如果两个物体被认为是同一个物体，返回True
        """
        # 如果两个物体都有有效的世界坐标位置，使用世界坐标距离判断
        if obj1.position != (0.0, 0.0, 0.0) and obj2.position != (0.0, 0.0, 0.0):
            distance = math.sqrt(
                (obj1.position[0] - obj2.position[0])**2 +
                (obj1.position[1] - obj2.position[1])**2
            )
            return distance < distance_threshold

        # 如果没有有效的世界坐标，使用像素坐标和时间差判断
        pixel_distance = math.sqrt(
            (obj1.pixel_position[0] - obj2.pixel_position[0])**2 +
            (obj1.pixel_position[1] - obj2.pixel_position[1])**2
        )
        time_diff = abs(obj1.timestamp - obj2.timestamp)

        # 像素距离小于50且时间差小于5秒，认为是同一个物体
        return pixel_distance < 50 and time_diff < 5.0

    def _is_same_object(self, obj1: RedObject, obj2: RedObject, distance_threshold=2.0) -> bool:
        """检测红色物体是否相同 - 使用通用函数"""
        return self._is_same_object_generic(obj1, obj2, distance_threshold)

    def _is_same_object_blue(self, obj1: BlueObject, obj2: BlueObject, distance_threshold=2.0) -> bool:
        """检测蓝色物体是否相同 - 使用通用函数"""
        return self._is_same_object_generic(obj1, obj2, distance_threshold)

    def _is_same_object_black(self, obj1: BlackObject, obj2: BlackObject, distance_threshold=2.0) -> bool:
        """检测黑色物体是否相同 - 使用通用函数"""
        return self._is_same_object_generic(obj1, obj2, distance_threshold)

    def _check_object_proximity_generic(self, current_pos: Tuple[float, float], color_type: str) -> bool:
        """
        通用的物体接近检测函数
        检查当前位置是否接近指定颜色类型的未访问物体
        
        Args:
            current_pos: 当前位置 (x, y)
            color_type: 颜色类型 ('red', 'blue', 'black')
        
        Returns:
            bool: 如果检测到接近物体并触发了访问，返回True；否则返回False
        """
        # 颜色类型配置映射
        PROXIMITY_CONFIG = {
            'red': {
                'objects_attr': 'red_objects',
                'exploration_config': config.INTELLIGENT_DECISION['RED_OBJECT_EXPLORATION'],
                'stats_key': 'red_objects_visited',
                'log_name': '红色物体',
                'event_type': 'red_object_visited',
                'inspection_state': FlightState.RED_OBJECT_INSPECTION,
            },
            'blue': {
                'objects_attr': 'blue_objects',
                'exploration_config': config.INTELLIGENT_DECISION['BLUE_OBJECT_EXPLORATION'],
                'stats_key': 'blue_objects_visited',
                'log_name': '蓝色物体',
                'event_type': 'blue_object_visited',
                'inspection_state': FlightState.BLUE_OBJECT_INSPECTION,
            },
            'black': {
                'objects_attr': 'black_objects',
                'exploration_config': config.INTELLIGENT_DECISION['BLACK_OBJECT_EXPLORATION'],
                'stats_key': 'black_objects_visited',
                'log_name': '黑色物体',
                'event_type': 'black_object_visited',
                'inspection_state': FlightState.BLACK_OBJECT_INSPECTION,
            }
        }
        
        if color_type not in PROXIMITY_CONFIG:
            raise ValueError(f"不支持的颜色类型: {color_type}，支持的类型: {list(PROXIMITY_CONFIG.keys())}")
        
        cfg = PROXIMITY_CONFIG[color_type]
        
        # 获取物体列表
        objects = getattr(self, cfg['objects_attr'])
        
        # 遍历所有未访问的物体
        for obj in objects:
            if not obj.visited:
                # 计算距离
                distance = math.sqrt(
                    (obj.position[0] - current_pos[0])**2 +
                    (obj.position[1] - current_pos[1])**2
                )
                
                # 获取最小接近距离
                min_distance = cfg['exploration_config']['MIN_DISTANCE']
                
                # 如果距离小于最小距离，标记为已访问
                if distance < min_distance:
                    obj.visited = True
                    obj.last_seen = time.time()
                    self.stats[cfg['stats_key']] += 1
                    
                    # 记录日志
                    self.logger.info(f"✅ 已访问{cfg['log_name']} #{obj.id} (距离: {distance:.1f}m)")
                    
                    # 记录事件到数据日志
                    if self.data_logger:
                        event_data = {
                            'object_id': obj.id,
                            'position': obj.position,
                            'distance': distance,
                            'timestamp': time.time()
                        }
                        self.data_logger.record_event(cfg['event_type'], event_data)
                    
                    # 改变状态为物体检查状态
                    self.change_state(cfg['inspection_state'])
                    return True
        
        return False

    def _check_red_object_proximity(self, current_pos):
        """检查红色物体接近 - 使用通用函数"""
        return self._check_object_proximity_generic(current_pos, 'red')

    def _check_blue_object_proximity(self, current_pos):
        """检查蓝色物体接近 - 使用通用函数"""
        return self._check_object_proximity_generic(current_pos, 'blue')

    def _check_black_object_proximity(self, current_pos):
        """检查黑色物体接近 - 使用通用函数"""
        return self._check_object_proximity_generic(current_pos, 'black')

    def get_depth_perception(self) -> PerceptionResult:
        result = PerceptionResult()
        self.stats['perception_cycles'] += 1

        try:
            if config.SYSTEM['ENABLE_HEALTH_CHECK']:
                current_time = time.time()
                if current_time - self.last_successful_loop > 10.0:
                    self.logger.warning("⚠️ 感知循环长时间无响应，尝试恢复...")
                    self._check_connection_health()

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

            depth_img = responses[0]
            depth_array = None
            if depth_img and hasattr(depth_img, 'image_data_float'):
                try:
                    depth_array = np.array(depth_img.image_data_float, dtype=np.float32)
                    depth_array = depth_array.reshape(depth_img.height, depth_img.width)

                    h, w = depth_array.shape

                    front_near = depth_array[h // 2:, w // 3:2 * w // 3]
                    min_front_distance = np.min(front_near) if front_near.size > 0 else 100

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

                    result.obstacle_positions = self._extract_obstacle_positions(depth_array, h, w)

                    ground_region = depth_array[3 * h // 4:, :]
                    if ground_region.size > 10:
                        row_variances = np.var(ground_region, axis=1)
                        result.terrain_slope = np.mean(row_variances) * 100

                    open_pixels = np.sum(depth_array[h // 2:, :] > self.depth_threshold_safe)
                    total_pixels = depth_array[h // 2:, :].size
                    result.open_space_score = open_pixels / total_pixels if total_pixels > 0 else 0

                    result.has_obstacle = min_front_distance < self.depth_threshold_near
                    result.obstacle_distance = min_front_distance
                    if result.has_obstacle:
                        self.stats['obstacles_detected'] += 1

                    if directions:
                        closest_dir = min(directions, key=lambda x: x[1])
                        result.obstacle_direction = closest_dir[0]

                    if result.terrain_slope > config.PERCEPTION['HEIGHT_STRATEGY']['SLOPE_THRESHOLD']:
                        result.recommended_height = config.PERCEPTION['HEIGHT_STRATEGY']['STEEP_SLOPE']
                    elif result.open_space_score > config.PERCEPTION['HEIGHT_STRATEGY']['OPENNESS_THRESHOLD']:
                        result.recommended_height = config.PERCEPTION['HEIGHT_STRATEGY']['OPEN_SPACE']

                except ValueError as e:
                    self.logger.error(f"❌ 深度图像数据转换错误: {e}")
                except Exception as e:
                    self.logger.error(f"❌ 深度图像处理异常: {e}")

            front_response = responses[1]
            if front_response and hasattr(front_response, 'image_data_uint8'):
                try:
                    img_array = np.frombuffer(front_response.image_data_uint8, dtype=np.uint8)

                    if len(img_array) > 0:
                        img_rgb = img_array.reshape(front_response.height, front_response.width, 3)
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                        current_time = time.time()

                        # 检测红色物体
                        if current_time - self.last_red_detection_time >= self.red_detection_interval:
                            red_objects, red_marked_image = self._detect_red_objects(img_bgr, depth_array)
                            result.red_objects = red_objects
                            result.red_objects_count = len(red_objects)
                            result.red_objects_image = red_marked_image
                            self.last_red_detection_time = current_time

                        # 检测蓝色物体
                        if current_time - self.last_blue_detection_time >= self.blue_detection_interval:
                            blue_objects, blue_marked_image = self._detect_blue_objects(img_bgr, depth_array)
                            result.blue_objects = blue_objects
                            result.blue_objects_count = len(blue_objects)
                            result.blue_objects_image = blue_marked_image
                            self.last_blue_detection_time = current_time

                        # 检测黑色物体
                        if current_time - self.last_black_detection_time >= self.black_detection_interval:
                            black_objects, black_marked_image = self._detect_black_objects(img_bgr, depth_array)
                            result.black_objects = black_objects
                            result.black_objects_count = len(black_objects)
                            result.black_objects_image = black_marked_image
                            self.last_black_detection_time = current_time

                        result.front_image = img_bgr

                        display_info = self._prepare_display_info(result)

                        self._update_exploration_grid(result)

                        self._record_flight_data(result)

                        # 更新信息窗口
                        self._update_info_window(result)

                        if self.front_window:
                            manual_info = None
                            if self.state == FlightState.MANUAL:
                                manual_info = self._get_manual_control_info()

                            # 内存优化：仅在需要标记时才复制图像
                            has_markers = (config.DISPLAY['FRONT_VIEW_WINDOW']['SHOW_RED_OBJECTS'] and result.red_objects_image is not None) or \
                                         (config.DISPLAY['FRONT_VIEW_WINDOW']['SHOW_BLUE_OBJECTS'] and result.blue_objects_image is not None) or \
                                         (config.DISPLAY['FRONT_VIEW_WINDOW']['SHOW_BLACK_OBJECTS'] and result.black_objects_image is not None)
                            
                            if has_markers:
                                display_image = img_bgr.copy()
                                if config.DISPLAY['FRONT_VIEW_WINDOW']['SHOW_RED_OBJECTS'] and result.red_objects_image is not None:
                                    red_mask = cv2.inRange(result.red_objects_image, (0, 100, 0), (0, 255, 255))
                                    display_image[red_mask > 0] = result.red_objects_image[red_mask > 0]

                                if config.DISPLAY['FRONT_VIEW_WINDOW']['SHOW_BLUE_OBJECTS'] and result.blue_objects_image is not None:
                                    blue_mask = cv2.inRange(result.blue_objects_image, (255, 100, 0), (255, 255, 255))
                                    display_image[blue_mask > 0] = result.blue_objects_image[blue_mask > 0]

                                if config.DISPLAY['FRONT_VIEW_WINDOW']['SHOW_BLACK_OBJECTS'] and result.black_objects_image is not None:
                                    black_mask = cv2.inRange(result.black_objects_image, (128, 128, 0), (128, 255, 255))
                                    display_image[black_mask > 0] = result.black_objects_image[black_mask > 0]
                            else:
                                # 没有标记时直接使用原图像引用
                                display_image = img_bgr

                            self.front_window.update_image(display_image, display_info, manual_info)
                            self.stats['front_image_updates'] += 1

                except Exception as e:
                    self.logger.warning(f"⚠️ 前视图像处理异常: {e}")

            self.last_successful_loop = time.time()

            if self.loop_count % 50 == 0 and config.DEBUG.get('LOG_DECISION_DETAILS', False):
                self.logger.debug(f"感知结果: 障碍={result.has_obstacle}, 距离={result.obstacle_distance:.1f}m, "
                                f"开阔度={result.open_space_score:.2f}, 红色物体={result.red_objects_count}个, "
                                f"蓝色物体={result.blue_objects_count}个, 黑色物体={result.black_objects_count}个")

        except Exception as e:
            if "ClientException" in str(type(e)) or "Connection" in str(e):
                self.logger.error(f"❌ AirSim客户端异常: {e}")
                self.stats['exceptions_caught'] += 1
                if self.data_logger:
                    self.data_logger.record_event('airsim_exception', {'error': str(e)})
                self._check_connection_health()
            else:
                self.logger.error(f"❌ 感知过程中发生未知异常: {e}")
                self.logger.debug(f"异常详情: {traceback.format_exc()}")
                self.stats['exceptions_caught'] += 1
                if self.data_logger:
                    self.data_logger.record_event('perception_exception', {'error': str(e)})

        return result

    def _record_flight_data(self, perception: PerceptionResult):
        if not config.DATA_RECORDING['ENABLED'] or not self.data_logger:
            return

        current_time = time.time()
        if current_time - self.last_data_record_time < self.data_record_interval:
            return

        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            orientation = state.kinematics_estimated.orientation

            roll, pitch, yaw = airsim.to_eularian_angles(orientation)

            cpu_usage = psutil.cpu_percent(interval=0) if config.PERFORMANCE['ENABLE_REALTIME_METRICS'] else 0.0
            memory_usage = psutil.virtual_memory().percent if config.PERFORMANCE['ENABLE_REALTIME_METRICS'] else 0.0

            red_objects_count = perception.red_objects_count
            red_objects_visited = sum(1 for obj in self.red_objects if obj.visited)

            blue_objects_count = perception.blue_objects_count
            blue_objects_visited = sum(1 for obj in self.blue_objects if obj.visited)

            black_objects_count = perception.black_objects_count
            black_objects_visited = sum(1 for obj in self.black_objects if obj.visited)

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
                'red_objects_count': red_objects_count,
                'red_objects_detected': self.stats['red_objects_detected'],
                'red_objects_visited': red_objects_visited,
                'blue_objects_count': blue_objects_count,
                'blue_objects_detected': self.stats['blue_objects_detected'],
                'blue_objects_visited': blue_objects_visited,
                'black_objects_count': black_objects_count,
                'black_objects_detected': self.stats['black_objects_detected'],
                'black_objects_visited': black_objects_visited,
            }

            self.data_logger.record_flight_data(data_dict)
            self.stats['data_points_recorded'] += 1
            self.last_data_record_time = current_time

        except Exception as e:
            self.logger.warning(f"⚠️ 记录飞行数据时出错: {e}")

    def _extract_obstacle_positions(self, depth_array, height, width):
        obstacles = []

        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            roll, pitch, yaw = airsim.to_eularian_angles(orientation)

            near_mask = depth_array < self.depth_threshold_near * 1.5

            step = 4
            for i in range(0, height, step):
                for j in range(0, width, step):
                    if near_mask[i, j]:
                        distance = depth_array[i, j]

                        fov_h = math.radians(90)
                        pixel_angle_x = (j - width/2) / (width/2) * (fov_h/2)
                        pixel_angle_y = (i - height/2) / (height/2) * (fov_h/2)

                        z = distance
                        x = z * math.tan(pixel_angle_x)
                        y = z * math.tan(pixel_angle_y)

                        world_x = x * math.cos(yaw) - y * math.sin(yaw) + pos.x_val
                        world_y = x * math.sin(yaw) + y * math.cos(yaw) + pos.y_val

                        obstacles.append((world_x, world_y))

            max_obstacles = 20
            if len(obstacles) > max_obstacles:
                obstacles = random.sample(obstacles, max_obstacles)

        except Exception as e:
            self.logger.warning(f"⚠️ 提取障碍物位置失败: {e}")

        return obstacles

    def _update_exploration_grid(self, perception: PerceptionResult):
        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position

            self.exploration_grid.update_position(pos.x_val, pos.y_val)

            if perception.obstacle_positions:
                self.exploration_grid.update_obstacles(perception.obstacle_positions)

            if perception.red_objects:
                self.exploration_grid.update_red_objects(perception.red_objects)

            if perception.blue_objects:
                self.exploration_grid.update_blue_objects(perception.blue_objects)

            if perception.black_objects:
                self.exploration_grid.update_black_objects(perception.black_objects)

            self.stats['grid_updates'] += 1

        except Exception as e:
            self.logger.warning(f"⚠️ 更新探索网格失败: {e}")

    def _prepare_display_info(self, perception: PerceptionResult) -> Dict:
        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position

            info = {
                'state': self.state.value,
                'obstacle_distance': perception.obstacle_distance,
                'position': (pos.x_val, pos.y_val, pos.z_val),
                'loop_count': self.loop_count,
                'red_objects_count': perception.red_objects_count,
                'red_objects_visited': sum(1 for obj in self.red_objects if obj.visited),
                'blue_objects_count': perception.blue_objects_count,
                'blue_objects_visited': sum(1 for obj in self.blue_objects if obj.visited),
                'black_objects_count': perception.black_objects_count,
                'black_objects_visited': sum(1 for obj in self.black_objects if obj.visited),
            }

            if hasattr(self, 'last_decision_info'):
                info['decision_info'] = self.last_decision_info

            if config.DATA_RECORDING['ENABLED']:
                info['data_points'] = self.stats['data_points_recorded']

            return info
        except:
            return {}

    def _get_manual_control_info(self):
        info_lines = []

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
                elif key == 32:
                    key_names.append("悬停")

            if key_names:
                info_lines.append(f"控制: {', '.join(key_names)}")
        else:
            info_lines.append("控制: 悬停")

        if self.red_objects:
            visited_count = sum(1 for obj in self.red_objects if obj.visited)
            info_lines.append(f"红色物体: {visited_count}/{len(self.red_objects)}")

        if self.blue_objects:
            visited_count = sum(1 for obj in self.blue_objects if obj.visited)
            info_lines.append(f"蓝色物体: {visited_count}/{len(self.blue_objects)}")

        if self.black_objects:
            visited_count = sum(1 for obj in self.black_objects if obj.visited)
            info_lines.append(f"黑色物体: {visited_count}/{len(self.black_objects)}")

        if self.manual_control_start > 0:
            elapsed = time.time() - self.manual_control_start
            info_lines.append(f"手动模式: {elapsed:.1f}秒")

        info_lines.append("ESC: 退出手动模式")

        return info_lines

    def apply_manual_control(self):
        if self.state != FlightState.MANUAL:
            return

        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation

            _, _, yaw = airsim.to_eularian_angles(orientation)

            vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0

            for key in list(self.control_keys.keys()):
                key_char = chr(key).lower() if 0 <= key <= 255 else ''

                if key_char == 'w':
                    vx += config.MANUAL['CONTROL_SPEED'] * math.cos(yaw)
                    vy += config.MANUAL['CONTROL_SPEED'] * math.sin(yaw)
                elif key_char == 's':
                    vx -= config.MANUAL['CONTROL_SPEED'] * math.cos(yaw)
                    vy -= config.MANUAL['CONTROL_SPEED'] * math.sin(yaw)

                if key_char == 'a':
                    vx += config.MANUAL['CONTROL_SPEED'] * math.cos(yaw + math.pi/2)
                    vy += config.MANUAL['CONTROL_SPEED'] * math.sin(yaw + math.pi/2)
                elif key_char == 'd':
                    vx += config.MANUAL['CONTROL_SPEED'] * math.cos(yaw - math.pi/2)
                    vy += config.MANUAL['CONTROL_SPEED'] * math.sin(yaw - math.pi/2)

                if key_char == 'q':
                    vz = -config.MANUAL['ALTITUDE_SPEED']
                elif key_char == 'e':
                    vz = config.MANUAL['ALTITUDE_SPEED']

                if key_char == 'z':
                    yaw_rate = -math.radians(config.MANUAL['YAW_SPEED'])
                elif key_char == 'x':
                    yaw_rate = math.radians(config.MANUAL['YAW_SPEED'])

                if key == 32:
                    self.client.hoverAsync(vehicle_name=self.drone_name)
                    self.control_keys = {}
                    return

            if config.MANUAL['SAFETY_ENABLED']:
                speed = math.sqrt(vx**2 + vy**2)
                if speed > config.MANUAL['MAX_MANUAL_SPEED']:
                    scale = config.MANUAL['MAX_MANUAL_SPEED'] / speed
                    vx *= scale
                    vy *= scale

                target_z = pos.z_val + vz * 0.1
                if target_z > config.MANUAL['MIN_ALTITUDE_LIMIT']:
                    vz = max(vz, (config.MANUAL['MIN_ALTITUDE_LIMIT'] - pos.z_val) * 10)
                if target_z < config.MANUAL['MAX_ALTITUDE_LIMIT']:
                    vz = min(vz, (config.MANUAL['MAX_ALTITUDE_LIMIT'] - pos.z_val) * 10)

            if vx != 0.0 or vy != 0.0 or vz != 0.0:
                self.client.moveByVelocityAsync(vx, vy, vz, 0.1, vehicle_name=self.drone_name)
            elif yaw_rate != 0.0:
                self.client.rotateByYawRateAsync(yaw_rate, 0.1, vehicle_name=self.drone_name)
            elif config.MANUAL['ENABLE_AUTO_HOVER'] and not self.control_keys:
                self.client.hoverAsync(vehicle_name=self.drone_name)

        except Exception as e:
            self.logger.warning(f"⚠️ 手动控制应用失败: {e}")

    def change_state(self, new_state: FlightState):
        if self.state != new_state:
            old_state = self.state.value
            self.logger.info(f"🔄 状态转换: {old_state} → {new_state.value}")
            self.state = new_state
            self.state_history.append((time.time(), new_state))
            self.stats['state_changes'] += 1

            if self.data_logger:
                event_data = {
                    'old_state': old_state,
                    'new_state': new_state.value,
                    'loop_count': self.loop_count
                }
                self.data_logger.record_event('state_change', event_data)

    def run_manual_control(self):
        self.logger.info("=" * 60)
        self.logger.info("启动手动控制模式")
        self.logger.info("=" * 60)

        if not self.front_window:
            self.logger.error("❌ 前视窗口未初始化")
            return

        try:
            self.change_state(FlightState.MANUAL)
            self.manual_control_start = time.time()

            self.front_window.set_manual_mode(True)

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

            self.control_keys = {}

            manual_active = True
            last_control_time = time.time()
            last_image_time = time.time()

            while manual_active and not self.emergency_flag:
                try:
                    if self.front_window.should_exit_manual():
                        self.logger.info("收到退出手动模式指令")
                        manual_active = False
                        break

                    if self.front_window:
                        window_keys = self.front_window.get_control_inputs()
                        self.control_keys = window_keys.copy()

                    if not self.front_window.display_active:
                        self.logger.info("前视窗口已关闭，退出手动模式")
                        manual_active = False
                        break

                    current_time = time.time()
                    if current_time - last_control_time >= 0.05:
                        self.apply_manual_control()
                        last_control_time = current_time

                    if current_time - last_image_time >= 0.1:
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

                                    if self.front_window:
                                        manual_info = self._get_manual_control_info()
                                        self.front_window.update_image(img_bgr, display_info, manual_info)
                                        last_image_time = current_time
                        except Exception as img_error:
                            pass

                    try:
                        state = self.client.getMultirotorState(vehicle_name=self.drone_name)
                        pos = state.kinematics_estimated.position
                        current_pos = (pos.x_val, pos.y_val)
                        self._check_red_object_proximity(current_pos)
                        self._check_blue_object_proximity(current_pos)
                        self._check_black_object_proximity(current_pos)
                        
                        # 更新信息窗口（手动控制模式下）
                        if current_time - last_image_time >= 0.1 and self.info_window:
                            # 创建一个简化的感知结果用于信息窗口更新
                            simple_perception = PerceptionResult()
                            simple_perception.obstacle_distance = 100.0
                            simple_perception.open_space_score = 1.0
                            simple_perception.has_obstacle = False
                            simple_perception.red_objects_count = len(self.red_objects)
                            simple_perception.blue_objects_count = len(self.blue_objects)
                            simple_perception.black_objects_count = len(self.black_objects)
                            self._update_info_window(simple_perception)
                    except:
                        pass

                    time.sleep(0.01)

                except KeyboardInterrupt:
                    self.logger.warning("⏹️ 用户中断手动控制")
                    manual_active = False
                    break
                except Exception as e:
                    self.logger.error(f"❌ 手动控制循环异常: {e}")
                    time.sleep(0.1)

            manual_time = time.time() - self.manual_control_start
            self.stats['manual_control_time'] = manual_time

            self.manual_control_start = 0
            self.control_keys = {}
            if self.front_window:
                self.front_window.set_manual_mode(False)

            try:
                self.client.hoverAsync(vehicle_name=self.drone_name).join()
            except:
                pass

            self.logger.info(f"⏱️  手动控制结束，持续时间: {manual_time:.1f}秒")

            self.change_state(FlightState.HOVERING)

            print("\n" + "="*60)
            print("手动控制模式已结束")
            print(f"控制时间: {manual_time:.1f}秒")
            print(f"检测到红色物体: {self.stats['red_objects_detected']}个")
            print(f"检测到蓝色物体: {self.stats['blue_objects_detected']}个")
            print("="*60)
            print("请选择下一步:")
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
        self.logger.info("=" * 60)
        self.logger.info("启动感知-决策-控制主循环")
        self.logger.info("=" * 60)

        try:
            self.logger.info("🚀 起飞中...")
            self.client.takeoffAsync(vehicle_name=self.drone_name).join()
            time.sleep(2)

            self.client.moveToZAsync(self.takeoff_height, 3, vehicle_name=self.drone_name).join()
            self.change_state(FlightState.HOVERING)
            time.sleep(2)

            exploration_start = time.time()

            while (time.time() - exploration_start < self.exploration_time and
                   not self.emergency_flag):

                self.loop_count += 1
                loop_start = time.time()

                perception = self.get_depth_perception()

                try:
                    state = self.client.getMultirotorState(vehicle_name=self.drone_name)
                    pos = state.kinematics_estimated.position
                    current_pos = (pos.x_val, pos.y_val)
                    if self._check_red_object_proximity(current_pos):
                        time.sleep(2)
                        self.change_state(FlightState.EXPLORING)
                    if self._check_blue_object_proximity(current_pos):
                        time.sleep(2)
                        self.change_state(FlightState.EXPLORING)
                    if self._check_black_object_proximity(current_pos):
                        time.sleep(2)
                        self.change_state(FlightState.EXPLORING)
                except:
                    pass

                decision = self.make_intelligent_decision(perception)

                self._execute_control_decision(decision)

                loop_time = time.time() - loop_start
                self.stats['average_loop_time'] = (self.stats['average_loop_time'] * (self.loop_count-1) + loop_time) / self.loop_count
                self.stats['max_loop_time'] = max(self.stats['max_loop_time'], loop_time)
                self.stats['min_loop_time'] = min(self.stats['min_loop_time'], loop_time)

                if self.data_logger:
                    self.data_logger.record_loop_time(loop_time)

                current_time = time.time()
                if current_time - self.last_performance_report >= self.performance_report_interval:
                    self._generate_performance_report()
                    self.last_performance_report = current_time

                if self.loop_count % config.SYSTEM.get('HEALTH_CHECK_INTERVAL', 20) == 0:
                    self._report_status(exploration_start, perception)
                    # 内存优化：定期垃圾回收
                    gc.collect()

                loop_time = time.time() - loop_start
                if loop_time < 0.1:
                    time.sleep(0.1 - loop_time)

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
        try:
            if not config.PERFORMANCE['ENABLE_REALTIME_METRICS']:
                return

            cpu_usage = psutil.cpu_percent(interval=0)
            memory_usage = psutil.virtual_memory().percent

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
                target_yaw = (time.time() % 10) * 0.2

                current_time = time.time()
                if (self.exploration_target is None or
                    current_time - self.target_update_time > self.target_lifetime):

                    self.exploration_target = self.exploration_grid.get_best_exploration_target(
                        (pos.x_val, pos.y_val),
                        perception.red_objects,
                        perception.blue_objects,
                        perception.black_objects
                    )
                    self.target_update_time = current_time

                    if self.exploration_target:
                        self.logger.info(f"🎯 新探索目标: {self.exploration_target[0]:.1f}, {self.exploration_target[1]:.1f}")

                if self.exploration_target:
                    self.change_state(FlightState.EXPLORING)

            elif self.state == FlightState.EXPLORING:
                if perception.has_obstacle:
                    self.change_state(FlightState.AVOIDING)
                    target_vx, target_vy = -vel.x_val * 2, -vel.y_val * 2
                else:
                    current_pos = (pos.x_val, pos.y_val)

                    if self.exploration_target is None:
                        self.exploration_target = self.exploration_grid.get_best_exploration_target(
                            current_pos,
                            perception.red_objects,
                            perception.blue_objects,
                            perception.black_objects
                        )
                        self.target_update_time = time.time()

                    vector = self.vector_planner.compute_vector(
                        current_pos,
                        self.exploration_target,
                        perception.obstacle_positions,
                        perception.red_objects,
                        perception.blue_objects,
                        perception.black_objects
                    )

                    speed_factor = self._calculate_adaptive_speed(perception, vector.magnitude())

                    target_speed = self.preferred_speed * speed_factor
                    current_speed = math.sqrt(vel.x_val**2 + vel.y_val**2)
                    speed_error = target_speed - current_speed
                    speed_adjustment = self.velocity_pid.update(speed_error)

                    final_vector = vector.normalize() * (target_speed + speed_adjustment)
                    target_vx = final_vector.x
                    target_vy = final_vector.y

                    self.stats['vector_field_updates'] += 1

                    self.last_decision_info = {
                        'vector_angle': math.atan2(vector.y, vector.x),
                        'vector_magnitude': vector.magnitude(),
                        'grid_score': len(self.exploration_grid.frontier_cells) / 100.0,
                        'speed_factor': speed_factor,
                        'red_objects_in_view': perception.red_objects_count,
                        'blue_objects_in_view': perception.blue_objects_count,
                        'black_objects_in_view': perception.black_objects_count,
                        'decision_time': time.time() - decision_start
                    }

                    if self.exploration_target:
                        distance_to_target = math.sqrt(
                            (self.exploration_target[0] - current_pos[0])**2 +
                            (self.exploration_target[1] - current_pos[1])**2
                        )
                        if distance_to_target < self.target_reached_distance:
                            self.exploration_target = None
                            self.change_state(FlightState.HOVERING)
                            self.logger.info("✅ 到达探索目标")

            elif self.state == FlightState.AVOIDING:
                if perception.has_obstacle:
                    current_pos = (pos.x_val, pos.y_val)

                    avoid_vector = self.vector_planner.compute_vector(
                        current_pos,
                        None,
                        perception.obstacle_positions,
                        perception.red_objects,
                        perception.blue_objects,
                        perception.black_objects
                    )

                    if avoid_vector.magnitude() > 0.1:
                        avoid_vector = avoid_vector.normalize() * 1.5
                        target_vx = avoid_vector.x
                        target_vy = avoid_vector.y

                    target_z = pos.z_val - 3
                else:
                    self.change_state(FlightState.EXPLORING)
                    time.sleep(1)

            elif self.state == FlightState.RED_OBJECT_INSPECTION:
                target_vx, target_vy = 0.0, 0.0
                time.sleep(2)
                self.change_state(FlightState.EXPLORING)

            elif self.state == FlightState.BLUE_OBJECT_INSPECTION:
                target_vx, target_vy = 0.0, 0.0
                time.sleep(2)
                self.change_state(FlightState.EXPLORING)

            elif self.state == FlightState.BLACK_OBJECT_INSPECTION:
                target_vx, target_vy = 0.0, 0.0
                time.sleep(2)
                self.change_state(FlightState.EXPLORING)

            elif self.state == FlightState.EMERGENCY:
                target_vx, target_vy, target_yaw = 0, 0, 0
                target_z = max(pos.z_val, -20)

            elif self.state == FlightState.PLANNING:
                target_vx, target_vy = 0, 0
                target_z = perception.recommended_height

            height_error = target_z - pos.z_val
            height_adjustment = self.height_pid.update(height_error)
            target_z += height_adjustment

            target_z = max(self.max_altitude, min(self.min_altitude, target_z))

            decision_time = time.time() - decision_start
            self.last_decision_info['total_decision_time'] = decision_time

            return target_vx, target_vy, target_z, target_yaw

        except Exception as e:
            self.logger.error(f"❌ 决策过程异常: {e}")
            if self.data_logger:
                self.data_logger.record_event('decision_exception', {'error': str(e)})
            return 0.0, 0.0, self.base_height, 0.0

    def _calculate_adaptive_speed(self, perception: PerceptionResult, vector_magnitude: float) -> float:
        if not config.INTELLIGENT_DECISION['ADAPTIVE_SPEED_ENABLED']:
            return 1.0

        open_factor = min(1.0, perception.open_space_score * 1.2)

        if perception.obstacle_distance < self.depth_threshold_near * 2:
            obs_factor = max(0.3, perception.obstacle_distance / (self.depth_threshold_near * 2))
        else:
            obs_factor = 1.0

        vector_factor = min(1.0, vector_magnitude * 2)

        red_factor = 0.8 if perception.red_objects_count > 0 else 1.0
        blue_factor = 0.8 if perception.blue_objects_count > 0 else 1.0
        black_factor = 0.8 if perception.black_objects_count > 0 else 1.0
        color_factor = min(red_factor, blue_factor, black_factor)

        speed_factor = open_factor * obs_factor * vector_factor * color_factor * 0.7

        speed_factor = max(
            config.INTELLIGENT_DECISION['MIN_SPEED_FACTOR'],
            min(config.INTELLIGENT_DECISION['MAX_SPEED_FACTOR'], speed_factor)
        )

        return speed_factor

    def _execute_control_decision(self, decision):
        try:
            target_vx, target_vy, target_z, target_yaw = decision

            if self.state in [FlightState.EXPLORING, FlightState.AVOIDING, FlightState.PLANNING,
                              FlightState.RED_OBJECT_INSPECTION, FlightState.BLUE_OBJECT_INSPECTION,
                              FlightState.BLACK_OBJECT_INSPECTION]:
                self.client.moveByVelocityZAsync(
                    target_vx, target_vy, target_z, 0.5,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=target_yaw),
                    vehicle_name=self.drone_name
                )
            else:
                self.client.moveToPositionAsync(
                    0, 0, target_z, 2,
                    vehicle_name=self.drone_name
                )

            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            self.visited_positions.append((pos.x_val, pos.y_val, pos.z_val))

        except Exception as e:
            self.logger.warning(f"⚠️ 控制指令执行失败: {e}")
            if self.data_logger:
                self.data_logger.record_event('control_exception', {'error': str(e)})
            try:
                self.client.hoverAsync(vehicle_name=self.drone_name).join()
            except:
                pass

    def _report_status(self, exploration_start, perception):
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
            self.logger.info(f"   红色物体: 检测到{perception.red_objects_count}个 "
                            f"| 已访问{self.stats['red_objects_visited']}个")
            self.logger.info(f"   蓝色物体: 检测到{perception.blue_objects_count}个 "
                            f"| 已访问{self.stats['blue_objects_visited']}个")
            self.logger.info(f"   黑色物体: 检测到{perception.black_objects_count}个 "
                            f"| 已访问{self.stats['black_objects_visited']}个")
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
        self.logger.info("=" * 60)
        self.logger.info("探索任务完成，开始返航程序")
        self.logger.info("=" * 60)

        self.change_state(FlightState.RETURNING)

        try:
            self.logger.info("↩️ 返回起始区域...")
            self.client.moveToPositionAsync(0, 0, -10, 4, vehicle_name=self.drone_name).join()
            time.sleep(2)

            self.logger.info("🛬 降落中...")
            self.change_state(FlightState.LANDING)
            self.client.landAsync(vehicle_name=self.drone_name).join()
            time.sleep(3)

        except Exception as e:
            self.logger.error(f"❌ 降落过程中出现异常: {e}")

        finally:
            self._cleanup_system()

            self._generate_summary_report()

    def _cleanup_system(self):
        self.logger.info("🧹 清理系统资源...")

        try:
            self.client.armDisarm(False, vehicle_name=self.drone_name)
            self.client.enableApiControl(False, vehicle_name=self.drone_name)
            self.logger.info("✅ 无人机控制已安全释放")
        except:
            self.logger.warning("⚠️ 释放控制时出现异常")

        if self.front_window:
            self.front_window.stop()
            self.logger.info("✅ 前视窗口已关闭")

        if self.info_window:
            self.info_window.stop()
            self.logger.info("✅ 信息窗口已关闭")

        if self.data_logger:
            self.logger.info("💾 正在保存飞行数据...")
            self.data_logger.save_json_data()

            if config.PERFORMANCE['SAVE_PERFORMANCE_REPORT']:
                performance_report = self.data_logger.generate_performance_report()
                self.logger.info(performance_report)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"performance_report_{timestamp}.txt"
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(performance_report)
                self.logger.info(f"📄 性能报告已保存至: {report_filename}")

    def _generate_summary_report(self):
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
        self.logger.info(f"   红色物体检测: {self.stats['red_objects_detected']}个")
        self.logger.info(f"   红色物体访问: {self.stats['red_objects_visited']}个")
        self.logger.info(f"   蓝色物体检测: {self.stats['blue_objects_detected']}个")
        self.logger.info(f"   蓝色物体访问: {self.stats['blue_objects_visited']}个")
        self.logger.info(f"   黑色物体检测: {self.stats['black_objects_detected']}个")
        self.logger.info(f"   黑色物体访问: {self.stats['black_objects_visited']}个")
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

        try:
            report_filename = f"mission_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("AirSimNH 无人机任务报告 (智能决策增强版 - 双窗口双色物体检测版)\n")
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
                f.write(f"红色物体检测总数: {self.stats['red_objects_detected']}个\n")
                f.write(f"红色物体已访问数: {self.stats['red_objects_visited']}个\n")
                f.write(f"蓝色物体检测总数: {self.stats['blue_objects_detected']}个\n")
                f.write(f"蓝色物体已访问数: {self.stats['blue_objects_visited']}个\n")
                f.write(f"黑色物体检测总数: {self.stats['black_objects_detected']}个\n")
                f.write(f"黑色物体已访问数: {self.stats['black_objects_visited']}个\n")
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
        if self.emergency_flag:
            return

        self.logger.error("\n🆘 紧急停止程序启动!")
        self.emergency_flag = True

        self.change_state(FlightState.EMERGENCY)

        try:
            self.client.hoverAsync(vehicle_name=self.drone_name).join()
            time.sleep(1)
            self.client.landAsync(vehicle_name=self.drone_name).join()
            time.sleep(2)
            self.logger.info("✅ 紧急降落指令已发送")
        except Exception as e:
            self.logger.error(f"⚠️ 紧急降落异常: {e}")

        if self.front_window:
            self.front_window.stop()

        if self.info_window:
            self.info_window.stop()

        self._cleanup_system()


# ==================== 主程序入口 ====================

def main():
    print("=" * 70)
    print("AirSimNH 无人机感知探索系统 - 智能决策增强版（双窗口双色物体检测版）")
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
    print("  • 红色与蓝色物体检测与记录")
    print("=" * 70)
    print("显示系统:")
    print("  • 双窗口模式: 前视窗口 + 信息窗口")
    print("  • 前视窗口: 摄像头画面、手动控制")
    print("  • 信息窗口: 系统状态、探索网格、物体统计")
    print("=" * 70)
    print("数据记录:")
    print(f"  • CSV格式: {config.DATA_RECORDING.get('SAVE_TO_CSV', False)}")
    print(f"  • JSON格式: {config.DATA_RECORDING.get('SAVE_TO_JSON', False)}")
    print(f"  • 性能监控: {config.DATA_RECORDING.get('PERFORMANCE_MONITORING', False)}")
    print(f"  • 红色物体记录: {config.DATA_RECORDING.get('RECORD_RED_OBJECTS', False)}")
    print(f"  • 蓝色物体记录: {config.DATA_RECORDING.get('RECORD_BLUE_OBJECTS', False)}")
    print("=" * 70)

    print("\n请选择运行模式:")
    print("  1. 智能探索模式 (AI自主决策，包含双色物体检测)")
    print("  2. 手动控制模式 (键盘控制)")
    print("  3. 混合模式 (先自动探索，后可切换)")
    print("=" * 50)

    mode_choice = input("请输入选择 (1/2/3): ").strip()

    explorer = None
    try:
        explorer = PerceptiveExplorer(drone_name="")

        def signal_handler(sig, frame):
            print("\n⚠️ 用户中断，正在安全停止...")
            if explorer:
                explorer.emergency_stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        if mode_choice == '1':
            print("\n" + "="*50)
            print("启动智能探索模式（含双色物体检测）")
            print("="*50)
            print("注意：将打开两个窗口:")
            print("  1. 前视窗口 - 显示摄像头画面")
            print("  2. 信息窗口 - 显示系统状态和探索信息")
            print("="*50)
            explorer.run_perception_loop()

        elif mode_choice == '2':
            print("\n" + "="*50)
            print("启动手动控制模式")
            print("="*50)

            print("正在起飞...")
            explorer.client.takeoffAsync(vehicle_name="").join()
            time.sleep(2)
            explorer.client.moveToZAsync(-10, 3, vehicle_name="").join()
            time.sleep(2)
            print("起飞完成，可以开始手动控制")
            print("请切换到无人机前视窗口，使用WSAD键控制")

            explorer.run_manual_control()

        elif mode_choice == '3':
            print("\n" + "="*50)
            print("启动混合模式")
            print("="*50)

            explorer.logger.info("🔍 开始智能探索（含双色物体检测）...")
            original_time = config.EXPLORATION['TOTAL_TIME']
            explorer.exploration_time = min(60, original_time)

            explorer.run_perception_loop()

            if not explorer.emergency_flag:
                print("\n" + "="*50)
                print("智能探索阶段结束")
                print(f"检测到红色物体: {explorer.stats['red_objects_detected']}个")
                print(f"检测到蓝色物体: {explorer.stats['blue_objects_detected']}个")
                print(f"检测到黑色物体: {explorer.stats['black_objects_detected']}个")
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

        try:
            if explorer and explorer.client:
                explorer.client.landAsync().join()
                explorer.client.armDisarm(False)
                explorer.client.enableApiControl(False)
        except:
            pass


if __name__ == "__main__":
    main()