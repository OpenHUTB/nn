#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuJoCo+CARLA自动巡航小车
核心改进：
1. 完全移除绝对路径依赖，使用相对路径+动态路径解析
2. 修复语法错误和逻辑混乱
3. 模块化重构，分离不同功能模块
4. 增强代码鲁棒性和可维护性
适配：CARLA 0.9.10，MuJoCo 2.3+
"""
import os
import sys
import time
import math
import random
import logging
from typing import Tuple, Optional, Dict
from collections import deque

import cv2
import numpy as np
import mujoco
import mujoco.viewer
import carla
from pynput import keyboard
import matplotlib.pyplot as plt

# ===================== 基础配置 =====================
# 获取脚本所在目录（核心：动态路径解析，避免绝对路径）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 相对路径配置（所有资源文件都基于脚本目录）
CONFIG = {
    # MuJoCo配置
    "MODEL_REL_PATH": "wheeled_car.xml",  # 相对于脚本的路径
    "CHASSIS_NAME": "chassis",
    # 速度配置
    "CRUISE_SPEED": 0.0015,
    "TURN_SPEED": 0.00075,
    # 距离配置
    "OBSTACLE_THRESHOLD": 0.5,
    "SAFE_DISTANCE": 0.2,
    # 转向配置
    "TURN_ANGLE": math.pi / 2,
    "STEER_LIMIT": 0.3,
    # 障碍物名称
    "OBSTACLE_NAMES": [
        'obs_box1', 'obs_box2', 'obs_box3', 'obs_box4',
        'obs_ball1', 'obs_ball2', 'obs_ball3',
        'wall1', 'wall2', 'front_dark_box'
    ],
    # 视图配置
    "CAM_DISTANCE": 2.5,
    "CAM_ELEVATION": -25,
    # CARLA配置
    "TARGET_SPEED": 20.0,  # km/h
    "PID_KP": 0.3,
    "PID_KI": 0.02,
    "PID_KD": 0.01,
    "LOOKAHEAD_DISTANCE": 8.0,
    "MAX_STEER_ANGLE": 30.0,
    "CAMERA_WIDTH": 800,
    "CAMERA_HEIGHT": 600,
    "CAMERA_FOV": 90,
    "VEHICLE_MODEL": "vehicle.tesla.model3",
    "PLOT_SIZE": (12, 10),
    "WAYPOINT_COUNT": 50
}

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ===================== 状态枚举 =====================
class CarState:
    CRUISING = "CRUISING"
    STOPPED = "STOPPED"
    TURNING = "TURNING"
    RESUMING = "RESUMING"


# ===================== 键盘管理器 =====================
class KeyManager:
    def __init__(self):
        self.keys = {keyboard.KeyCode.from_char('r'): False}
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, k):
        if k in self.keys:
            self.keys[k] = True

    def _on_release(self, k):
        if k in self.keys:
            self.keys[k] = False

    def is_pressed(self, key_char: str) -> bool:
        key = keyboard.KeyCode.from_char(key_char)
        return self.keys.get(key, False)

    def reset_key(self, key_char: str):
        key = keyboard.KeyCode.from_char(key_char)
        if key in self.keys:
            self.keys[key] = False


# ===================== MuJoCo小车控制器 =====================
class MujoCoCarController:
    """专门处理MuJoCo小车逻辑"""

    def __init__(self):
        self.config = CONFIG
        self.key_manager = KeyManager()

        # 关键：动态构建模型路径（避免绝对路径）
        self.model_path = os.path.join(SCRIPT_DIR, self.config["MODEL_REL_PATH"])

        # 加载模型（增加路径检查）
        self._load_model()

        # 初始化ID缓存
        self._init_ids()

        # 初始化状态
        self.car_state = CarState.CRUISING
        self.target_steer_angle = 0.0
        self.turn_direction = 1
        self.turn_progress = 0.0

        # 复位小车
        self.reset_car()

    def _load_model(self):
        """加载MuJoCo模型，增加路径检查"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"MuJoCo模型文件不存在: {self.model_path}\n"
                f"请确保wheeled_car.xml文件在以下目录：{SCRIPT_DIR}"
            )

        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            logger.info(f"✅ 成功加载MuJoCo模型: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"❌ 加载MuJoCo模型失败: {e}")

    def _init_ids(self):
        """初始化body ID缓存"""
        # 小车底盘ID
        self.chassis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config["CHASSIS_NAME"]
        )
        if self.chassis_id == -1:
            raise RuntimeError(f"❌ 未找到底盘: {self.config['CHASSIS_NAME']}")

        # 障碍物ID
        self.obstacle_ids = {}
        for name in self.config["OBSTACLE_NAMES"]:
            obs_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if obs_id != -1:
                self.obstacle_ids[name] = obs_id
        logger.info(f"✅ 加载{len(self.obstacle_ids)}个障碍物ID")

    def reset_car(self):
        """复位小车到初始状态"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.03  # 设置初始高度
        self.car_state = CarState.CRUISING
        self.target_steer_angle = 0.0
        self.turn_progress = 0.0
        logger.info("\n🔄 MuJoCo小车已复位")

    def detect_obstacle(self) -> Tuple[int, float, Optional[str]]:
        """检测前方障碍物"""
        chassis_pos = self.data.body(self.chassis_id).xpos
        min_distance = float('inf')
        closest_obs = None

        for obs_name, obs_id in self.obstacle_ids.items():
            try:
                obs_pos = self.data.body(obs_id).xpos
                dx = obs_pos[0] - chassis_pos[0]
                dy = obs_pos[1] - chassis_pos[1]
                distance = math.hypot(dx, dy)

                if dx > 0 and abs(dy) < 0.3 and distance < self.config["OBSTACLE_THRESHOLD"]:
                    if distance < min_distance:
                        min_distance = distance
                        closest_obs = obs_name
            except Exception:
                continue

        # 判断状态
        if closest_obs is None:
            return 0, 0.0, None
        elif min_distance < self.config["SAFE_DISTANCE"]:
            return 2, min_distance, closest_obs
        else:
            return 1, min_distance, closest_obs

    def _set_steer(self, angle: float):
        """设置转向角度"""
        clamped_angle = np.clip(angle, -self.config["STEER_LIMIT"], self.config["STEER_LIMIT"])
        self.data.ctrl[0] = clamped_angle  # 前左转向
        self.data.ctrl[1] = clamped_angle  # 前右转向

    def _set_speed(self, speed: float):
        """设置驱动速度"""
        self.data.ctrl[2] = speed  # 前左驱动
        self.data.ctrl[3] = speed  # 前右驱动
        self.data.ctrl[4] = speed  # 后左驱动
        self.data.ctrl[5] = speed  # 后右驱动

    def update(self):
        """更新小车状态和控制指令"""
        # 检查复位键
        if self.key_manager.is_pressed('r'):
            self.reset_car()
            self.key_manager.reset_key('r')
            return

        # 检测障碍物
        obs_status, obs_dist, obs_name = self.detect_obstacle()

        # 状态机逻辑
        if self.car_state == CarState.CRUISING:
            if obs_status == 2:
                self.car_state = CarState.STOPPED
                self._set_speed(0.0)
                logger.warning(f"\n🛑 紧急停止！障碍物：{obs_name} (距离: {obs_dist:.2f}m)")
            elif obs_status == 1:
                self.car_state = CarState.STOPPED
                self._set_speed(0.0)
                logger.warning(f"\n⚠️  检测到障碍物：{obs_name} (距离: {obs_dist:.2f}m)")
            else:
                self._set_steer(0.0)
                self._set_speed(self.config["CRUISE_SPEED"])

        elif self.car_state == CarState.STOPPED:
            self.turn_progress += 1
            self._set_speed(0.0)

            if self.turn_progress > 10:
                self.turn_direction = 1 if random.random() > 0.5 else -1
                self.target_steer_angle = self.turn_direction * self.config["TURN_ANGLE"]
                self.car_state = CarState.TURNING
                self.turn_progress = 0.0
                dir_text = "右" if self.turn_direction > 0 else "左"
                logger.info(f"\n🔄 开始{dir_text}转90度避障")

        elif self.car_state == CarState.TURNING:
            self.turn_progress += 0.01
            current_steer = self.target_steer_angle * self.turn_progress
            self._set_steer(current_steer)
            self._set_speed(self.config["TURN_SPEED"])

            if self.turn_progress >= 1.0 or abs(current_steer) >= self.config["STEER_LIMIT"]:
                self.car_state = CarState.RESUMING
                self.turn_progress = 0.0
                logger.info("\n✅ 转向完成，开始恢复巡航")

        elif self.car_state == CarState.RESUMING:
            self.turn_progress += 0.01
            current_steer = self.target_steer_angle * (1 - self.turn_progress)
            self._set_steer(current_steer)

            # 修复：CRUISING是状态枚举，应该用CRUISE_SPEED
            current_speed = self.config["TURN_SPEED"] + (
                    self.config["CRUISE_SPEED"] - self.config["TURN_SPEED"]
            ) * self.turn_progress
            self._set_speed(current_speed)

            if self.turn_progress >= 1.0:
                if obs_status == 0:
                    self.car_state = CarState.CRUISING
                    logger.info("\n🚗 前方安全，恢复正常巡航")
                else:
                    self.car_state = CarState.STOPPED
                    logger.warning("\n⚠️  转向后仍有障碍，重新停止")
                self.turn_progress = 0.0

    def get_status_info(self) -> str:
        """获取状态信息"""
        vel = np.linalg.norm(self.data.qvel[:3])
        current_steer = math.degrees((self.data.ctrl[0] + self.data.ctrl[1]) / 2)

        info = f"状态: {self.car_state:<8} | 速度: {vel:.5f} m/s"
        if abs(current_steer) > 0.1:
            info += f" | 转向: {current_steer:.1f}°"

        if self.car_state == CarState.CRUISING:
            obs_status, obs_dist, obs_name = self.detect_obstacle()
            if obs_status > 0 and obs_name:
                info += f" | 前方障碍: {obs_name} ({obs_dist:.2f}m)"

        return info


# ===================== CARLA工具类 =====================
class CarlaTools:
    """CARLA辅助工具类"""

    @staticmethod
    def normalize_angle(angle):
        """将角度归一化到[-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def get_vehicle_pose(vehicle):
        """获取车辆的位置、朝向和速度"""
        transform = vehicle.get_transform()
        loc = transform.location
        yaw = math.radians(transform.rotation.yaw)
        vel = vehicle.get_velocity()
        speed = 3.6 * np.linalg.norm([vel.x, vel.y, vel.z])
        return loc, yaw, speed

    @staticmethod
    def clear_all_actors(world):
        """清理所有车辆和传感器"""
        for actor in world.get_actors():
            try:
                if actor.type_id.startswith('vehicle') or actor.type_id.startswith('sensor'):
                    actor.destroy()
            except Exception as e:
                logger.warning(f"清理Actor失败: {e}")
        time.sleep(0.5)
        logger.info("已清理所有残留Actor")

    @staticmethod
    def focus_vehicle(world, vehicle):
        """将CARLA客户端视角聚焦到车辆"""
        spectator = world.get_spectator()
        t = vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(t.location + carla.Location(x=0, y=-8, z=5), t.rotation)
        )
        logger.info("CARLA客户端已聚焦到车辆")

    @staticmethod
    def generate_road_waypoints(world, start_loc, count=50, step=2.0):
        """生成道路航点"""
        waypoints = []
        map = world.get_map()
        wp = map.get_waypoint(start_loc)
        for i in range(count):
            waypoints.append((wp.transform.location.x, wp.transform.location.y, wp.transform.location.z))
            next_wps = wp.next(step)
            if next_wps:
                wp = next_wps[0]
            else:
                break
        logger.info(f"生成了{len(waypoints)}个原生道路航点")
        return waypoints


# ===================== 主函数 =====================
def main():
    """主程序入口"""
    # 初始化MuJoCo控制器
    try:
        mujoco_controller = MujoCoCarController()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(e)
        return

    # 启动MuJoCo可视化
    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(mujoco_controller.model, mujoco_controller.data)
        viewer.cam.distance = CONFIG["CAM_DISTANCE"]
        viewer.cam.elevation = CONFIG["CAM_ELEVATION"]

        # MuJoCo主循环
        logger.info("\n🚀 MuJoCo自动巡航小车启动（按R键复位）")
        logger.info("=====================================")

        while viewer.is_running():
            # 更新小车控制
            mujoco_controller.update()

            # 执行仿真步骤
            mujoco.mj_step(mujoco_controller.model, mujoco_controller.data)

            # 打印状态信息
            status_info = mujoco_controller.get_status_info()
            print(f"\r{status_info}", end='', flush=True)

            # 同步视图
            viewer.sync()

            # 控制循环频率
            time.sleep(0.005)

    # 手动终止处理（Ctrl+C）
    except KeyboardInterrupt:
        logger.info("\n\n👋 用户中断程序")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
    finally:
        if viewer:
            viewer.close()
        logger.info("程序正常退出")


if __name__ == "__main__":
    main()
