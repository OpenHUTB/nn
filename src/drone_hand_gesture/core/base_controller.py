# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from .logger import Logger
from .config import ConfigManager


class BaseDroneController(ABC):
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.logger = Logger()

        self.connected: bool = False
        self.state: Dict[str, Any] = {
            'position': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0]),
            'battery': 100.0,
            'armed': False,
            'mode': 'DISARMED'
        }

        self.trajectory: List[tuple] = []
        self.max_trajectory_points: int = self.config.get(
            "simulation.max_trajectory_points", 500
        )

        self._gesture_commands: Dict[str, str] = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "pointing_up": "up",
            "pointing_down": "down",
            "victory": "forward",
            "thumb_up": "backward",
            "thumb_down": "stop",
            "ok_sign": "hover",
            "left": "left",
            "right": "right"
        }

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def takeoff(self, altitude: Optional[float] = None) -> bool:
        pass

    @abstractmethod
    def land(self) -> bool:
        pass

    @abstractmethod
    def hover(self):
        pass

    @abstractmethod
    def move_by_velocity(self, vx: float, vy: float, vz: float, duration: float = 0.5):
        pass

    def send_command(self, command: str, intensity: float = 1.0):
        self.logger.info(f"收到命令: {command}, 强度: {intensity}")

        if command == "takeoff":
            self.takeoff()
        elif command == "land":
            self.land()
        elif command == "hover":
            self.hover()
        elif command == "forward":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(speed, 0, 0)
        elif command == "backward":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(-speed, 0, 0)
        elif command == "left":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(0, -speed, 0)
        elif command == "right":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(0, speed, 0)
        elif command == "up":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(0, 0, -speed)
        elif command == "down":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(0, 0, speed)
        elif command == "stop":
            self.move_by_velocity(0, 0, 0)
            self.state['armed'] = False
            self.state['mode'] = 'DISARMED'

    def get_state(self) -> Dict[str, Any]:
        return {
            'position': self.state['position'].copy(),
            'velocity': self.state['velocity'].copy(),
            'orientation': self.state['orientation'].copy(),
            'battery': self.state['battery'],
            'armed': self.state['armed'],
            'mode': self.state['mode']
        }

    def get_trajectory(self) -> List[tuple]:
        return self.trajectory.copy()

    def get_status_string(self) -> str:
        pos = self.state['position']
        return (f"模式: {self.state['mode']} | "
                f"位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
                f"电池: {self.state['battery']:.1f}% | "
                f"解锁: {'是' if self.state['armed'] else '否'}")

    def _record_trajectory(self):
        self.trajectory.append(tuple(self.state['position']))
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)

    def reset(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None):
        if position is not None:
            self.state['position'] = np.array(position)
        else:
            self.state['position'] = np.array([0.0, 0.0, 0.0])

        if orientation is not None:
            self.state['orientation'] = np.array(orientation)
        else:
            self.state['orientation'] = np.array([0.0, 0.0, 0.0])

        self.state['velocity'] = np.array([0.0, 0.0, 0.0])
        self.state['battery'] = 100.0
        self.state['armed'] = False
        self.state['mode'] = 'DISARMED'
        self.trajectory.clear()
        self.logger.info("无人机状态已重置")

    @property
    def gesture_commands(self) -> Dict[str, str]:
        return self._gesture_commands
