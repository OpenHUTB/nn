# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    DEFAULT_CONFIG: Dict[str, Any] = {
        "drone": {
            "max_speed": 2.0,
            "max_altitude": 10.0,
            "takeoff_altitude": 2.0,
            "hover_threshold": 0.1,
            "battery_drain_rate": 0.01,
            "mass": 1.0,
            "gravity": 9.81
        },
        "camera": {
            "default_id": 1,
            "width": 640,
            "height": 480,
            "fps": 30
        },
        "gesture": {
            "threshold": 0.6,
            "command_cooldown": 1.5,
            "use_ml": True
        },
        "simulation": {
            "window_width": 1024,
            "window_height": 768,
            "target_fps": 60,
            "max_trajectory_points": 500
        },
        "airsim": {
            "ip_address": "127.0.0.1",
            "port": 41451,
            "vehicle_name": ""
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs"
        }
    }

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = self.DEFAULT_CONFIG.copy()
        self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self._merge_config(self.config, loaded_config)
                print(f"[Config] 已从 {self.config_path} 加载配置")
            except Exception as e:
                print(f"[Config] 加载配置失败，使用默认配置: {e}")
        else:
            print(f"[Config] 配置文件不存在，使用默认配置")
            self.save_config()

    def _merge_config(self, base: Dict, update: Dict):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def save_config(self):
        try:
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"[Config] 配置已保存到 {self.config_path}")
        except Exception as e:
            print(f"[Config] 保存配置失败: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any):
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        return self.config.copy()

    def reset(self):
        self.config = self.DEFAULT_CONFIG.copy()
        self.save_config()
