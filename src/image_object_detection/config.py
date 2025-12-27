# config.py
# 功能：定义项目全局配置参数，便于统一管理和修改
# 支持从外部 config.yaml 文件加载配置；若文件不存在，则使用内置默认值。

import os
import yaml
from pathlib import Path

class ConfigError(Exception):
    """配置相关异常的基类"""
    pass

class Config:
    """
    应用程序的配置类。
    所有核心参数（如模型路径、默认图像路径、阈值等）集中在此初始化，
    便于维护和部署时调整。
    
    配置优先级（从高到低）：
      1. 外部 config.yaml 文件（位于项目根目录）
      2. 内置默认值（硬编码在本类中）
    """
    def __init__(self, config_file="config.yaml"):
        # 获取当前 config.py 文件所在目录的绝对路径
        # 使用 os.path.abspath 确保路径在不同操作系统下一致
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # 构建配置文件的完整路径（相对于项目根目录）
        config_path = base_dir / config_file

        # 内置默认配置（与原逻辑一致）
        default_config = {
            "default_image_path": str(base_dir / "data" / "test.jpg"),
            "model_path": "yolov8n.pt",
            "confidence_threshold": 0.25,
            "camera_index": 0,
            "output_interval": 1.0,
        }

        # 尝试从外部 YAML 文件加载用户配置
        if config_path.is_file():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                # 合并：用户配置覆盖默认值
                default_config.update(user_config)
            except yaml.YAMLError as e:
                raise ConfigError(f"Invalid YAML syntax in {config_file}: {e}")
            except Exception as e:
                raise ConfigError(f"Failed to read {config_file}: {e}")

        # 提取配置项（保持与原字段名一致）
        self.default_image_path = default_config["default_image_path"]
        self.model_path = default_config["model_path"]
        self.confidence_threshold = float(default_config["confidence_threshold"])
        self.camera_index = int(default_config["camera_index"])
        self.output_interval = float(default_config["output_interval"])

        # 路径标准化：若 default_image_path 是相对路径，则基于项目根目录解析
        if not os.path.isabs(self.default_image_path):
            self.default_image_path = str(base_dir / self.default_image_path)

        # 验证置信度阈值
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigError("confidence_threshold must be in range [0.0, 1.0]")

        # 验证摄像头索引
        if self.camera_index < 0:
            raise ConfigError("camera_index must be a non-negative integer")

        # 验证输出间隔
        if self.output_interval <= 0:
            raise ConfigError("output_interval must be positive")
