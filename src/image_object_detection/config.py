# config.py
# 功能：定义项目全局配置参数，便于统一管理和修改
# 支持从外部 config.yaml 文件加载配置；若文件不存在，则使用内置默认值。
# 支持环境变量覆盖配置项。

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
      1. 环境变量（格式: YOLO_<SECTION>_<KEY>，如 YOLO_MODEL_PATH）
      2. 外部 config.yaml 文件（位于项目根目录）
      3. 内置默认值（硬编码在本类中）
    """
    
    def __init__(self, config_file="config.yaml"):
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        config_path = base_dir / config_file

        default_config = self._get_default_config(base_dir)

        if config_path.is_file():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                default_config.update(user_config)
            except yaml.YAMLError as e:
                raise ConfigError(f"Invalid YAML syntax in {config_file}: {e}")
            except Exception as e:
                raise ConfigError(f"Failed to read {config_file}: {e}")

        self._apply_env_overrides(default_config)

        self._load_config(default_config)
        self._validate_config()

    def _get_default_config(self, base_dir):
        """获取默认配置"""
        return {
            # 模型配置
            "model": {
                "path": "yolov8n.pt",
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "device": "auto",
                "half_precision": False,
            },
            
            # 输入配置
            "input": {
                "default_image_path": str(base_dir / "data" / "test.jpg"),
                "camera_index": 0,
                "input_dir": str(base_dir / "data" / "input"),
                "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
            },
            
            # 输出配置
            "output": {
                "output_dir": str(base_dir / "data" / "output"),
                "save_results": True,
                "output_format": "jpg",
                "auto_create_dir": True,
            },
            
            # 性能配置
            "performance": {
                "num_threads": 4,
                "batch_size": 1,
                "warmup_runs": 3,
            },
            
            # 显示配置
            "display": {
                "show_confidence": True,
                "show_distance": True,
                "show_danger_level": True,
                "show_fps": False,
                "window_size": (1280, 720),
            },
            
            # 危险等级配置
            "danger": {
                "danger_threshold": 10.0,
                "warning_threshold": 20.0,
                "known_height": 1.6,
                "focal_length": 700,
            },
            
            # 日志配置
            "logging": {
                "log_level": "INFO",
                "log_file": None,
                "log_to_console": True,
            },
        }

    def _apply_env_overrides(self, config):
        """应用环境变量覆盖配置"""
        env_prefix = "YOLO_"
        
        def _apply_section(section_name, section_config):
            if isinstance(section_config, dict):
                for key, value in section_config.items():
                    env_key = f"{env_prefix}{section_name.upper()}_{key.upper()}"
                    env_value = os.environ.get(env_key)
                    if env_value is not None:
                        section_config[key] = self._parse_env_value(env_value, type(value))
        
        for section, config_dict in config.items():
            _apply_section(section, config_dict)

    def _parse_env_value(self, value, target_type):
        """解析环境变量值并转换为目标类型"""
        if target_type == bool:
            return value.lower() in ("true", "1", "yes")
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == list:
            return value.split(',')
        else:
            return value

    def _load_config(self, config):
        """加载配置到实例属性"""
        # 模型配置
        self.model_path = config["model"]["path"]
        self.confidence_threshold = float(config["model"]["confidence_threshold"])
        self.iou_threshold = float(config["model"]["iou_threshold"])
        self.device = config["model"]["device"]
        self.half_precision = bool(config["model"]["half_precision"])
        
        # 输入配置
        self.default_image_path = config["input"]["default_image_path"]
        self.camera_index = int(config["input"]["camera_index"])
        self.input_dir = config["input"]["input_dir"]
        self.supported_formats = config["input"]["supported_formats"]
        
        # 输出配置
        self.output_dir = config["output"]["output_dir"]
        self.save_results = bool(config["output"]["save_results"])
        self.output_format = config["output"]["output_format"]
        self.auto_create_dir = bool(config["output"]["auto_create_dir"])
        
        # 性能配置
        self.num_threads = int(config["performance"]["num_threads"])
        self.batch_size = int(config["performance"]["batch_size"])
        self.warmup_runs = int(config["performance"]["warmup_runs"])
        
        # 显示配置
        self.show_confidence = bool(config["display"]["show_confidence"])
        self.show_distance = bool(config["display"]["show_distance"])
        self.show_danger_level = bool(config["display"]["show_danger_level"])
        self.show_fps = bool(config["display"]["show_fps"])
        self.window_size = tuple(config["display"]["window_size"])
        
        # 危险等级配置
        self.danger_threshold = float(config["danger"]["danger_threshold"])
        self.warning_threshold = float(config["danger"]["warning_threshold"])
        self.known_height = float(config["danger"]["known_height"])
        self.focal_length = float(config["danger"]["focal_length"])
        
        # 日志配置
        self.log_level = config["logging"]["log_level"]
        self.log_file = config["logging"]["log_file"]
        self.log_to_console = bool(config["logging"]["log_to_console"])

        # 路径标准化
        if not os.path.isabs(self.default_image_path):
            base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.default_image_path = str(base_dir / self.default_image_path)
        if not os.path.isabs(self.input_dir):
            base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.input_dir = str(base_dir / self.input_dir)
        if not os.path.isabs(self.output_dir):
            base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.output_dir = str(base_dir / self.output_dir)

    def _validate_config(self):
        """验证配置参数的有效性"""
        # 验证置信度阈值
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigError("confidence_threshold must be in range [0.0, 1.0]")
        
        # 验证IOU阈值
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ConfigError("iou_threshold must be in range [0.0, 1.0]")
        
        # 验证摄像头索引
        if self.camera_index < 0:
            raise ConfigError("camera_index must be a non-negative integer")
        
        # 验证设备类型
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.device.lower() not in valid_devices:
            raise ConfigError(f"device must be one of: {valid_devices}")
        
        # 验证日志级别
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigError(f"log_level must be one of: {valid_log_levels}")
        
        # 验证危险阈值
        if self.danger_threshold >= self.warning_threshold:
            raise ConfigError("danger_threshold must be less than warning_threshold")

    def get_model_config(self):
        """获取模型相关配置字典"""
        return {
            "path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "half_precision": self.half_precision,
        }

    def get_input_config(self):
        """获取输入相关配置字典"""
        return {
            "default_image_path": self.default_image_path,
            "camera_index": self.camera_index,
            "input_dir": self.input_dir,
            "supported_formats": self.supported_formats,
        }

    def get_output_config(self):
        """获取输出相关配置字典"""
        return {
            "output_dir": self.output_dir,
            "save_results": self.save_results,
            "output_format": self.output_format,
            "auto_create_dir": self.auto_create_dir,
        }

    def get_display_config(self):
        """获取显示相关配置字典"""
        return {
            "show_confidence": self.show_confidence,
            "show_distance": self.show_distance,
            "show_danger_level": self.show_danger_level,
            "show_fps": self.show_fps,
            "window_size": self.window_size,
        }

    def get_danger_config(self):
        """获取危险等级相关配置字典"""
        return {
            "danger_threshold": self.danger_threshold,
            "warning_threshold": self.warning_threshold,
            "known_height": self.known_height,
            "focal_length": self.focal_length,
        }

    def get_logging_config(self):
        """获取日志相关配置字典"""
        return {
            "log_level": self.log_level,
            "log_file": self.log_file,
            "log_to_console": self.log_to_console,
        }

    def __repr__(self):
        """返回配置的字符串表示"""
        return f"Config(model_path={self.model_path!r}, conf_threshold={self.confidence_threshold})"

    def print_summary(self):
        """打印配置摘要"""
        print("=" * 60)
        print("📋 Configuration Summary")
        print("=" * 60)
        
        print("\n[Model]")
        print(f"  Path: {self.model_path}")
        print(f"  Confidence Threshold: {self.confidence_threshold}")
        print(f"  IOU Threshold: {self.iou_threshold}")
        print(f"  Device: {self.device}")
        
        print("\n[Input]")
        print(f"  Default Image: {self.default_image_path}")
        print(f"  Camera Index: {self.camera_index}")
        print(f"  Input Directory: {self.input_dir}")
        
        print("\n[Output]")
        print(f"  Output Directory: {self.output_dir}")
        print(f"  Save Results: {self.save_results}")
        
        print("\n[Danger Levels]")
        print(f"  Danger Threshold: {self.danger_threshold}m")
        print(f"  Warning Threshold: {self.warning_threshold}m")
        
        print("\n" + "=" * 60)
