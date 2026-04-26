# -*- coding: utf-8 -*-
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 获取当前脚本所在目录的父目录（drone_hand_gesture 目录）
BASE_DIR = Path(__file__).resolve().parent.parent


class Logger:
    _instance: Optional['Logger'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "drone_system", log_dir: Optional[str] = None):
        if self._initialized:
            return

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # 如果没有指定日志目录，使用相对于项目根目录的默认路径
        if log_dir is None:
            log_path = BASE_DIR / "logs"
        else:
            log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"drone_{timestamp}.log"

        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self._initialized = True

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

    def get_logger(self):
        return self.logger
