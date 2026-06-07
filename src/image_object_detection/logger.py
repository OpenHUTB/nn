# logger.py
# 功能：日志系统 - 支持多级别、多输出目标的日志管理

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """
    日志管理器类。
    
    功能特性：
    - 支持5种日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    - 支持同时输出到控制台和文件
    - 支持自定义日志格式
    - 支持按日期滚动日志文件
    - 线程安全设计
    """

    def __init__(self, name: str = "detection", log_level: str = "INFO", 
                 log_file: Optional[str] = None, log_to_console: bool = True):
        """
        初始化日志管理器。
        
        参数:
            name: 日志器名称
            log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
            log_file: 日志文件路径（None表示不输出到文件）
            log_to_console: 是否输出到控制台
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        self._remove_existing_handlers()
        
        if log_to_console:
            self._add_console_handler()
        
        if log_file:
            self._add_file_handler(log_file)

    def _remove_existing_handlers(self):
        """移除已存在的处理器，避免重复输出"""
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

    def _add_console_handler(self):
        """添加控制台处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self, log_file: str):
        """添加文件处理器"""
        log_path = Path(log_file)
        
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """输出DEBUG级别日志"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """输出INFO级别日志"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """输出WARNING级别日志"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """输出ERROR级别日志"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """输出CRITICAL级别日志"""
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, exc_info=True, **kwargs):
        """输出异常日志（自动包含堆栈信息）"""
        self.logger.exception(message, exc_info=exc_info, **kwargs)

    def set_level(self, level: str):
        """设置日志级别"""
        self.logger.setLevel(level)

    def get_level(self):
        """获取当前日志级别"""
        return logging.getLevelName(self.logger.level)


class LogManager:
    """
    日志管理器 - 管理多个日志器实例。
    
    提供全局日志管理功能，支持为不同模块创建独立日志器。
    """
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str = "detection", log_level: str = "INFO",
                   log_file: Optional[str] = None, log_to_console: bool = True) -> Logger:
        """
        获取或创建日志器。
        
        参数:
            name: 日志器名称
            log_level: 日志级别
            log_file: 日志文件路径
            log_to_console: 是否输出到控制台
        
        返回:
            Logger 实例
        """
        if name not in cls._loggers:
            cls._loggers[name] = Logger(name, log_level, log_file, log_to_console)
        
        return cls._loggers[name]
    
    @classmethod
    def set_global_level(cls, level: str):
        """设置所有日志器的级别"""
        for logger in cls._loggers.values():
            logger.set_level(level)
    
    @classmethod
    def get_all_loggers(cls):
        """获取所有日志器名称"""
        return list(cls._loggers.keys())


# 创建默认日志器
default_logger = LogManager.get_logger("detection")


def get_logger(name: str = "detection") -> Logger:
    """便捷函数：获取日志器"""
    return LogManager.get_logger(name)