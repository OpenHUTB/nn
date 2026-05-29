"""
配置管理模块

支持从命令行参数、YAML配置文件和环境变量加载配置
"""

import argparse
import json
import os
from typing import Dict, Any, Optional

import yaml


class Config:
    """配置类，支持多种配置来源"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
    
    def __getattr__(self, key: str) -> Any:
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"配置项 '{key}' 不存在")
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config
    
    def save(self, filepath: str) -> None:
        """保存配置到文件"""
        _, ext = os.path.splitext(filepath)
        if ext.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(self._config, f, indent=2)
        else:
            with open(filepath, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """从文件加载配置"""
        _, ext = os.path.splitext(filepath)
        if ext.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_cli(cls, parser: argparse.ArgumentParser) -> 'Config':
        """从命令行参数创建配置"""
        args = parser.parse_args()
        return cls(vars(args))


class FrozenLakeConfig(Config):
    """FrozenLake环境配置"""
    
    DEFAULT_CONFIG = {
        'epochs': 10000,
        'map_size': 4,
        'alpha': 0.8,
        'gamma': 0.9,
        'radr': 0.001,
        'log_interval': 1000,
        'test': False,
        'render': False,
        'save_path': 'frozen_lake_q_table.npy',
        'algorithm': 'q_learning'
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        merged = self.DEFAULT_CONFIG.copy()
        if config_dict:
            merged.update(config_dict)
        super().__init__(merged)


class CartPoleConfig(Config):
    """CartPole环境配置"""
    
    DEFAULT_CONFIG = {
        'epochs': 50000,
        'num_bins': 50,
        'alpha': 0.1,
        'gamma': 0.995,
        'log_interval': 1000,
        'training': False,
        'testing': False,
        'render': False,
        'save_path': 'cartpole_q_table.npy',
        'algorithm': 'q_learning'
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        merged = self.DEFAULT_CONFIG.copy()
        if config_dict:
            merged.update(config_dict)
        super().__init__(merged)


def get_config_parser(env_name: str) -> argparse.ArgumentParser:
    """获取特定环境的命令行参数解析器"""
    parser = argparse.ArgumentParser(description=f'{env_name}训练配置')
    
    if env_name.lower() == 'frozenlake':
        parser.add_argument('--epochs', type=int, default=10000, help='训练轮数')
        parser.add_argument('--map-size', type=int, default=4, help='地图大小')
        parser.add_argument('--alpha', type=float, default=0.8, help='学习率')
        parser.add_argument('--gamma', type=float, default=0.9, help='折扣因子')
        parser.add_argument('--radr', type=float, default=0.001, help='探索率衰减率')
        parser.add_argument('--log-interval', type=int, default=1000, help='日志输出间隔')
        parser.add_argument('--test', action='store_true', help='测试已训练的模型')
        parser.add_argument('--render', action='store_true', help='渲染测试过程')
        parser.add_argument('--save-path', type=str, default='frozen_lake_q_table.npy', help='Q表保存路径')
        parser.add_argument('--algorithm', type=str, default='q_learning', choices=['q_learning', 'sarsa'], help='算法选择')
    
    elif env_name.lower() == 'cartpole':
        parser.add_argument('--epochs', type=int, default=50000, help='训练轮数')
        parser.add_argument('--num-bins', type=int, default=50, help='每个状态维度的区间数')
        parser.add_argument('--alpha', type=float, default=0.1, help='学习率')
        parser.add_argument('--gamma', type=float, default=0.995, help='折扣因子')
        parser.add_argument('--log-interval', type=int, default=1000, help='日志输出间隔')
        parser.add_argument('--training', action='store_true', help='进行训练')
        parser.add_argument('--testing', action='store_true', help='进行测试')
        parser.add_argument('--render', action='store_true', help='渲染环境')
        parser.add_argument('--save-path', type=str, default='cartpole_q_table.npy', help='Q表保存路径')
        parser.add_argument('--algorithm', type=str, default='q_learning', choices=['q_learning', 'sarsa'], help='算法选择')
    
    return parser
