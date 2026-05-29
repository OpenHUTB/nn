"""
强化学习训练工具模块

提供统一的训练框架、日志记录和性能评估功能
"""

import json
import os
import time
from typing import Dict, List, Optional, Union

import numpy as np


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.start_time = time.time()
        self.records = []
    
    def log(self, epoch: int, metrics: Dict[str, float]) -> None:
        """记录训练指标"""
        record = {
            "epoch": epoch,
            "timestamp": time.time() - self.start_time,
            **metrics
        }
        self.records.append(record)
    
    def save(self, filename: str = "training_log.json") -> None:
        """保存日志到文件"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.records, f, indent=2)
        print(f"训练日志已保存到: {filepath}")
    
    def print_summary(self) -> None:
        """打印训练摘要"""
        if not self.records:
            print("没有训练记录")
            return
        
        final = self.records[-1]
        print(f"\n{'='*50}")
        print(f"训练摘要")
        print(f"{'='*50}")
        print(f"总训练时间: {final['timestamp']:.2f} 秒")
        print(f"最终指标:")
        for key, value in final.items():
            if key not in ["epoch", "timestamp"]:
                print(f"  {key}: {value:.4f}")


class PerformanceEvaluator:
    """性能评估器"""
    
    @staticmethod
    def calculate_success_rate(rewards: List[float], threshold: float = 0.0) -> float:
        """计算成功率"""
        successes = sum(1 for r in rewards if r > threshold)
        return successes / len(rewards) * 100
    
    @staticmethod
    def calculate_mean_reward(rewards: List[float]) -> float:
        """计算平均奖励"""
        return np.mean(rewards)
    
    @staticmethod
    def calculate_std_reward(rewards: List[float]) -> float:
        """计算奖励标准差"""
        return np.std(rewards)
    
    @staticmethod
    def evaluate(env, agent, episodes: int = 100, render: bool = False) -> Dict[str, float]:
        """评估代理性能"""
        rewards = []
        for _ in range(episodes):
            state = env.reset()[0]
            done = False
            total_reward = 0
            
            while not done:
                if render:
                    env.render()
                
                action = agent.get_action(state)
                state, reward, done, trunc, info = env.step(action)
                total_reward += reward
            
            rewards.append(total_reward)
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(PerformanceEvaluator.calculate_success_rate(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards))
        }


class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, param_grid: Dict[str, List]):
        """
        参数:
            param_grid: 超参数网格，如 {'alpha': [0.1, 0.5, 0.8], 'gamma': [0.9, 0.99]}
        """
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = float('-inf')
    
    def generate_combinations(self) -> List[Dict]:
        """生成所有参数组合"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = [{}]
        for key, vals in zip(keys, values):
            temp = []
            for combo in combinations:
                for val in vals:
                    new_combo = combo.copy()
                    new_combo[key] = val
                    temp.append(new_combo)
            combinations = temp
        
        return combinations
    
    def tune(self, train_func, eval_func, verbose: bool = True) -> Dict:
        """
        执行超参数搜索
        
        参数:
            train_func: 训练函数，接受params返回模型
            eval_func: 评估函数，接受模型返回分数
            verbose: 是否输出详细信息
        """
        combinations = self.generate_combinations()
        
        for params in combinations:
            if verbose:
                print(f"\n测试参数: {params}")
            
            model = train_func(params)
            score = eval_func(model)
            
            if verbose:
                print(f"得分: {score:.4f}")
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                if verbose:
                    print(f"新的最佳参数! 得分: {self.best_score:.4f}")
        
        print(f"\n{'='*50}")
        print(f"超参数搜索完成")
        print(f"最佳参数: {self.best_params}")
        print(f"最佳得分: {self.best_score:.4f}")
        
        return self.best_params


def load_q_table(filepath: str) -> np.ndarray:
    """加载Q表"""
    return np.load(filepath)


def save_q_table(q_table: np.ndarray, filepath: str) -> None:
    """保存Q表"""
    np.save(filepath, q_table)
    print(f"Q表已保存到: {filepath}")


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def print_progress(epoch: int, total_epochs: int, metrics: Dict, bar_length: int = 40):
    """打印训练进度条"""
    progress = epoch / total_epochs
    bar = '=' * int(progress * bar_length) + ' ' * (bar_length - int(progress * bar_length))
    
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"\r[{bar}] {epoch}/{total_epochs} | {metrics_str}", end='')
    
    if epoch == total_epochs:
        print()
