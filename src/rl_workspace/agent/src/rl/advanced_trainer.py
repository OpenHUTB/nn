"""
高级训练模块

提供带进度条、早停、学习率调度的高级训练功能
"""

import os
import sys
import time
from typing import Optional, Callable, Dict, List
import numpy as np


class ProgressBar:
    """训练进度条"""
    
    def __init__(self, total: int, prefix: str = "训练进度", length: int = 50):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current: int, metrics: Optional[Dict[str, float]] = None):
        """更新进度条"""
        self.current = current
        percent = current / self.total
        filled = int(self.length * percent)
        bar = '█' * filled + '-' * (self.length - filled)
        
        elapsed = time.time() - self.start_time
        eta = elapsed / percent - elapsed if percent > 0 else 0
        
        metrics_str = ""
        if metrics:
            metrics_str = " | " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        
        print(f"\r{self.prefix}: [{bar}] {percent*100:.1f}% | ETA: {eta:.1f}s{metrics_str}", end='')
        
        if current >= self.total:
            print()
    
    def close(self):
        """关闭进度条"""
        print()


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_iteration = 0
    
    def __call__(self, score: float, iteration: int) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            self.best_iteration = iteration
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_iteration = iteration
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, initial_lr: float, schedule_type: str = 'exponential', 
                 decay_rate: float = 0.95, decay_steps: int = 1000):
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_lr = initial_lr
    
    def step(self, step: int):
        """更新学习率"""
        if self.schedule_type == 'exponential':
            self.current_lr = self.initial_lr * (self.decay_rate ** (step / self.decay_steps))
        elif self.schedule_type == 'step':
            self.current_lr = self.initial_lr * (self.decay_rate ** (step // self.decay_steps))
        elif self.schedule_type == 'cosine':
            self.current_lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
        
        return self.current_lr
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = {}
        self.windows = {}
    
    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
                self.windows[key] = []
            
            self.history[key].append(value)
            self.windows[key].append(value)
            
            if len(self.windows[key]) > self.window_size:
                self.windows[key].pop(0)
    
    def get_mean(self, key: str) -> float:
        """获取滑动平均"""
        if key not in self.windows or len(self.windows[key]) == 0:
            return 0.0
        return float(np.mean(self.windows[key]))
    
    def get_std(self, key: str) -> float:
        """获取标准差"""
        if key not in self.windows or len(self.windows[key]) == 0:
            return 0.0
        return float(np.std(self.windows[key]))
    
    def get_history(self, key: str) -> List[float]:
        """获取历史记录"""
        return self.history.get(key, [])
    
    def save(self, filepath: str):
        """保存指标历史"""
        np.save(filepath, self.history, allow_pickle=True)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, save_dir: str = "checkpoints", max_keep: int = 5):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.checkpoints = []
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, iteration: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{iteration}.npz")
        
        checkpoint_data = {
            'iteration': iteration,
            'metrics': metrics,
            'model': model
        }
        np.savez(checkpoint_path, **checkpoint_data)
        
        self.checkpoints.append((iteration, checkpoint_path))
        
        if len(self.checkpoints) > self.max_keep:
            old_iter, old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
        
        print(f"\n检查点已保存: {checkpoint_path}")
    
    def load_best(self) -> Optional[dict]:
        """加载最佳检查点"""
        if not self.checkpoints:
            return None
        
        best_iter, best_path = max(self.checkpoints, key=lambda x: x[0])
        return np.load(best_path, allow_pickle=True)


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.start_time = None
        self.metrics_history = []
        self.callbacks = []
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        print(f"\n训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log(self, iteration: int, metrics: Dict[str, float]):
        """记录指标"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        log_entry = {
            'iteration': iteration,
            'elapsed': elapsed,
            **metrics
        }
        self.metrics_history.append(log_entry)
        
        for callback in self.callbacks:
            callback(iteration, metrics)
    
    def add_callback(self, callback: Callable):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        if not self.metrics_history:
            return {}
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            'total_iterations': len(self.metrics_history),
            'total_time': total_time,
            'iterations_per_second': len(self.metrics_history) / total_time if total_time > 0 else 0
        }
        
        for key in self.metrics_history[0].keys():
            if key not in ['iteration', 'elapsed']:
                values = [entry[key] for entry in self.metrics_history if key in entry]
                summary[f'{key}_mean'] = float(np.mean(values))
                summary[f'{key}_final'] = float(values[-1])
        
        return summary
    
    def print_summary(self):
        """打印训练摘要"""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print("训练摘要")
        print(f"{'='*60}")
        print(f"总迭代次数: {summary.get('total_iterations', 0)}")
        print(f"总训练时间: {summary.get('total_time', 0):.2f} 秒")
        print(f"迭代速度: {summary.get('iterations_per_second', 0):.2f} iter/s")
        
        print(f"\n指标统计:")
        for key, value in summary.items():
            if '_mean' in key or '_final' in key:
                metric_name = key.replace('_mean', ' (平均)').replace('_final', ' (最终)')
                print(f"  {metric_name}: {value:.4f}")
        
        print(f"{'='*60}")
