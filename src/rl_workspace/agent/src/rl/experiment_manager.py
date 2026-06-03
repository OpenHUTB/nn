"""
实验管理和超参数优化模块

提供实验跟踪、结果保存和自动超参数搜索功能
扩展：自动保存 matplotlib 图表到实验目录
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


@dataclass
class ExperimentConfig:
    """实验配置"""
    algorithm: str
    env_name: str
    hyperparameters: Dict[str, Any]
    seed: int = 42
    timestamp: Optional[str] = None


@dataclass
class ExperimentResult:
    """实验结果"""
    config: ExperimentConfig
    metrics: Dict[str, float]
    training_time: float
    success: bool
    error_message: Optional[str] = None


class ExperimentTracker:
    """
    实验跟踪器
    
    记录每次实验的配置、指标和结果
    扩展：自动保存图表到实验子目录
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = experiment_dir
        self.current_experiment = None
        self.experiments = []
        os.makedirs(experiment_dir, exist_ok=True)
    
    def start_experiment(self, config: ExperimentConfig):
        """开始新实验"""
        config.timestamp = datetime.now().isoformat()
        self.current_experiment = {
            'config': config,
            'start_time': time.time(),
            'metrics': {},
            'checkpoints': []
        }
        # 创建实验专属的 figures 子目录
        exp_id = self._generate_exp_id(config)
        self.current_experiment['figure_dir'] = os.path.join(self.experiment_dir, exp_id, 'figures')
        os.makedirs(self.current_experiment['figure_dir'], exist_ok=True)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """记录指标"""
        if self.current_experiment is None:
            return
        
        if name not in self.current_experiment['metrics']:
            self.current_experiment['metrics'][name] = []
        
        self.current_experiment['metrics'][name].append({
            'value': value,
            'step': step,
            'timestamp': time.time() - self.current_experiment['start_time']
        })
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """批量记录指标"""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def add_checkpoint(self, checkpoint_data: Any, metadata: Optional[Dict] = None):
        """添加检查点"""
        if self.current_experiment is None:
            return
        
        checkpoint = {
            'data': checkpoint_data,
            'metadata': metadata or {},
            'timestamp': time.time() - self.current_experiment['start_time']
        }
        self.current_experiment['checkpoints'].append(checkpoint)
    
    def save_figure(self, fig, filename: str):
        """
        自动保存 matplotlib 图表到当前实验的 figures 目录
        参数：
            fig: matplotlib.figure.Figure 对象
            filename: 文件名（如 'training_curve.png'）
        """
        if self.current_experiment is None:
            raise RuntimeError("没有正在进行的实验，请先调用 start_experiment")
        filepath = os.path.join(self.current_experiment['figure_dir'], filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath, dpi=150)
        print(f"图表已保存到: {filepath}")
    
    def end_experiment(self, success: bool = True, error_message: Optional[str] = None):
        """结束实验"""
        if self.current_experiment is None:
            return
        
        elapsed_time = time.time() - self.current_experiment['start_time']
        
        result = ExperimentResult(
            config=self.current_experiment['config'],
            metrics={name: [m['value'] for m in values] 
                    for name, values in self.current_experiment['metrics'].items()},
            training_time=elapsed_time,
            success=success,
            error_message=error_message
        )
        
        self.experiments.append(result)
        self._save_experiment(result)
        
        self.current_experiment = None
        return result
    
    def _save_experiment(self, result: ExperimentResult):
        """保存实验结果"""
        exp_id = self._generate_exp_id(result.config)
        exp_path = os.path.join(self.experiment_dir, exp_id)
        os.makedirs(exp_path, exist_ok=True)
        
        with open(os.path.join(exp_path, 'config.json'), 'w') as f:
            json.dump({
                'algorithm': result.config.algorithm,
                'env_name': result.config.env_name,
                'hyperparameters': result.config.hyperparameters,
                'seed': result.config.seed,
                'timestamp': result.config.timestamp
            }, f, indent=2)
        
        with open(os.path.join(exp_path, 'results.json'), 'w') as f:
            json.dump({
                'metrics': result.metrics,
                'training_time': result.training_time,
                'success': result.success,
                'error_message': result.error_message
            }, f, indent=2)
    
    def _generate_exp_id(self, config: ExperimentConfig) -> str:
        """生成实验ID"""
        config_str = json.dumps(config.hyperparameters, sort_keys=True)
        hash_str = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{config.algorithm}_{config.env_name}_{hash_str}"
    
    def get_best_experiment(self, metric: str = 'success_rate', 
                           mode: str = 'max') -> Optional[ExperimentResult]:
        """获取最佳实验"""
        if not self.experiments:
            return None
        
        best = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for exp in self.experiments:
            if metric in exp.metrics and exp.metrics[metric]:
                values = exp.metrics[metric]
                value = np.mean(values) if isinstance(values, list) else values
                
                if mode == 'max' and value > best_value:
                    best_value = value
                    best = exp
                elif mode == 'min' and value < best_value:
                    best_value = value
                    best = exp
        
        return best
    
    def get_summary(self) -> Dict:
        """获取实验摘要"""
        if not self.experiments:
            return {}
        
        summary = {
            'total_experiments': len(self.experiments),
            'successful_experiments': sum(1 for e in self.experiments if e.success),
            'average_training_time': np.mean([e.training_time for e in self.experiments])
        }
        
        return summary


# ========== 以下 HyperparameterOptimizer 和 ExperimentSuite 保持不变 ==========

class HyperparameterOptimizer:
    """
    超参数优化器
    
    支持网格搜索、随机搜索和贝叶斯优化
    """
    
    def __init__(self, param_space: Dict[str, Any]):
        self.param_space = param_space
        self.results = []
    
    def grid_search(self, train_fn: Callable, n_trials: Optional[int] = None):
        """网格搜索"""
        import itertools
        
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        
        param_combinations = list(itertools.product(*values))
        
        if n_trials:
            param_combinations = param_combinations[:n_trials]
        
        for params in param_combinations:
            param_dict = dict(zip(keys, params))
            print(f"\n测试参数: {param_dict}")
            
            try:
                result = train_fn(param_dict)
                self.results.append({
                    'params': param_dict,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                self.results.append({
                    'params': param_dict,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        return self.get_best_params()
    
    def random_search(self, train_fn: Callable, n_trials: int = 20):
        """随机搜索"""
        for i in range(n_trials):
            params = {key: np.random.choice(values) 
                     for key, values in self.param_space.items()}
            
            print(f"\n试验 {i+1}/{n_trials}: {params}")
            
            try:
                result = train_fn(params)
                self.results.append({
                    'params': params,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                self.results.append({
                    'params': params,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        return self.get_best_params()
    
    def get_best_params(self, metric: str = 'reward', mode: str = 'max') -> Optional[Dict]:
        """获取最佳参数"""
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            return None
        
        if isinstance(successful_results[0]['result'], dict):
            if metric not in successful_results[0]['result']:
                metric = list(successful_results[0]['result'].keys())[0]
        
        best = max(successful_results, 
                  key=lambda x: x['result'][metric] if isinstance(x['result'], dict) else x['result'])
        
        print(f"\n最佳参数: {best['params']}")
        print(f"最佳结果: {best['result']}")
        
        return best['params']
    
    def save_results(self, filepath: str):
        """保存优化结果"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)


class ExperimentSuite:
    """
    实验套件
    
    管理和运行一组相关实验
    """
    
    def __init__(self, name: str = "experiment_suite"):
        self.name = name
        self.tracker = ExperimentTracker()
        self.optimizers = {}
    
    def add_experiment(self, config: ExperimentConfig, train_fn: Callable):
        """添加实验"""
        self.tracker.start_experiment(config)
        
        try:
            result = train_fn(config.hyperparameters)
            self.tracker.log_metrics(result.get('metrics', {}))
            self.tracker.end_experiment(success=True)
        except Exception as e:
            self.tracker.end_experiment(success=False, error_message=str(e))
    
    def run_grid_search(self, param_space: Dict, train_fn: Callable, 
                       algorithm: str = "grid_search"):
        """运行网格搜索"""
        optimizer = HyperparameterOptimizer(param_space)
        best_params = optimizer.grid_search(train_fn)
        self.optimizers[algorithm] = optimizer
        return best_params
    
    def run_random_search(self, param_space: Dict, train_fn: Callable,
                         n_trials: int = 20, algorithm: str = "random_search"):
        """运行随机搜索"""
        optimizer = HyperparameterOptimizer(param_space)
        best_params = optimizer.random_search(train_fn, n_trials)
        self.optimizers[algorithm] = optimizer
        return best_params
    
    def generate_report(self, filepath: str = "experiment_report.md"):
        """生成实验报告"""
        summary = self.tracker.get_summary()
        best_exp = self.tracker.get_best_experiment()
        
        report = f"""# {self.name} 实验报告

## 实验摘要
- 总实验数: {summary.get('total_experiments', 0)}
- 成功实验: {summary.get('successful_experiments', 0)}
- 平均训练时间: {summary.get('average_training_time', 0):.2f}秒

## 最佳实验
"""
        if best_exp:
            report += f"""
- 算法: {best_exp.config.algorithm}
- 环境: {best_exp.config.env_name}
- 训练时间: {best_exp.training_time:.2f}秒
- 配置: {json.dumps(best_exp.config.hyperparameters, indent=2)}

### 指标
"""
            for metric, values in best_exp.metrics.items():
                report += f"- {metric}: {np.mean(values):.4f}\n"
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"实验报告已保存到: {filepath}")
        
        return report