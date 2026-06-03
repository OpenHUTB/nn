"""
强化学习训练工具模块

提供统一的训练框架、日志记录和性能评估功能
扩展：可视化回调管理、实时训练曲线集成
"""

import json
import os
import time
from typing import Dict, List, Optional, Union, Callable, Any

import numpy as np

# 从 visualizer 导入新类
from ..rl.visualizer import LivePlot, PolicyHeatmap, ValueSurface


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


# ========== 新增：可视化集成辅助类 ==========

class VisualizationManager:
    """
    可视化管理器，用于在训练过程中管理 LivePlot、PolicyHeatmap 等
    用法：
        viz = VisualizationManager(env_name="CartPole", num_actions=2)
        viz.setup_live_plot(xlabel='Episode', ylabel='Reward')
        viz.setup_heatmap(num_actions=2)
        for episode in range(N):
            viz.update_live_plot('reward', episode, episode_reward)
            probs = agent.get_action_probs(state)  # 需要算法提供
            viz.update_heatmap(probs, episode)
    """

    def __init__(self, env_name: str = "RLEnv", log_dir: str = "visualizations"):
        self.env_name = env_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.live_plot: Optional[LivePlot] = None
        self.heatmap: Optional[PolicyHeatmap] = None
        self.value_surface: Optional[ValueSurface] = None

    def setup_live_plot(self, xlabel: str = 'Episode', ylabel: str = 'Reward', title: str = 'Training Curve'):
        """初始化实时曲线"""
        self.live_plot = LivePlot(xlabel=xlabel, ylabel=ylabel, title=title)

    def setup_heatmap(self, num_actions: int):
        """初始化策略热力图"""
        self.heatmap = PolicyHeatmap(num_actions=num_actions, env_name=self.env_name)

    def setup_value_surface(self, bounds=((-3,3),(-3,3)), resolution=30):
        """初始化价值曲面图"""
        self.value_surface = ValueSurface(bounds=bounds, resolution=resolution)

    def update_live_plot(self, name: str, x: float, y: float):
        if self.live_plot:
            self.live_plot.update(name, x, y)

    def update_heatmap(self, probs: np.ndarray, step: int):
        if self.heatmap:
            self.heatmap.update(probs, step)

    def update_value_surface(self, value_func: Callable[[np.ndarray], float], step: int):
        if self.value_surface:
            self.value_surface.update(value_func, step)

    def save_all(self, prefix: str = "final"):
        """保存所有当前图表"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if self.live_plot:
            self.live_plot.save(os.path.join(self.log_dir, f"{prefix}_live_plot_{timestamp}.png"))
        if self.heatmap:
            self.heatmap.save(os.path.join(self.log_dir, f"{prefix}_heatmap_{timestamp}.png"))
        if self.value_surface:
            self.value_surface.save(os.path.join(self.log_dir, f"{prefix}_surface_{timestamp}.png"))

    def close_all(self):
        """关闭所有图形窗口"""
        if self.live_plot:
            self.live_plot.close()
        if self.heatmap:
            self.heatmap.close()
        if self.value_surface:
            self.value_surface.close()


# ========== 示例训练函数（展示如何集成可视化） ==========

def train_with_visualization(agent, env, n_episodes: int = 500, 
                             visualize: bool = True, 
                             plot_interval: int = 10,
                             log_dir: str = "training_logs"):
    """
    示例训练函数，展示如何使用 VisualizationManager 集成实时曲线和热力图
    
    参数:
        agent: 强化学习代理，需要实现 get_action, update 等方法，并且如果是离散策略需要提供 get_action_probs
        env: gym 环境
        n_episodes: 训练轮数
        visualize: 是否启用可视化
        plot_interval: 每隔多少 episode 更新一次热力图
        log_dir: 日志保存目录
    """
    logger = TrainingLogger(log_dir=os.path.join(log_dir, "logs"))
    
    viz = None
    if visualize:
        # 检查环境动作空间是否为离散
        if hasattr(env.action_space, 'n'):
            num_actions = env.action_space.n
            viz = VisualizationManager(env_name=env.spec.id if env.spec else "RLEnv", 
                                       log_dir=os.path.join(log_dir, "figures"))
            viz.setup_live_plot(xlabel='Episode', ylabel='Total Reward', title='Training Progress')
            viz.setup_heatmap(num_actions=num_actions)
        else:
            print("警告：连续动作空间暂不支持热力图，将只显示实时曲线")
            viz = VisualizationManager(env_name=env.spec.id if env.spec else "RLEnv",
                                       log_dir=os.path.join(log_dir, "figures"))
            viz.setup_live_plot(xlabel='Episode', ylabel='Total Reward', title='Training Progress')
    
    all_rewards = []
    
    for episode in range(1, n_episodes + 1):
        state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, trunc, info = env.step(action)
            total_reward += reward
            
            # 假设 agent 有 update 方法 (如 QLearner 的 get_next_action_with_Q_table_update)
            if hasattr(agent, 'update'):
                agent.update(state, action, reward, next_state, done)
            elif hasattr(agent, 'get_next_action_with_Q_table_update'):
                # 针对 QLearner/SARSALearner 的集成
                agent.get_next_action_with_Q_table_update(next_state, reward)
            
            state = next_state
            step += 1
        
        all_rewards.append(total_reward)
        logger.log(episode, {"reward": total_reward, "avg_reward": np.mean(all_rewards[-50:])})
        
        # 更新实时曲线
        if viz and viz.live_plot:
            viz.update_live_plot('reward', episode, total_reward)
            viz.update_live_plot('avg_reward', episode, np.mean(all_rewards[-50:]))
        
        # 定期更新策略热力图（如果代理有 get_action_probs 方法）
        if visualize and viz and viz.heatmap and episode % plot_interval == 0:
            if hasattr(agent, 'get_action_probs'):
                # 注意：需要算法类实现 get_action_probs，以下为示例
                probs = agent.get_action_probs(state)  # 期望返回形状 (num_actions,)
                viz.update_heatmap(probs, episode)
            else:
                # 对于 Q-Learning 类，可以从 Q 表计算 softmax 概率
                if hasattr(agent, 'Q') and agent.Q is not None:
                    q_values = agent.Q[state] if isinstance(state, tuple) else agent.Q[state]
                    probs = np.exp(q_values) / np.sum(np.exp(q_values))
                    viz.update_heatmap(probs, episode)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Avg Reward: {np.mean(all_rewards[-50:]):.2f}")
    
    # 训练结束，保存所有可视化图表
    if viz:
        viz.save_all(prefix="final")
        viz.close_all()
    
    logger.save()
    logger.print_summary()
    return all_rewards


# ========== 原有的辅助函数 ==========

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