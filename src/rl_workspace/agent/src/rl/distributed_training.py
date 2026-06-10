"""
分布式训练模块

支持多进程并行训练和结果聚合
"""

import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import queue
import time


@dataclass
class WorkerResult:
    """工作进程结果"""
    worker_id: int
    rewards: List[float]
    final_reward: float
    training_time: float
    success: bool
    error: Optional[str] = None


class ParallelTrainer:
    """并行训练器"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.results = []
    
    def train_worker(self, worker_id: int, train_fn: Callable,
                   epochs: int, shared_config: dict) -> WorkerResult:
        """单个工作进程训练"""
        start_time = time.time()
        
        try:
            rewards = train_fn(epochs=epochs, **shared_config)
            training_time = time.time() - start_time
            final_reward = np.mean(rewards[-100:]) if len(rewards) > 0 else 0
            
            return WorkerResult(
                worker_id=worker_id,
                rewards=rewards,
                final_reward=final_reward,
                training_time=training_time,
                success=True
            )
        except Exception as e:
            training_time = time.time() - start_time
            return WorkerResult(
                worker_id=worker_id,
                rewards=[],
                final_reward=0,
                training_time=training_time,
                success=False,
                error=str(e)
            )
    
    def parallel_train(self, train_fn: Callable, total_epochs: int,
                     shared_config: dict = None) -> List[WorkerResult]:
        """并行训练"""
        if shared_config is None:
            shared_config = {}
        
        epochs_per_worker = total_epochs // self.num_workers
        extra_epochs = total_epochs % self.num_workers
        
        processes = []
        result_queue = mp.Queue()
        
        for worker_id in range(self.num_workers):
            epochs = epochs_per_worker + (1 if worker_id < extra_epochs else 0)
            
            p = mp.Process(
                target=self._worker_wrapper,
                args=(worker_id, train_fn, epochs, shared_config, result_queue)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        results.sort(key=lambda x: x.worker_id)
        self.results = results
        
        return results
    
    def _worker_wrapper(self, worker_id: int, train_fn: Callable,
                       epochs: int, shared_config: dict, result_queue: mp.Queue):
        """工作进程包装器"""
        result = self.train_worker(worker_id, train_fn, epochs, shared_config)
        result_queue.put(result)
    
    def aggregate_results(self) -> dict:
        """聚合结果"""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {'error': '所有工作进程训练失败'}
        
        all_rewards = []
        for r in successful_results:
            all_rewards.extend(r.rewards)
        
        return {
            'num_workers': self.num_workers,
            'successful_workers': len(successful_results),
            'total_epochs': sum(r.epochs for r in successful_results),
            'avg_reward': np.mean([r.final_reward for r in successful_results]),
            'max_reward': max(r.final_reward for r in successful_results),
            'min_reward': min(r.final_reward for r in successful_results),
            'total_time': max(r.training_time for r in successful_results),
            'all_rewards': all_rewards
        }
    
    def print_summary(self):
        """打印摘要"""
        agg = self.aggregate_results()
        
        print(f"\n{'='*60}")
        print("并行训练摘要")
        print(f"{'='*60}")
        print(f"工作进程数: {agg.get('num_workers', 0)}")
        print(f"成功进程数: {agg.get('successful_workers', 0)}")
        print(f"总训练轮数: {agg.get('total_epochs', 0)}")
        print(f"平均奖励: {agg.get('avg_reward', 0):.2f}")
        print(f"最大奖励: {agg.get('max_reward', 0):.2f}")
        print(f"最小奖励: {agg.get('min_reward', 0):.2f}")
        print(f"总训练时间: {agg.get('total_time', 0):.2f}秒")
        print(f"{'='*60}")


def parallel_q_learning_worker(worker_id: int, epochs: int, config: dict) -> List[float]:
    """Q-Learning并行工作函数"""
    import gymnasium as gym
    from rl.algorithms import QLearner
    
    env = gym.make(config['env_name'])
    learner = QLearner(
        states=env.observation_space.n,
        actions=env.action_space.n,
        alpha=config.get('alpha', 0.8),
        gamma=config.get('gamma', 0.9),
        radr=config.get('radr', 0.001)
    )
    
    rewards = []
    
    for episode in range(epochs):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        
        action = learner.get_next_action_without_Q_table_update(state)
        
        while not done:
            new_state, reward, done, trunc, info = env.step(action)
            action = learner.get_next_action_with_Q_table_update(new_state, reward)
            episode_reward += reward
        
        learner.decay_rar(episode)
        rewards.append(episode_reward)
    
    env.close()
    return rewards


def parallel_train_q_learning(total_epochs: int = 10000, num_workers: int = 4,
                             env_name: str = "FrozenLake-v1"):
    """并行训练Q-Learning"""
    print(f"开始并行训练Q-Learning ({num_workers}个工作进程)...")
    
    config = {
        'env_name': env_name,
        'alpha': 0.8,
        'gamma': 0.9,
        'radr': 0.001
    }
    
    trainer = ParallelTrainer(num_workers=num_workers)
    
    def worker_fn(epochs, **kwargs):
        return parallel_q_learning_worker(0, epochs, kwargs)
    
    results = trainer.parallel_train(worker_fn, total_epochs, config)
    trainer.print_summary()
    
    return trainer.aggregate_results()


if __name__ == "__main__":
    results = parallel_train_q_learning(total_epochs=2000, num_workers=2)
