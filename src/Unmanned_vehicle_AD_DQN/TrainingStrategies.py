# TrainingStrategies.py
import os
import math
import pickle
import numpy as np
from datetime import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# 静态障碍物检测器
class StaticObstacleDetector:
    def __init__(self):
        self.static_obstacle_history = deque(maxlen=50)
        self.collision_patterns = []
        
    def detect_pattern(self, location, heading, action, reward):
        """检测静态障碍物碰撞模式"""
        if reward < -20:
            pattern = {
                'location': location,
                'heading': heading,
                'action': action,
                'timestamp': datetime.now()
            }
            self.static_obstacle_history.append(pattern)
            
            if len(self.static_obstacle_history) >= 10:
                self.analyze_collision_patterns()
                
    def analyze_collision_patterns(self):
        """分析碰撞模式"""
        if len(self.static_obstacle_history) == 0:
            return
            
        print(f"静态障碍物碰撞分析: 总次数={len(self.static_obstacle_history)}")
        
    def get_safe_action_suggestion(self, current_location, current_heading):
        """获取安全动作建议"""
        suggestions = []
        
        if len(self.static_obstacle_history) > 0:
            for pattern in list(self.static_obstacle_history)[-5:]:
                loc = pattern['location']
                distance = math.sqrt(
                    (current_location[0] - loc[0])**2 + 
                    (current_location[1] - loc[1])**2
                )
                
                if distance < 10.0:
                    dangerous_action = pattern['action']
                    suggestions.append({
                        'avoid_action': dangerous_action,
                        'suggested_actions': [0, 1, 2] if dangerous_action in [3, 4] else [3, 4],
                        'reason': '历史碰撞区域'
                    })
        
        return suggestions


# 课程学习管理器
class CurriculumManager:
    def __init__(self, env):
        self.env = env
        self.current_stage = 0
        self.stage_thresholds = [0.3, 0.5, 0.7, 0.85, 0.9]
        
        self.stage_configs = [
            # 阶段0: 入门
            {
                'pedestrian_cross': 2,
                'pedestrian_normal': 1,
                'static_obstacle_penalty': 0.5,
                'max_episode_steps': 800,
                'success_threshold': 0.3,
                'difficulty_name': '入门'
            },
            # 阶段1: 简单
            {
                'pedestrian_cross': 4,
                'pedestrian_normal': 2,
                'static_obstacle_penalty': 1.0,
                'max_episode_steps': 1000,
                'success_threshold': 0.4,
                'difficulty_name': '简单'
            },
            # 阶段2: 中等
            {
                'pedestrian_cross': 6,
                'pedestrian_normal': 3,
                'static_obstacle_penalty': 2.0,
                'max_episode_steps': 1200,
                'success_threshold': 0.5,
                'difficulty_name': '中等'
            },
            # 阶段3: 困难
            {
                'pedestrian_cross': 8,
                'pedestrian_normal': 4,
                'static_obstacle_penalty': 3.0,
                'max_episode_steps': 1500,
                'success_threshold': 0.6,
                'difficulty_name': '困难'
            },
            # 阶段4: 专家
            {
                'pedestrian_cross': 10,
                'pedestrian_normal': 6,
                'static_obstacle_penalty': 4.0,
                'max_episode_steps': 1800,
                'success_threshold': 0.7,
                'difficulty_name': '专家'
            },
            # 阶段5: 大师
            {
                'pedestrian_cross': 12,
                'pedestrian_normal': 8,
                'static_obstacle_penalty': 5.0,
                'max_episode_steps': 2400,
                'success_threshold': 0.8,
                'difficulty_name': '大师'
            }
        ]
        
        # 训练历史
        self.success_history = deque(maxlen=20)
        self.reward_history = deque(maxlen=50)
        self.reaction_time_history = deque(maxlen=50)
        self.static_collision_history = deque(maxlen=20)
        
        # 静态障碍物检测器
        self.static_detector = StaticObstacleDetector()
        
    def update_stage(self, success, reward, reaction_time=None, static_collision=False):
        """更新训练阶段"""
        self.success_history.append(1 if success else 0)
        self.reward_history.append(reward)
        if reaction_time is not None:
            self.reaction_time_history.append(reaction_time)
        if static_collision:
            self.static_collision_history.append(1)
        else:
            self.static_collision_history.append(0)
        
        if len(self.success_history) >= 10:
            success_rate = sum(self.success_history) / len(self.success_history)
            avg_reward = np.mean(self.reward_history) if self.reward_history else 0
            
            static_collision_rate = sum(self.static_collision_history) / len(self.static_collision_history) if self.static_collision_history else 0
            
            if len(self.success_history) % 20 == 0:
                stage_info = self.get_current_config()
                print(f"课程学习 - 阶段: {self.current_stage}({stage_info['difficulty_name']})")
                print(f"  成功率: {success_rate:.2f}, 平均奖励: {avg_reward:.2f}")
                print(f"  静态碰撞率: {static_collision_rate:.2f}")
                if self.reaction_time_history:
                    avg_rt = np.mean(self.reaction_time_history)
                    print(f"  平均反应时间: {avg_rt:.2f}秒")
            
            if self.current_stage < len(self.stage_configs) - 1:
                next_stage_threshold = self.stage_configs[self.current_stage]['success_threshold']
                
                can_advance = (
                    success_rate >= next_stage_threshold and 
                    avg_reward > 3 and
                    static_collision_rate < 0.2
                )
                
                if can_advance:
                    self.current_stage += 1
                    print(f"🎉 课程学习: 进阶到阶段 {self.current_stage}!")
                    print(f"   新配置: {self.stage_configs[self.current_stage]['difficulty_name']}")
                    return True
                    
            if self.current_stage > 0 and (
                success_rate < 0.2 or 
                static_collision_rate > 0.4 or
                (self.reaction_time_history and np.mean(self.reaction_time_history) > 2.0)
            ):
                self.current_stage -= 1
                print(f"⚠️ 课程学习: 退回阶段 {self.current_stage}")
                return True
        
        return False
    
    def get_current_config(self):
        """获取当前阶段的配置"""
        return self.stage_configs[min(self.current_stage, len(self.stage_configs) - 1)]
    
    def apply_to_environment(self):
        """将当前阶段配置应用到环境"""
        config = self.get_current_config()
        return config


# 多目标优化器
class MultiObjectiveOptimizer:
    def __init__(self):
        self.objectives = {
            'reaction_time': {
                'weight': 0.20,
                'description': '快速反应避障',
                'metrics': ['reaction_time', 'proactive_actions']
            },
            'safety': {
                'weight': 0.35,
                'description': '安全避障和避免碰撞',
                'metrics': ['collision_avoidance', 'pedestrian_distance', 'static_obstacle_distance']
            },
            'efficiency': {
                'weight': 0.20,
                'description': '快速到达目的地',
                'metrics': ['progress_speed', 'total_time']
            },
            'comfort': {
                'weight': 0.15,
                'description': '平稳驾驶体验',
                'metrics': ['smoothness', 'steering_changes']
            },
            'static_avoidance': {
                'weight': 0.10,
                'description': '避免静态障碍物',
                'metrics': ['static_collision', 'static_distance']
            }
        }
        
        self.metrics_history = {
            'reaction_time': [],
            'safety': [],
            'efficiency': [],
            'comfort': [],
            'static_avoidance': []
        }
        
    def compute_composite_reward(self, metrics):
        """计算综合奖励值"""
        composite = 0
        
        for obj_name, obj_info in self.objectives.items():
            if obj_name in metrics:
                normalized_value = self._normalize_metric(metrics[obj_name], obj_name)
                composite += normalized_value * obj_info['weight']
                
                self.metrics_history[obj_name].append(normalized_value)
        
        # 特殊奖励/惩罚项
        if metrics.get('collision', False):
            composite -= 12
        if metrics.get('static_collision', False):
            composite -= 15
        if metrics.get('off_road', False):
            composite -= 8
            
        if 'reaction_time' in metrics:
            rt = metrics['reaction_time']
            if rt < 0.3:
                composite += 3
            elif rt > 1.2:
                composite -= 4
        
        if metrics.get('proactive_action', False):
            composite += 2.0
            
        if metrics.get('static_distance', 100) > 15:
            composite += 1.0
        elif metrics.get('static_distance', 100) < 5:
            composite -= 3.0
            
        return composite
    
    def _normalize_metric(self, value, metric_name):
        """归一化指标值到[0, 1]范围"""
        normalization_rules = {
            'reaction_time': lambda x: max(0, 1 - x/3),
            'safety': lambda x: min(max(x / 10, 0), 1),
            'efficiency': lambda x: min(max(x / 100, 0), 1),
            'comfort': lambda x: min(max((x + 5) / 10, 0), 1),
            'static_avoidance': lambda x: min(max(1 - x/5, 0), 1)
        }
        
        if metric_name in normalization_rules:
            return normalization_rules[metric_name](value)
        return min(max(value, 0), 1)
    
    def adjust_weights(self, performance_feedback):
        """根据性能反馈动态调整权重"""
        recent_performance = {}
        for obj in self.objectives:
            if len(self.metrics_history[obj]) >= 10:
                recent_avg = np.mean(self.metrics_history[obj][-10:])
                recent_performance[obj] = recent_avg
        
        if recent_performance:
            worst_obj = min(recent_performance, key=recent_performance.get)
            best_obj = max(recent_performance, key=recent_performance.get)
            
            if recent_performance[worst_obj] < 0.3:
                adjustment = 0.04
                self.objectives[worst_obj]['weight'] += adjustment
                self.objectives[best_obj]['weight'] -= adjustment
                
                total = sum(obj['weight'] for obj in self.objectives.values())
                for obj in self.objectives:
                    self.objectives[obj]['weight'] /= total
                
                if adjustment != 0:
                    print(f"动态权重调整: {worst_obj}权重↑ {adjustment:.3f}, {best_obj}权重↓ {adjustment:.3f}")
    
    def get_performance_report(self):
        """生成性能报告"""
        report = "多目标优化性能报告:\n"
        report += "=" * 50 + "\n"
        
        for obj_name, obj_info in self.objectives.items():
            history = self.metrics_history[obj_name]
            if history:
                avg = np.mean(history[-20:]) if len(history) >= 20 else np.mean(history)
                report += f"{obj_name}(权重:{obj_info['weight']:.2f}): 平均得分={avg:.3f}\n"
                report += f"  描述: {obj_info['description']}\n"
        
        return report


# 模仿学习管理器
class ImitationLearningManager:
    def __init__(self, expert_data_path=None):
        self.expert_data_path = expert_data_path
        self.expert_data = []
        self.is_pretrained = False
        
    def load_expert_data(self, path):
        """加载专家示范数据"""
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.expert_data = pickle.load(f)
                print(f"已加载 {len(self.expert_data)} 条专家示范数据")
                return True
            else:
                print(f"专家数据文件不存在: {path}")
                return False
        except Exception as e:
            print(f"加载专家数据失败: {e}")
            return False


# 优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, max_size=20000, alpha=0.7, beta_start=0.5, beta_frames=50000):
        self.max_size = max_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
    def __len__(self):
        return len(self.buffer)
    
    def beta(self):
        """线性递增的beta值，用于重要性采样权重"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, experience, error=None):
        """添加经验到缓冲区"""
        if error is None:
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = (abs(error) + 1e-5) ** self.alpha
            
        state, action, reward, next_state, done = experience
        
        # 静态碰撞检测
        if reward < -20:
            priority *= 2.0
            
        elif reward < -5:
            priority *= 1.5
            
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        """从缓冲区中采样一批经验"""
        if len(self.buffer) == 0:
            return [], [], []
            
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
        
        samples = [self.buffer[i] for i in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta())
        weights /= weights.max()
        
        self.frame += 1
        
        return indices, samples, weights
    
    def update_priorities(self, indices, errors):
        """更新采样经验的优先级"""
        for idx, error in zip(indices, errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha