# TrainingStrategies.py
import os
import math
import pickle
import numpy as np
from datetime import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# 课程学习管理器
class CurriculumManager:
    def __init__(self, env):
        self.env = env
        self.current_stage = 0
        self.stage_thresholds = [0.3, 0.5, 0.7, 0.85]  # 成功率阈值
        self.stage_configs = [
            # 阶段0: 入门 - 更早引入行人，但数量少速度慢
            {
                'pedestrian_cross': 3,      # 增加一点难度
                'pedestrian_normal': 2,     
                'pedestrian_speed_min': 0.3,  # 降低行人速度
                'pedestrian_speed_max': 0.8,  
                'max_episode_steps': 900,    # 减少最大步数，加速训练
                'success_threshold': 0.3,
                'difficulty_name': '简单'
            },
            # 阶段1: 初级
            {
                'pedestrian_cross': 5,      
                'pedestrian_normal': 3,
                'pedestrian_speed_min': 0.5,
                'pedestrian_speed_max': 1.0,
                'max_episode_steps': 1200,   
                'success_threshold': 0.5,
                'difficulty_name': '初级'
            },
            # 阶段2: 中级 - 增加反应时间训练
            {
                'pedestrian_cross': 7,
                'pedestrian_normal': 4,
                'pedestrian_speed_min': 0.7,
                'pedestrian_speed_max': 1.3,
                'max_episode_steps': 1500,   
                'success_threshold': 0.6,  # 提高阈值
                'difficulty_name': '中级'
            },
            # 阶段3: 高级
            {
                'pedestrian_cross': 9,
                'pedestrian_normal': 5,
                'pedestrian_speed_min': 0.9,
                'pedestrian_speed_max': 1.6,
                'max_episode_steps': 1800,   
                'success_threshold': 0.7,
                'difficulty_name': '高级'
            },
            # 阶段4: 专家 (正常难度)
            {
                'pedestrian_cross': 10,     
                'pedestrian_normal': 6,
                'pedestrian_speed_min': 1.0,
                'pedestrian_speed_max': 2.0,
                'max_episode_steps': 2400,
                'success_threshold': 0.8,
                'difficulty_name': '专家'
            },
            # 阶段5: 大师 (挑战)
            {
                'pedestrian_cross': 12,     
                'pedestrian_normal': 8,
                'pedestrian_speed_min': 1.2,
                'pedestrian_speed_max': 2.5,
                'max_episode_steps': 3000,
                'success_threshold': 0.85,
                'difficulty_name': '大师'
            }
        ]
        
        # 训练历史
        self.success_history = deque(maxlen=20)  # 记录最近20轮的成功情况
        self.reward_history = deque(maxlen=50)   # 记录最近50轮的奖励
        self.reaction_time_history = deque(maxlen=50)  # 记录反应时间
        
    def update_stage(self, success, reward, reaction_time=None):
        """更新训练阶段"""
        # 记录历史
        self.success_history.append(1 if success else 0)
        self.reward_history.append(reward)
        if reaction_time is not None:
            self.reaction_time_history.append(reaction_time)
        
        # 计算最近成功率
        if len(self.success_history) >= 10:
            success_rate = sum(self.success_history) / len(self.success_history)
            avg_reward = np.mean(self.reward_history) if self.reward_history else 0
            
            # 每20轮打印一次
            if len(self.success_history) % 20 == 0:
                stage_info = self.get_current_config()
                print(f"课程学习 - 阶段: {self.current_stage}({stage_info['difficulty_name']}), "
                      f"成功率: {success_rate:.2f}, 平均奖励: {avg_reward:.2f}")
                if self.reaction_time_history:
                    avg_rt = np.mean(self.reaction_time_history)
                    print(f"  平均反应时间: {avg_rt:.2f}秒")
            
            # 检查是否可以进入下一阶段
            if self.current_stage < len(self.stage_configs) - 1:
                next_stage_threshold = self.stage_configs[self.current_stage]['success_threshold']
                
                # 不仅要看成功率，还要看反应时间（如果可用）
                can_advance = success_rate >= next_stage_threshold and avg_reward > 5
                if can_advance and self.reaction_time_history:
                    avg_rt = np.mean(self.reaction_time_history)
                    # 要求反应时间小于1秒
                    if avg_rt < 1.0:
                        self.current_stage += 1
                        print(f"🎉 课程学习: 进阶到阶段 {self.current_stage}!")
                        print(f"   新配置: {self.stage_configs[self.current_stage]['difficulty_name']}")
                        return True
                elif can_advance:
                    self.current_stage += 1
                    print(f"🎉 课程学习: 进阶到阶段 {self.current_stage}!")
                    print(f"   新配置: {self.stage_configs[self.current_stage]['difficulty_name']}")
                    return True
                    
            # 如果表现太差或反应时间太长，退回上一阶段
            if self.current_stage > 0 and (success_rate < 0.2 or 
                (self.reaction_time_history and np.mean(self.reaction_time_history) > 2.0)):
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
        # 定义优化目标及其权重（可动态调整）
        self.objectives = {
            'reaction_time': {
                'weight': 0.25,  # 新增：反应时间权重
                'description': '快速反应避障',
                'metrics': ['reaction_time', 'proactive_actions']
            },
            'safety': {
                'weight': 0.30,  
                'description': '安全避障和避免碰撞',
                'metrics': ['collision_avoidance', 'pedestrian_distance']
            },
            'efficiency': {
                'weight': 0.25,  
                'description': '快速到达目的地',
                'metrics': ['progress_speed', 'total_time']
            },
            'comfort': {
                'weight': 0.15,
                'description': '平稳驾驶体验',
                'metrics': ['smoothness', 'steering_changes']
            },
            'rule_following': {
                'weight': 0.05,
                'description': '遵守交通规则',
                'metrics': ['lane_keeping', 'speed_limit']
            }
        }
        
        # 指标跟踪
        self.metrics_history = {
            'reaction_time': [],
            'safety': [],
            'efficiency': [],
            'comfort': [],
            'rule_following': []
        }
        
    def compute_composite_reward(self, metrics):
        """计算综合奖励值"""
        composite = 0
        
        for obj_name, obj_info in self.objectives.items():
            if obj_name in metrics:
                # 归一化处理每个目标的贡献
                normalized_value = self._normalize_metric(metrics[obj_name], obj_name)
                composite += normalized_value * obj_info['weight']
                
                # 记录指标历史
                self.metrics_history[obj_name].append(normalized_value)
        
        # 特殊奖励/惩罚项
        if metrics.get('collision', False):
            composite -= 10  # 增加碰撞惩罚
        if metrics.get('off_road', False):
            composite -= 5   # 增加偏离道路惩罚
        if metrics.get('dangerous_action', False):
            composite -= 3   # 增加危险动作惩罚
            
        # 新增：反应时间相关奖励/惩罚
        if 'reaction_time' in metrics:
            rt = metrics['reaction_time']
            if rt < 0.5:  # 快速反应
                composite += 2
            elif rt > 1.5:  # 反应太慢
                composite -= 3
        
        # 新增：主动避障奖励
        if metrics.get('proactive_action', False):
            composite += 1.5
            
        return composite
    
    def _normalize_metric(self, value, metric_name):
        """归一化指标值到[0, 1]范围"""
        # 不同指标的归一化方式不同
        normalization_rules = {
            'reaction_time': lambda x: max(0, 1 - x/3),  # 反应时间越短越好
            'safety': lambda x: min(max(x / 10, 0), 1),
            'efficiency': lambda x: min(max(x / 100, 0), 1),
            'comfort': lambda x: min(max((x + 5) / 10, 0), 1),
            'rule_following': lambda x: min(max(x, 0), 1)
        }
        
        if metric_name in normalization_rules:
            return normalization_rules[metric_name](value)
        return min(max(value, 0), 1)  # 默认截断到[0, 1]
    
    def adjust_weights(self, performance_feedback):
        """根据性能反馈动态调整权重"""
        # 如果某个目标表现持续较差，增加其权重
        recent_performance = {}
        for obj in self.objectives:
            if len(self.metrics_history[obj]) >= 10:
                recent_avg = np.mean(self.metrics_history[obj][-10:])
                recent_performance[obj] = recent_avg
        
        if recent_performance:
            # 找到表现最差的目标
            worst_obj = min(recent_performance, key=recent_performance.get)
            best_obj = max(recent_performance, key=recent_performance.get)
            
            # 如果最差目标表现低于阈值，增加其权重
            if recent_performance[worst_obj] < 0.3:
                adjustment = 0.03  # 调整幅度
                self.objectives[worst_obj]['weight'] += adjustment
                self.objectives[best_obj]['weight'] -= adjustment
                
                # 确保权重总和为1
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


# 模仿学习管理器（保持不变，略作修改）
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


# 优先经验回放缓冲区（增加对危险经验的优先级）
class PrioritizedReplayBuffer:
    def __init__(self, max_size=15000, alpha=0.7, beta_start=0.5, beta_frames=50000):
        self.max_size = max_size
        self.alpha = alpha  # 优先级程度 (0 = 均匀采样, 1 = 完全优先级)
        self.beta_start = beta_start  # 重要性采样权重起始值
        self.beta_frames = beta_frames  # beta线性增长的帧数
        self.frame = 1
        
        # 使用循环缓冲区
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
            
        # 如果是危险经验（负奖励较大），增加优先级
        reward = experience[2]
        if reward < -2:  # 危险经验
            priority *= 1.5
            
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        """从缓冲区中采样一批经验"""
        if len(self.buffer) == 0:
            return [], [], []
            
        # 计算采样概率
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
        
        # 获取样本
        samples = [self.buffer[i] for i in indices]
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta())
        weights /= weights.max()  # 归一化
        
        # 更新帧计数器
        self.frame += 1
        
        return indices, samples, weights
    
    def update_priorities(self, indices, errors):
        """更新采样经验的优先级"""
        for idx, error in zip(indices, errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha