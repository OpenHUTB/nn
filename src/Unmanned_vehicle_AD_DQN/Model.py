# Model.py
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, \
    Flatten, Dropout, BatchNormalization, MaxPooling2D, Multiply, Add, Lambda, Subtract, UpSampling2D, Conv2DTranspose, \
    Reshape, Layer, LayerNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow.keras.backend as backend
from threading import Thread, Lock
from Environment import *
from Hyperparameters import *
import pickle
import json
from datetime import datetime
from scipy import ndimage


# 增强的TensorBoard类
class EnhancedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log_write_dir = self.log_dir
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.lock = Lock()

    def set_model(self, model):
        self.model = model
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter
        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def update_stats(self, **stats):
        with self.lock:
            with self.writer.as_default():
                for key, value in stats.items():
                    tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


# 增强的优先经验回放缓冲区
class EnhancedPrioritizedReplayBuffer:
    def __init__(self, max_size=REPLAY_MEMORY_SIZE, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.max_size = max_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # 使用分段存储以提高效率
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.obstacle_experiences = []  # 存储避障相关经验
        self.success_experiences = []   # 存储成功经验
        
    def __len__(self):
        return len(self.buffer)
    
    def beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, experience, error=None, is_obstacle=False, is_success=False):
        """添加经验到缓冲区"""
        if error is None:
            if self.priorities:
                priority = max(self.priorities) * 0.8
            else:
                priority = 1.0
        else:
            priority = (abs(error) + 1e-5) ** self.alpha
            
        self.buffer.append(experience)
        self.priorities.append(priority)
        
        # 分类存储特殊经验
        if is_obstacle:
            self.obstacle_experiences.append((experience, error))
            if len(self.obstacle_experiences) > self.max_size // 10:
                self.obstacle_experiences.pop(0)
                
        if is_success:
            self.success_experiences.append((experience, error))
            if len(self.success_experiences) > self.max_size // 10:
                self.success_experiences.pop(0)
                
        self.frame += 1
        
    def sample(self, batch_size, obstacle_ratio=0.3, success_ratio=0.2):
        """改进的采样策略，确保包含避障和成功经验"""
        if len(self.buffer) == 0:
            return [], [], [], []
            
        # 基础采样（来自普通缓冲区）
        base_size = int(batch_size * (1 - obstacle_ratio - success_ratio))
        probs = None
        if base_size > 0:
            priorities = np.array(self.priorities, dtype=np.float32)
            if len(priorities) != len(self.buffer):
                # 如果长度不匹配，重新初始化 priorities
                self.priorities = [1.0] * len(self.buffer)
                priorities = np.array(self.priorities, dtype=np.float32)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            base_indices = np.random.choice(len(self.buffer), base_size, p=probs)
        else:
            base_indices = []
            
        # 避障经验采样
        obstacle_size = int(batch_size * obstacle_ratio)
        obstacle_indices = []
        if self.obstacle_experiences and obstacle_size > 0:
            obstacle_size = min(obstacle_size, len(self.obstacle_experiences))
            obstacle_samples = random.sample(self.obstacle_experiences, obstacle_size)
            # 找到这些样本在缓冲区中的索引（近似）
            for exp, _ in obstacle_samples:
                try:
                    idx = list(self.buffer).index(exp)
                    obstacle_indices.append(idx)
                except:
                    pass
                    
        # 成功经验采样
        success_size = int(batch_size * success_ratio)
        success_indices = []
        if self.success_experiences and success_size > 0:
            success_size = min(success_size, len(self.success_experiences))
            success_samples = random.sample(self.success_experiences, success_size)
            for exp, _ in success_samples:
                try:
                    idx = list(self.buffer).index(exp)
                    success_indices.append(idx)
                except:
                    pass
        
        # 合并所有索引
        all_indices = list(base_indices) + obstacle_indices + success_indices
        
        # 如果数量不够，用基础采样补足
        if len(all_indices) < batch_size:
            additional = batch_size - len(all_indices)
            if probs is not None:
                additional_indices = np.random.choice(len(self.buffer), additional, p=probs)
            else:
                additional_indices = np.random.choice(len(self.buffer), additional)
            all_indices.extend(additional_indices)
            
        # 获取样本
        samples = [self.buffer[i] for i in all_indices]
        
        # 计算重要性采样权重
        total = len(self.buffer)
        priorities_array = np.array(self.priorities, dtype=np.float32)
        if len(priorities_array) != total:
            # 如果长度不匹配，重新初始化
            self.priorities = [1.0] * total
            priorities_array = np.array(self.priorities, dtype=np.float32)
        
        weights = (total * (priorities_array[all_indices] ** self.alpha) / (priorities_array ** self.alpha).sum()) ** (-self.beta())
        
        # 归一化权重
        max_weight = weights.max()
        if max_weight > 0:
            weights /= max_weight
        
        return all_indices, samples, weights
    
    def update_priorities(self, indices, errors):
        """更新采样经验的优先级"""
        for idx, error in zip(indices, errors):
            if 0 <= idx < len(self.priorities):
                # 对于避障相关的经验，给予更高的优先级权重
                is_obstacle_exp = any(idx == list(self.buffer).index(exp[0]) for exp in self.obstacle_experiences if exp[0] in self.buffer)
                if is_obstacle_exp:
                    error *= 1.5  # 提高避障经验的优先级
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha


# 障碍物注意力模块
class ObstacleAttentionLayer(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ObstacleAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        # 空间注意力
        self.spatial_conv = Conv2D(self.filters, (self.kernel_size, self.kernel_size), 
                                   padding='same', activation='relu')
        self.spatial_attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid')
        
        # 通道注意力
        self.channel_gap = GlobalAveragePooling2D()
        self.channel_fc1 = Dense(self.filters // 8, activation='relu')
        self.channel_fc2 = Dense(self.filters, activation='sigmoid')
        
        super(ObstacleAttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # 空间注意力
        spatial_features = self.spatial_conv(inputs)
        spatial_attention = self.spatial_attention(spatial_features)
        
        # 通道注意力
        channel_weights = self.channel_gap(inputs)
        channel_weights = self.channel_fc1(channel_weights)
        channel_weights = self.channel_fc2(channel_weights)
        channel_weights = Reshape((1, 1, self.filters))(channel_weights)
        
        # 合并注意力
        attended = Multiply()([inputs, spatial_attention])
        attended = Multiply()([attended, channel_weights])
        
        # 残差连接
        output = Add()([inputs, attended])
        return output


# 课程学习管理器 - 增强版
class EnhancedCurriculumManager:
    def __init__(self, env):
        self.env = env
        self.current_stage = 0
        self.stage_progress = 0.0  # 阶段内进度（0-1）
        
        # 增强的阶段配置，更专注于避障训练
        self.stage_configs = [
            # 阶段0: 基础避障训练
            {
                'name': '基础避障',
                'pedestrian_cross': 2,
                'pedestrian_normal': 1,
                'pedestrian_speed_min': 0.3,
                'pedestrian_speed_max': 0.8,
                'max_episode_steps': 900,
                'success_threshold': 0.4,
                'obstacle_focus': 0.8,  # 避障训练权重
                'speed_limit': 25
            },
            # 阶段1: 简单场景
            {
                'name': '简单场景',
                'pedestrian_cross': 4,
                'pedestrian_normal': 2,
                'pedestrian_speed_min': 0.5,
                'pedestrian_speed_max': 1.2,
                'max_episode_steps': 1200,
                'success_threshold': 0.5,
                'obstacle_focus': 0.7,
                'speed_limit': 30
            },
            # 阶段2: 中等难度
            {
                'name': '中等难度',
                'pedestrian_cross': 6,
                'pedestrian_normal': 3,
                'pedestrian_speed_min': 0.7,
                'pedestrian_speed_max': 1.5,
                'max_episode_steps': 1800,
                'success_threshold': 0.6,
                'obstacle_focus': 0.6,
                'speed_limit': 35
            },
            # 阶段3: 复杂场景
            {
                'name': '复杂场景',
                'pedestrian_cross': 8,
                'pedestrian_normal': 4,
                'pedestrian_speed_min': 0.8,
                'pedestrian_speed_max': 1.8,
                'max_episode_steps': 2400,
                'success_threshold': 0.7,
                'obstacle_focus': 0.5,
                'speed_limit': 40
            },
            # 阶段4: 挑战模式
            {
                'name': '挑战模式',
                'pedestrian_cross': 10,
                'pedestrian_normal': 5,
                'pedestrian_speed_min': 1.0,
                'pedestrian_speed_max': 2.2,
                'max_episode_steps': 3000,
                'success_threshold': 0.75,
                'obstacle_focus': 0.4,
                'speed_limit': 45
            },
            # 阶段5: 专家模式
            {
                'name': '专家模式',
                'pedestrian_cross': 12,
                'pedestrian_normal': 6,
                'pedestrian_speed_min': 1.2,
                'pedestrian_speed_max': 2.5,
                'max_episode_steps': 3600,
                'success_threshold': 0.8,
                'obstacle_focus': 0.3,
                'speed_limit': 50
            }
        ]
        
        # 训练历史
        self.success_history = deque(maxlen=20)
        self.reward_history = deque(maxlen=50)
        self.obstacle_avoidance_history = deque(maxlen=30)  # 避障成功率历史
        
    def update_stage(self, success, reward, obstacle_avoidance_score=0):
        """更新训练阶段"""
        # 记录历史
        self.success_history.append(1 if success else 0)
        self.reward_history.append(reward)
        if obstacle_avoidance_score > 0:
            self.obstacle_avoidance_history.append(obstacle_avoidance_score)
        
        # 计算统计数据
        if len(self.success_history) >= 10:
            success_rate = sum(self.success_history) / len(self.success_history)
            avg_reward = np.mean(self.reward_history) if self.reward_history else 0
            obstacle_rate = np.mean(self.obstacle_avoidance_history) if self.obstacle_avoidance_history else 0
            
            # 动态更新阶段进度
            current_config = self.get_current_config()
            target_threshold = current_config['success_threshold']
            
            # 计算阶段内进度
            if target_threshold > 0:
                self.stage_progress = min(1.0, success_rate / target_threshold)
            
            # 检查是否可以进入下一阶段
            if self.current_stage < len(self.stage_configs) - 1:
                next_stage = self.current_stage + 1
                next_threshold = self.stage_configs[next_stage]['success_threshold']
                
                # 进阶条件：达到成功率阈值且有一定的避障表现
                if (success_rate >= current_config['success_threshold'] and 
                    avg_reward > 3 and 
                    obstacle_rate > 0.6):
                    
                    self.current_stage = next_stage
                    self.stage_progress = 0.0
                    print(f"🎉 课程学习: 进阶到阶段 {self.current_stage} - {self.stage_configs[self.current_stage]['name']}!")
                    return True, 'advance'
                    
            # 如果表现持续不佳，退回上一阶段
            if (self.current_stage > 0 and 
                success_rate < 0.2 and 
                len(self.success_history) >= 15):
                
                self.current_stage -= 1
                self.stage_progress = 0.5  # 退回后给予中等进度
                print(f"⚠️ 课程学习: 退回阶段 {self.current_stage} - {self.stage_configs[self.current_stage]['name']}")
                return True, 'regress'
        
        return False, 'maintain'
    
    def get_current_config(self):
        """获取当前阶段的配置"""
        return self.stage_configs[min(self.current_stage, len(self.stage_configs) - 1)]
    
    def get_stage_info(self):
        """获取阶段信息"""
        config = self.get_current_config()
        return {
            'stage': self.current_stage,
            'name': config['name'],
            'progress': self.stage_progress,
            'pedestrian_total': config['pedestrian_cross'] + config['pedestrian_normal'],
            'difficulty': self.current_stage / len(self.stage_configs)
        }


# 多目标优化器 - 增强版，专注于避障
class EnhancedMultiObjectiveOptimizer:
    def __init__(self):
        # 动态权重调整，初期更注重安全
        self.objectives = {
            'safety': {
                'weight': 0.50,  # 提高安全权重
                'description': '安全避障和避免碰撞',
                'target_value': 0.9,
                'current_performance': 0.0,
                'improvement_rate': 0.0
            },
            'efficiency': {
                'weight': 0.20,
                'description': '快速到达目的地',
                'target_value': 0.7,
                'current_performance': 0.0,
                'improvement_rate': 0.0
            },
            'comfort': {
                'weight': 0.15,
                'description': '平稳驾驶体验',
                'target_value': 0.6,
                'current_performance': 0.0,
                'improvement_rate': 0.0
            },
            'obstacle_avoidance': {  # 新增：专门针对障碍物避让
                'weight': 0.15,
                'description': '有效避让行人和建筑物',
                'target_value': 0.8,
                'current_performance': 0.0,
                'improvement_rate': 0.0
            }
        }
        
        # 指标跟踪
        self.metrics_history = {
            'safety': deque(maxlen=100),
            'efficiency': deque(maxlen=100),
            'comfort': deque(maxlen=100),
            'obstacle_avoidance': deque(maxlen=100)
        }
        
        # 避障特别奖励参数
        self.obstacle_avoidance_bonus = {
            'near_miss': 2.0,  # 成功避让奖励
            'safe_distance': 1.0,  # 保持安全距离奖励
            'collision_penalty': -10.0,  # 碰撞惩罚
            'danger_zone_penalty': -3.0  # 进入危险区域惩罚
        }
        
    def compute_composite_reward(self, metrics, obstacle_info=None):
        """计算综合奖励值，特别关注避障"""
        composite = 0
        
        # 基础目标计算
        for obj_name, obj_info in self.objectives.items():
            if obj_name in metrics:
                normalized_value = self._normalize_metric(metrics[obj_name], obj_name)
                composite += normalized_value * obj_info['weight']
                
                # 更新性能记录
                self.metrics_history[obj_name].append(normalized_value)
                self.objectives[obj_name]['current_performance'] = np.mean(
                    self.metrics_history[obj_name]) if self.metrics_history[obj_name] else 0
        
        # 特别避障奖励
        if obstacle_info:
            composite += self._compute_obstacle_avoidance_reward(obstacle_info)
        
        # 特殊惩罚项（增强）
        if metrics.get('collision', False):
            composite += self.obstacle_avoidance_bonus['collision_penalty']
        if metrics.get('off_road', False):
            composite -= 5
        if metrics.get('dangerous_action', False):
            composite -= 4
        if metrics.get('near_miss', False):  # 成功避让奖励
            composite += self.obstacle_avoidance_bonus['near_miss']
            
        return composite
    
    def _compute_obstacle_avoidance_reward(self, obstacle_info):
        """计算避障特别奖励"""
        reward = 0
        
        # 根据障碍物距离给予奖励
        min_distance = obstacle_info.get('min_distance', float('inf'))
        if min_distance < 100:  # 只考虑100米内的障碍物
            if min_distance > 15:  # 非常安全
                reward += self.obstacle_avoidance_bonus['safe_distance'] * 0.5
            elif min_distance > 10:  # 安全
                reward += self.obstacle_avoidance_bonus['safe_distance'] * 0.3
            elif min_distance > 5:  # 警告距离
                reward -= self.obstacle_avoidance_bonus['danger_zone_penalty'] * 0.5
            else:  # 危险距离
                reward -= self.obstacle_avoidance_bonus['danger_zone_penalty']
        
        # 成功避让奖励
        if obstacle_info.get('avoidance_success', False):
            reward += self.obstacle_avoidance_bonus['near_miss']
        
        return reward
    
    def _normalize_metric(self, value, metric_name):
        """归一化指标值"""
        normalization_rules = {
            'safety': lambda x: min(max(x / 10, 0), 1),
            'efficiency': lambda x: min(max(x / 100, 0), 1),
            'comfort': lambda x: min(max((x + 5) / 10, 0), 1),
            'obstacle_avoidance': lambda x: min(max(x, 0), 1)
        }
        
        if metric_name in normalization_rules:
            return normalization_rules[metric_name](value)
        return min(max(value, 0), 1)
    
    def adjust_weights(self, performance_feedback, training_stage=0):
        """动态调整权重，考虑训练阶段"""
        # 首先更新指标历史
        if performance_feedback:
            for obj_name, value in performance_feedback.items():
                if obj_name in self.metrics_history:
                    self.metrics_history[obj_name].append(value)
        
        # 根据训练阶段调整基础权重
        if training_stage < 2:  # 初期阶段更注重安全
            self.objectives['safety']['weight'] = 0.6
            self.objectives['efficiency']['weight'] = 0.15
            self.objectives['obstacle_avoidance']['weight'] = 0.15
            self.objectives['comfort']['weight'] = 0.10
        elif training_stage < 4:  # 中期平衡
            self.objectives['safety']['weight'] = 0.5
            self.objectives['efficiency']['weight'] = 0.2
            self.objectives['obstacle_avoidance']['weight'] = 0.15
            self.objectives['comfort']['weight'] = 0.15
        else:  # 后期注重效率
            self.objectives['safety']['weight'] = 0.4
            self.objectives['efficiency']['weight'] = 0.25
            self.objectives['obstacle_avoidance']['weight'] = 0.2
            self.objectives['comfort']['weight'] = 0.15
        
        # 基于性能反馈微调
        recent_performance = {}
        for obj in self.objectives:
            if len(self.metrics_history[obj]) >= 10:
                recent_avg = np.mean(list(self.metrics_history[obj])[-10:])
                recent_performance[obj] = recent_avg
        
        if recent_performance:
            # 如果某个目标表现持续低于阈值，增加其权重
            for obj_name, obj_info in self.objectives.items():
                if obj_name in recent_performance:
                    performance = recent_performance[obj_name]
                    target = obj_info['target_value']
                    
                    if performance < target * 0.7:  # 表现严重不足
                        adjustment = 0.03
                        obj_info['weight'] += adjustment
                        # 从表现最好的目标中扣除
                        best_obj = max(recent_performance, key=recent_performance.get)
                        if best_obj != obj_name:
                            self.objectives[best_obj]['weight'] -= adjustment
            
            # 确保权重总和为1
            total = sum(obj['weight'] for obj in self.objectives.values())
            for obj in self.objectives:
                self.objectives[obj]['weight'] /= total
    
    def get_performance_report(self):
        """生成性能报告"""
        report = "多目标优化性能报告 (增强避障版):\n"
        report += "=" * 60 + "\n"
        
        for obj_name, obj_info in self.objectives.items():
            history = self.metrics_history[obj_name]
            if history:
                history_list = list(history)
                avg = np.mean(history_list[-20:]) if len(history_list) >= 20 else np.mean(history_list)
                trend = "↑" if len(history_list) >= 2 and history_list[-1] > history_list[-2] else "↓"
                report += f"{obj_name:20s} 权重:{obj_info['weight']:.2f} 得分:{avg:.3f}{trend}\n"
                report += f"  目标值:{obj_info['target_value']:.2f} - {obj_info['description']}\n"
        
        # 计算总体避障成功率
        if self.metrics_history['obstacle_avoidance']:
            avoidance_rate = np.mean(self.metrics_history['obstacle_avoidance'])
            report += f"\n总体避障成功率: {avoidance_rate:.2%}\n"
        
        return report


# 模仿学习管理器 - 增强版
class EnhancedImitationLearningManager:
    def __init__(self, expert_data_path=None):
        self.expert_data_path = expert_data_path
        self.expert_data = []
        self.is_pretrained = False
        self.avoidance_demos = []  # 专门存储避障演示
        
    def load_expert_data(self, path):
        """加载专家示范数据"""
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'demonstrations' in data:
                        self.expert_data = data['demonstrations']
                        if 'avoidance_demos' in data:
                            self.avoidance_demos = data['avoidance_demos']
                    else:
                        self.expert_data = data
                print(f"已加载 {len(self.expert_data)} 条专家示范数据")
                if self.avoidance_demos:
                    print(f"已加载 {len(self.avoidance_demos)} 条避障专用演示")
                return True
            else:
                print(f"专家数据文件不存在: {path}")
                return False
        except Exception as e:
            print(f"加载专家数据失败: {e}")
            return False
    
    def collect_expert_demonstration(self, env, num_episodes=3, focus_avoidance=True):
        """收集专家示范数据，特别关注避障场景"""
        print(f"开始收集专家示范数据 ({num_episodes}个episodes)...")
        
        demonstrations = []
        avoidance_demos = []
        
        for episode in range(num_episodes):
            print(f"收集专家示范 Episode {episode + 1}/{num_episodes}")
            
            # 调整环境难度，专注于避障
            if focus_avoidance:
                config = {'pedestrian_cross': 8, 'pedestrian_normal': 4}
                env.spawn_pedestrians_with_config(config)
            
            state = env.reset(episode)
            done = False
            episode_data = []
            obstacle_encountered = False
            
            while not done:
                # 使用增强的规则控制器，特别注重避障
                action = self._enhanced_rule_based_controller(env)
                
                new_state, reward, done, _ = env.step(action)
                
                # 检查是否遇到障碍物
                min_distance = getattr(env, 'last_ped_distance', float('inf'))
                if min_distance < 8.0:
                    obstacle_encountered = True
                
                # 只保存避障相关的示范数据
                if min_distance < 8.0:  # 只在有障碍物时保存数据
                    demo_entry = {
                        'state': state.copy(),
                        'action': action,
                        'reward': reward,
                        'next_state': new_state.copy(),
                        'done': done,
                        'obstacle_nearby': min_distance < 8.0,
                        'obstacle_distance': min_distance
                    }
                    
                    episode_data.append(demo_entry)
                    avoidance_demos.append(demo_entry)
                
                state = new_state
            
            # 如果整个episode中有避障场景，保存为避障演示
            if obstacle_encountered:
                avoidance_demos.extend(episode_data)
            
            demonstrations.extend(episode_data)
            env.cleanup_actors()
        
        # 保存专家数据
        self.expert_data = avoidance_demos  # 只保存避障演示
        self.avoidance_demos = avoidance_demos
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_data = {
            'demonstrations': avoidance_demos,  # 只保存避障演示
            'avoidance_demos': avoidance_demos,
            'collection_date': timestamp,
            'num_episodes': num_episodes,
            'focus_avoidance': focus_avoidance
        }
        
        save_path = f"expert_data_enhanced_{timestamp}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"专家示范数据已保存到: {save_path}")
        print(f"共 {len(demonstrations)} 条记录，其中 {len(avoidance_demos)} 条避障专用演示")
        return True
    
    def _enhanced_rule_based_controller(self, env):
        """增强的基于规则的控制器，特别注重避障"""
        # 获取车辆状态
        vehicle_location = env.vehicle.get_location()
        velocity = env.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
        
        # 检查前方障碍物
        obstacle_info = self._check_obstacles_ahead(env)
        has_obstacle = obstacle_info['has_obstacle']
        obstacle_distance = obstacle_info['distance']
        obstacle_direction = obstacle_info['direction']  # 'left', 'right', 'center'
        
        # 避障优先级最高
        if has_obstacle:
            if obstacle_distance < 5.0:  # 紧急避让
                if obstacle_direction == 'left':
                    return 4  # 右转避让
                elif obstacle_direction == 'right':
                    return 3  # 左转避让
                else:  # 正前方
                    return 0  # 紧急制动
            elif obstacle_distance < 10.0:  # 预警距离
                if speed_kmh > 20:
                    return 0  # 减速
                elif obstacle_direction == 'center':
                    # 轻微转向避让
                    return 3 if random.random() > 0.5 else 4
                else:
                    return 1  # 保持警惕
        
        # 速度控制
        if speed_kmh < 15:
            return 2  # 加速
        elif speed_kmh > 35:
            return 0  # 减速
        else:
            return 1  # 保持
    
    def _check_obstacles_ahead(self, env):
        """检查前方障碍物，返回详细信息"""
        vehicle_location = env.vehicle.get_location()
        
        has_obstacle = False
        min_distance = float('inf')
        obstacle_direction = 'center'
        
        # 检查行人
        for walker in env.walker_list:
            if not walker.is_alive:
                continue
                
            ped_location = walker.get_location()
            dx = ped_location.x - vehicle_location.x
            dy = ped_location.y - vehicle_location.y
            
            # 只考虑前方的行人（车辆朝向为0度）
            if dx > 0 and abs(dy) < 20:  # 前方20米内
                distance = math.sqrt(dx**2 + dy**2)
                if distance < min_distance:
                    min_distance = distance
                    has_obstacle = True
                    
                    # 判断障碍物方向
                    if dy > 2:  # 在车辆右侧
                        obstacle_direction = 'right'
                    elif dy < -2:  # 在车辆左侧
                        obstacle_direction = 'left'
                    else:
                        obstacle_direction = 'center'
        
        return {
            'has_obstacle': has_obstacle,
            'distance': min_distance,
            'direction': obstacle_direction
        }
    
    def pretrain_with_behavioral_cloning(self, model, epochs=20, focus_avoidance=True):
        """使用行为克隆进行预训练，可选专注避障"""
        if not self.expert_data:
            print("没有专家数据可用，跳过预训练")
            return model
        
        print(f"开始行为克隆预训练 ({epochs}个epochs)...")
        
        # 选择训练数据
        if focus_avoidance and self.avoidance_demos:
            print(f"使用 {len(self.avoidance_demos)} 条避障专用演示进行训练")
            training_data = self.avoidance_demos
        else:
            print(f"使用 {len(self.expert_data)} 条常规演示进行训练")
            training_data = self.expert_data
        
        # 准备训练数据
        states = []
        actions = []
        
        for demo in training_data:
            states.append(demo['state'])
            actions.append(demo['action'])
        
        # 数据增强：对图像进行轻微变换以增强泛化
        augmented_states = []
        augmented_actions = []
        
        for state, action in zip(states, actions):
            # 原始数据
            augmented_states.append(state)
            augmented_actions.append(action)
            
            # 添加轻微亮度变化
            if random.random() > 0.7:
                bright_state = np.clip(state * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
                augmented_states.append(bright_state)
                augmented_actions.append(action)
            
            # 添加轻微对比度变化
            if random.random() > 0.7:
                contrast = random.uniform(0.8, 1.2)
                mean = np.mean(state)
                contrast_state = np.clip((state - mean) * contrast + mean, 0, 255).astype(np.uint8)
                augmented_states.append(contrast_state)
                augmented_actions.append(action)
        
        # 将状态归一化
        states_array = np.array(augmented_states) / 255.0
        
        # 将动作转换为one-hot编码
        actions_onehot = tf.keras.utils.to_categorical(augmented_actions, num_classes=5)
        
        # 备份原始编译设置
        original_loss = model.loss
        original_optimizer = model.optimizer
        original_metrics = model.metrics_names
        
        # 重新编译模型用于分类任务
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 添加回调函数
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # 训练模型模仿专家行为
        history = model.fit(
            states_array, actions_onehot,
            batch_size=16,
            epochs=epochs,
            validation_split=0.2,
            verbose=1,
            callbacks=callbacks
        )
        
        print(f"预训练完成 - 最终准确率: {history.history['accuracy'][-1]:.3f}")
        
        # 恢复原始编译设置
        model.compile(
            optimizer=original_optimizer,
            loss=original_loss,
            metrics=original_metrics
        )
        
        self.is_pretrained = True
        return model


# DQN智能体类 - 完全重写，专注于避障
class EnhancedDQNAgent:
    def __init__(self, use_dueling=True, use_per=True, use_curriculum=True, 
                 use_multi_objective=True, use_attention=True, use_enhanced_model=True):
        
        # 配置参数
        self.use_dueling = use_dueling
        self.use_per = use_per
        self.use_curriculum = use_curriculum
        self.use_multi_objective = use_multi_objective
        self.use_attention = use_attention
        self.use_enhanced_model = use_enhanced_model
        
        # 创建主网络和目标网络
        if use_enhanced_model:
            self.model = self.create_enhanced_model()
            self.target_model = self.create_enhanced_model()
        elif use_attention:
            self.model = self.create_attention_model()
            self.target_model = self.create_attention_model()
        elif use_dueling:
            self.model = self.create_dueling_model()
            self.target_model = self.create_dueling_model()
        else:
            self.model = self.create_basic_model()
            self.target_model = self.create_basic_model()
            
        self.target_model.set_weights(self.model.get_weights())

        # 经验回放缓冲区
        if use_per:
            self.replay_buffer = EnhancedPrioritizedReplayBuffer(max_size=REPLAY_MEMORY_SIZE)
        else:
            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # 自定义TensorBoard
        self.tensorboard = EnhancedTensorBoard(
            log_dir=f"logs/{MODEL_NAME}-enhanced-{int(time.time())}",
            histogram_freq=0,
            write_graph=True,
            write_images=False
        )
        
        self.target_update_counter = 0
        self.train_step_counter = 0

        # 训练控制标志
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
        self.training_paused = False
        
        # 训练策略组件
        self.curriculum_manager = None
        self.multi_objective_optimizer = None
        self.imitation_manager = None
        
        # 避障性能跟踪
        self.obstacle_avoidance_stats = {
            'success_count': 0,
            'total_encounters': 0,
            'near_misses': 0,
            'collisions': 0
        }
        
        # 锁用于线程安全
        self.training_lock = Lock()
        
    def setup_training_strategies(self, env=None):
        """设置训练策略组件"""
        if self.use_curriculum and env:
            self.curriculum_manager = EnhancedCurriculumManager(env)
            print("增强版课程学习管理器已启用")
        
        if self.use_multi_objective:
            self.multi_objective_optimizer = EnhancedMultiObjectiveOptimizer()
            print("增强版多目标优化器已启用")
        
        # 模仿学习管理器
        self.imitation_manager = EnhancedImitationLearningManager()
        print("增强版模仿学习管理器已启用")
        
    def create_basic_model(self):
        """创建基础深度Q网络模型"""
        inputs = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        
        # 特征提取层
        x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 展平层
        x = Flatten()(x)
        
        # 全连接层
        x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # 输出层
        outputs = Dense(5, activation='linear')(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            loss="huber", 
            optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            metrics=["mae"]
        )
        return model
    
    def create_dueling_model(self):
        """创建Dueling DQN模型"""
        inputs = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        
        # 共享的特征提取层
        x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 展平层
        x = Flatten()(x)
        
        # 共享的全连接层
        shared = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
        shared = Dropout(0.3)(shared)
        shared = Dense(256, activation='relu')(shared)
        
        # 价值流
        value_stream = Dense(128, activation='relu')(shared)
        value_stream = Dropout(0.2)(value_stream)
        value = Dense(1, activation='linear', name='value')(value_stream)
        
        # 优势流
        advantage_stream = Dense(128, activation='relu')(shared)
        advantage_stream = Dropout(0.2)(advantage_stream)
        advantage = Dense(5, activation='linear', name='advantage')(advantage_stream)
        
        # 合并: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        mean_advantage = Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        advantage_centered = Subtract()([advantage, mean_advantage])
        q_values = Add()([value, advantage_centered])
        
        # 创建模型
        model = Model(inputs=inputs, outputs=q_values)
        
        # 编译模型
        model.compile(
            loss="huber",
            optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            metrics=["mae"]
        )
        
        return model
    
    def create_attention_model(self):
        """创建带注意力机制的模型"""
        inputs = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        
        # 第一卷积块
        x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 第二卷积块 + 注意力
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = ObstacleAttentionLayer(64)(x)  # 添加注意力层
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 第三卷积块 + 注意力
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = ObstacleAttentionLayer(128)(x)  # 第二层注意力
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 展平层
        x = Flatten()(x)
        
        # 全连接层
        x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Dueling架构
        # 价值流
        value_stream = Dense(128, activation='relu')(x)
        value_stream = Dropout(0.2)(value_stream)
        value = Dense(1, activation='linear', name='value')(value_stream)
        
        # 优势流
        advantage_stream = Dense(128, activation='relu')(x)
        advantage_stream = Dropout(0.2)(advantage_stream)
        advantage = Dense(5, activation='linear', name='advantage')(advantage_stream)
        
        # 合并
        mean_advantage = Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        advantage_centered = Subtract()([advantage, mean_advantage])
        q_values = Add()([value, advantage_centered])
        
        # 创建模型
        model = Model(inputs=inputs, outputs=q_values)
        
        # 编译模型
        model.compile(
            loss="huber",
            optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            metrics=["mae"]
        )
        
        return model
    
    def create_enhanced_model(self):
        """创建增强版模型，专门用于避障"""
        inputs = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        
        # 多尺度特征提取
        # 分支1: 大感受野，检测远处障碍物
        branch1 = Conv2D(32, (7, 7), strides=(2, 2), padding='same')(inputs)
        branch1 = Activation('relu')(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)
        
        # 分支2: 中等感受野
        branch2 = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(inputs)
        branch2 = Activation('relu')(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)
        
        # 分支3: 小感受野，检测近处细节
        branch3 = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
        branch3 = Activation('relu')(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = MaxPooling2D(pool_size=(2, 2))(branch3)
        
        # 合并分支
        merged = Concatenate()([branch1, branch2, branch3])
        
        # 深度特征提取
        x = Conv2D(128, (3, 3), padding='same')(merged)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = ObstacleAttentionLayer(128)(x)  # 注意力机制
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = ObstacleAttentionLayer(256)(x)  # 第二层注意力
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 展平层
        x = Flatten()(x)
        
        # 密集连接层
        x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.4)(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # 双流输出（Dueling DQN）
        # 价值流
        value_stream = Dense(128, activation='relu')(x)
        value_stream = Dropout(0.2)(value_stream)
        value = Dense(1, activation='linear', name='value')(value_stream)
        
        # 优势流（特别关注避障动作）
        advantage_stream = Dense(128, activation='relu')(x)
        advantage_stream = Dropout(0.2)(advantage_stream)
        advantage = Dense(5, activation='linear', name='advantage')(advantage_stream)
        
        # 合并: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        mean_advantage = Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        advantage_centered = Subtract()([advantage, mean_advantage])
        q_values = Add()([value, advantage_centered])
        
        # 创建模型
        model = Model(inputs=inputs, outputs=q_values)
        
        # 编译模型
        optimizer = Adam(
            learning_rate=LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        model.compile(
            loss="huber",
            optimizer=optimizer,
            metrics=["mae"]
        )
        
        print("增强版模型创建完成（专门用于避障）")
        return model
    
    def update_replay_memory(self, transition, is_obstacle=False, is_success=False):
        """更新经验回放缓冲区"""
        if self.use_per:
            self.replay_buffer.add(transition, error=1.0, 
                                  is_obstacle=is_obstacle, 
                                  is_success=is_success)
        else:
            self.replay_memory.append(transition)
            
            # 如果是重要经验，额外存储
            if is_obstacle or is_success:
                self.replay_memory.append(transition)  # 重要经验重复存储
    
    def train(self):
        """训练DQN网络"""
        with self.training_lock:
            if self.training_paused:
                return
                
            if self.use_per:
                if len(self.replay_buffer) < MIN_REPLAY_MEMORY_SIZE:
                    return
                    
                # PER采样，特别关注避障经验
                indices, minibatch, weights = self.replay_buffer.sample(
                    MINIBATCH_SIZE, 
                    obstacle_ratio=0.3,  # 30%避障经验
                    success_ratio=0.2     # 20%成功经验
                )
                if len(minibatch) == 0:
                    return
            else:
                if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                    return
                    
                # 标准采样
                minibatch = random.sample(self.replay_memory, 
                                         min(MINIBATCH_SIZE, len(self.replay_memory)))
                weights = np.ones(len(minibatch))

            # 准备训练数据
            current_states = np.array([transition[0] for transition in minibatch]) / 255
            current_qs_list = self.model.predict(current_states, 
                                                batch_size=PREDICTION_BATCH_SIZE,
                                                verbose=0)

            new_current_states = np.array([transition[3] for transition in minibatch]) / 255
            future_qs_list = self.target_model.predict(new_current_states, 
                                                      batch_size=PREDICTION_BATCH_SIZE,
                                                      verbose=0)

            x = []  # 输入状态
            y = []  # 目标Q值
            errors = []  # TD误差

            # 计算目标Q值（Double DQN风格）
            for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
                if not done:
                    # Double DQN: 使用主网络选择动作，目标网络评估
                    next_qs = self.model.predict(np.array([new_state]) / 255, verbose=0)[0]
                    best_action = np.argmax(next_qs)
                    max_future_q = future_qs_list[index][best_action]
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward

                current_qs = current_qs_list[index].copy()
                old_q = current_qs[action]
                current_qs[action] = new_q
                
                # 计算TD误差
                td_error = abs(new_q - old_q)
                errors.append(td_error)

                x.append(current_state)
                y.append(current_qs)

            # PER: 更新优先级
            if self.use_per and len(errors) > 0:
                self.replay_buffer.update_priorities(indices, errors)

            # 记录日志
            log_this_step = False
            if self.tensorboard.step > self.last_logged_episode:
                log_this_step = True
                self.last_logged_episode = self.tensorboard.step

            # 训练模型
            self.model.fit(
                np.array(x) / 255, 
                np.array(y),
                batch_size=TRAINING_BATCH_SIZE,
                sample_weight=weights if self.use_per else None,
                verbose=0, 
                shuffle=False,
                callbacks=[self.tensorboard] if log_this_step else None
            )

            self.train_step_counter += 1

            # 更新目标网络
            if log_this_step:
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                print("目标网络已更新")
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
                
    def train_in_loop(self):
        """在单独线程中持续训练"""
        # 预热训练
        x = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 5)).astype(np.float32)

        self.model.fit(x, y, verbose=False, batch_size=1)
        self.training_initialized = True

        print("训练线程已启动")
        
        # 持续训练循环
        while True:
            if self.terminate:
                print("训练线程终止")
                return
                
            try:
                self.train()
                time.sleep(0.005)  # 更高的训练频率
            except Exception as e:
                print(f"训练过程中出错: {e}")
                time.sleep(0.1)

    def get_qs(self, state):
        """获取状态的Q值"""
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255, 
                                verbose=0)[0]
    
    def update_obstacle_stats(self, success, is_collision=False, is_near_miss=False):
        """更新避障统计"""
        self.obstacle_avoidance_stats['total_encounters'] += 1
        if success:
            self.obstacle_avoidance_stats['success_count'] += 1
        if is_collision:
            self.obstacle_avoidance_stats['collisions'] += 1
        if is_near_miss:
            self.obstacle_avoidance_stats['near_misses'] += 1
    
    def get_obstacle_avoidance_rate(self):
        """获取避障成功率"""
        total = self.obstacle_avoidance_stats['total_encounters']
        if total == 0:
            return 0.0
        return self.obstacle_avoidance_stats['success_count'] / total
    
    def pause_training(self):
        """暂停训练"""
        self.training_paused = True
        
    def resume_training(self):
        """恢复训练"""
        self.training_paused = False
    
    def save_model(self, path, include_stats=True):
        """保存模型"""
        try:
            # 尝试使用Keras的标准保存方法
            self.model.save(path)
            print(f"模型已保存到: {path}")
        except Exception as e:
            print(f"标准保存失败，尝试替代方法: {e}")
            try:
                # 尝试保存权重
                weights_path = path.replace('.model', '_weights.h5')
                self.model.save_weights(weights_path)
                print(f"模型权重已保存到: {weights_path}")

                # 保存模型架构
                config_path = path.replace('.model', '_config.json')
                with open(config_path, 'w') as f:
                    f.write(self.model.to_json())
                print(f"模型架构已保存到: {config_path}")
            except Exception as e2:
                print(f"权重保存也失败: {e2}")
                print("跳过模型保存，继续训练...")
        
        if include_stats:
            # 保存训练统计
            stats = {
                'obstacle_avoidance_stats': self.obstacle_avoidance_stats,
                'train_step_counter': self.train_step_counter,
                'model_config': {
                    'use_dueling': self.use_dueling,
                    'use_per': self.use_per,
                    'use_attention': self.use_attention,
                    'use_enhanced_model': self.use_enhanced_model
                }
            }
            
            stats_path = path.replace('.model', '_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
            print(f"训练统计已保存到: {stats_path}")
    
    def load_model(self, path):
        """加载模型"""
        try:
            # 定义自定义层
            custom_objects = {
                'ObstacleAttentionLayer': ObstacleAttentionLayer,
                'Add': Add,
                'Subtract': Subtract,
                'Lambda': Lambda,
                'Multiply': Multiply
            }
            
            self.model = tf.keras.models.load_model(path, custom_objects=custom_objects)
            self.target_model.set_weights(self.model.get_weights())
            print(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False