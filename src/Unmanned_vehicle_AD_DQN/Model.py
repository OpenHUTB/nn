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
    Flatten, Dropout, BatchNormalization, MaxPooling2D, Multiply, Add, Lambda, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow.keras.backend as backend
from threading import Thread
from Environment import *
from Hyperparameters import *
import pickle
import json
from datetime import datetime


# 自定义TensorBoard类
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log_write_dir = self.log_dir
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

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
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


# 优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, max_size=REPLAY_MEMORY_SIZE, alpha=0.6, beta_start=0.4, beta_frames=100000):
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
            
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        """从缓冲区中采样一批经验"""
        if len(self.buffer) == 0:
            return [], [], [], []
            
        # 计算采样概率
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
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


# 课程学习管理器
class CurriculumManager:
    def __init__(self, env):
        self.env = env
        self.current_stage = 0
        self.stage_thresholds = [0.3, 0.5, 0.7, 0.85]  # 成功率阈值
        self.stage_configs = [
            # 阶段0: 入门
            {
                'pedestrian_cross': 4,      # 十字路口行人数量
                'pedestrian_normal': 2,     # 普通路段行人数量
                'pedestrian_speed_min': 0.5,  # 行人最低速度
                'pedestrian_speed_max': 1.0,  # 行人最高速度
                'max_episode_steps': 1200,   # 最大步数 (20秒 * 60FPS)
                'success_threshold': 0.3     # 进入下一阶段成功率
            },
            # 阶段1: 初级
            {
                'pedestrian_cross': 6,
                'pedestrian_normal': 3,
                'pedestrian_speed_min': 0.7,
                'pedestrian_speed_max': 1.3,
                'max_episode_steps': 1800,   # 30秒
                'success_threshold': 0.5
            },
            # 阶段2: 中级
            {
                'pedestrian_cross': 8,
                'pedestrian_normal': 4,
                'pedestrian_speed_min': 0.8,
                'pedestrian_speed_max': 1.5,
                'max_episode_steps': 2400,   # 40秒
                'success_threshold': 0.7
            },
            # 阶段3: 高级 (正常难度)
            {
                'pedestrian_cross': 10,
                'pedestrian_normal': 5,
                'pedestrian_speed_min': 1.0,
                'pedestrian_speed_max': 2.0,
                'max_episode_steps': 3600,   # 60秒
                'success_threshold': 0.85
            },
            # 阶段4: 专家 (挑战)
            {
                'pedestrian_cross': 12,
                'pedestrian_normal': 6,
                'pedestrian_speed_min': 1.2,
                'pedestrian_speed_max': 2.5,
                'max_episode_steps': 3600,
                'success_threshold': 0.9
            }
        ]
        
        # 训练历史
        self.success_history = deque(maxlen=20)  # 记录最近20轮的成功情况
        self.reward_history = deque(maxlen=50)   # 记录最近50轮的奖励
        
    def update_stage(self, success, reward):
        """更新训练阶段"""
        # 记录历史
        self.success_history.append(1 if success else 0)
        self.reward_history.append(reward)
        
        # 计算最近成功率
        if len(self.success_history) >= 10:
            success_rate = sum(self.success_history) / len(self.success_history)
            avg_reward = np.mean(self.reward_history) if self.reward_history else 0
            
            print(f"课程学习 - 当前阶段: {self.current_stage}, 成功率: {success_rate:.2f}, 平均奖励: {avg_reward:.2f}")
            
            # 检查是否可以进入下一阶段
            if self.current_stage < len(self.stage_configs) - 1:
                next_stage_threshold = self.stage_configs[self.current_stage]['success_threshold']
                if success_rate >= next_stage_threshold and avg_reward > 5:
                    self.current_stage += 1
                    print(f"🎉 课程学习: 进阶到阶段 {self.current_stage}!")
                    return True
                    
            # 如果表现太差，退回上一阶段
            if self.current_stage > 0 and success_rate < 0.2:
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
        # 注意：这里需要修改Environment.py中的行人生成逻辑来支持这些参数
        # 暂时返回配置，由外部调用者处理
        return config


# 多目标优化器
class MultiObjectiveOptimizer:
    def __init__(self):
        # 定义优化目标及其权重（可动态调整）
        self.objectives = {
            'safety': {
                'weight': 0.4,
                'description': '安全避障和避免碰撞',
                'metrics': ['collision_avoidance', 'pedestrian_distance']
            },
            'efficiency': {
                'weight': 0.25,
                'description': '快速到达目的地',
                'metrics': ['progress_speed', 'total_time']
            },
            'comfort': {
                'weight': 0.2,
                'description': '平稳驾驶体验',
                'metrics': ['smoothness', 'steering_changes']
            },
            'rule_following': {
                'weight': 0.15,
                'description': '遵守交通规则',
                'metrics': ['lane_keeping', 'speed_limit']
            }
        }
        
        # 指标跟踪
        self.metrics_history = {
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
        
        # 特殊惩罚项
        if metrics.get('collision', False):
            composite -= 10
        if metrics.get('off_road', False):
            composite -= 5
        if metrics.get('dangerous_action', False):
            composite -= 3
            
        return composite
    
    def _normalize_metric(self, value, metric_name):
        """归一化指标值到[0, 1]范围"""
        # 不同指标的归一化方式不同
        normalization_rules = {
            'safety': lambda x: min(max(x / 10, 0), 1),  # 假设安全分满分10
            'efficiency': lambda x: min(max(x / 100, 0), 1),  # 效率分满分100
            'comfort': lambda x: min(max((x + 5) / 10, 0), 1),  # 舒适度[-5, 5] -> [0, 1]
            'rule_following': lambda x: min(max(x, 0), 1)  # 规则遵循度[0, 1]
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
                adjustment = 0.05
                self.objectives[worst_obj]['weight'] += adjustment
                self.objectives[best_obj]['weight'] -= adjustment
                
                # 确保权重总和为1
                total = sum(obj['weight'] for obj in self.objectives.values())
                for obj in self.objectives:
                    self.objectives[obj]['weight'] /= total
                
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
    
    def collect_expert_demonstration(self, env, num_episodes=10):
        """收集专家示范数据（可以手动控制或使用规则控制器）"""
        print(f"开始收集专家示范数据 ({num_episodes}个episodes)...")
        
        demonstrations = []
        
        for episode in range(num_episodes):
            print(f"收集专家示范 Episode {episode + 1}/{num_episodes}")
            
            state = env.reset(episode)
            done = False
            episode_data = []
            
            while not done:
                # 这里可以使用规则控制器或手动控制
                # 示例：简单的规则控制器
                action = self._rule_based_controller(env)
                
                new_state, reward, done, _ = env.step(action)
                
                # 保存示范数据
                episode_data.append({
                    'state': state.copy(),
                    'action': action,
                    'reward': reward,
                    'next_state': new_state.copy(),
                    'done': done
                })
                
                state = new_state
            
            demonstrations.extend(episode_data)
            env.cleanup_actors()
        
        # 保存专家数据
        self.expert_data = demonstrations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"expert_data_{timestamp}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(demonstrations, f)
        
        print(f"专家示范数据已保存到: {save_path}, 共 {len(demonstrations)} 条记录")
        return True
    
    def _rule_based_controller(self, env):
        """基于规则的控制器（作为专家示范）"""
        # 获取车辆状态
        vehicle_location = env.vehicle.get_location()
        velocity = env.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
        
        # 简单规则：保持速度在20-40 km/h，避免障碍物
        if speed_kmh < 20:
            return 2  # 加速
        elif speed_kmh > 40:
            return 0  # 减速
        else:
            # 检查前方障碍物
            has_obstacle_ahead = self._check_obstacle_ahead(env)
            if has_obstacle_ahead:
                return 0  # 减速
            else:
                return 1  # 保持
        
        return 1  # 默认保持
    
    def _check_obstacle_ahead(self, env):
        """检查前方是否有障碍物（简化版本）"""
        # 这里可以添加更复杂的障碍物检测逻辑
        # 暂时返回False
        return False
    
    def pretrain_with_behavioral_cloning(self, model, epochs=20):
        """使用行为克隆进行预训练"""
        if not self.expert_data:
            print("没有专家数据可用，跳过预训练")
            return model
        
        print(f"开始行为克隆预训练 ({epochs}个epochs)...")
        
        # 准备训练数据
        states = []
        actions = []
        
        for demo in self.expert_data:
            states.append(demo['state'])
            actions.append(demo['action'])
        
        # 将状态归一化
        states = np.array(states) / 255.0
        
        # 将动作转换为one-hot编码
        actions_onehot = tf.keras.utils.to_categorical(actions, num_classes=5)
        
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
        
        # 训练模型模仿专家行为
        history = model.fit(
            states, actions_onehot,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
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
    
    def train_with_dagger(self, model, env, iterations=5, episodes_per_iter=5):
        """使用DAgger算法进行训练"""
        print(f"开始DAgger训练 ({iterations}次迭代，每次{episodes_per_iter}个episodes)...")
        
        aggregated_data = self.expert_data.copy()
        
        for iteration in range(iterations):
            print(f"\nDAgger 迭代 {iteration + 1}/{iterations}")
            
            # 使用当前策略收集数据
            new_demos = []
            
            for episode in range(episodes_per_iter):
                print(f"  收集数据 Episode {episode + 1}/{episodes_per_iter}")
                
                state = env.reset(episode)
                done = False
                
                while not done:
                    # 使用当前策略选择动作
                    qs = model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]
                    action = np.argmax(qs)
                    
                    # 执行动作
                    new_state, reward, done, _ = env.step(action)
                    
                    # 专家纠正（这里可以添加专家纠正逻辑）
                    # 如果策略动作与专家建议不同，使用专家动作
                    expert_action = self._rule_based_controller(env)
                    
                    # 保存数据（使用专家纠正后的动作）
                    new_demos.append({
                        'state': state.copy(),
                        'action': expert_action,  # 使用专家动作
                        'reward': reward,
                        'next_state': new_state.copy(),
                        'done': done
                    })
                    
                    state = new_state
                
                env.cleanup_actors()
            
            # 合并数据
            aggregated_data.extend(new_demos)
            
            # 在合并数据上重新训练
            states = [d['state'] for d in aggregated_data]
            actions = [d['action'] for d in aggregated_data]
            
            states = np.array(states) / 255.0
            actions_onehot = tf.keras.utils.to_categorical(actions, num_classes=5)
            
            # 训练模型
            history = model.fit(
                states, actions_onehot,
                batch_size=32,
                epochs=10,
                validation_split=0.1,
                verbose=0
            )
            
            print(f"  训练完成 - 准确率: {history.history['accuracy'][-1]:.3f}")
        
        print("DAgger训练完成!")
        return model


# DQN智能体类 - 升级版（整合训练策略）
class DQNAgent:
    def __init__(self, use_dueling=True, use_per=True, use_curriculum=True, use_multi_objective=True):
        # 创建主网络和目标网络
        self.use_dueling = use_dueling
        self.use_per = use_per
        self.use_curriculum = use_curriculum
        self.use_multi_objective = use_multi_objective
        
        if use_dueling:
            self.model = self.create_dueling_model()
            self.target_model = self.create_dueling_model()
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()
            
        self.target_model.set_weights(self.model.get_weights())

        # 经验回放缓冲区 - 使用PER或标准缓冲区
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(max_size=REPLAY_MEMORY_SIZE)
        else:
            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # 自定义TensorBoard
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0  # 目标网络更新计数器

        # 训练控制标志
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
        
        # 训练策略组件
        self.curriculum_manager = None
        self.multi_objective_optimizer = None
        self.imitation_manager = None
        
    def setup_training_strategies(self, env=None):
        """设置训练策略组件"""
        if self.use_curriculum and env:
            self.curriculum_manager = CurriculumManager(env)
            print("课程学习管理器已启用")
        
        if self.use_multi_objective:
            self.multi_objective_optimizer = MultiObjectiveOptimizer()
            print("多目标优化器已启用")
        
        # 模仿学习管理器（需要时手动启用）
        self.imitation_manager = ImitationLearningManager()

    def create_model(self):
        """创建标准深度Q网络模型"""
        # 使用函数式API
        inputs = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        
        # 第一卷积块
        x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 第二卷积块
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 第三卷积块
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 空间注意力机制
        attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        x = Multiply()([x, attention])
        
        # 展平层
        x = Flatten()(x)
        
        # 全连接层
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # 输出层 - 5个动作
        outputs = Dense(5, activation='linear')(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(loss="huber", optimizer=Adam(learning_rate=LEARNING_RATE), metrics=["mae"])
        return model
    
    def create_dueling_model(self):
        """创建Dueling DQN模型架构"""
        inputs = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        
        # 共享的特征提取层
        # 第一卷积块
        x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 第二卷积块
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 第三卷积块
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 空间注意力机制
        attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        x = Multiply()([x, attention])
        
        # 展平层
        x = Flatten()(x)
        
        # 共享的全连接层
        shared = Dense(512, activation='relu')(x)
        shared = Dropout(0.3)(shared)
        shared = Dense(256, activation='relu')(shared)
        
        # 价值流 (V(s))
        value_stream = Dense(128, activation='relu')(shared)
        value_stream = Dropout(0.2)(value_stream)
        value = Dense(1, activation='linear', name='value')(value_stream)
        
        # 优势流 (A(s,a))
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
        model.compile(loss="huber", optimizer=Adam(learning_rate=LEARNING_RATE), metrics=["mae"])
        
        return model

    def update_replay_memory(self, transition):
        """更新经验回放缓冲区"""
        # transition = (当前状态, 动作, 奖励, 新状态, 完成标志)
        if self.use_per:
            # PER: 初始添加时使用最大优先级
            self.replay_buffer.add(transition, error=1.0)  # 初始误差设为1.0
        else:
            self.replay_memory.append(transition)

    def minibatch_chooser(self):
        """改进的经验采样策略"""
        if self.use_per:
            # PER采样
            if len(self.replay_buffer) < MIN_REPLAY_MEMORY_SIZE:
                return [], [], [], []
                
            indices, samples, weights = self.replay_buffer.sample(MINIBATCH_SIZE)
            return indices, samples, weights
        else:
            # 标准采样
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return random.sample(self.replay_memory, min(len(self.replay_memory), MINIBATCH_SIZE))
                
            # 分类经验样本
            positive_samples = []    # 高奖励经验
            negative_samples = []    # 负奖励/碰撞经验
            neutral_samples = []     # 中性奖励经验
            
            for sample in self.replay_memory:
                _, _, reward, _, done = sample
                
                if done and reward < -5:  # 碰撞或严重错误
                    negative_samples.append(sample)
                elif reward > 1:  # 积极经验
                    positive_samples.append(sample)
                else:  # 中性经验
                    neutral_samples.append(sample)
            
            # 平衡采样
            batch = []
            
            # 采样负经验 (20%)
            num_negative = min(len(negative_samples), MINIBATCH_SIZE // 5)
            batch.extend(random.sample(negative_samples, num_negative))
            
            # 采样正经验 (30%)
            num_positive = min(len(positive_samples), MINIBATCH_SIZE // 3)
            batch.extend(random.sample(positive_samples, num_positive))
            
            # 用中性经验补全批次
            remaining = MINIBATCH_SIZE - len(batch)
            if remaining > 0:
                batch.extend(random.sample(neutral_samples, min(remaining, len(neutral_samples))))
            
            # 如果还不够，从整个记忆库随机采样
            if len(batch) < MINIBATCH_SIZE:
                additional = MINIBATCH_SIZE - len(batch)
                batch.extend(random.sample(self.replay_memory, additional))
                
            random.shuffle(batch)  # 打乱批次
            return batch

    def train(self):
        """训练DQN网络"""
        if self.use_per:
            if len(self.replay_buffer) < MIN_REPLAY_MEMORY_SIZE:
                return
                
            # PER: 采样并获取权重
            indices, minibatch, weights = self.replay_buffer.sample(MINIBATCH_SIZE)
            if len(minibatch) == 0:
                return
        else:
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return
                
            # 标准采样
            minibatch = self.minibatch_chooser()
            weights = np.ones(len(minibatch))  # 标准训练权重为1

        # 准备训练数据
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states, batch_size=PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states, batch_size=PREDICTION_BATCH_SIZE)

        x = []  # 输入状态
        y = []  # 目标Q值
        errors = []  # TD误差（用于PER）

        # 计算目标Q值
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                # 使用贝尔曼方程计算目标Q值
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward  # 终止状态

            current_qs = current_qs_list[index].copy()
            old_q = current_qs[action]  # 用于计算TD误差
            current_qs[action] = new_q  # 更新对应动作的Q值
            
            # 计算TD误差
            td_error = abs(new_q - old_q)
            errors.append(td_error)

            x.append(current_state)
            y.append(current_qs)

        # PER: 更新优先级
        if self.use_per and len(errors) > 0:
            self.replay_buffer.update_priorities(indices, errors)

        # 记录日志判断
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        # 训练模型（带样本权重）
        self.model.fit(np.array(x) / 255, np.array(y), 
                      batch_size=TRAINING_BATCH_SIZE, 
                      sample_weight=weights if self.use_per else None,
                      verbose=0, shuffle=False,
                      callbacks=[self.tensorboard] if log_this_step else None)

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
        y = np.random.uniform(size=(1, 5)).astype(np.float32)  # 改为5个输出

        self.model.fit(x, y, verbose=False, batch_size=1)
        self.training_initialized = True

        # 持续训练循环
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)  # 控制训练频率

    def get_qs(self, state):
        """获取状态的Q值"""
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]