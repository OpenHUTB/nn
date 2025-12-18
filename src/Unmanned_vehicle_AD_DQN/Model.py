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
    Flatten, Dropout, BatchNormalization, MaxPooling2D, Multiply, Add, Lambda, Subtract, Reshape, LayerNormalization, \
    SpatialDropout2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow.keras.backend as backend
from threading import Thread

# 导入本地模块
from TrainingStrategies import PrioritizedReplayBuffer

# 导入超参数
try:
    from Hyperparameters import *
except ImportError:
    DISCOUNT = 0.97
    MEMORY_FRACTION = 0.35
    REPLAY_MEMORY_SIZE = 10000
    MIN_REPLAY_MEMORY_SIZE = 3000
    MINIBATCH_SIZE = 64
    PREDICTION_BATCH_SIZE = 1
    TRAINING_BATCH_SIZE = 16
    UPDATE_TARGET_EVERY = 20
    LEARNING_RATE = 0.00005
    IM_HEIGHT = 120
    IM_WIDTH = 160
    MODEL_NAME = "YY_Optimized_v3"

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

# 注意力机制模块
class AttentionModule:
    @staticmethod
    def channel_attention(input_feature, ratio=8):
        """通道注意力机制"""
        channel_axis = -1
        channel = input_feature.shape[channel_axis]
        
        shared_layer_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', 
                                 use_bias=True, bias_initializer='zeros')
        shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, 
                                 bias_initializer='zeros')
        
        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)
        
        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)
        
        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        
        return Multiply()([input_feature, cbam_feature])
    
    @staticmethod
    def spatial_attention(input_feature):
        """空间注意力机制"""
        avg_pool = tf.reduce_mean(input_feature, axis=3, keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=3, keepdims=True)
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', 
                          kernel_initializer='he_normal', use_bias=False)(concat)
        
        return Multiply()([input_feature, attention])
    
    @staticmethod
    def cbam_block(input_feature, ratio=8):
        """完整的CBAM注意力模块"""
        x = AttentionModule.channel_attention(input_feature, ratio)
        x = AttentionModule.spatial_attention(x)
        return x


class EnhancedStateProcessor:
    """增强状态处理器 - 处理多模态输入"""
    
    @staticmethod
    def process_state(state_dict):
        """处理增强的状态字典"""
        # 提取图像
        image_state = state_dict['image']
        
        # 提取其他状态信息
        location = state_dict.get('location', np.zeros(2))  # 2维
        speed = state_dict.get('speed', np.array([0]))      # 1维
        heading = state_dict.get('heading', np.array([0]))  # 1维
        last_action = state_dict.get('last_action', np.array([1]))  # 1维
        
        # 归一化处理
        location_norm = location / np.array([200, 200])  # 假设最大范围
        speed_norm = speed / 100.0  # 假设最大速度100km/h
        heading_norm = heading / 180.0  # 归一化到[-1, 1]
        
        # 动作历史向量
        action_history = np.zeros(5)
        if last_action.size > 0:
            action_idx = int(last_action[0])
            if 0 <= action_idx < 5:
                action_history[action_idx] = 1.0
        
        # 合并状态向量 - 总共10维
        vector_state = np.concatenate([
            location_norm,      # 2维
            speed_norm,         # 1维
            heading_norm,       # 1维
            action_history      # 5维
        ])
        
        # 填充到10维
        if len(vector_state) != 10:
            vector_state = np.pad(vector_state, (0, 10 - len(vector_state)), 'constant')
        
        return image_state, vector_state
    
    @staticmethod
    def batch_process(states):
        """批量处理状态"""
        batch_images = []
        batch_vectors = []
        
        for state in states:
            if isinstance(state, dict):
                img, vec = EnhancedStateProcessor.process_state(state)
            else:
                img = state
                vec = np.zeros(10)  # 10维向量
            
            batch_images.append(img)
            batch_vectors.append(vec)
        
        return np.array(batch_images), np.array(batch_vectors)


# DQN智能体类 - 支持多模态输入
class DQNAgent:
    def __init__(self, use_dueling=True, use_per=True, use_curriculum=True, use_multi_objective=True):
        self.use_dueling = use_dueling
        self.use_per = use_per
        self.use_curriculum = use_curriculum
        self.use_multi_objective = use_multi_objective
        
        if use_dueling:
            self.model = self.create_enhanced_dueling_model()
            self.target_model = self.create_enhanced_dueling_model()
        else:
            self.model = self.create_enhanced_model()
            self.target_model = self.create_enhanced_model()
            
        self.target_model.set_weights(self.model.get_weights())

        # 经验回放缓冲区
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(max_size=REPLAY_MEMORY_SIZE)
        else:
            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # 自定义TensorBoard
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        # 训练控制标志
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
        
        # 训练策略组件
        self.curriculum_manager = None
        self.multi_objective_optimizer = None
        self.imitation_manager = None
        
        # 状态处理器
        self.state_processor = EnhancedStateProcessor()
        
        # 反应时间追踪
        self.reaction_times = deque(maxlen=100)
        self.last_obstacle_distance = float('inf')
        
    def setup_training_strategies(self, env=None):
        """设置训练策略组件"""
        if self.use_curriculum and env:
            from TrainingStrategies import CurriculumManager
            self.curriculum_manager = CurriculumManager(env)
            print("课程学习管理器已启用")
        
        if self.use_multi_objective:
            from TrainingStrategies import MultiObjectiveOptimizer
            self.multi_objective_optimizer = MultiObjectiveOptimizer()
            print("多目标优化器已启用")
        
        from TrainingStrategies import ImitationLearningManager
        self.imitation_manager = ImitationLearningManager()

    def create_enhanced_model(self):
        """创建增强的深度Q网络模型 - 支持多模态输入"""
        # 图像输入流
        image_input = Input(shape=(IM_HEIGHT, IM_WIDTH, 3), name='image_input')
        
        # 第一卷积块
        x = SeparableConv2D(32, (3, 3), strides=(1, 1), padding='same')(image_input)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 第二卷积块
        x = SeparableConv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 第三卷积块
        x = SeparableConv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # 注意力机制
        x = AttentionModule.cbam_block(x)
        
        # 第四卷积块 - 静态障碍物特征提取
        x = SeparableConv2D(256, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(0.1)(x)
        
        # 展平图像特征
        image_features = Flatten()(x)
        image_features = Dense(256, activation='relu')(image_features)
        image_features = Dropout(0.3)(image_features)
        
        # 向量输入流（位置、速度、方向等）- 10维
        vector_input = Input(shape=(10,), name='vector_input')
        
        # 向量处理
        vector_features = Dense(64, activation='relu')(vector_input)
        vector_features = BatchNormalization()(vector_features)
        vector_features = Dense(32, activation='relu')(vector_features)
        vector_features = Dropout(0.2)(vector_features)
        
        # 合并特征
        merged = Concatenate()([image_features, vector_features])
        
        # 决策层
        x = Dense(256, activation='relu')(merged)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # 输出层 - 5个动作
        outputs = Dense(5, activation='linear')(x)
        
        # 创建模型
        model = Model(inputs=[image_input, vector_input], outputs=outputs)
        
        # 编译模型
        model.compile(loss="huber", 
                     optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), 
                     metrics=["mae"])
        return model
    
    def create_enhanced_dueling_model(self):
        """创建增强的Dueling DQN模型"""
        # 图像输入流
        image_input = Input(shape=(IM_HEIGHT, IM_WIDTH, 3), name='image_input')
        
        # 共享的图像特征提取
        x = SeparableConv2D(32, (3, 3), strides=(1, 1), padding='same')(image_input)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = SeparableConv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = SeparableConv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = AttentionModule.cbam_block(x)
        
        x = SeparableConv2D(256, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(0.1)(x)
        
        # 展平图像特征
        image_features = Flatten()(x)
        image_features = Dense(256, activation='relu')(image_features)
        image_features = Dropout(0.3)(image_features)
        
        # 向量输入流 - 10维
        vector_input = Input(shape=(10,), name='vector_input')
        
        # 向量处理
        vector_features = Dense(64, activation='relu')(vector_input)
        vector_features = BatchNormalization()(vector_features)
        vector_features = Dense(32, activation='relu')(vector_features)
        vector_features = Dropout(0.2)(vector_features)
        
        # 合并特征
        merged = Concatenate()([image_features, vector_features])
        
        # 共享层
        shared = Dense(256, activation='relu')(merged)
        shared = Dropout(0.3)(shared)
        shared = Dense(128, activation='relu')(shared)
        
        # 价值流 (V(s))
        value_stream = Dense(64, activation='relu')(shared)
        value_stream = Dropout(0.2)(value_stream)
        value = Dense(1, activation='linear', name='value')(value_stream)
        
        # 优势流 (A(s,a))
        advantage_stream = Dense(64, activation='relu')(shared)
        advantage_stream = Dropout(0.2)(advantage_stream)
        advantage = Dense(5, activation='linear', name='advantage')(advantage_stream)
        
        # 合并: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        mean_advantage = Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        advantage_centered = Subtract()([advantage, mean_advantage])
        q_values = Add()([value, advantage_centered])
        
        # 创建模型
        model = Model(inputs=[image_input, vector_input], outputs=q_values)
        
        # 编译模型
        model.compile(loss="huber", 
                     optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), 
                     metrics=["mae"])
        
        return model

    def update_replay_memory(self, transition, reaction_time=None):
        """更新经验回放缓冲区"""
        if reaction_time is not None:
            self.reaction_times.append(reaction_time)
            
            if reaction_time > 1.0:
                transition = list(transition)
                transition[2] -= 0.5
                transition = tuple(transition)
        
        if self.use_per:
            self.replay_buffer.add(transition, error=1.0)
        else:
            self.replay_memory.append(transition)

    def minibatch_chooser(self):
        """改进的经验采样策略"""
        if self.use_per:
            if len(self.replay_buffer) < MIN_REPLAY_MEMORY_SIZE:
                return []
                
            indices, samples, weights = self.replay_buffer.sample(MINIBATCH_SIZE)
            return samples
        else:
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return random.sample(self.replay_memory, min(len(self.replay_memory), MINIBATCH_SIZE))
                
            # 分类经验样本
            static_collision_samples = []
            danger_samples = []
            collision_samples = []
            positive_samples = []
            neutral_samples = []
            
            for sample in self.replay_memory:
                state, action, reward, next_state, done = sample
                
                if done and reward < -20:
                    static_collision_samples.append(sample)
                elif done and reward < -5:
                    collision_samples.append(sample)
                elif reward < -2 and not done:
                    danger_samples.append(sample)
                elif reward > 2:
                    positive_samples.append(sample)
                else:
                    neutral_samples.append(sample)
            
            # 平衡采样
            batch = []
            
            # 采样静态障碍物碰撞经验
            num_static = min(len(static_collision_samples), MINIBATCH_SIZE // 5)
            batch.extend(random.sample(static_collision_samples, num_static))
            
            # 采样行人碰撞经验
            num_collision = min(len(collision_samples), MINIBATCH_SIZE * 3 // 20)
            batch.extend(random.sample(collision_samples, num_collision))
            
            # 采样危险经验
            num_danger = min(len(danger_samples), MINIBATCH_SIZE // 4)
            batch.extend(random.sample(danger_samples, num_danger))
            
            # 采样成功避障经验
            num_positive = min(len(positive_samples), MINIBATCH_SIZE // 4)
            batch.extend(random.sample(positive_samples, num_positive))
            
            # 用中性经验补全批次
            remaining = MINIBATCH_SIZE - len(batch)
            if remaining > 0:
                batch.extend(random.sample(neutral_samples, min(remaining, len(neutral_samples))))
            
            if len(batch) < MINIBATCH_SIZE:
                additional = MINIBATCH_SIZE - len(batch)
                batch.extend(random.sample(self.replay_memory, additional))
                
            random.shuffle(batch)
            return batch

    def train(self):
        """训练DQN网络"""
        if self.use_per:
            if len(self.replay_buffer) < MIN_REPLAY_MEMORY_SIZE:
                return
                
            indices, minibatch, weights = self.replay_buffer.sample(MINIBATCH_SIZE)
            if len(minibatch) == 0:
                return
        else:
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return
                
            minibatch = self.minibatch_chooser()
            weights = np.ones(len(minibatch))

        # 处理批次数据
        batch_images = []
        batch_vectors = []
        next_batch_images = []
        next_batch_vectors = []
        
        for transition in minibatch:
            current_state, action, reward, new_state, done = transition
            
            # 处理当前状态
            if isinstance(current_state, dict):
                img, vec = self.state_processor.process_state(current_state)
            else:
                img = current_state
                vec = np.zeros(10)  # 10维向量
            
            batch_images.append(img)
            batch_vectors.append(vec)
            
            # 处理下一个状态
            if isinstance(new_state, dict):
                next_img, next_vec = self.state_processor.process_state(new_state)
            else:
                next_img = new_state
                next_vec = np.zeros(10)
            
            next_batch_images.append(next_img)
            next_batch_vectors.append(next_vec)
        
        # 归一化图像
        batch_images = np.array(batch_images) / 255
        next_batch_images = np.array(next_batch_images) / 255
        batch_vectors = np.array(batch_vectors)
        next_batch_vectors = np.array(next_batch_vectors)
        
        # 预测Q值
        current_qs_list = self.model.predict([batch_images, batch_vectors], 
                                            batch_size=PREDICTION_BATCH_SIZE, 
                                            verbose=0)
        future_qs_list = self.target_model.predict([next_batch_images, next_batch_vectors], 
                                                  batch_size=PREDICTION_BATCH_SIZE, 
                                                  verbose=0)

        x_images = []
        x_vectors = []
        y = []
        errors = []

        # 计算目标Q值
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index].copy()
            old_q = current_qs[action]
            current_qs[action] = new_q
            
            td_error = abs(new_q - old_q)
            errors.append(td_error)

            # 处理当前状态用于训练
            if isinstance(current_state, dict):
                img, vec = self.state_processor.process_state(current_state)
            else:
                img = current_state
                vec = np.zeros(10)
            
            x_images.append(img)
            x_vectors.append(vec)
            y.append(current_qs)

        # PER: 更新优先级
        if self.use_per and len(errors) > 0:
            self.replay_buffer.update_priorities(indices, errors)

        # 记录日志判断
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        # 训练模型
        self.model.fit([np.array(x_images)/255, np.array(x_vectors)], 
                      np.array(y), 
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
        dummy_image = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        dummy_vector = np.random.uniform(size=(1, 10)).astype(np.float32)
        y = np.random.uniform(size=(1, 5)).astype(np.float32)

        self.model.fit([dummy_image, dummy_vector], y, verbose=False, batch_size=1)
        self.training_initialized = True

        # 持续训练循环
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

    def get_qs(self, state):
        """获取状态的Q值"""
        if isinstance(state, dict):
            image_state, vector_state = self.state_processor.process_state(state)
            image_input = np.array(image_state).reshape(-1, *image_state.shape) / 255
            vector_input = np.array(vector_state).reshape(-1, *vector_state.shape)
            return self.model.predict([image_input, vector_input], verbose=0)[0]
        else:
            # 向后兼容
            image_input = np.array(state).reshape(-1, *state.shape) / 255
            dummy_vector = np.zeros((1, 10))
            return self.model.predict([image_input, dummy_vector], verbose=0)[0]
    
    def get_average_reaction_time(self):
        """获取平均反应时间"""
        if len(self.reaction_times) > 0:
            return np.mean(self.reaction_times)
        return 0