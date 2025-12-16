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
from TrainingStrategies import *


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