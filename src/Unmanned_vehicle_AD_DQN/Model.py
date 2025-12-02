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
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, \
    Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow.keras.backend as backend
from threading import Thread
from Environment import *
from Hyperparameters import *


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


# DQN智能体类
class DQNAgent:
    def __init__(self):
        # 创建主网络和目标网络
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # 经验回放缓冲区
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # 自定义TensorBoard
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0  # 目标网络更新计数器
        self.graph = tf.compat.v1.get_default_graph()

        # 训练控制标志
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        """创建深度Q网络模型 - 扩展为5个输出"""
        model = Sequential()
        
        # 第一卷积块
        model.add(Conv2D(32, (5, 5), strides=(2, 2), input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())  # 批归一化
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # 第二卷积块
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # 第三卷积块
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # 第四卷积块
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        # 展平层
        model.add(Flatten())
        
        # 全连接层
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))  # 防止过拟合
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        
        # 输出层 - 扩展为5个动作: 0-减速, 1-保持, 2-加速, 3-左转, 4-右转
        model.add(Dense(5, activation='linear'))
        
        # 编译模型，使用Huber损失和Adam优化器
        model.compile(loss="huber", optimizer=Adam(lr=LEARNING_RATE), metrics=["mae"])
        return model

    def update_replay_memory(self, transition):
        """更新经验回放缓冲区"""
        # transition = (当前状态, 动作, 奖励, 新状态, 完成标志)
        self.replay_memory.append(transition)

    def minibatch_chooser(self):
        """改进的经验采样策略"""
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
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # 选择小批量经验
        minibatch = self.minibatch_chooser()

        # 准备训练数据
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states, batch_size=PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states, batch_size=PREDICTION_BATCH_SIZE)

        x = []  # 输入状态
        y = []  # 目标Q值

        # 计算目标Q值
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                # 使用贝尔曼方程计算目标Q值
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward  # 终止状态

            current_qs = current_qs_list[index]
            current_qs[action] = new_q  # 更新对应动作的Q值

            x.append(current_state)
            y.append(current_qs)

        # 记录日志判断
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        # 训练模型
        self.model.fit(np.array(x) / 255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
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