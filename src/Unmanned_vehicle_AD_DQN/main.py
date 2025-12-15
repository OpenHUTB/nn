# main.py
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
    Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow.keras.backend as backend
from threading import Thread

from tqdm import tqdm

import Hyperparameters
from Environment import *
from Model import *
from Hyperparameters import *

if __name__ == '__main__':
    FPS = 60  # 帧率
    ep_rewards = [-200]  # 存储每轮奖励

    # 为了结果可重复性（注释掉）
    # random.seed(1)
    # np.random.seed(1)
    # tf.compat.v1.set_random_seed(1)

    # GPU内存配置，主要用于多智能体训练
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # 创建模型保存目录
    if not os.path.isdir('models'):
        os.makedirs('models')

    # 创建智能体和环境
    agent = DQNAgent()
    env = CarEnv()

    # 启动训练线程并等待训练初始化完成
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # 预热Q网络
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # 训练统计变量
    best_score = -float('inf')  # 最佳得分
    success_count = 0  # 成功次数计数
    scores = []  # 存储每轮得分
    avg_scores = []  # 存储平均得分
    
    # 迭代训练轮次
    epds = []
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []  # 重置碰撞历史
        agent.tensorboard.step = episode  # 设置TensorBoard步数

        # 课程学习 - 随训练进度调整难度
        if episode > EPISODES // 2:
            # 训练后期增加行人数量以提高难度
            env.spawn_pedestrians_general(8, True)  # 减少数量
            env.spawn_pedestrians_general(4, False)  # 减少数量
        else:
            # 训练前期减少行人数量以降低难度
            env.spawn_pedestrians_general(6, True)  # 减少数量
            env.spawn_pedestrians_general(3, False)  # 减少数量

        # 重置每轮统计 - 重置得分和步数
        score = 0
        step = 1

        # 重置环境并获取初始状态
        current_state = env.reset()

        # 重置完成标志并开始迭代直到本轮结束
        done = False
        episode_start = time.time()

        # 单次episode内的最大步数限制
        max_steps_per_episode = SECONDS_PER_EPISODE * FPS

        # 仅在给定秒数内运行
        while not done and step < max_steps_per_episode:

            # 选择动作策略
            if np.random.random() > Hyperparameters.EPSILON:
                # 从Q网络获取动作（利用）
                qs = agent.get_qs(current_state)
                action = np.argmax(qs)
                print(f'动作: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}] {action}')
            else:
                # 随机选择动作（探索）- 扩展为5个动作
                action = np.random.randint(0, 5)
                # 添加延迟以匹配60FPS
                time.sleep(1 / FPS)

            # 更频繁的状态更新
            if step % 5 == 0:
                new_state, reward, done, _ = env.step(action)
                
                score += reward  # 累加奖励
                agent.update_replay_memory((current_state, action, reward, new_state, done))  # 更新经验回放
                current_state = new_state  # 更新当前状态

            step += 1

            if done:
                break

        # 本轮结束 - 销毁所有actor
        env.cleanup_actors()

        # 更新成功计数
        if score > 5:  # 成功完成的阈值
            success_count += 1
        
        # 动态保存最佳模型
        if score > best_score:
            best_score = score
            agent.model.save(f'models/{MODEL_NAME}_best_{score:.2f}.model')

        # 记录得分统计
        scores.append(score)
        avg_scores.append(np.mean(scores[-10:]))  # 计算最近10轮平均分

        # 定期聚合统计信息
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = np.mean(scores[-AGGREGATE_STATS_EVERY:])  # 平均奖励
            min_reward = min(scores[-AGGREGATE_STATS_EVERY:])  # 最小奖励
            max_reward = max(scores[-AGGREGATE_STATS_EVERY:])  # 最大奖励
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=Hyperparameters.EPSILON)

            # 保存模型，仅当最小奖励达到设定值时
            if min_reward >= MIN_REWARD and (episode not in epds):
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        epds.append(episode)
        print('轮次: ', episode, '得分 %.2f' % score, '成功次数:', success_count)
        
        # 衰减探索率
        if Hyperparameters.EPSILON > Hyperparameters.MIN_EPSILON:
            Hyperparameters.EPSILON *= Hyperparameters.EPSILON_DECAY
            Hyperparameters.EPSILON = max(Hyperparameters.MIN_EPSILON, Hyperparameters.EPSILON)

    # 设置训练线程终止标志并等待其结束
    agent.terminate = True
    trainer_thread.join()
    # 保存最终模型
    agent.model.save(
        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # 绘制训练曲线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(scores)  # 得分曲线
    plt.plot(avg_scores)  # 平均得分曲线
    plt.ylabel('得分')
    plt.xlabel('训练轮次')
    plt.show()