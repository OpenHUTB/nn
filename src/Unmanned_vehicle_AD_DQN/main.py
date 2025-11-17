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
    FPS = 60
    ep_rewards = [-200]

    # For more repetitive results
    # random.seed(1)
    # np.random.seed(1)
    # tf.compat.v1.set_random_seed(1)

    # Memory fraction, used mostly when training multiple agents
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # 添加训练统计
    best_score = -float('inf')
    success_count = 0
    scores = []
    avg_scores = []
    
    # Iterate over episodes
    epds = []
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []
        agent.tensorboard.step = episode

        # 课程学习 - 随训练进度调整难度
        if episode > EPISODES // 2:
            # 后期增加行人数量
            env.spawn_pedestrians_general(40, True)
            env.spawn_pedestrians_general(15, False)
        else:
            # 前期减少行人数量
            env.spawn_pedestrians_general(25, True)
            env.spawn_pedestrians_general(8, False)

        # Restarting episode - reset episode reward and step number
        score = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # 单次episode内的步数限制
        max_steps_per_episode = SECONDS_PER_EPISODE * FPS

        # Play for given number of seconds only
        while not done and step < max_steps_per_episode:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > Hyperparameters.EPSILON:
                # Get action from Q table
                qs = agent.get_qs(current_state)
                action = np.argmax(qs)
                print(f'Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')
            else:
                # Get random action
                action = np.random.randint(0, 3)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1 / FPS)

            # 更频繁的状态更新
            if step % 5 == 0:
                new_state, reward, done, _ = env.step(action)
                
                score += reward
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                current_state = new_state

            step += 1

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        # 更新成功计数
        if score > 5:  # 成功完成的阈值
            success_count += 1
        
        # 动态保存最佳模型
        if score > best_score:
            best_score = score
            agent.model.save(f'models/{MODEL_NAME}_best_{score:.2f}.model')

        scores.append(score)
        avg_scores.append(np.mean(scores[-10:]))

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = np.mean(scores[-AGGREGATE_STATS_EVERY:])
            min_reward = min(scores[-AGGREGATE_STATS_EVERY:])
            max_reward = max(scores[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=Hyperparameters.EPSILON)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD and (episode not in epds):
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        epds.append(episode)
        print('episode: ', episode, 'score %.2f' % score, 'success_count:', success_count)
        
        # Decay epsilon
        if Hyperparameters.EPSILON > Hyperparameters.MIN_EPSILON:
            Hyperparameters.EPSILON *= Hyperparameters.EPSILON_DECAY
            Hyperparameters.EPSILON = max(Hyperparameters.MIN_EPSILON, Hyperparameters.EPSILON)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(
        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()