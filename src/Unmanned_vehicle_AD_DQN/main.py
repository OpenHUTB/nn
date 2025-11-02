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
    # For stats
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

    # Iterate over episodes
    epds = []
    scores = []
    avg_scores = []
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        env.collision_hist = []
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        score = 0
        step = 1


        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:

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

            new_state, reward, done, _ = env.step(action)

            # Transform new continuous state to new discrete state and count reward
            score += reward

            if score < REWARD_OFFSET:
                done = True

            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

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
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_score:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        epds.append(episode)
        print('episode: ', episode, 'score %.2f' % score)
        # Decay epsilon
        if Hyperparameters.EPSILON > Hyperparameters.MIN_EPSILON:
            Hyperparameters.EPSILON *= Hyperparameters.EPSILON_DECAY
            Hyperparameters.EPSILON = max(Hyperparameters.MIN_EPSILON, Hyperparameters.EPSILON)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(
        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_score:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()