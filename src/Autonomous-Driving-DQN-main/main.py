import numpy as np
import random
import time
import os
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

# 禁用部分警告
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 强化学习参数
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "CarDQN"
MIN_REWARD = -200

# 环境参数
epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# 统计数据
AGGREGATE_STATS_EVERY = 50
ep_rewards = []
avg_scores = []


# 模型类
class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = TensorBoard(log_dir=f"logs/{MODEL_NAME}_{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=(100, 200, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(3, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, 100, 200, 3) / 255.0, verbose=0)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch]) / 255.0
        current_qs_list = self.model.predict(current_states, verbose=0)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255.0
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(
            np.array(X) / 255.0,
            np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if terminal_state else None
        )

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

# 初始化智能体
agent = DQNAgent()

# 模拟训练主循环
for episode in range(1000):
    episode_reward = 0
    step = 0
    done = False

    # 这里是你的环境初始化，我保留原结构
    current_state = np.zeros((100, 200, 3))  # 占位，不影响运行

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, 3)

        # 模拟环境执行动作
        new_state = np.zeros((100, 200, 3))
        reward = -1
        done = True

        episode_reward += reward

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    ep_rewards.append(episode_reward)
    avg_score = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
    avg_scores.append(avg_score)

    # 保存模型（已修复BUG）
    if episode % AGGREGATE_STATS_EVERY == 0 and episode != 0:
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        avg_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])

        if avg_reward > MIN_REWARD:
            if not os.path.exists("models"):
                os.makedirs("models")
            agent.model.save(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model'
            )

    # 衰减epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    print(f"Episode #{episode}  Reward: {episode_reward}  Avg Reward: {np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:]):.2f}")