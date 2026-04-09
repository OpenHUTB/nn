import numpy as np
import random
import time
import os
from collections import deque
import matplotlib.pyplot as plt  # 画图用
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ================== 参数 ==================
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "CarDQN"
MIN_REWARD = -200

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50
ep_rewards = []
avg_reward_list = []  # 用来画图


# ================== DQN 模型 ==================
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
        model.add(Conv2D(256, (3,3), input_shape=(100,200,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(256, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(3, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1,100,200,3)/255, verbose=0)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([trans[0] for trans in minibatch])/255
        current_qs = self.model.predict(current_states, verbose=0)
        new_states = np.array([trans[3] for trans in minibatch])/255
        future_qs = self.target_model.predict(new_states, verbose=0)

        X, y = [], []
        for i, (state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_q = np.max(future_qs[i])
                new_q = reward + DISCOUNT * max_q
            else:
                new_q = reward
            current_qs[i][action] = new_q
            X.append(state)
            y.append(current_qs[i])

        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter +=1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

# ================== 训练主循环 ==================
agent = DQNAgent()

for episode in range(1000):
    episode_reward = 0
    done = False
    current_state = np.zeros((100,200,3))

    while not done:
        if np.random.rand() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0,3)

        new_state = np.zeros((100,200,3))
        reward = -1
        done = True
        episode_reward += reward

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, episode)
        current_state = new_state

    ep_rewards.append(episode_reward)
    avg_r = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
    avg_reward_list.append(avg_r)

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    print(f"Episode {episode:3d} | Reward: {episode_reward:4.1f} | Avg: {avg_r:4.2f}")

# ================== ✅ 自动生成训练曲线图 ==================
plt.figure(figsize=(10,5))
plt.plot(avg_reward_list, label="Average Reward", color="#1f77b4", linewidth=2)
plt.title("DQN Autonomous Driving Training Curve", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Average Reward", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("training_plot.png", dpi=300)  # 直接生成图片
plt.close()

print("\n✅ 训练完成！曲线图已保存为：training_plot.png")