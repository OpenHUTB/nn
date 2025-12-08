import numpy as np
import random
from collections import deque
from datetime import datetime
import logging


# for building the DQN model
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam


class CarAgent:
    def __init__(
            self,
            action_size,
            action_space,
            state_size,
            discount_factor=0.95,  # Gamma
            learning_rate=0.01,  # Alpha
            epsilon=1,
            epsilon_decay=0.99,
            epsilon_min=0.01,
            memory_size=1000000,
            batch_size=32,
    ):

        self.logger = logging.getLogger("CarRacing_Logger")
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space

        self.model = self.build_model()

    def get_action(self, observation):
        if np.random.rand() <= self.epsilon:
            # Exploration: Random action from possible actions
            self.logger.debug("Taking Exploratory action")
            temp_action = random.choice(self.action_space)
        else:
            # Choose action with the highest Q-value: q(s, a)
            observation = observation.reshape(1, self.state_size)
            self.logger.debug("Taking Exploited action")
            val_ = self.model.predict(observation).tolist()[0]
            temp_action = self.action_space[np.argmax(val_)]
        return temp_action

    def append_sample(self, observation, action, reward, next_observation, terminated):
        # Append the tuple (s, a, r, s', terminated) to memory (replay buffer) after every action
        # Analysis of each Sample going inside Replay Buffer / Memory
        # observation_temp ==> (1, 27648) ==> Numpy Array
        # next_observation_temp ==> (1, 27648) ==> Numpy Array
        # reward ==> 0.09 ==> Float Value
        # action ==> [-0.9, 0.1, 0.1] ==> List Value
        # self.logger.debug(f"Appending sample to memory")
        self.memory.append(
            (
                observation.reshape(1, self.state_size),
                action,
                reward,
                next_observation.reshape(1, self.state_size),
                terminated
            )
        )

    def build_model(self):
        model = Sequential(
            [
                Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'),
                Dense(32, activation='relu', kernel_initializer='he_uniform'),
                Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')
            ]
        )
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def train_model(self):
        if len(self.memory) > self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)

            current_state_data = np.zeros((self.batch_size, self.state_size))
            next_state_data = np.zeros((self.batch_size, self.state_size))

            rewards_data = []
            terminated_data = []
            action_data = []

            # Iterate over the minibatch (of 32) selected from Replay Buffer / Memory
            for i in range(self.batch_size):
                observation, action, reward, next_observation, terminated_boolean = minibatch[i]

                current_state_data[i] = observation
                next_state_data[i] = next_observation

                action_data.append(action)
                rewards_data.append(reward)

                terminated_data.append(terminated_boolean)

            # Predict Q-values from Current State (s)
            target = self.model.predict(current_state_data)
            # self.logger.debug(f"Shape of Current State Q-Value: {target.shape}")

            # Predict Q-values from Next State (s`)
            target_qval = self.model.predict(next_state_data)

            # update the target values
            for i in range(self.batch_size):
                if action_data[i] in self.action_space:
                    index_val = self.action_space.index(action_data[i])

                if terminated_data[i]:
                    # For Terminal State
                    target[i][index_val] = rewards_data[i]
                else:
                    # For Non-Terminal State
                    target[i][index_val] = rewards_data[i] + self.discount_factor * np.max(target_qval[i])

            # model fit
            self.model.fit(
                current_state_data,
                target,
                batch_size=self.batch_size,
                epochs=1,
                verbose=0
            )

    def save_model_weights(self, episode):
        self.model.save_weights(f"data/Episode_{episode}_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".h5")
