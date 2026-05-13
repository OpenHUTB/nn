import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
import QLearner as ql

# for step in range(50):
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, done, trunc, info = env.step(action)
#     # print(observation)  # (cart position, cart velocity, pole angle, pole angular velocity)
#     time.sleep(0.1)
#
# env.close()


def create_bins(num_bins_per_obs=10):
    bins_cart_position = np.linspace(-4.8, 4.8, num_bins_per_obs)
    bins_cart_velocity = np.linspace(-5, 5, num_bins_per_obs)
    bins_pole_angle = np.linspace(-0.418, 0.418, num_bins_per_obs)
    bins_pole_angular_velocity = np.linspace(-5, 5, num_bins_per_obs)

    bins = np.array(
        [
            bins_cart_position,
            bins_cart_velocity,
            bins_pole_angle,
            bins_pole_angular_velocity,
        ]
    )

    return bins


# this function is really flexible where you can adjust how agent is rewarded/punished
def fail(done, points, reward):
    if done and points < 500:
        reward = -600

    return reward


def discretize_observation(observations, bins):
    binned_observations = []

    for i, observation in enumerate(observations):
        discretized_observation = np.digitize(
            observation, bins[i]
        )  # cool function to find which bin a value lies in given an array of bins
        discretized_observation = min(bins.shape[1] - 1, discretized_observation)

        binned_observations.append(discretized_observation)

    return tuple(binned_observations)


####### Visualization ################
log_interval = 50
render_interval = 30000

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.canvas.draw()
#####################################


def train(env):
    points_log = []
    mean_points_log = []
    epochs = []

    debug = True

    rewards = []
    log_interval = 1000

    # # of bins for cart_position, # of bins of cart_velocity, # of bins for pole angle, # of bins for pole angular velocity
    learner = ql.QLearner(
        states=(NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS),
        actions=env.action_space.n,
        alpha=0.1,
        gamma=0.995,
    )

    for epoch in range(EPOCHS):
        if debug:
            print(
                f"============= running episode: {epoch} of {EPOCHS} ================="
            )

        initial_state = env.reset()[0]
        discretized_state = discretize_observation(initial_state, BINS)

        done = False
        points = 0
        epochs.append(epoch)

        action = learner.get_next_action_without_Q_table_update(discretized_state)

        while not done:
            # state, reward... env.step()
            new_state, reward, done, trunc, info = env.step(action)

            # discretize the state
            discretized_state = discretize_observation(new_state, BINS)

            # custom reward
            reward = fail(done, points, reward)

            # get next action
            # print(f"last action: {action}, new state: {new_state}, new discretized state: {discretized_state}")
            action = learner.get_next_action_with_Q_table_update(
                discretized_state, reward
            )
            # print(action)

            # track rewards
            points += 1

        # if debug: print(learner.Q)

        # decay the random action rate
        learner.decay_rar(epoch)

        points_log.append(points)
        running_mean = round(np.mean(points_log[-30:]), 2)
        mean_points_log.append(running_mean)

        #####################################################
        if epoch % log_interval == 0:
            print(f"current mean rewards: {running_mean}")
            ax.clear()
            ax.scatter(epochs, points_log)
            ax.plot(epochs, points_log)
            ax.plot(epochs, mean_points_log, label=f"Running Mean: {running_mean}")
            plt.legend()
            fig.canvas.draw()
            plt.show()

    env.close()
    return learner.Q


# lets see how the agent is performing after training
def check_performance(env, use_qable=True, q_table=None):
    total_reward = 0
    done = False
    state = env.reset()[0]

    for steps in range(500):
        env.render()
        if use_qable == True:
            discrete_state = discretize_observation(state, BINS)  # get bins
            action = np.argmax(
                q_table[discrete_state]
            )  # and chose action from the Q-Table
        else:
            action = env.action_space.sample()
        state, reward, done, trunc, info = env.step(
            action
        )  # Finally perform the action
        total_reward += 1
        if done:
            break

    print(f"You got {total_reward} points!")
    env.close()


# points_log = []
# mean_points_log = []
# epochs = []
#
# for epoch in range(EPOCHS):
#
#     print(f"\n======================= running epoch: {epoch} ======================")
#
#     initial_state = env.reset()[0]
#     discretized_state = discretize_observation(initial_state, BINS)
#     done = False
#     points = 0
#
#     epochs.append(epoch)
#
#     # play game
#     while not done:
#         action = epsilon_greedy_action_selection(epsilon, q_table, discretized_state)
#         # print(f"action: {action}")
#         next_state, reward, done, trunc, info = env.step(action)
#
#         reward = fail(done, points, reward)
#
#         next_state_discretized = discretize_observation(next_state, BINS)
#         old_q_value = q_table[discretized_state + (action,)]
#         next_optimal_q_value = np.max(q_table[next_state_discretized])
#
#         next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)
#         q_table[discretized_state + (action,)] = next_q
#
#         discretized_state = next_state_discretized
#         points += 1
#
#     print(f"\t points: {points}")
#     epsilon = reduce_epsilon(epsilon, epoch)
#     points_log.append(points)
#     running_mean = round(np.mean(points_log[-30:]), 2)
#     mean_points_log.append(running_mean)
#
#     #####################################################
#     if epoch % log_interval == 0:
#         ax.clear()
#         ax.scatter(epochs, points_log)
#         ax.plot(epochs, points_log)
#         ax.plot(epochs, mean_points_log, label=f"Running Mean: {running_mean}")
#         plt.legend()
#         fig.canvas.draw()
#         plt.show()

# env.close()

if __name__ == "__main__":
    NUM_BINS = 50
    BINS = create_bins(NUM_BINS)

    training = False
    testing = True

    if training == True:
        # create the environment
        env = gym.make("CartPole-v1", render_mode="rgb_array")  # ["human", "rgb_array"]
        env.reset()

        # run the training
        EPOCHS = 50000  # episodes, how many times the agents plays the game until it hits done
        q_table = train(env)

        np.save("C:\\Users\\riteshm\omscs\\vip\\RL\\cartpole_q_table.npy", q_table)

    if testing == True:
        # print("====== using random actions... ")
        # env = gym.make("CartPole-v1", render_mode="human")
        # env.reset()
        # check_performance(env, use_qable=False)
        #
        # time.sleep(2)

        print("====== using Q-table actions... ")
        env = gym.make("CartPole-v1", render_mode="human")
        env.reset()
        q_table = np.load("C:\\Users\\riteshm\omscs\\vip\\RL\\cartpole_q_table.npy")
        check_performance(env, use_qable=True, q_table=q_table)
