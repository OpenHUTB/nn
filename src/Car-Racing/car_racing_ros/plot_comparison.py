import matplotlib.pyplot as plt
import csv
import numpy as np
import os

# Paths to log files
dqn_log = 'training\\logs\\DQN_log_test.csv'
double_dqn_log = 'training\\logs\\DoubleDQN_log_test.csv'

def read_rewards(log_path):
    if not os.path.exists(log_path):
        print(f"Warning: Log file {log_path} not found. Returning empty list.")
        return []
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
        reward_row = [float(x) for x in rows[2][1:] if x]
        if len(rows) < 3:
            print(f"Warning: Invalid log file {log_path}. Returning empty list.")
            return []
        reward_row = [float(x) for x in rows[2][1:] if x]  # Reward is the third row (index 2)
        return reward_row

dqn_rewards = read_rewards(dqn_log)
double_dqn_rewards = read_rewards(double_dqn_log)

# Plot if data exists
if dqn_rewards or double_dqn_rewards:
    plt.figure(figsize=(10, 6))
    if dqn_rewards:
        plt.plot(dqn_rewards, label='DQN', color='#1f77b4')
    if double_dqn_rewards:
        plt.plot(double_dqn_rewards, label='Double DQN', color='#ff7f0e')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN vs Double DQN Reward Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('training\\comparison_plot.png')
    plt.show()
else:
    print("No reward data available to plot.")