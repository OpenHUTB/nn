import os
import sys

# Adjust sys.path to include the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import modules with their full file names
import dqn_model.DQN_model as DQN
import double_dqn_model.double_dqn as DDQN

# Load and evaluate DQN
try:
    dqn_driver = DQN.Agent(
        state_space_shape=(4, 84, 84),
        action_n=5,
        load_state=True,
        load_model='DQN.pt'
    )
except Exception as e:
    print(f"Error loading DQN model: {e}. Initializing with default parameters.")
    dqn_driver = DQN.Agent(
        state_space_shape=(4, 84, 84),
        action_n=5,
        config_path='configs/dqn.yaml'
    )

# Load and evaluate Double DQN - Add this block
try:
    ddqn_driver = DDQN.DoubleDQNAgent(
        state_space_shape=(4, 84, 84),
        action_n=5,
        load_state=True,
        load_model='DoubleDQN.pt'
    )
except Exception as e:
    print(f"Error loading Double DQN model: {e}. Initializing with default parameters.")
    ddqn_driver = DDQN.DoubleDQNAgent(
        state_space_shape=(4, 84, 84),
        action_n=5,
        config_path='configs/double_dqn.yaml'
    )

def evaluate_dqn(agent, num_episodes=5):  # Adjust num_episodes as needed
    import gymnasium as gym
    from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, RecordVideo
    from dqn_model.DQN_model import SkipFrame  # Import SkipFrame from DQN_model
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = RecordVideo(env, video_folder='videos\\DQN')
    env = SkipFrame(env, skip=4)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    agent.epsilon = 0
    scores = []
    steps = []
    for episode in range(num_episodes):
        state, info = env.reset(seed=episode)
        score = 0
        step_count = 0
        while True:
            action = agent.take_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
            step_count += 1
            if terminated or truncated:
                break
        scores.append(score)
        steps.append(step_count)
        print(f"Evaluation Episode {episode+1}/{num_episodes} | Seed: {episode} | Score: {score:.1f}")
    env.close()
    return sum(scores) / len(scores), sum(1 for s in scores if s > 0) / len(scores), sum(steps) / len(steps)

def evaluate_ddqn(agent, num_episodes=5):  # Adjust num_episodes as needed
    import gymnasium as gym
    from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, RecordVideo
    from dqn_model.DQN_model import SkipFrame  # Import SkipFrame from DQN_model
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = RecordVideo(env, video_folder='C:\\Users\\25012\\Desktop\\rl-car-ddqn-main\\videos\\DoubleDQN')
    env = SkipFrame(env, skip=4)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    agent.epsilon = 0
    scores = []
    steps = []
    for episode in range(num_episodes):
        state, info = env.reset(seed=episode)
        score = 0
        step_count = 0
        while True:
            action = agent.take_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
            step_count += 1
            if terminated or truncated:
                break
        scores.append(score)
        steps.append(step_count)
        print(f"Evaluation Episode {episode+1}/{num_episodes} | Seed: {episode} | Score: {score:.1f}")
    env.close()
    return sum(scores) / len(scores), sum(1 for s in scores if s > 0) / len(scores), sum(steps) / len(steps)

# Run evaluations
dqn_avg_score, dqn_success_rate, dqn_avg_steps = evaluate_dqn(dqn_driver)
ddqn_avg_score, ddqn_success_rate, ddqn_avg_steps = evaluate_ddqn(ddqn_driver)

# Print comparison
print("Model Comparison:")
print(f"DQN - Avg Score: {dqn_avg_score:.1f}, Success Rate: {dqn_success_rate:.2%}, Avg Steps: {dqn_avg_steps:.1f}")
print(f"Double DQN - Avg Score: {ddqn_avg_score:.1f}, Success Rate: {ddqn_success_rate:.2%}, Avg Steps: {ddqn_avg_steps:.1f}")