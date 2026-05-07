import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import doubledqn_agent as DDQN
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, RecordVideo

# Load the saved model
model_file = 'DoubleDQN.pt'
model_path = Path(__file__).resolve().parents[1] / 'training' / 'saved_models' / model_file
driver = DDQN.DoubleDQNAgent(
    state_space_shape=(4, 84, 84),
    action_n=5,
    load_state=model_path.exists(),
    load_model=model_file if model_path.exists() else None
)

# Evaluate
def evaluate_agent(agent, num_episodes=5, render=True):
    env = gym.make("CarRacing-v2", continuous=False, render_mode="rgb_array")
    try:
        import moviepy  # noqa: F401
        env = RecordVideo(env, video_folder=os.path.join('videos', 'DoubleDQN'))
    except Exception:
        pass
    env = DDQN.SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, num_stack=4)
    agent.epsilon = 0
    seeds_list = [i for i in range(num_episodes)]
    scores = []
    for episode, seed in enumerate(seeds_list):
        state, info = env.reset(seed=seed)
        score = 0
        updating = True
        while updating:
            action = agent.take_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
            updating = not (terminated or truncated)
        scores.append(score)
        print(f"Evaluation Episode {episode+1}/{num_episodes} | Seed: {seed} | Score: {score:.1f}")
    env.close()
    return sum(scores) / len(scores)

avg_score = evaluate_agent(driver, num_episodes=5, render=True)
print(f"Average evaluation score: {avg_score:.1f}")
