import os
import DQN_model as DQN

# Load the saved model
save_dir = 'training\saved_models'  # Match DQN_model.py
model_file = 'DQN.pt'  # Specify the exact file
driver = DQN.Agent(
    state_space_shape=(4, 84, 84),  # Adjust to your state shape
    action_n=5,  # Match the saved model's action_n
    load_state=True,
    load_model=model_file
)

# Evaluate
def evaluate_agent(agent, num_episodes=5, render=True):
    import gymnasium as gym
    from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
    env = gym.make("CarRacing-v3", continuous=False, render_mode="human" if render else "rgb_array")
    env = DQN.SkipFrame(env, skip=4)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
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