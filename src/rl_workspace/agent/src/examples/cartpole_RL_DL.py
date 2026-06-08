import gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")  # , render_mode='human')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)

vec_env = model.get_env()
obs = vec_env.reset()
rewards = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    rewards += 1
    # VecEnv resets automatically
    if done:
        break
    # obs = env.reset()
print(f"reward: {rewards}")
env.close()
