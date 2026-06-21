# 模型评估测试脚本
import gymnasium as gym
from stable_baselines3 import PPO

def evaluate_model(model_path, episodes=10):
    env = gym.make("BipedalWalker-v3", render_mode="human")
    model = PPO.load(model_path)
    total_reward_sum = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_reward_sum += ep_reward
        print(f"第{ep+1}轮奖励：{ep_reward:.2f}")
    
    avg_reward = total_reward_sum / episodes
    print(f"平均奖励：{avg_reward:.2f}")
    env.close()
    return avg_reward

if __name__ == "__main__":
    evaluate_model("bipedalwalker_ppo_model.zip")