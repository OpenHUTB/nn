import os
from stable_baselines3 import PPO
from env_utils import make_env


def main():
    # 简单示例：训练一个非常小的 PPO 模型用于快速验证
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    env = make_env(env_name='BipedalWalker-v3', hardcore=False, use_monitor=False, norm_obs=False, norm_reward=False)

    model = PPO('MlpPolicy', env, verbose=1)
    timesteps = 1000
    print(f"Training for {timesteps} timesteps (quick demo)...")
    model.learn(total_timesteps=timesteps)

    model_path = os.path.join(model_dir, 'ppo_bipedalwalker_example')
    model.save(model_path)
    print(f"Saved example model to: {model_path}.zip")

    env.close()


if __name__ == '__main__':
    main()
