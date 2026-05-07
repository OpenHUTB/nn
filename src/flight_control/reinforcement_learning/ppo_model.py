from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from .drone_env import DroneEnv
import os

class PPOModel:
    def __init__(self, env=None, model_path=None):
        self.env = env or DroneEnv()
        self.model = None
        self.model_path = model_path
        
        # 如果提供了模型路径，加载现有模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 创建新模型
            self.create_model()
    
    def create_model(self):
        # 创建PPO模型
        self.model = PPO(
            "CnnPolicy",
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5
        )
    
    def train(self, total_timesteps=1000000, save_freq=10000):
        # 创建检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path="./models/",
            name_prefix="ppo_drone"
        )
        
        # 确保保存目录存在
        os.makedirs("./models/", exist_ok=True)
        
        # 开始训练
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback
        )
        
        # 保存最终模型
        self.save_model("./models/ppo_drone_final.zip")
    
    def save_model(self, path):
        # 保存模型
        self.model.save(path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        # 加载模型
        self.model = PPO.load(path, env=self.env)
        print(f"模型已从: {path} 加载")
    
    def predict(self, observation):
        # 预测动作
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def evaluate(self, episodes=10):
        # 评估模型
        total_reward = 0
        for episode in range(episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.predict(obs)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            total_reward += episode_reward
            print(f"Episode {episode+1}: {episode_reward:.2f}")
        
        average_reward = total_reward / episodes
        print(f"平均奖励: {average_reward:.2f}")
        return average_reward
