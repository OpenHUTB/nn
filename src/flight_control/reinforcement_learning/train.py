from .ppo_model import PPOModel

if __name__ == "__main__":
    # 创建模型
    model = PPOModel()
    
    # 开始训练
    print("开始训练模型...")
    model.train(total_timesteps=1000000, save_freq=50000)
    
    # 评估模型
    print("评估模型...")
    model.evaluate(episodes=10)
