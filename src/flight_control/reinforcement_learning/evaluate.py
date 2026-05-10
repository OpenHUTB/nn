from .ppo_model import PPOModel

if __name__ == "__main__":
    # 加载训练好的模型
    model_path = "./models/ppo_drone_final.zip"
    model = PPOModel(model_path=model_path)
    
    # 评估模型
    print("评估训练好的模型...")
    model.evaluate(episodes=20)
