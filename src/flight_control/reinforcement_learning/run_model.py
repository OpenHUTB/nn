from .ppo_model import PPOModel
import time

class RLController:
    def __init__(self, model_path=None, client=None):
        self.model_path = model_path
        self.client = client
        self.model = None
        self.obs = None
        self.running = False
    
    def initialize(self):
        # 初始化模型
        if self.model_path:
            from .drone_env import DroneEnv
            env = DroneEnv(client=self.client)
            self.model = PPOModel(env=env, model_path=self.model_path)
            print("强化学习模型已初始化")
            # 重置环境获取初始观察
            self.obs = self.model.env.reset()
        else:
            print("未提供模型路径")
    
    def step(self):
        """执行一个强化学习步骤 - 供主循环调用"""
        if not self.model or not self.running:
            return None
        
        try:
            # 预测动作
            action = self.model.predict(self.obs)
            
            # 执行动作
            self.obs, reward, done, info = self.model.env.step(action)
            
            # 打印信息
            print(f"Action: {action}, Reward: {reward:.2f}, Target: {info['current_target']}")
            
            # 检查是否完成
            if done:
                self.obs = self.model.env.reset()
            
            return (action, reward, info)
        except Exception as e:
            print(f"强化学习步骤执行失败: {e}")
            return None
    
    def start(self):
        # 开始强化学习控制
        if not self.model:
            print("模型未初始化")
            return
        
        self.running = True
        print("开始强化学习控制...")
        
        # 重置环境
        self.obs = self.model.env.reset()
        
        while self.running:
            # 预测动作
            action = self.model.predict(self.obs)
            
            # 执行动作
            self.obs, reward, done, info = self.model.env.step(action)
            
            # 打印信息
            print(f"Action: {action}, Reward: {reward:.2f}, Target: {info['current_target']}")
            
            # 检查是否完成
            if done:
                self.obs = self.model.env.reset()
            
            # 短暂延迟
            time.sleep(0.1)
    
    def stop(self):
        # 停止强化学习控制
        self.running = False
        print("强化学习控制已停止")

    def close(self):
        # 关闭环境
        if self.model:
            self.model.env.close()
            self.model = None
