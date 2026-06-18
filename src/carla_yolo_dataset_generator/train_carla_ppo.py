import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
import queue
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class CarlaEnv(gym.Env):
    """
    自定义的 CARLA 强化学习环境，符合 Gymnasium 标准。
    包含车辆生成、传感器数据获取、动作执行和奖励计算。
    """
    def __init__(self, host='127.0.0.1', port=2000, max_steps=1000):
        super(CarlaEnv, self).__init__()
        
        # 1. 连接 CARLA 客户端
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.max_steps = max_steps
        self.current_step = 0
        self.actor_list = []
        
        # 2. 定义动作空间 (Action Space)
        # [转向 (Steer), 油门 (Throttle)]
        # Steer: [-1.0, 1.0], Throttle: [0.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # 3. 定义观察空间 (Observation Space)
        # 使用 RGB 摄像头图像作为输入，尺寸缩小为 84x84 以加速 CNN 训练
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(84, 84, 3), 
            dtype=np.uint8
        )
        
        # 传感器队列
        self.image_queue = queue.Queue()
        self.collision_history = []
        
        # 设置同步模式 (对于强化学习非常重要，确保物理引擎与推理同步)
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05 # 20 FPS
        self.world.apply_settings(self.settings)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cleanup()
        self.current_step = 0
        self.collision_history = []
        self.image_queue = queue.Queue()

        # 1. 生成车辆 (以截图中的 Tesla Model 3 为例)
        vehicle_bp = self.blueprint_library.filter('model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        self.actor_list.append(self.vehicle)

        # 2. 设置 RGB 摄像头
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '84')
        camera_bp.set_attribute('image_size_y', '84')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: self._process_image(data))

        # 3. 设置碰撞传感器
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_history.append(event))

        # 等待第一帧图像
        self.world.tick()
        obs = self._get_observation()
        
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        # 1. 执行动作
        steer = float(action[0])
        throttle = float(action[1])
        brake = 0.0
        
        control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        self.vehicle.apply_control(control)
        
        # 2. 推进世界状态
        self.world.tick()
        
        # 3. 获取状态与计算奖励
        obs = self._get_observation()
        v = self.vehicle.get_velocity()
        speed_kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # --- 奖励函数设计 (Reward Function) ---
        # 鼓励保持速度，惩罚碰撞。你可以根据截图中的 Center deviance 进一步优化。
        reward = 0.0
        terminated = False
        truncated = False
        
        if len(self.collision_history) > 0:
            terminated = True
            reward = -200.0 # 碰撞严重惩罚
        else:
            # 基础速度奖励
            reward = speed_kmh / 20.0 
            
            # 如果静止不动，给予惩罚
            if speed_kmh < 1.0:
                reward -= 1.0
                
        # 超时截断
        if self.current_step >= self.max_steps:
            truncated = True
            
        info = {'speed_kmh': speed_kmh}
        
        return obs, reward, terminated, truncated, info

    def _process_image(self, image):
        # 处理 CARLA 的原始图像数据，转换为 numpy array
        raw_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        raw_data = np.reshape(raw_data, (image.height, image.width, 4))
        rgb = raw_data[:, :, :3] # 去掉 Alpha 通道
        self.image_queue.put(rgb)

    def _get_observation(self):
        # 确保我们获取到最新的一帧图像
        while True:
            try:
                img = self.image_queue.get(timeout=2.0)
                return img
            except queue.Empty:
                print("Warning: Camera image queue timeout!")
                return np.zeros((84, 84, 3), dtype=np.uint8)

    def cleanup(self):
        # 清理世界中的 Actor，防止内存泄漏
        for actor in self.actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list.append([])

    def close(self):
        self.cleanup()
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)


if __name__ == '__main__':
    print("正在初始化 CARLA 环境...")
    try:
        # 创建环境
        env = CarlaEnv(host='127.0.0.1', port=2000, max_steps=1000)
        
        print("环境初始化成功，开始构建 PPO 模型...")
        # 使用 PPO 算法，CnnPolicy 专门用于处理图像观察输入
        # device='cuda' 确保使用 GPU 加速（如果有的话）
        model = PPO(
            "CnnPolicy", 
            env, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1, 
            tensorboard_log="./carla_ppo_tensorboard/"
        )
        
        # 设置检查点，每隔 10000 步保存一次模型
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path='./models/',
            name_prefix='carla_ppo_model'
        )
        
        print("开始训练！(可以使用 Tensorboard 查看训练曲线)...")
        # 开始训练，假设训练 500,000 步
        model.learn(total_timesteps=500000, callback=checkpoint_callback)
        
        # 保存最终模型
        model.save("carla_ppo_final")
        print("训练结束，模型已保存。")
        
    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        if 'env' in locals():
            env.close()
            print("环境已安全清理关闭。")