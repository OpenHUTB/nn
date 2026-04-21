import gym
import numpy as np
import airsim
import time
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self, client=None):
        super(DroneEnv, self).__init__()
        
        # 连接到AirSim
        if client is None:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        else:
            self.client = client
        
        # 定义动作空间：9个离散动作
        # 0: 前进, 1: 后退, 2: 左移, 3: 右移, 4: 上升, 5: 下降, 6: 左转, 7: 右转, 8: 悬停
        self.action_space = spaces.Discrete(9)
        
        # 定义观察空间：360x240 RGB图像
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 360, 3), dtype=np.uint8)
        
        # 飞行参数
        self.speed = 2.0
        self.height = -3.0
        
        # 目标框位置（模拟）
        self.targets = [
            (0, 0, -3.0),
            (10, 0, -3.0),
            (10, 10, -3.0),
            (0, 10, -3.0),
            (0, 0, -3.0)
        ]
        self.current_target_idx = 0
        
        # 重置无人机
        self.reset()
    
    def reset(self):
        # 重置无人机位置到当前目标点
        target = self.targets[self.current_target_idx]
        
        # 重置无人机
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPositionAsync(target[0], target[1], target[2], 2).join()
        self.client.hoverAsync().join()
        time.sleep(1)
        
        # 获取初始观察
        observation = self._get_observation()
        return observation
    
    def step(self, action):
        # 执行动作
        self._take_action(action)
        
        # 获取新的观察
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否完成任务
        done = self._check_done()
        
        # 信息
        info = {
            'current_target': self.current_target_idx,
            'position': self._get_position()
        }
        
        return observation, reward, done, info
    
    def _get_observation(self):
        # 获取RGB图像
        try:
            responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
            response = responses[0]
            
            # 处理图像数据
            image_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            image_data = image_data.reshape(response.height, response.width, 3)
            
            # 确保图像尺寸正确
            if image_data.shape != (240, 360, 3):
                import cv2
                image_data = cv2.resize(image_data, (360, 240))
            
            return image_data
        except Exception as e:
            # 返回空图像或之前的图像
            print(f"获取图像失败: {e}")
            # 返回默认的黑色图像
            return np.zeros((240, 360, 3), dtype=np.uint8)
    
    def _take_action(self, action):
        # 根据动作执行相应的操作
        try:
            if action == 0:  # 前进
                self.client.moveByVelocityBodyFrameAsync(self.speed, 0, 0, 0.5)
            elif action == 1:  # 后退
                self.client.moveByVelocityBodyFrameAsync(-self.speed*0.7, 0, 0, 0.5)
            elif action == 2:  # 左移
                self.client.moveByVelocityBodyFrameAsync(0, -self.speed, 0, 0.5)
            elif action == 3:  # 右移
                self.client.moveByVelocityBodyFrameAsync(0, self.speed, 0, 0.5)
            elif action == 4:  # 上升
                self.height -= 0.5
                self.client.moveToZAsync(self.height, 0.8)
            elif action == 5:  # 下降
                self.height += 0.5
                self.client.moveToZAsync(self.height, 0.8)
            elif action == 6:  # 左转
                self.client.rotateByYawRateAsync(-30, 0.5)
            elif action == 7:  # 右转
                self.client.rotateByYawRateAsync(30, 0.5)
            elif action == 8:  # 悬停
                self.client.hoverAsync()
            
            # 等待动作执行
            time.sleep(0.5)
        except Exception as e:
            print(f"执行动作失败: {e}")
    
    def _get_position(self):
        # 获取无人机当前位置
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            return (pos.x_val, pos.y_val, pos.z_val)
        except Exception as e:
            print(f"获取位置失败: {e}")
            # 返回默认位置
            return (0, 0, -3.0)
    
    def _calculate_reward(self):
        # 计算奖励
        try:
            current_pos = self._get_position()
            target = self.targets[self.current_target_idx]
            
            # 计算到目标的距离
            distance = np.sqrt(
                (current_pos[0] - target[0])**2 +
                (current_pos[1] - target[1])**2 +
                (current_pos[2] - target[2])**2
            )
            
            # 基础奖励：距离越近奖励越高
            reward = max(0, 10 - distance)
            
            # 如果接近目标，给予额外奖励
            if distance < 1.0:
                reward += 50
                self.current_target_idx = (self.current_target_idx + 1) % len(self.targets)
            
            return reward
        except Exception as e:
            print(f"计算奖励失败: {e}")
            return 0
    
    def _check_done(self):
        # 检查是否完成所有目标
        return False  # 持续运行
    
    def close(self):
        # 关闭环境
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception as e:
            print(f"关闭环境失败: {e}")
