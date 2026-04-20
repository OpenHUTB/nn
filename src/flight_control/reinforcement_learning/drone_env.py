import gym
import numpy as np
import airsim
import time
import cv2
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self, client=None, use_yolo=False, yolo_model_path=None):
        super(DroneEnv, self).__init__()

        # 连接到AirSim
        if client is None:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        else:
            self.client = client

        # YOLO 配置
        self.use_yolo = use_yolo
        self.yolo_model_path = yolo_model_path
        self.yolo = None

        if self.use_yolo:
            self._init_yolo()

        # 定义动作空间：9个离散动作
        # 0: 前进, 1: 后退, 2: 左移, 3: 右移, 4: 上升, 5: 下降, 6: 左转, 7: 右转, 8: 悬停
        self.action_space = spaces.Discrete(9)

        # 定义观察空间
        if self.use_yolo:
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(240, 360, 3), dtype=np.uint8),
                'detection_info': spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
            })
        else:
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

        self.last_detection_info = {
            'has_target': False,
            'target_center_offset': (0.0, 0.0),
            'target_distance': 1.0,
            'target_size': 0.0,
            'num_detections': 0
        }

        # 重置无人机
        self.reset()

    def _init_yolo(self):
        try:
            from .yolo_inference import YOLOInference
            self.yolo = YOLOInference(model_path=self.yolo_model_path)
            print(f"YOLO initialized successfully (use_yolo=True)")
        except ImportError:
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(__file__))
                from yolo_inference import YOLOInference
                self.yolo = YOLOInference(model_path=self.yolo_model_path)
                print(f"YOLO initialized successfully (use_yolo=True)")
            except Exception as e:
                print(f"YOLO initialization failed: {e}, falling back to no YOLO")
                self.use_yolo = False
    
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
        self._take_action(action)

        observation = self._get_observation()

        reward = self._calculate_reward()

        done = self._check_done()

        info = {
            'current_target': self.current_target_idx,
            'position': self._get_position(),
            'detection_info': self.last_detection_info
        }

        return observation, reward, done, info
    
    def _get_observation(self):
        image_data = self._get_raw_image()

        if self.use_yolo and self.yolo is not None:
            detections = self.yolo.detect(image_data)
            self.last_detection_info = self.yolo.get_detection_info(image_data.shape, detections)

            detection_array = np.array([
                1.0 if self.last_detection_info['has_target'] else 0.0,
                self.last_detection_info['target_center_offset'][0],
                self.last_detection_info['target_center_offset'][1],
                self.last_detection_info['target_distance'],
                self.last_detection_info['target_size']
            ], dtype=np.float32)

            return {
                'image': image_data,
                'detection_info': detection_array
            }

        return image_data

    def _get_raw_image(self):
        try:
            responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
            response = responses[0]

            image_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            image_data = image_data.reshape(response.height, response.width, 3)

            if image_data.shape != (240, 360, 3):
                image_data = cv2.resize(image_data, (360, 240))

            return image_data
        except Exception as e:
            print(f"获取图像失败: {e}")
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
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception as e:
            print(f"关闭环境失败: {e}")

    def get_annotated_frame(self):
        image = self._get_raw_image()
        if self.use_yolo and self.yolo is not None:
            detections = self.yolo.detect(image)
            return self.yolo.annotate_image(image, detections)
        return image