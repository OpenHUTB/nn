import numpy as np
import mujoco
from .base import PerceptionModule
from src.box.utils.rendering import Camera  # 复用仿真器的Camera类

class VisionPerception(PerceptionModule):
    """视觉感知模块，支持RGB/深度图采集"""
    def __init__(self, model, data, camera_id="rgb_camera", resolution=(640, 480), 
                 capture_depth=False, normalize=True, **kwargs):
        super().__init__(modality="vision", model=model, data=data, **kwargs)
        
        # 相机配置
        self.camera_id = camera_id
        self.resolution = resolution
        self.capture_depth = capture_depth
        self.normalize = normalize
        
        # 初始化相机（复用仿真器的Camera类）
        self.camera = Camera(
            context=self.kwargs.get("rendering_context"),
            model=model,
            data=data,
            camera_id=camera_id,
            dt=self.kwargs.get("dt", 0.01)
        )
        self.cameras = [self.camera]  # 关联到模块相机列表

    def get_observation(self, model, data, info=None):
        """获取视觉观测数据（RGB/深度图）"""
        # 渲染相机画面
        rgb_img, depth_img = self.camera.render()
        
        # 拼接观测（RGB为主，可选深度）
        if self.capture_depth:
            # 归一化深度图
            depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
            obs = np.concatenate([rgb_img, depth_img[..., np.newaxis]], axis=-1)
        else:
            obs = rgb_img
        
        # 归一化到[0,1]（可选）
        if self.normalize:
            obs = obs.astype(np.float32) / 255.0
        
        # 展平为一维向量（适配RL观测空间）
        return obs.flatten()

    def get_observation_space_params(self):
        """定义视觉观测空间参数（适配Gymnasium）"""
        # 计算观测维度：RGB(3通道) + 深度(可选1通道)
        channels = 3 + (1 if self.capture_depth else 0)
        shape = (self.resolution[0] * self.resolution[1] * channels,)
        return {
            "low": 0.0,
            "high": 1.0,
            "shape": shape
        }

    def get_renders(self):
        """返回原始RGB图像（用于仿真器渲染）"""
        rgb_img, _ = self.camera.render()
        return [rgb_img]