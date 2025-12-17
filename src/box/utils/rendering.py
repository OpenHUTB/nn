# src/box/utils/rendering.py
import mujoco
import numpy as np

class Context:
    """渲染上下文类，用于管理渲染配置"""
    def __init__(self, model, max_resolution):
        self.model = model
        self.max_resolution = max_resolution  # 格式：[宽度, 高度]
# 必须确保此文件的路径和内容完全匹配
class Context:
    def __init__(self, model, max_resolution):
        self.model = model
        self.max_resolution = max_resolution

class Camera:
    def __init__(self, context, model, data, camera_id, dt):
        self.context = context
        self.model = model
        self.data = data
        self.camera_id = camera_id
        self.dt = dt
        self._fps = 1.0 / dt

    def render(self):
        return (None, None)