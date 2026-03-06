# -*- coding: utf-8 -*-
import unittest
import sys
import os
import math

# 确保能导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.geometry import get_speed

class TestAutonomousLogic(unittest.TestCase):
    """
    单元测试类：校验自动驾驶核心算法逻辑
    """

    def test_speed_calculation(self):
        """测试车速转换逻辑是否正确 (m/s 到 km/h)"""
        # 模拟一个简单的速度对象 (CARLA 风格)
        class MockVelocity:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class MockVehicle:
            def get_velocity(self):
                return MockVelocity(10.0, 0, 0) # 10 m/s

        vehicle = MockVehicle()
        speed = get_speed(vehicle)
        # 10 m/s 应该等于 36 km/h
        self.assertAlmostEqual(speed, 36.0, places=1)
        print(f"单元测试通过: 模拟速度 10m/s -> 计算结果 {speed}km/h")

    def test_config_paths(self):
        """测试模型路径配置是否存在"""
        from config import MODEL_CFG
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cfg_path = os.path.join(root, MODEL_CFG)
        self.assertTrue(os.path.exists(cfg_path))
        print(f"路径测试通过: 找到模型配置文件 {MODEL_CFG}")

if __name__ == '__main__':
    unittest.main()