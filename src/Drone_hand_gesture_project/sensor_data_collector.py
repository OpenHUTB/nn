# -*- coding: utf-8 -*-
"""
传感器数据采集模块
从 AirSim 采集多种传感器数据
"""

import numpy as np
import time
from typing import Dict, Optional, List


class SensorDataCollector:
    """传感器数据采集器"""
    
    def __init__(self, airsim_controller=None):
        """
        初始化采集器
        
        Args:
            airsim_controller: AirSim 控制器实例
        """
        self.controller = airsim_controller
        self.is_collecting = False
        self.collected_data = {
            'images': [],
            'depth_maps': [],
            'imu_data': [],
            'lidar_data': [],
            'position_data': []
        }
    
    def connect(self, ip: str = "127.0.0.1", port: int = 41451) -> bool:
        """连接到 AirSim"""
        if self.controller:
            return self.controller.connected
        
        try:
            from airsim_controller import AirSimController
            self.controller = AirSimController(ip, port)
            return self.controller.connect()
        except Exception as e:
            print(f"❌ 连接失败：{e}")
            return False
    
    def collect_image(self, camera_id: int = 0, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """采集相机图像"""
        if not self.controller:
            print("❌ 未连接控制器")
            return None
        
        try:
            img = self.controller.get_camera_image(camera_id, "scene")
            if img is not None:
                self.collected_data['images'].append(img)
                
                if save_path:
                    import cv2
                    cv2.imwrite(save_path, img)
                    print(f"✅ 图像已保存：{save_path}")
                
                return img
        except Exception as e:
            print(f"⚠️ 采集图像失败：{e}")
        return None
    
    def collect_depth_map(self, camera_id: int = 0) -> Optional[np.ndarray]:
        """采集深度图"""
        if not self.controller:
            return None
        
        try:
            depth_img = self.controller.get_camera_image(camera_id, "depth")
            if depth_img is not None:
                self.collected_data['depth_maps'].append(depth_img)
                return depth_img
        except Exception as e:
            print(f"⚠️ 采集深度图失败：{e}")
        return None
    
    def collect_imu_data(self) -> Optional[Dict]:
        """采集 IMU 数据"""
        if not self.controller:
            return None
        
        try:
            state = self.controller.get_state()
            imu_data = {
                'timestamp': time.time(),
                'position': state['position'],
                'velocity': state['velocity'],
                'orientation': state['orientation'],
                'angular_velocity': state.get('angular_velocity', np.zeros(3))
            }
            
            self.collected_data['imu_data'].append(imu_data)
            return imu_data
        except Exception as e:
            print(f"⚠️ 采集 IMU 数据失败：{e}")
        return None
    
    def continuous_collect(self, duration: float = 10.0, interval: float = 0.1) -> List[Dict]:
        """连续采集数据"""
        if not self.controller:
            print("❌ 未连接控制器")
            return []
        
        print(f"📊 开始连续采集 {duration} 秒...")
        self.is_collecting = True
        
        start_time = time.time()
        data_sequence = []
        
        while self.is_collecting and (time.time() - start_time) < duration:
            data_point = {
                'timestamp': time.time() - start_time,
                'imu': self.collect_imu_data(),
                'image': None,  # 可选，采集会较慢
                'depth': None
            }
            
            data_sequence.append(data_point)
            time.sleep(interval)
        
        self.is_collecting = False
        print(f"✅ 采集完成，共 {len(data_sequence)} 个数据点")
        return data_sequence
    
    def stop_collecting(self):
        """停止采集"""
        self.is_collecting = False
        print("⏹ 停止数据采集")
    
    def get_statistics(self) -> Dict:
        """获取采集统计"""
        return {
            'images': len(self.collected_data['images']),
            'depth_maps': len(self.collected_data['depth_maps']),
            'imu_samples': len(self.collected_data['imu_data']),
            'lidar_scans': len(self.collected_data['lidar_data'])
        }
    
    def save_all_data(self, base_path: str = "sensor_data"):
        """保存所有采集的数据"""
        import os
        import json
        
        os.makedirs(base_path, exist_ok=True)
        print(f"💾 保存数据到 {base_path}/...")
        
        # 保存统计数据
        stats = self.get_statistics()
        with open(f"{base_path}/statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 保存 IMU 数据
        if self.collected_data['imu_data']:
            imu_array = []
            for item in self.collected_data['imu_data']:
                imu_array.append({
                    'timestamp': item['timestamp'],
                    'position': item['position'].tolist(),
                    'velocity': item['velocity'].tolist(),
                    'orientation': item['orientation'].tolist()
                })
            
            np.save(f"{base_path}/imu_data.npy", imu_array)
            print(f"  ✅ IMU 数据：{len(imu_array)} 条")
        
        print("✅ 数据保存完成")


def test_collector():
    """测试传感器采集"""
    print("=" * 60)
    print("传感器数据采集测试")
    print("=" * 60)
    
    collector = SensorDataCollector()
    
    # 模拟连接（如果没有 AirSim）
    print("\n尝试连接 AirSim...")
    if not collector.connect():
        print("⚠️ 无法连接 AirSim，使用模拟数据")
        # 创建模拟数据
        for i in range(10):
            mock_data = {
                'timestamp': i * 0.1,
                'position': np.array([i * 0.1, 0, 2.0]),
                'velocity': np.array([0.1, 0, 0]),
                'orientation': np.array([0, 0, 0])
            }
            collector.collected_data['imu_data'].append(mock_data)
    else:
        # 真实采集
        collector.collect_imu_data()
    
    # 显示统计
    stats = collector.get_statistics()
    print("\n采集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 保存数据
    collector.save_all_data()
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    test_collector()
