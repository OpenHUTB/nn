# -*- coding: utf-8 -*-
"""
飞行数据记录器
记录和分析无人机飞行数据
"""

import numpy as np
import time
import json
from datetime import datetime
from typing import List, Dict, Optional


class FlightRecorder:
    """飞行数据记录器"""
    
    def __init__(self, save_dir: str = "flight_logs"):
        """
        初始化飞行记录器
        
        Args:
            save_dir: 数据保存目录
        """
        self.save_dir = save_dir
        self.is_recording = False
        self.start_time = None
        self.flight_data = []
        self.metadata = {}
        
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def start_recording(self, metadata: Optional[Dict] = None):
        """开始记录"""
        self.is_recording = True
        self.start_time = time.time()
        self.flight_data = []
        self.metadata = metadata or {}
        print("📊 开始记录飞行数据...")
    
    def stop_recording(self):
        """停止记录"""
        self.is_recording = False
        duration = time.time() - self.start_time if self.start_time else 0
        print(f"⏹ 停止记录，总时长：{duration:.1f}秒")
    
    def add_data_point(self, position: np.ndarray, velocity: np.ndarray, 
                       orientation: np.ndarray, **kwargs):
        """
        添加数据点
        
        Args:
            position: 位置 (x, y, z)
            velocity: 速度 (vx, vy, vz)
            orientation: 姿态 (roll, pitch, yaw)
            **kwargs: 其他数据
        """
        if not self.is_recording:
            return
        
        timestamp = time.time() - self.start_time
        
        data_point = {
            'timestamp': timestamp,
            'position': position.tolist() if hasattr(position, 'tolist') else list(position),
            'velocity': velocity.tolist() if hasattr(velocity, 'tolist') else list(velocity),
            'orientation': orientation.tolist() if hasattr(orientation, 'tolist') else list(orientation),
            **kwargs
        }
        
        self.flight_data.append(data_point)
    
    def get_statistics(self) -> Dict:
        """获取飞行统计信息"""
        if not self.flight_data:
            return {}
        
        positions = np.array([d['position'] for d in self.flight_data])
        velocities = np.array([d['velocity'] for d in self.flight_data])
        
        stats = {
            'total_points': len(self.flight_data),
            'duration': self.flight_data[-1]['timestamp'] - self.flight_data[0]['timestamp'],
            'max_altitude': float(positions[:, 2].max()),
            'avg_altitude': float(positions[:, 2].mean()),
            'max_speed': float(np.linalg.norm(velocities, axis=1).max()),
            'avg_speed': float(np.linalg.norm(velocities, axis=1).mean()),
            'total_distance': self._calculate_distance(positions)
        }
        
        return stats
    
    def _calculate_distance(self, positions: np.ndarray) -> float:
        """计算总飞行距离"""
        total = 0.0
        for i in range(len(positions) - 1):
            total += np.linalg.norm(positions[i+1] - positions[i])
        return float(total)
    
    def save_to_npy(self, filename: Optional[str] = None) -> str:
        """保存为 NPY 格式"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/flight_{timestamp}.npy"
        
        np.save(filename, self.flight_data)
        print(f"✅ 数据已保存：{filename}")
        return filename
    
    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """保存为 CSV 格式"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/flight_{timestamp}.csv"
        
        try:
            import pandas as pd
            df = pd.DataFrame(self.flight_data)
            
            # 展开嵌套的列表
            for col in ['position', 'velocity', 'orientation']:
                if col in df.columns:
                    expanded = pd.DataFrame(df[col].tolist(), 
                                           columns=[f'{col}_x', f'{col}_y', f'{col}_z'])
                    df = pd.concat([df.drop(col, axis=1), expanded], axis=1)
            
            df.to_csv(filename, index=False)
            print(f"✅ CSV 已保存：{filename}")
        except ImportError:
            print("⚠️ 未安装 pandas，使用 JSON 格式保存")
            return self.save_to_json(filename.replace('.csv', '.json'))
        
        return filename
    
    def save_to_json(self, filename: Optional[str] = None) -> str:
        """保存为 JSON 格式"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/flight_{timestamp}.json"
        
        data = {
            'metadata': self.metadata,
            'statistics': self.get_statistics(),
            'data': self.flight_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON 已保存：{filename}")
        return filename
    
    def plot_trajectory(self, save_path: Optional[str] = None):
        """绘制飞行轨迹"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            if not self.flight_data:
                print("⚠️ 没有数据可绘制")
                return
            
            positions = np.array([d['position'] for d in self.flight_data])
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制轨迹
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'b-', linewidth=2, label='飞行轨迹')
            
            # 标记起点和终点
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                      c='green', s=100, label='起点', marker='o')
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                      c='red', s=100, label='终点', marker='x')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('无人机飞行轨迹')
            ax.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"✅ 轨迹图已保存：{save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("⚠️ 未安装 matplotlib，无法绘制轨迹")


def test_flight_recorder():
    """测试飞行记录器"""
    print("=" * 60)
    print("飞行记录器测试")
    print("=" * 60)
    
    recorder = FlightRecorder()
    
    # 模拟飞行数据
    recorder.start_recording({'pilot': 'test', 'mode': 'simulation'})
    
    for t in range(100):
        position = np.array([t * 0.1, t * 0.05, 2.0 + np.sin(t * 0.1)])
        velocity = np.array([0.1, 0.05, np.cos(t * 0.1) * 0.1])
        orientation = np.array([0.0, 0.0, t * 0.01])
        
        recorder.add_data_point(position, velocity, orientation)
        time.sleep(0.01)
    
    recorder.stop_recording()
    
    # 获取统计信息
    stats = recorder.get_statistics()
    print("\n飞行统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 保存数据
    recorder.save_to_npy()
    recorder.save_to_csv()
    recorder.save_to_json()
    
    # 绘制轨迹
    recorder.plot_trajectory(save_path="flight_trajectory.png")
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    test_flight_recorder()
