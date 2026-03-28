#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA 真实数据加载器
"""

import numpy as np
from pathlib import Path
import json

class CarlaDataLoader:
    """CARLA 真实数据加载器"""
    
    def __init__(self, data_path="data/"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.data = None
        self.metadata = None
    
    def load_data(self, town='Town01', weather='ClearNoon'):
        """
        加载真实的 CARLA 数据
        
        参数:
            town: CARLA 城镇名称 (Town01, Town02, Town03, Town04, Town05)
            weather: 天气条件 (ClearNoon, ClearSunset, CloudyNoon, WetNoon, MidRainyNoon)
        """
        data_file = self.data_path / f'carla_{town}_{weather}.npy'
        meta_file = self.data_path / f'carla_{town}_{weather}_metadata.json'
        
        if data_file.exists():
            self.data = np.load(data_file, allow_pickle=True)
            
            # 加载元数据
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            print("="*60)
            print("✅ 已加载真实的 CARLA 仿真数据")
            print("="*60)
            print(f"   数据来源: {self.metadata.get('source', 'CARLA_Simulator')}")
            print(f"   城镇: {town}")
            print(f"   天气: {weather}")
            print(f"   数据量: {len(self.data)} 帧")
            print(f"   时长: {self.data[-1]['timestamp']:.1f} 秒")
            
            if self.metadata:
                print(f"   包含场景: {', '.join(self.metadata.get('scenes', []))}")
            
            print("="*60)
            return self.data
        else:
            print(f"⚠️ 未找到 CARLA 数据文件: {data_file}")
            print("请先运行数据采集程序:")
            print("  python data/download_real_carla_data.py")
            return None
    
    def get_data_summary(self):
        """获取数据摘要"""
        if self.data is None:
            return {}
        
        speeds = [d['speed'] for d in self.data]
        violations = [d for d in self.data if d.get('is_violation', False)]
        
        # 统计各类型违规
        violation_stats = {}
        for v in violations:
            scene_type = v.get('scene_type', 'unknown')
            if scene_type not in violation_stats:
                violation_stats[scene_type] = 0
            violation_stats[scene_type] += 1
        
        return {
            'source': 'CARLA_Simulator',
            'total_frames': len(self.data),
            'duration': self.data[-1]['timestamp'],
            'avg_speed': np.mean(speeds),
            'max_speed': max(speeds),
            'min_speed': min(speeds),
            'total_violations': len(violations),
            'violation_stats': violation_stats,
            'town': self.data[0].get('town', 'Unknown'),
            'weather': self.data[0].get('weather', 'Unknown')
        }
    
    def get_ground_truth(self):
        """获取真实违规标签（用于评估）"""
        if self.data is None:
            return {}
        
        ground_truth = {
            'speeding': [],
            'red_light': [],
            'lane_violation': []
        }
        
        for d in self.data:
            if d.get('is_violation', False):
                scene_type = d.get('scene_type')
                if scene_type == 'speeding':
                    ground_truth['speeding'].append({
                        'timestamp': d['timestamp'],
                        'speed': d['speed'],
                        'limit': d['speed_limit']
                    })
                elif scene_type == 'red_light':
                    ground_truth['red_light'].append({
                        'timestamp': d['timestamp'],
                        'speed': d['speed']
                    })
                elif scene_type == 'lane':
                    ground_truth['lane_violation'].append({
                        'timestamp': d['timestamp'],
                        'lane_type': d['lane_type']
                    })
        
        return ground_truth
    
    def list_available_datasets(self):
        """列出所有可用的 CARLA 数据集"""
        datasets = []
        
        for file in self.data_path.glob('carla_*.npy'):
            # 解析文件名
            parts = file.stem.split('_')
            if len(parts) >= 3:
                town = parts[1]
                weather = parts[2]
                datasets.append({
                    'file': file.name,
                    'town': town,
                    'weather': weather,
                    'size': file.stat().st_size
                })
        
        return datasets


if __name__ == "__main__":
    loader = CarlaDataLoader()
    
    # 显示可用数据集
    datasets = loader.list_available_datasets()
    if datasets:
        print("📁 可用的 CARLA 数据集:")
        for ds in datasets:
            print(f"   - {ds['town']} / {ds['weather']} ({ds['file']})")
    
    # 加载数据
    data = loader.load_data(town='Town01', weather='ClearNoon')
    
    if data is not None:
        summary = loader.get_data_summary()
        print("\n📊 数据摘要:")
        for key, value in summary.items():
            print(f"   {key}: {value}")