from utils.logger import logger
from utils.screen import message
from src.data import Data, PilotData
import numpy as np
import os
import json
import datetime

class DataProcessor:
    """
    数据处理模块：在训练前对数据进行分析、清洗和预处理
    """
    
    def __init__(self):
        self.steering_angles = []
        self.throttles = []
        self.brakes = []
        self.speeds = []
        self.data_stats = {}
        
    def analyze_data(self, data):
        """
        分析数据集的统计信息
        """
        logger.info('Starting data analysis...')
        message('正在分析数据集...')
        
        # 提取所有数据
        for item in data:
            self.steering_angles.append(item.steering)
            self.throttles.append(item.throttle)
            self.brakes.append(item.brake)
            self.speeds.append(item.speed)
        
        # 计算统计信息
        self.data_stats = {
            'total_samples': len(data),
            'speed': {
                'min': np.min(self.speeds),
                'max': np.max(self.speeds),
                'mean': np.mean(self.speeds),
                'std': np.std(self.speeds),
                'median': np.median(self.speeds),
                'stationary_count': int(np.sum(np.array(self.speeds) == 0)),
                'stationary_percentage': float(np.sum(np.array(self.speeds) == 0) / len(self.speeds) * 100)
            },
            'steering': {
                'min': np.min(self.steering_angles),
                'max': np.max(self.steering_angles),
                'mean': np.mean(self.steering_angles),
                'std': np.std(self.steering_angles),
                'median': np.median(self.steering_angles)
            },
            'throttle': {
                'min': np.min(self.throttles),
                'max': np.max(self.throttles),
                'mean': np.mean(self.throttles),
                'std': np.std(self.throttles),
                'median': np.median(self.throttles)
            },
            'brake': {
                'min': np.min(self.brakes),
                'max': np.max(self.brakes),
                'mean': np.mean(self.brakes),
                'std': np.std(self.brakes),
                'median': np.median(self.brakes)
            }
        }
        
        logger.info(f'Data analysis completed. Total samples: {self.data_stats["total_samples"]}')
        return self.data_stats
    
    def detect_outliers(self, threshold=3.0):
        """
        检测异常值（使用Z-score方法）
        """
        message('正在检测异常值...')
        
        # 计算Z-score
        steering_z = np.abs((np.array(self.steering_angles) - np.mean(self.steering_angles)) / np.std(self.steering_angles))
        throttle_z = np.abs((np.array(self.throttles) - np.mean(self.throttles)) / np.std(self.throttles))
        brake_z = np.abs((np.array(self.brakes) - np.mean(self.brakes)) / np.std(self.brakes))
        
        # 找出异常值索引
        outliers = np.where((steering_z > threshold) | (throttle_z > threshold) | (brake_z > threshold))[0]
        
        self.data_stats['outliers'] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(self.steering_angles)) * 100,
            'indices': outliers.tolist()[:10]  # 只保存前10个索引
        }
        
        logger.info(f'Outlier detection completed. Found {len(outliers)} outliers ({(len(outliers)/len(self.steering_angles)*100):.2f}%)')
        return self.data_stats['outliers']
    
    def clean_data(self, data, remove_outliers=True):
        """
        清洗数据，去除异常值
        """
        message('正在清洗数据...')
        
        if remove_outliers and 'outliers' in self.data_stats:
            outlier_indices = set(self.data_stats['outliers']['indices'])
            cleaned_data = [item for i, item in enumerate(data) if i not in outlier_indices]
            
            # 更新统计信息
            cleaned_steering = [item.steering for item in cleaned_data]
            cleaned_throttle = [item.throttle for item in cleaned_data]
            cleaned_brake = [item.brake for item in cleaned_data]
            
            self.data_stats['cleaned'] = {
                'total_samples': len(cleaned_data),
                'removed_samples': len(data) - len(cleaned_data),
                'steering': {
                    'min': np.min(cleaned_steering),
                    'max': np.max(cleaned_steering),
                    'mean': np.mean(cleaned_steering),
                    'std': np.std(cleaned_steering)
                },
                'throttle': {
                    'min': np.min(cleaned_throttle),
                    'max': np.max(cleaned_throttle),
                    'mean': np.mean(cleaned_throttle),
                    'std': np.std(cleaned_throttle)
                },
                'brake': {
                    'min': np.min(cleaned_brake),
                    'max': np.max(cleaned_brake),
                    'mean': np.mean(cleaned_brake),
                    'std': np.std(cleaned_brake)
                }
            }
            
            logger.info(f'Data cleaning completed. Removed {len(data) - len(cleaned_data)} outlier samples')
            return cleaned_data
        else:
            return data
    
    def balance_data(self, data, max_samples_per_bin=1000):
        """
        平衡数据集，避免某些数据分布过于集中
        """
        message('正在平衡数据集...')

        # 按转向角度分箱
        bins = np.linspace(-1, 1, 21)  # 20个箱子
        digitized = np.digitize(self.steering_angles, bins)

        # 创建平衡后的数据
        balanced_data = []
        bin_indices = [[] for _ in range(len(bins) + 1)]

        for i, item in enumerate(data):
            bin_idx = digitized[i]
            if 0 <= bin_idx < len(bin_indices):
                bin_indices[bin_idx].append(i)

        # 对每个箱子进行采样
        for idx_list in bin_indices:
            if len(idx_list) > max_samples_per_bin:
                selected = np.random.choice(idx_list, max_samples_per_bin, replace=False)
            else:
                selected = idx_list

            for idx in selected:
                balanced_data.append(data[idx])

        self.data_stats['balanced'] = {
            'total_samples': len(balanced_data),
            'original_samples': len(data)
        }

        logger.info(f'Data balancing completed. Original: {len(data)}, Balanced: {len(balanced_data)}')
        return balanced_data
    
    def filter_stationary_data(self, data, speed_threshold=0.1):
        """
        过滤静止数据，基于速度阈值判断
        
        Args:
            data: 输入数据列表
            speed_threshold: 速度阈值，低于此值认为是静止状态
        
        Returns:
            过滤后的数据
        """
        message(f'正在过滤静止数据 (速度阈值 < {speed_threshold})...')
        
        stationary_indices = []
        for i, item in enumerate(data):
            if item.speed < speed_threshold:
                stationary_indices.append(i)
        
        filtered_data = [item for i, item in enumerate(data) if i not in stationary_indices]
        
        self.data_stats['stationary_filtered'] = {
            'total_samples': len(filtered_data),
            'original_samples': len(data),
            'removed_samples': len(data) - len(filtered_data),
            'removed_percentage': float((len(data) - len(filtered_data)) / len(data) * 100),
            'speed_threshold': speed_threshold
        }
        
        logger.info(f'Stationary data filtering completed. Removed {len(stationary_indices)} stationary samples ({(len(stationary_indices)/len(data)*100):.2f}%)')
        return filtered_data
    
    def save_stats(self, filename=None):
        """
        保存统计信息到JSON文件
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'data_stats_{timestamp}.json'
        
        os.makedirs('data_stats/', exist_ok=True)
        save_path = f'data_stats/{filename}'
        
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_native(self.data_stats), f, indent=4, ensure_ascii=False)
        
        logger.info(f'Data statistics saved to: {save_path}')
        return save_path
    
    def print_stats(self):
        """
        打印统计信息到控制台
        """
        stats = self.data_stats
        
        print('\n' + '='*60)
        print('                    数据统计报告')
        print('='*60)
        
        print(f"\n[基本信息]")
        print(f"  总样本数: {stats['total_samples']}")
        
        if 'speed' in stats:
            print(f"\n[速度分布]")
            print(f"  最小值: {stats['speed']['min']:.4f}")
            print(f"  最大值: {stats['speed']['max']:.4f}")
            print(f"  平均值: {stats['speed']['mean']:.4f}")
            print(f"  标准差: {stats['speed']['std']:.4f}")
            print(f"  中位数: {stats['speed']['median']:.4f}")
            print(f"  静止样本数: {stats['speed']['stationary_count']} ({stats['speed']['stationary_percentage']:.2f}%)")
        
        print(f"\n[转向角度分布]")
        print(f"  最小值: {stats['steering']['min']:.4f}")
        print(f"  最大值: {stats['steering']['max']:.4f}")
        print(f"  平均值: {stats['steering']['mean']:.4f}")
        print(f"  标准差: {stats['steering']['std']:.4f}")
        print(f"  中位数: {stats['steering']['median']:.4f}")
        
        print(f"\n[油门压力分布]")
        print(f"  最小值: {stats['throttle']['min']:.4f}")
        print(f"  最大值: {stats['throttle']['max']:.4f}")
        print(f"  平均值: {stats['throttle']['mean']:.4f}")
        print(f"  标准差: {stats['throttle']['std']:.4f}")
        print(f"  中位数: {stats['throttle']['median']:.4f}")
        
        print(f"\n[刹车压力分布]")
        print(f"  最小值: {stats['brake']['min']:.4f}")
        print(f"  最大值: {stats['brake']['max']:.4f}")
        print(f"  平均值: {stats['brake']['mean']:.4f}")
        print(f"  标准差: {stats['brake']['std']:.4f}")
        print(f"  中位数: {stats['brake']['median']:.4f}")
        
        if 'outliers' in stats:
            print(f"\n[异常值检测]")
            print(f"  异常值数量: {stats['outliers']['count']}")
            print(f"  占比: {stats['outliers']['percentage']:.2f}%")
        
        if 'cleaned' in stats:
            print(f"\n[数据清洗后]")
            print(f"  清洗后样本数: {stats['cleaned']['total_samples']}")
            print(f"  移除样本数: {stats['cleaned']['removed_samples']}")
        
        if 'balanced' in stats:
            print(f"\n[数据平衡后]")
            print(f"  平衡后样本数: {stats['balanced']['total_samples']}")
        
        if 'stationary_filtered' in stats:
            print(f"\n[静止数据过滤后]")
            print(f"  过滤后样本数: {stats['stationary_filtered']['total_samples']}")
            print(f"  移除样本数: {stats['stationary_filtered']['removed_samples']} ({stats['stationary_filtered']['removed_percentage']:.2f}%)")
            print(f"  速度阈值: {stats['stationary_filtered']['speed_threshold']}")
        
        print('\n' + '='*60)
    
    @staticmethod
    def process(data, enable_cleaning=True, enable_balancing=True, enable_stationary_filtering=False, speed_threshold=0.1):
        """
        完整的数据处理流程
        
        Args:
            data: 输入数据列表
            enable_cleaning: 是否启用数据清洗（去除异常值）
            enable_balancing: 是否启用数据平衡
            enable_stationary_filtering: 是否启用静止数据过滤
            speed_threshold: 静止判定速度阈值
        
        Returns:
            处理后的数据和统计信息
        """
        processor = DataProcessor()
        
        # 分析数据
        processor.analyze_data(data)
        
        # 过滤静止数据
        if enable_stationary_filtering:
            data = processor.filter_stationary_data(data, speed_threshold=speed_threshold)
        
        # 检测异常值
        processor.detect_outliers()
        
        # 清洗数据
        if enable_cleaning:
            data = processor.clean_data(data, remove_outliers=True)
        
        # 平衡数据
        if enable_balancing:
            data = processor.balance_data(data)
        
        # 打印统计信息
        processor.print_stats()
        
        # 保存统计信息
        processor.save_stats()
        
        return data, processor.data_stats
