#!/usr/bin/env python3
"""
CVIPS 行人安全增强数据收集系统 - 主程序（增强版）
"""
import sys
import os
import time
import random
import argparse
import traceback
import math
import threading
import json
from datetime import datetime
import gc
import psutil
import numpy as np
import signal

from carla_utils import setup_carla_path, import_carla_module, setup_carla_environment
from config_manager import ConfigManager
from annotation_generator import AnnotationGenerator
from data_validator import DataValidator
from lidar_processor import LidarProcessor, MultiSensorFusion
from multi_vehicle_manager import MultiVehicleManager
from pedestrian_safety_monitor import PedestrianSafetyMonitor
from scene_manager import SceneManager
from sensor_enhancer import SensorDataEnhancer, EnhancementConfig, WeatherType, EnhancementMethod
from v2x_communication import V2XCommunication

# 设置CARLA环境
carla_module, remaining_argv = setup_carla_environment()
carla = carla_module


class PerformanceMonitor:
    """性能监控器（增强版）"""

    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.frame_times = []
        self.first_frame_time = None
        self.last_frame_time = None
        self.gc_stats = []
        self.network_stats = []

        # 性能阈值
        self.warning_thresholds = {
            'memory_mb': config.get('performance', {}).get('memory_management', {}).get('early_stop_threshold', 450),
            'cpu_percent': 90,
            'frame_time_ms': 200  # 5 FPS
        }

        # 警告记录
        self.warnings = []

    def sample_memory(self):
        """采样内存使用"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            memory_mb = memory_info.rss / 1024 / 1024
            self.memory_samples.append(memory_mb)

            # 检查内存警告
            if memory_mb > self.warning_thresholds['memory_mb'] * 0.8:
                if not any('内存使用高' in w for w in self.warnings[-5:]):  # 避免重复警告
                    self.warnings.append(f"内存使用高: {memory_mb:.1f}MB")

            system_memory = psutil.virtual_memory()

            return {
                'process_mb': memory_mb,
                'process_vms_mb': memory_info.vms / 1024 / 1024,
                'system_total_mb': system_memory.total / 1024 / 1024,
                'system_used_percent': system_memory.percent,
                'system_available_mb': system_memory.available / 1024 / 1024,
                'warning': memory_mb > self.warning_thresholds['memory_mb'] * 0.8
            }
        except:
            memory_mb = 0
            self.memory_samples.append(memory_mb)
            return {
                'process_mb': memory_mb,
                'system_total_mb': 0,
                'system_used_percent': 0,
                'system_available_mb': 0,
                'warning': False
            }

    def sample_cpu(self):
        """采样CPU使用"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)

            # 检查CPU警告
            if cpu_percent > self.warning_thresholds['cpu_percent']:
                if not any('CPU使用高' in w for w in self.warnings[-5:]):
                    self.warnings.append(f"CPU使用高: {cpu_percent:.1f}%")

            per_core = psutil.cpu_percent(interval=0.1, percpu=True)

            return {
                'total_percent': cpu_percent,
                'per_core': per_core,
                'count': psutil.cpu_count(),
                'warning': cpu_percent > self.warning_thresholds['cpu_percent']
            }
        except:
            cpu_percent = 0
            self.cpu_samples.append(cpu_percent)
            return {
                'total_percent': cpu_percent,
                'per_core': [0],
                'count': 1,
                'warning': False
            }

    def sample_gc(self):
        """采样垃圾回收统计"""
        try:
            gc.collect()
            gc_stats = {
                'collected': gc.get_count()[0],
                'uncollectable': len(gc.garbage),
                'thresholds': gc.get_threshold(),
                'enabled': gc.isenabled()
            }
            self.gc_stats.append(gc_stats)
            return gc_stats
        except:
            return {}

    def record_frame_time(self, frame_time):
        """记录帧时间"""
        self.frame_times.append(frame_time)

        # 检查帧时间警告
        frame_time_ms = frame_time * 1000
        if frame_time_ms > self.warning_thresholds['frame_time_ms']:
            if not any('帧时间过长' in w for w in self.warnings[-5:]):
                self.warnings.append(f"帧时间过长: {frame_time_ms:.1f}ms")

        if self.first_frame_time is None:
            self.first_frame_time = time.time()
        self.last_frame_time = time.time()

    def get_performance_summary(self):
        """获取性能摘要"""
        if not self.frame_times or len(self.frame_times) < 2:
            avg_frame_time = 0
            fps = 0
        else:
            avg_frame_time = np.mean(self.frame_times)
            fps = len(self.frame_times) / max(0.1, (self.last_frame_time - self.first_frame_time))

        summary = {
            'total_runtime': time.time() - self.start_time,
            'memory_statistics': {
                'average_memory_mb': np.mean(self.memory_samples) if self.memory_samples else 0,
                'max_memory_mb': max(self.memory_samples) if self.memory_samples else 0,
                'min_memory_mb': min(self.memory_samples) if self.memory_samples else 0,
                'memory_warnings': len(
                    [m for m in self.memory_samples if m > self.warning_thresholds['memory_mb'] * 0.8])
            },
            'cpu_statistics': {
                'average_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0,
                'max_cpu_percent': max(self.cpu_samples) if self.cpu_samples else 0,
                'cpu_warnings': len([c for c in self.cpu_samples if c > self.warning_thresholds['cpu_percent']])
            },
            'frame_statistics': {
                'average_frame_time': avg_frame_time,
                'frames_per_second': fps,
                'total_frames': len(self.frame_times),
                'frame_time_warnings': len(
                    [f for f in self.frame_times if f * 1000 > self.warning_thresholds['frame_time_ms']])
            },
            'gc_statistics': {
                'total_collections': len(self.gc_stats),
                'average_collected': np.mean([s.get('collected', 0) for s in self.gc_stats]) if self.gc_stats else 0
            },
            'warnings': self.warnings[-10:]  # 最近10个警告
        }

        if self.frame_times and len(self.frame_times) >= 2:
            summary['frame_time_stats'] = {
                'p50': np.percentile(self.frame_times, 50),
                'p75': np.percentile(self.frame_times, 75),
                'p95': np.percentile(self.frame_times, 95),
                'p99': np.percentile(self.frame_times, 99) if len(self.frame_times) > 1 else 0,
                'std': np.std(self.frame_times)
            }
        else:
            summary['frame_time_stats'] = {
                'p50': 0,
                'p75': 0,
                'p95': 0,
                'p99': 0,
                'std': 0
            }

        return summary

    def check_emergency_stop(self):
        """检查是否需要紧急停止"""
        if not self.memory_samples:
            return False

        current_memory = self.memory_samples[-1]
        return current_memory > self.warning_thresholds['memory_mb']


class Log:
    """日志系统（增强版）"""

    LOG_LEVELS = {
        'DEBUG': 0,
        'INFO': 1,
        'WARNING': 2,
        'ERROR': 3,
        'CRITICAL': 4
    }

    current_level = 'INFO'
    log_file = None

    @classmethod
    def setup(cls, config):
        """设置日志系统"""
        cls.current_level = config.get('monitoring', {}).get('log_level', 'INFO')
        log_file = config.get('monitoring', {}).get('log_file', 'cvips.log')

        if config.get('monitoring', {}).get('enable_logging', True):
            cls.log_file = open(log_file, 'a', encoding='utf-8')

    @classmethod
    def _should_log(cls, level):
        """检查是否应该记录"""
        return cls.LOG_LEVELS[level] >= cls.LOG_LEVELS[cls.current_level]

    @staticmethod
    def info(msg):
        """信息日志"""
        if not Log._should_log('INFO'):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[INFO][{timestamp}] {msg}"
        print(f"\033[92m{log_msg}\033[0m")

        if Log.log_file:
            Log.log_file.write(log_msg + '\n')
            Log.log_file.flush()

    @staticmethod
    def warning(msg):
        """警告日志"""
        if not Log._should_log('WARNING'):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[WARNING][{timestamp}] {msg}"
        print(f"\033[93m{log_msg}\033[0m")

        if Log.log_file:
            Log.log_file.write(log_msg + '\n')
            Log.log_file.flush()

    @staticmethod
    def error(msg):
        """错误日志"""
        if not Log._should_log('ERROR'):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[ERROR][{timestamp}] {msg}"
        print(f"\033[91m{log_msg}\033[0m")

        if Log.log_file:
            Log.log_file.write(log_msg + '\n')
            Log.log_file.flush()

    @staticmethod
    def debug(msg):
        """调试日志"""
        if not Log._should_log('DEBUG'):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[DEBUG][{timestamp}] {msg}"
        print(f"\033[90m{log_msg}\033[0m")

        if Log.log_file:
            Log.log_file.write(log_msg + '\n')
            Log.log_file.flush()

    @staticmethod
    def performance(msg):
        """性能日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[PERF][{timestamp}] {msg}"
        print(f"\033[96m{log_msg}\033[0m")

        if Log.log_file:
            Log.log_file.write(log_msg + '\n')
            Log.log_file.flush()

    @staticmethod
    def safety(msg):
        """安全日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[SAFETY][{timestamp}] {msg}"
        print(f"\033[95m{log_msg}\033[0m")

        if Log.log_file:
            Log.log_file.write(log_msg + '\n')
            Log.log_file.flush()

    @staticmethod
    def critical(msg):
        """关键日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[CRITICAL][{timestamp}] {msg}"
        print(f"\033[91;1m{log_msg}\033[0m")

        if Log.log_file:
            Log.log_file.write(log_msg + '\n')
            Log.log_file.flush()

    @classmethod
    def cleanup(cls):
        """清理日志资源"""
        if cls.log_file:
            cls.log_file.close()
            cls.log_file = None


class EmergencyHandler:
    """紧急处理程序"""

    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.emergency_stop = False
        self.last_emergency_time = 0

        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理"""
        Log.critical(f"收到信号 {signum}，正在安全关闭...")
        self.emergency_stop = True
        self.data_collector.is_running = False

    def check_emergency(self):
        """检查紧急情况"""
        current_time = time.time()

        # 避免过于频繁的紧急检查
        if current_time - self.last_emergency_time < 5.0:
            return False

        self.last_emergency_time = current_time

        # 检查内存使用
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            memory_threshold = self.data_collector.config['performance'].get(
                'memory_management', {}).get('early_stop_threshold', 450)

            if memory_mb > memory_threshold:
                Log.critical(f"内存使用超过阈值: {memory_mb:.1f}MB > {memory_threshold}MB")
                return True

        except:
            pass

        return False

    def emergency_cleanup(self):
        """紧急清理"""
        Log.critical("执行紧急清理...")

        try:
            # 强制垃圾回收
            gc.collect()

            # 清理传感器数据
            for sensor_manager in self.data_collector.sensor_managers.values():
                if hasattr(sensor_manager, 'image_processor'):
                    if hasattr(sensor_manager.image_processor, 'image_cache'):
                        sensor_manager.image_processor.image_cache.clear()

            Log.critical("紧急清理完成")

        except Exception as e:
            Log.error(f"紧急清理失败: {e}")


class WeatherSystem:
    """天气系统（增强版）"""

    WEATHER_PRESETS = {
        'clear': {
            'cloudiness': 10,
            'precipitation': 0,
            'wind': 5,
            'fog_density': 0,
            'wetness': 0,
            'sun_altitude': 75
        },
        'rainy': {
            'cloudiness': 90,
            'precipitation': 80,
            'wind': 15,
            'fog_density': 20,
            'wetness': 80,
            'sun_altitude': 30
        },
        'cloudy': {
            'cloudiness': 70,
            'precipitation': 10,
            'wind': 10,
            'fog_density': 10,
            'wetness': 20,
            'sun_altitude': 45
        },
        'foggy': {
            'cloudiness': 50,
            'precipitation': 0,
            'fog_density': 60,
            'wind': 3,
            'wetness': 30,
            'sun_altitude': 20
        },
        'wet': {
            'cloudiness': 40,
            'precipitation': 0,
            'wind': 8,
            'fog_density': 10,
            'wetness': 60,
            'sun_altitude': 50
        }
    }

    TIME_PRESETS = {
        'noon': {'sun_altitude': 75, 'sun_azimuth': 0},
        'sunset': {'sun_altitude': 15, 'sun_azimuth': 270},
        'night': {'sun_altitude': -20, 'sun_azimuth': 0},
        'dawn': {'sun_altitude': 15, 'sun_azimuth': 90}
    }

    @staticmethod
    def create_weather(weather_type, time_of_day):
        """创建天气（增强版）"""
        weather = carla.WeatherParameters()

        if weather_type in WeatherSystem.WEATHER_PRESETS:
            preset = WeatherSystem.WEATHER_PRESETS[weather_type]
            weather.cloudiness = preset.get('cloudiness', 30)
            weather.precipitation = preset.get('precipitation', 0)
            weather.precipitation_deposits = preset.get('wetness', 0)
            weather.wind_intensity = preset.get('wind', 5)
            weather.fog_density = preset.get('fog_density', 0)
            weather.fog_distance = 5.0 if preset.get('fog_density', 0) > 50 else 15.0
            weather.wetness = preset.get('wetness', 0)

        if time_of_day in WeatherSystem.TIME_PRESETS:
            time_preset = WeatherSystem.TIME_PRESETS[time_of_day]
            weather.sun_altitude_angle = time_preset['sun_altitude']
            weather.sun_azimuth_angle = time_preset['sun_azimuth']

        # 根据时间调整亮度
        if time_of_day == 'night':
            weather.cloudiness = min(weather.cloudiness + 20, 100)

        return weather

    @staticmethod
    def create_dynamic_weather(base_weather, variation=0.1):
        """创建动态变化的天气"""
        dynamic = carla.WeatherParameters()

        # 基于基础天气添加随机变化
        for attr in ['cloudiness', 'precipitation', 'wind_intensity', 'fog_density']:
            if hasattr(base_weather, attr):
                base_value = getattr(base_weather, attr)
                variation_range = base_value * variation
                new_value = max(0, min(100, base_value + random.uniform(-variation_range, variation_range)))
                setattr(dynamic, attr, new_value)

        # 保持其他属性
        for attr in ['sun_altitude_angle', 'sun_azimuth_angle', 'precipitation_deposits', 'wetness']:
            if hasattr(base_weather, attr):
                setattr(dynamic, attr, getattr(base_weather, attr))

        return dynamic


class ImageProcessor:
    """图像处理器（增强版）"""

    def __init__(self, output_dir, config=None):
        self.output_dir = output_dir
        self.stitched_dir = os.path.join(output_dir, "stitched")
        os.makedirs(self.stitched_dir, exist_ok=True)

        self.config = config or {}
        img_config = self.config.get('image_processing', {})

        self.compress_images = img_config.get('compress_images', True)
        self.compression_quality = img_config.get('compression_quality', 85)
        self.enable_memory_cache = img_config.get('enable_memory_cache', True)
        self.image_cache = {}
        self.max_cache_size = img_config.get('max_cache_size', 50)

        self.stitch_layouts = {
            'vehicle_4cam': {
                'canvas_size': (640 * 2 + 20, 360 * 2 + 20),
                'positions': [(10, 10), (660, 10), (10, 390), (660, 390)],
                'tile_size': (640, 360)
            },
            'infrastructure_4cam': {
                'canvas_size': (640 * 2 + 20, 360 * 2 + 20),
                'positions': [(10, 10), (660, 10), (10, 390), (660, 390)],
                'tile_size': (640, 360)
            },
            'wide_view': {
                'canvas_size': (1280, 720),
                'positions': [(0, 0)],
                'tile_size': (1280, 720)
            }
        }

        self.stats = {
            'images_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0,
            'stitching_errors': 0,
            'saved_size_mb': 0
        }

    def stitch(self, image_paths, frame_num, view_type="vehicle"):
        """拼接图像（增强版）"""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            Log.warning("PIL未安装，跳过图像拼接")
            return False

        start_time = time.time()

        # 检查缓存
        cache_key = f"{view_type}_{frame_num}"
        if self.enable_memory_cache and cache_key in self.image_cache:
            cached_image = self.image_cache[cache_key]
            output_path = os.path.join(self.stitched_dir, f"{view_type}_{frame_num:06d}.jpg")

            try:
                if self.compress_images:
                    cached_image.save(output_path, "JPEG",
                                      quality=self.compression_quality,
                                      optimize=True,
                                      progressive=True)
                else:
                    cached_image.save(output_path, "PNG", optimize=True)

                self.stats['cache_hits'] += 1
                self.stats['images_processed'] += 1
                processing_time = time.time() - start_time
                self.stats['total_processing_time'] += processing_time

                return True
            except Exception as e:
                Log.error(f"保存缓存图像失败: {e}")
                return False

        self.stats['cache_misses'] += 1

        # 获取布局配置
        layout = self.stitch_layouts.get(view_type, self.stitch_layouts['vehicle_4cam'])
        canvas_size = layout['canvas_size']
        positions = layout['positions']
        tile_size = layout['tile_size']

        # 创建画布
        canvas = Image.new('RGB', canvas_size, (40, 40, 40))
        draw = ImageDraw.Draw(canvas)

        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        images_loaded = 0
        image_items = list(image_paths.items())

        for idx, (cam_name, img_path) in enumerate(image_items[:len(positions)]):
            if img_path and os.path.exists(img_path):
                try:
                    # 检查缓存
                    if img_path in self.image_cache:
                        img = self.image_cache[img_path]
                    else:
                        # 加载图像
                        img = Image.open(img_path)
                        img.load()

                        # 调整大小
                        if img.size != tile_size:
                            img = img.resize(tile_size, Image.Resampling.LANCZOS)

                        # 缓存图像
                        if self.enable_memory_cache:
                            self.image_cache[img_path] = img
                            if len(self.image_cache) > self.max_cache_size:
                                self._cleanup_cache()

                    # 添加到画布
                    if idx < len(positions):
                        canvas.paste(img, positions[idx])

                        # 添加摄像机标签
                        label = cam_name.replace('_', ' ').title()
                        draw.text((positions[idx][0] + 5, positions[idx][1] + 5),
                                  label, fill=(255, 255, 200), font=font)

                        images_loaded += 1

                except Exception as e:
                    Log.warning(f"加载图像失败 {cam_name}: {e}")

                    # 创建占位图像
                    placeholder = Image.new('RGB', tile_size, (80, 80, 80))
                    if idx < len(positions):
                        canvas.paste(placeholder, positions[idx])
                        draw.text((positions[idx][0] + 5, positions[idx][1] + 5),
                                  f"Error: {cam_name}", fill=(255, 100, 100), font=font)
            else:
                # 创建占位图像
                placeholder = Image.new('RGB', tile_size, (80, 80, 80))
                if idx < len(positions):
                    canvas.paste(placeholder, positions[idx])
                    draw.text((positions[idx][0] + 5, positions[idx][1] + 5),
                              f"Missing: {cam_name}", fill=(255, 200, 100), font=font)

        if images_loaded == 0:
            Log.warning(f"没有图像加载成功: {view_type}")
            self.stats['stitching_errors'] += 1
            return False

        # 添加帧信息和时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = f"Frame: {frame_num:06d} | {timestamp} | {view_type} | Images: {images_loaded}/{len(image_items)}"
        draw.text((10, canvas_size[1] - 25), info_text, fill=(200, 200, 255), font=font)

        # 保存图像
        output_path = os.path.join(self.stitched_dir, f"{view_type}_{frame_num:06d}.jpg")

        try:
            if self.compress_images:
                canvas.save(output_path, "JPEG",
                            quality=self.compression_quality,
                            optimize=True,
                            progressive=True,
                            subsampling='4:2:0')
            else:
                canvas.save(output_path, "PNG", optimize=True)

            # 计算节省的空间
            original_size = sum([os.path.getsize(p) for p in image_paths.values() if p and os.path.exists(p)])
            stitched_size = os.path.getsize(output_path)
            if original_size > 0:
                size_saving = (original_size - stitched_size) / (1024 * 1024)
                self.stats['saved_size_mb'] += max(0, size_saving)

        except Exception as e:
            Log.error(f"保存图像失败: {e}")
            self.stats['stitching_errors'] += 1
            return False

        # 更新缓存
        if self.enable_memory_cache:
            self.image_cache[cache_key] = canvas.copy()

        # 清理
        del canvas
        gc.collect()

        # 更新统计
        self.stats['images_processed'] += 1
        processing_time = time.time() - start_time
        self.stats['total_processing_time'] += processing_time

        # 定期报告
        if self.stats['images_processed'] % 50 == 0:
            self._report_statistics()

        return True

    def _cleanup_cache(self):
        """清理缓存"""
        if len(self.image_cache) > self.max_cache_size:
            keys_to_remove = list(self.image_cache.keys())[:len(self.image_cache) - self.max_cache_size]
            for key in keys_to_remove:
                del self.image_cache[key]
            Log.debug(f"清理图像缓存: 移除了{len(keys_to_remove)}个缓存项")

    def _report_statistics(self):
        """报告统计信息"""
        total_operations = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_operations > 0:
            cache_hit_rate = self.stats['cache_hits'] / total_operations
        else:
            cache_hit_rate = 0

        avg_time = 0
        if self.stats['images_processed'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['images_processed']

        Log.performance(f"图像处理统计: "
                        f"处理{self.stats['images_processed']}张, "
                        f"平均{avg_time:.3f}秒/张, "
                        f"缓存命中率{cache_hit_rate:.1%}, "
                        f"节省空间{self.stats['saved_size_mb']:.1f}MB")

    def get_stats(self):
        """获取统计信息"""
        total_operations = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_operations > 0:
            cache_hit_rate = self.stats['cache_hits'] / total_operations
        else:
            cache_hit_rate = 0

        avg_time = 0
        if self.stats['images_processed'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['images_processed']

        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'average_processing_time': avg_time,
            'current_cache_size': len(self.image_cache),
            'stitching_success_rate': (self.stats['images_processed'] - self.stats['stitching_errors']) /
                                      max(1, self.stats['images_processed'])
        }


class TrafficManager:
    """交通管理器（增强版）"""

    def __init__(self, world, config):
        self.world = world
        self.config = config
        self.vehicles = []
        self.pedestrians = []
        self.bicycles = []
        self.motorcycles = []

        # 设置随机种子
        seed = config['scenario'].get('seed', random.randint(1, 10000))
        random.seed(seed)
        np.random.seed(seed)
        Log.info(f"随机种子: {seed}")

        self.batch_spawn = config.get('batch_spawn', True)
        self.max_spawn_attempts = config.get('max_spawn_attempts', 10)

        # 行人安全相关
        self.pedestrian_safety_zones = config.get('pedestrian_safety_zones', True)
        self.crosswalk_density = config.get('crosswalk_density', 'medium')

        self.spawn_stats = {
            'total_attempts': 0,
            'successful_spawns': 0,
            'failed_spawns': 0,
            'total_spawn_time': 0,
            'vehicle_types': {},
            'pedestrian_types': {}
        }

    def spawn_ego_vehicle(self):
        """生成主车（增强版）"""
        blueprint_lib = self.world.get_blueprint_library()

        # 优先选择的车辆类型
        preferred_vehicles = [
            'vehicle.tesla.model3',
            'vehicle.audi.tt',
            'vehicle.nissan.patrol',
            'vehicle.mercedes.coupe',
            'vehicle.bmw.grandtourer',
            'vehicle.lincoln.mkz2017',
            'vehicle.chevrolet.impala'
        ]

        vehicle_bp = None
        for vtype in preferred_vehicles:
            if blueprint_lib.filter(vtype):
                vehicle_bp = random.choice(blueprint_lib.filter(vtype))
                break

        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))

        # 设置车辆属性
        if vehicle_bp.has_attribute('color'):
            colors = vehicle_bp.get_attribute('color').recommended_values
            if colors:
                vehicle_bp.set_attribute('color', random.choice(colors))

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            Log.error("没有可用的生成点")
            return None

        # 尝试生成
        for attempt in range(self.max_spawn_attempts):
            self.spawn_stats['total_attempts'] += 1
            spawn_point = random.choice(spawn_points)

            try:
                start_time = time.time()
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

                # 设置自动驾驶
                vehicle.set_autopilot(True)

                # 设置初始控制
                control = carla.VehicleControl()
                control.throttle = 0.3
                control.steer = 0.0
                vehicle.apply_control(control)

                spawn_time = time.time() - start_time
                self.spawn_stats['total_spawn_time'] += spawn_time
                self.spawn_stats['successful_spawns'] += 1

                # 记录车辆类型
                vehicle_type = vehicle.type_id
                self.spawn_stats['vehicle_types'][vehicle_type] = \
                    self.spawn_stats['vehicle_types'].get(vehicle_type, 0) + 1

                Log.info(f"主车生成: {vehicle_type}, 位置: {spawn_point.location}, 时间: {spawn_time:.3f}秒")

                return vehicle

            except Exception as e:
                self.spawn_stats['failed_spawns'] += 1
                if attempt == self.max_spawn_attempts - 1:
                    Log.warning(f"主车生成失败: {e}")
                else:
                    time.sleep(0.1)

        return None

    def spawn_traffic(self, center_location):
        """生成交通（增强版）"""
        start_time = time.time()

        # 生成车辆
        vehicles_spawned = self._spawn_vehicles_batch() if self.batch_spawn else self._spawn_vehicles()

        # 生成行人
        pedestrians_spawned = self._spawn_pedestrians(center_location)

        # 生成自行车
        bicycles_spawned = self._spawn_bicycles(center_location)

        # 生成摩托车
        motorcycles_spawned = self._spawn_motorcycles(center_location)

        total_time = time.time() - start_time

        Log.info(f"交通生成完成: "
                 f"{vehicles_spawned}辆车, "
                 f"{pedestrians_spawned}个行人, "
                 f"{bicycles_spawned}辆自行车, "
                 f"{motorcycles_spawned}辆摩托车, "
                 f"用时: {total_time:.2f}秒")

        # 生成统计报告
        success_rate = self.spawn_stats['successful_spawns'] / max(1, self.spawn_stats['total_attempts'])
        avg_spawn_time = self.spawn_stats['total_spawn_time'] / max(1, self.spawn_stats['successful_spawns'])

        Log.performance(f"生成统计: "
                        f"成功率{success_rate:.1%}, "
                        f"平均生成时间{avg_spawn_time:.3f}秒, "
                        f"车辆类型: {len(self.spawn_stats['vehicle_types'])}种, "
                        f"行人类型: {len(self.spawn_stats['pedestrian_types'])}种")

        return vehicles_spawned + pedestrians_spawned + bicycles_spawned + motorcycles_spawned

    def _spawn_vehicles_batch(self):
        """批量生成车辆"""
        blueprint_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            Log.warning("没有可用的车辆生成点")
            return 0

        num_vehicles = min(self.config['traffic']['background_vehicles'], 20)
        spawned = 0

        # 准备批处理命令
        batch_commands = []
        available_points = spawn_points.copy()
        random.shuffle(available_points)

        for i in range(num_vehicles):
            if i >= len(available_points):
                break

            try:
                # 选择车辆类型
                vehicle_types = self.config['traffic']['vehicle_types']
                available_types = [vt for vt in vehicle_types if blueprint_lib.filter(vt)]

                if not available_types:
                    vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
                else:
                    vehicle_bp = random.choice(blueprint_lib.filter(random.choice(available_types)))

                # 设置颜色
                if vehicle_bp.has_attribute('color'):
                    colors = vehicle_bp.get_attribute('color').recommended_values
                    if colors:
                        vehicle_bp.set_attribute('color', random.choice(colors))

                spawn_point = available_points[i]

                # 添加随机偏移
                offset_x = random.uniform(-3.0, 3.0)
                offset_y = random.uniform(-3.0, 3.0)
                location = carla.Location(
                    x=spawn_point.location.x + offset_x,
                    y=spawn_point.location.y + offset_y,
                    z=spawn_point.location.z
                )

                # 稍微调整方向
                rotation = carla.Rotation(
                    pitch=spawn_point.rotation.pitch,
                    yaw=spawn_point.rotation.yaw + random.uniform(-10, 10),
                    roll=spawn_point.rotation.roll
                )

                transform = carla.Transform(location, rotation)
                batch_commands.append((vehicle_bp, transform))

            except Exception as e:
                Log.debug(f"准备车辆生成命令失败: {e}")
                continue

        # 执行批处理生成
        for vehicle_bp, transform in batch_commands:
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, transform)
                vehicle.set_autopilot(True)

                # 设置初始速度
                control = carla.VehicleControl()
                control.throttle = random.uniform(0.2, 0.5)
                vehicle.apply_control(control)

                self.vehicles.append(vehicle)
                spawned += 1

                self.spawn_stats['successful_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1

                # 记录类型
                vehicle_type = vehicle.type_id
                self.spawn_stats['vehicle_types'][vehicle_type] = \
                    self.spawn_stats['vehicle_types'].get(vehicle_type, 0) + 1

            except Exception as e:
                self.spawn_stats['failed_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
                Log.debug(f"车辆生成失败: {e}")

            # 小延迟以避免过载
            if spawned % 3 == 0:
                time.sleep(0.05)

        return spawned

    def _spawn_pedestrians(self, center_location):
        """生成行人（增强版）"""
        blueprint_lib = self.world.get_blueprint_library()

        num_peds = min(self.config['traffic']['pedestrians'], 30)
        spawned = 0

        # 获取行人类型
        pedestrian_types = self.config['traffic']['pedestrian_types']
        available_types = [pt for pt in pedestrian_types if blueprint_lib.filter(pt)]

        if not available_types:
            available_types = list(blueprint_lib.filter('walker.pedestrian.*'))

        if not available_types:
            Log.warning("没有可用的行人生成蓝图")
            return 0

        for i in range(num_peds):
            try:
                # 选择行人类型
                ped_bp = random.choice(available_types)

                # 根据场景类型调整位置
                scenario_name = self.config['scenario']['name'].lower()

                if 'school' in scenario_name or 'crossing' in scenario_name:
                    # 学校或人行横道场景：行人在中心区域更密集
                    angle = random.uniform(0, 2 * math.pi)
                    if i < num_peds * 0.7:  # 70%的行人在中心区域
                        distance = random.uniform(3.0, 10.0)
                    else:
                        distance = random.uniform(10.0, 20.0)
                else:
                    # 普通场景
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(5.0, 25.0)

                # 计算位置
                location = carla.Location(
                    x=center_location.x + distance * math.cos(angle),
                    y=center_location.y + distance * math.sin(angle),
                    z=center_location.z + 0.5
                )

                # 确保位置在地面上
                waypoint = self.world.get_map().get_waypoint(location)
                if waypoint:
                    location.z = waypoint.transform.location.z + 0.5

                # 创建行人
                pedestrian = self.world.spawn_actor(ped_bp, carla.Transform(location))

                # 设置行人控制器
                controller_bp = blueprint_lib.find('controller.ai.walker')
                if controller_bp:
                    controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
                    controller.start()

                    # 设置目标位置
                    target_location = carla.Location(
                        x=center_location.x + random.uniform(-20, 20),
                        y=center_location.y + random.uniform(-20, 20),
                        z=location.z
                    )
                    controller.go_to_location(target_location)

                    # 设置速度
                    controller.set_max_speed(random.uniform(0.5, 2.0))

                self.pedestrians.append(pedestrian)
                spawned += 1

                self.spawn_stats['successful_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1

                # 记录类型
                pedestrian_type = pedestrian.type_id
                self.spawn_stats['pedestrian_types'][pedestrian_type] = \
                    self.spawn_stats['pedestrian_types'].get(pedestrian_type, 0) + 1

            except Exception as e:
                self.spawn_stats['failed_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
                Log.debug(f"行人生成失败: {e}")

        return spawned

    def _spawn_bicycles(self, center_location):
        """生成自行车"""
        blueprint_lib = self.world.get_blueprint_library()

        num_bikes = min(self.config['traffic'].get('bicycles', 0), 10)
        spawned = 0

        # 查找自行车蓝图
        bike_bps = list(blueprint_lib.filter('vehicle.bh*')) + list(blueprint_lib.filter('vehicle.diamondback*'))

        if not bike_bps:
            Log.debug("没有可用的自行车蓝图")
            return 0

        for i in range(num_bikes):
            try:
                bike_bp = random.choice(bike_bps)

                # 计算位置（在行人附近）
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(8.0, 15.0)

                location = carla.Location(
                    x=center_location.x + distance * math.cos(angle),
                    y=center_location.y + distance * math.sin(angle),
                    z=center_location.z + 0.3
                )

                # 确保位置在地面上
                waypoint = self.world.get_map().get_waypoint(location)
                if waypoint:
                    location.z = waypoint.transform.location.z + 0.3

                # 创建自行车
                rotation = carla.Rotation(0, random.uniform(0, 360), 0)
                bicycle = self.world.spawn_actor(bike_bp, carla.Transform(location, rotation))

                self.bicycles.append(bicycle)
                spawned += 1

                self.spawn_stats['successful_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1

            except Exception as e:
                self.spawn_stats['failed_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
                Log.debug(f"自行车生成失败: {e}")

        return spawned

    def _spawn_motorcycles(self, center_location):
        """生成摩托车"""
        blueprint_lib = self.world.get_blueprint_library()

        num_motos = min(self.config['traffic'].get('motorcycles', 0), 5)
        spawned = 0

        # 查找摩托车蓝图
        moto_bps = list(blueprint_lib.filter('vehicle.harley-davidson*')) + \
                   list(blueprint_lib.filter('vehicle.yamaha*')) + \
                   list(blueprint_lib.filter('vehicle.kawasaki*'))

        if not moto_bps:
            Log.debug("没有可用的摩托车蓝图")
            return 0

        for i in range(num_motos):
            try:
                moto_bp = random.choice(moto_bps)

                # 计算位置
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(10.0, 20.0)

                location = carla.Location(
                    x=center_location.x + distance * math.cos(angle),
                    y=center_location.y + distance * math.sin(angle),
                    z=center_location.z + 0.5
                )

                # 确保位置在地面上
                waypoint = self.world.get_map().get_waypoint(location)
                if waypoint:
                    location.z = waypoint.transform.location.z + 0.5

                # 创建摩托车
                rotation = carla.Rotation(0, random.uniform(0, 360), 0)
                motorcycle = self.world.spawn_actor(moto_bp, carla.Transform(location, rotation))
                motorcycle.set_autopilot(True)

                self.motorcycles.append(motorcycle)
                spawned += 1

                self.spawn_stats['successful_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1

            except Exception as e:
                self.spawn_stats['failed_spawns'] += 1
                self.spawn_stats['total_attempts'] += 1
                Log.debug(f"摩托车生成失败: {e}")

        return spawned

    def add_pedestrian_safety_features(self, center_location):
        """添加行人安全设施"""
        features = []

        try:
            # 添加交通锥
            if self.pedestrian_safety_zones:
                from scene_manager import SceneManager
                cones = SceneManager.spawn_traffic_cones(self.world, center_location, num_cones=8)
                features.extend(cones)

            # 添加行人横道标记
            if self.crosswalk_density in ['medium', 'high']:
                from scene_manager import SceneManager
                crosswalks = SceneManager.spawn_pedestrian_safety_features(
                    self.world, center_location, 'crosswalk')
                features.extend(crosswalks)

            Log.info(f"添加了 {len(features)} 个行人安全设施")

        except Exception as e:
            Log.warning(f"添加行人安全设施失败: {e}")

        return features

    def cleanup(self):
        """清理交通管理器"""
        Log.info("开始清理交通管理器...")

        cleanup_stats = {
            'vehicles_destroyed': 0,
            'pedestrians_destroyed': 0,
            'bicycles_destroyed': 0,
            'motorcycles_destroyed': 0,
            'errors': 0
        }

        try:
            # 清理行人
            for pedestrian in self.pedestrians:
                try:
                    if pedestrian and pedestrian.is_alive:
                        pedestrian.destroy()
                        cleanup_stats['pedestrians_destroyed'] += 1
                except:
                    cleanup_stats['errors'] += 1

            self.pedestrians.clear()

            # 清理自行车
            for bicycle in self.bicycles:
                try:
                    if bicycle and bicycle.is_alive:
                        bicycle.destroy()
                        cleanup_stats['bicycles_destroyed'] += 1
                except:
                    cleanup_stats['errors'] += 1

            self.bicycles.clear()

            # 清理摩托车
            for motorcycle in self.motorcycles:
                try:
                    if motorcycle and motorcycle.is_alive:
                        motorcycle.destroy()
                        cleanup_stats['motorcycles_destroyed'] += 1
                except:
                    cleanup_stats['errors'] += 1

            self.motorcycles.clear()

            # 清理车辆
            for vehicle in self.vehicles:
                try:
                    if vehicle and vehicle.is_alive:
                        vehicle.destroy()
                        cleanup_stats['vehicles_destroyed'] += 1
                except:
                    cleanup_stats['errors'] += 1

            self.vehicles.clear()

            Log.info(f"交通管理器清理完成: "
                     f"车辆{cleanup_stats['vehicles_destroyed']}, "
                     f"行人{cleanup_stats['pedestrians_destroyed']}, "
                     f"自行车{cleanup_stats['bicycles_destroyed']}, "
                     f"摩托车{cleanup_stats['motorcycles_destroyed']}, "
                     f"错误{cleanup_stats['errors']}")

        except Exception as e:
            Log.error(f"清理交通管理器失败: {e}")


class SensorManager:
    """传感器管理器（增强版）"""

    def __init__(self, world, config, data_dir):
        self.world = world
        self.config = config
        self.data_dir = data_dir
        self.sensors = []

        self.frame_counter = 0
        self.last_capture_time = 0
        self.last_performance_sample = 0

        # 帧率控制
        self.target_fps = config['performance'].get('frame_rate_limit', 8.0)
        self.min_frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.125
        self.last_frame_time = 0
        self.frame_skip_count = 0
        self.max_frame_skip = 3

        # 数据缓冲
        self.vehicle_buffer = {}
        self.infra_buffer = {}
        self.buffer_lock = threading.RLock()

        # 状态控制
        self.is_running = True
        self.is_paused = False

        # 性能监控
        self.performance_monitor = PerformanceMonitor(config)

        # 处理器
        self.image_processor = ImageProcessor(data_dir, config.get('image_processing', {}))
        self.lidar_processor = None
        self.fusion_manager = None

        # 配置参数
        self.batch_size = config['performance'].get('batch_size', 8)
        self.enable_async_processing = config['performance'].get('enable_async_processing', True)
        self.max_workers = config['performance'].get('max_workers', 4)

        # LiDAR处理
        if config['sensors'].get('lidar_sensors', 0) > 0:
            lidar_config = config['performance'].get('lidar_processing', {})
            self.lidar_processor = LidarProcessor(data_dir, lidar_config)

        # 融合处理
        if config['output'].get('save_fusion', False):
            fusion_config = config['performance'].get('fusion', {})
            self.fusion_manager = MultiSensorFusion(data_dir, fusion_config)

        # 统计
        self.sensor_stats = {
            'total_images': 0,
            'total_lidar_frames': 0,
            'image_capture_times': [],
            'lidar_processing_times': [],
            'sensor_errors': 0,
            'dropped_frames': 0
        }

    def setup_cameras(self, vehicle, center_location, vehicle_id=0):
        """设置摄像头（增强版）"""
        vehicle_cams = self._setup_vehicle_cameras(vehicle, vehicle_id)
        infra_cams = self._setup_infrastructure_cameras(center_location)

        Log.info(f"摄像头设置完成: {vehicle_cams}车辆摄像头 + {infra_cams}基础设施摄像头")
        return vehicle_cams + infra_cams

    def _setup_vehicle_cameras(self, vehicle, vehicle_id):
        """设置车辆摄像头"""
        if not vehicle:
            return 0

        camera_configs = {
            'front_wide': {
                'loc': (2.0, 0, 1.8),
                'rot': (0, -3, 0),
                'fov': 120,
                'description': '前视广角摄像头'
            },
            'front_narrow': {
                'loc': (2.0, 0, 1.6),
                'rot': (0, 0, 0),
                'fov': 60,
                'description': '前视窄角摄像头'
            },
            'right_side': {
                'loc': (0.5, 1.0, 1.5),
                'rot': (0, -2, 90),
                'fov': 100,
                'description': '右侧摄像头'
            },
            'left_side': {
                'loc': (0.5, -1.0, 1.5),
                'rot': (0, -2, -90),
                'fov': 100,
                'description': '左侧摄像头'
            },
            'rear': {
                'loc': (-1.5, 0, 1.7),
                'rot': (0, -5, 180),
                'fov': 100,
                'description': '后视摄像头'
            }
        }

        installed = 0
        for cam_name, config_data in camera_configs.items():
            if self._create_camera(cam_name, config_data, vehicle, 'vehicle', vehicle_id):
                installed += 1

        return installed

    def _setup_infrastructure_cameras(self, center_location):
        """设置基础设施摄像头"""
        camera_configs = [
            {
                'name': 'north',
                'offset': (0, -25, 15),
                'rotation': (0, -30, 180),
                'description': '北向路口摄像头'
            },
            {
                'name': 'south',
                'offset': (0, 25, 15),
                'rotation': (0, -30, 0),
                'description': '南向路口摄像头'
            },
            {
                'name': 'east',
                'offset': (25, 0, 15),
                'rotation': (0, -30, -90),
                'description': '东向路口摄像头'
            },
            {
                'name': 'west',
                'offset': (-25, 0, 15),
                'rotation': (0, -30, 90),
                'description': '西向路口摄像头'
            },
            {
                'name': 'overhead',
                'offset': (0, 0, 30),
                'rotation': (-90, 0, 0),
                'description': '俯视摄像头'
            }
        ]

        installed = 0
        for cam_config in camera_configs:
            sensor_config = {
                'loc': (
                    center_location.x + cam_config['offset'][0],
                    center_location.y + cam_config['offset'][1],
                    center_location.z + cam_config['offset'][2]
                ),
                'rot': cam_config['rotation'],
                'fov': 90,
                'description': cam_config['description']
            }

            if self._create_camera(cam_config['name'], sensor_config, None, 'infrastructure'):
                installed += 1

        return installed

    def setup_lidar(self, vehicle, vehicle_id=0):
        """设置LiDAR传感器"""
        if not vehicle or not self.config['sensors'].get('lidar_sensors', 0) > 0:
            return 0

        try:
            blueprint_lib = self.world.get_blueprint_library()
            lidar_bp = blueprint_lib.find('sensor.lidar.ray_cast')

            if not lidar_bp:
                Log.warning("未找到LiDAR传感器蓝图")
                return 0

            lidar_config = self.config['sensors'].get('lidar_config', {})

            # 设置LiDAR参数
            lidar_bp.set_attribute('channels', str(lidar_config.get('channels', 32)))
            lidar_bp.set_attribute('range', str(lidar_config.get('range', 100)))
            lidar_bp.set_attribute('points_per_second', str(lidar_config.get('points_per_second', 100000)))
            lidar_bp.set_attribute('rotation_frequency', str(lidar_config.get('rotation_frequency', 10)))

            # 设置视野
            lidar_bp.set_attribute('upper_fov', str(lidar_config.get('upper_fov', 15)))
            lidar_bp.set_attribute('lower_fov', str(lidar_config.get('lower_fov', -25)))
            lidar_bp.set_attribute('horizontal_fov', '360')

            # 设置位置和方向
            lidar_location = carla.Location(x=0, y=0, z=2.5)
            lidar_rotation = carla.Rotation(0, 0, 0)
            lidar_transform = carla.Transform(lidar_location, lidar_rotation)

            # 生成LiDAR传感器
            lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

            def lidar_callback(lidar_data):
                """LiDAR数据回调"""
                if not self.is_running or self.is_paused:
                    return

                try:
                    current_time = time.time()
                    frame_start_time = time.time()

                    # 检查采集间隔
                    if current_time - self.last_capture_time >= self.config['sensors']['capture_interval']:
                        if self.lidar_processor:
                            try:
                                start_process = time.time()

                                # 处理LiDAR数据
                                metadata = self.lidar_processor.process_lidar_data(lidar_data, self.frame_counter)

                                process_time = time.time() - start_process
                                self.sensor_stats['lidar_processing_times'].append(process_time)
                                self.sensor_stats['total_lidar_frames'] += 1

                                # 创建融合文件
                                if metadata and self.fusion_manager:
                                    vehicle_image_path = None
                                    with self.buffer_lock:
                                        if self.vehicle_buffer:
                                            for cam_name, img_path in self.vehicle_buffer.items():
                                                if os.path.exists(img_path):
                                                    vehicle_image_path = img_path
                                                    break

                                    sensor_data = {
                                        'lidar': os.path.join(self.data_dir, "lidar",
                                                              f"lidar_{self.frame_counter:06d}.bin")
                                    }
                                    if vehicle_image_path:
                                        sensor_data['camera'] = vehicle_image_path

                                    self.fusion_manager.create_synchronization_file(
                                        self.frame_counter, sensor_data)

                            except Exception as e:
                                self.sensor_stats['sensor_errors'] += 1
                                if self.is_running:
                                    Log.error(f"LiDAR处理失败: {e}")

                    # 记录帧时间
                    frame_time = time.time() - frame_start_time
                    self.performance_monitor.record_frame_time(frame_time)

                except Exception as e:
                    self.sensor_stats['sensor_errors'] += 1
                    if self.is_running:
                        Log.error(f"LiDAR回调错误: {e}")

            # 开始监听
            lidar_sensor.listen(lidar_callback)
            self.sensors.append(lidar_sensor)

            Log.info(f"LiDAR传感器已安装: {lidar_bp.id}, 通道数: {lidar_config.get('channels', 32)}")
            return 1

        except Exception as e:
            Log.error(f"LiDAR安装失败: {e}")
            return 0

    def _create_camera(self, name, config, parent, sensor_type, vehicle_id=0):
        """创建摄像头"""
        try:
            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')

            if not blueprint:
                Log.warning(f"未找到摄像头蓝图: sensor.camera.rgb")
                return False

            # 设置摄像头参数
            img_size = self.config['sensors'].get('image_size', [1280, 720])
            blueprint.set_attribute('image_size_x', str(img_size[0]))
            blueprint.set_attribute('image_size_y', str(img_size[1]))
            blueprint.set_attribute('fov', str(config.get('fov', 90)))

            # 设置后处理效果
            camera_config = self.config['sensors'].get('camera_config', {})
            if camera_config.get('post_processing') == 'semantic':
                blueprint.set_attribute('enable_postprocess_effects', 'True')

            # 设置位置和方向
            location = carla.Location(config['loc'][0], config['loc'][1], config['loc'][2])
            rotation = carla.Rotation(config['rot'][0], config['rot'][1], config['rot'][2])
            transform = carla.Transform(location, rotation)

            # 生成摄像头
            if parent:
                camera = self.world.spawn_actor(blueprint, transform, attach_to=parent)
            else:
                camera = self.world.spawn_actor(blueprint, transform)

            # 创建保存目录
            if sensor_type == 'vehicle' and vehicle_id > 0:
                save_dir = os.path.join(self.data_dir, "raw", f"vehicle_{vehicle_id}", name)
            else:
                save_dir = os.path.join(self.data_dir, "raw", sensor_type, name)

            os.makedirs(save_dir, exist_ok=True)

            # 创建回调函数
            callback = self._create_callback(save_dir, name, sensor_type, vehicle_id)
            camera.listen(callback)

            self.sensors.append(camera)

            Log.debug(f"摄像头创建成功: {name} ({sensor_type}), 位置: {location}")
            return True

        except Exception as e:
            Log.warning(f"创建摄像头 {name} 失败: {e}")
            return False

    def _create_callback(self, save_dir, name, sensor_type, vehicle_id=0):
        """创建回调函数"""
        capture_interval = self.config['sensors']['capture_interval']

        def callback(image):
            """摄像头数据回调"""
            if not self.is_running or self.is_paused:
                return

            try:
                current_time = time.time()
                frame_start_time = time.time()

                # 帧率控制
                time_since_last_frame = current_time - self.last_frame_time
                if time_since_last_frame < self.min_frame_interval:
                    self.frame_skip_count += 1
                    if self.frame_skip_count > self.max_frame_skip:
                        self.frame_skip_count = 0
                    else:
                        self.sensor_stats['dropped_frames'] += 1
                        return
                else:
                    self.frame_skip_count = 0

                # 检查采集间隔
                if current_time - self.last_capture_time >= capture_interval:
                    self.frame_counter += 1
                    self.last_capture_time = current_time
                    self.last_frame_time = current_time

                    # 保存图像
                    capture_start = time.time()
                    filename = os.path.join(save_dir, f"{name}_{self.frame_counter:06d}.png")
                    image.save_to_disk(filename, carla.ColorConverter.Raw)
                    capture_time = time.time() - capture_start

                    self.sensor_stats['image_capture_times'].append(capture_time)
                    self.sensor_stats['total_images'] += 1

                    # 缓冲图像路径
                    with self.buffer_lock:
                        if sensor_type == 'vehicle':
                            self.vehicle_buffer[name] = filename

                            # 检查是否触发拼接
                            if len(self.vehicle_buffer) >= 4:
                                self._trigger_stitching(self.vehicle_buffer.copy(),
                                                        self.frame_counter,
                                                        f'vehicle_{vehicle_id}')
                                self.vehicle_buffer.clear()
                        else:
                            self.infra_buffer[name] = filename

                            # 检查是否触发拼接
                            if len(self.infra_buffer) >= 4:
                                self._trigger_stitching(self.infra_buffer.copy(),
                                                        self.frame_counter,
                                                        'infrastructure')
                                self.infra_buffer.clear()

                    # 性能监控
                    if current_time - self.last_performance_sample >= 5.0:
                        memory_info = self.performance_monitor.sample_memory()
                        cpu_info = self.performance_monitor.sample_cpu()

                        if self.frame_counter % 20 == 0:
                            Log.performance(f"系统监控 - "
                                            f"内存: {memory_info['process_mb']:.1f}MB, "
                                            f"CPU: {cpu_info['total_percent']:.1f}%, "
                                            f"线程数: {threading.active_count()}, "
                                            f"帧数: {self.frame_counter}")

                        self.last_performance_sample = current_time

                    # 记录帧时间
                    frame_time = time.time() - frame_start_time
                    self.performance_monitor.record_frame_time(frame_time)

                    # 定期垃圾回收
                    if self.frame_counter % 50 == 0:
                        gc.collect()

                        # 报告传感器统计
                        if self.sensor_stats['total_images'] > 0:
                            avg_capture_time = np.mean(self.sensor_stats['image_capture_times'][-100:]) \
                                if len(self.sensor_stats['image_capture_times']) > 0 else 0
                            Log.performance(f"传感器统计: "
                                            f"图像{self.sensor_stats['total_images']}张, "
                                            f"平均捕获时间{avg_capture_time:.3f}秒, "
                                            f"错误{self.sensor_stats['sensor_errors']}个")

            except Exception as e:
                self.sensor_stats['sensor_errors'] += 1
                if self.is_running:
                    Log.error(f"传感器回调错误: {e}")

        return callback

    def _trigger_stitching(self, image_buffer, frame_num, view_type):
        """触发图像拼接"""
        if self.enable_async_processing:
            try:
                from concurrent.futures import ThreadPoolExecutor

                # 使用线程池异步处理
                with ThreadPoolExecutor(max_workers=min(2, self.max_workers)) as executor:
                    future = executor.submit(
                        self.image_processor.stitch,
                        image_buffer,
                        frame_num,
                        view_type
                    )
                    # 可以在这里添加回调处理结果
            except:
                # 异步处理失败，回退到同步处理
                self.image_processor.stitch(image_buffer, frame_num, view_type)
        else:
            # 同步处理
            self.image_processor.stitch(image_buffer, frame_num, view_type)

    def get_frame_count(self):
        """获取帧计数"""
        return self.frame_counter

    def pause(self):
        """暂停传感器"""
        self.is_paused = True

    def resume(self):
        """恢复传感器"""
        self.is_paused = False

    def generate_sensor_summary(self):
        """生成传感器摘要"""
        summary = {
            'total_sensors': len(self.sensors),
            'frame_count': self.frame_counter,
            'lidar_data': None,
            'fusion_data': None,
            'performance': self.performance_monitor.get_performance_summary(),
            'sensor_stats': self.sensor_stats,
            'image_processor_stats': self.image_processor.get_stats()
        }

        # 添加图像捕获统计
        if self.sensor_stats['image_capture_times']:
            summary['sensor_stats']['avg_image_capture_time'] = \
                np.mean(self.sensor_stats['image_capture_times'])
            summary['sensor_stats']['max_image_capture_time'] = \
                max(self.sensor_stats['image_capture_times'])

        # 添加LiDAR处理统计
        if self.sensor_stats['lidar_processing_times']:
            summary['sensor_stats']['avg_lidar_process_time'] = \
                np.mean(self.sensor_stats['lidar_processing_times'])
            summary['sensor_stats']['max_lidar_process_time'] = \
                max(self.sensor_stats['lidar_processing_times'])

        # LiDAR数据摘要
        if self.lidar_processor:
            summary['lidar_data'] = self.lidar_processor.generate_lidar_summary()

        # 融合数据摘要
        if self.fusion_manager:
            summary['fusion_data'] = self.fusion_manager.generate_fusion_report()

        return summary

    def cleanup(self):
        """清理传感器"""
        Log.info(f"安全清理 {len(self.sensors)} 个传感器...")

        self.is_running = False

        # 停止所有传感器
        for sensor in self.sensors:
            try:
                if hasattr(sensor, 'stop'):
                    sensor.stop()
                    time.sleep(0.001)
            except:
                pass

        time.sleep(0.2)

        # 刷新LiDAR批处理数据
        if self.lidar_processor:
            try:
                self.lidar_processor.flush_batch()
                self.lidar_processor.cleanup()
            except:
                pass

        # 销毁传感器
        for i, sensor in enumerate(self.sensors):
            try:
                if hasattr(sensor, 'destroy'):
                    sensor.destroy()
            except:
                pass

            if i % 5 == 0:
                time.sleep(0.01)

        self.sensors.clear()

        # 清理图像缓存
        if hasattr(self.image_processor, 'image_cache'):
            try:
                self.image_processor.image_cache.clear()
            except:
                pass

        # 清理数据缓冲
        with self.buffer_lock:
            self.vehicle_buffer.clear()
            self.infra_buffer.clear()

        # 垃圾回收
        gc.collect()

        Log.info("传感器清理完成")


class DataCollector:
    """数据收集器（增强版）"""

    def __init__(self, config):
        self.config = config
        self.client = None
        self.world = None
        self.ego_vehicles = []
        self.scene_center = None

        # 设置输出目录
        self.setup_directories()

        # 管理器
        self.traffic_manager = None
        self.sensor_managers = {}
        self.multi_vehicle_manager = None
        self.v2x_communication = None
        self.safety_monitor = None
        self.scene_manager = None

        # 状态
        self.start_time = None
        self.is_running = False
        self.collected_frames = 0

        # 性能监控
        self.performance_monitor = PerformanceMonitor(config)

        # 紧急处理
        self.emergency_handler = EmergencyHandler(self)

        # 输出格式
        self.output_format = config.get('output_format', 'standard')

        # 行人安全增强
        self.pedestrian_safety_mode = config['scenario'].get('pedestrian_safety_mode', True)

    def setup_directories(self):
        """设置输出目录"""
        scenario = self.config['scenario']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = os.path.join(
            self.config['output']['data_dir'],
            f"{scenario['name']}_{scenario['town']}_{timestamp}"
        )

        # 创建必要目录
        directories = [
            "raw/vehicle_1",
            "raw/vehicle_2",
            "raw/infrastructure",
            "stitched",
            "lidar",
            "fusion",
            "calibration",
            "cooperative/v2x_messages",
            "cooperative/shared_perception",
            "v2xformer_format",
            "kitti_format",
            "metadata",
            "safety_reports",
            "risk_maps",
            "emergency_events",
            "enhanced"
        ]

        for subdir in directories:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

        Log.info(f"数据目录: {self.output_dir}")

    def connect(self):
        """连接到CARLA服务器"""
        for attempt in range(1, 6):
            try:
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(15.0)

                town = self.config['scenario']['town']
                self.world = self.client.load_world(town)

                # 设置世界参数
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)

                Log.info(f"连接成功: {town} (尝试 {attempt}/5)")
                return True

            except Exception as e:
                Log.warning(f"连接尝试 {attempt}/5 失败: {str(e)[:100]}")
                time.sleep(3)

        return False

    def setup_scene(self):
        """设置场景（增强版）"""
        # 设置天气
        weather_cfg = self.config['scenario']
        weather = WeatherSystem.create_weather(weather_cfg['weather'], weather_cfg['time_of_day'])
        self.world.set_weather(weather)
        Log.info(f"天气: {weather_cfg['weather']}, 时间: {weather_cfg['time_of_day']}")

        # 获取场景中心
        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            self.scene_center = spawn_points[len(spawn_points) // 2].location
        else:
            self.scene_center = carla.Location(0, 0, 0)

        # 初始化交通管理器
        self.traffic_manager = TrafficManager(self.world, self.config)

        # 生成主车
        num_ego_vehicles = min(self.config['cooperative'].get('num_coop_vehicles', 2) + 1, 3)
        for i in range(num_ego_vehicles):
            ego_vehicle = self.traffic_manager.spawn_ego_vehicle()
            if ego_vehicle:
                self.ego_vehicles.append(ego_vehicle)
                Log.info(f"主车 {i + 1} 生成: {ego_vehicle.type_id}")
            else:
                Log.warning(f"主车 {i + 1} 生成失败")

        if not self.ego_vehicles:
            Log.error("没有主车生成成功")
            return False

        # 生成交通
        traffic_count = self.traffic_manager.spawn_traffic(self.scene_center)
        Log.info(f"生成交通: {traffic_count} 个对象")

        # 添加行人安全设施
        if self.pedestrian_safety_mode:
            safety_features = self.traffic_manager.add_pedestrian_safety_features(self.scene_center)
            Log.info(f"添加行人安全设施: {len(safety_features)} 个")

        # 初始化V2X通信
        if self.config['v2x']['enabled']:
            self.v2x_communication = V2XCommunication(self.config['v2x'])

            for i, vehicle in enumerate(self.ego_vehicles):
                location = vehicle.get_location()
                self.v2x_communication.register_node(
                    f'vehicle_{vehicle.id}',
                    (location.x, location.y, location.z),
                    {'type': 'vehicle', 'capabilities': ['bsm', 'rsm', 'emergency_brake']}
                )

        # 初始化多车辆管理器
        self.multi_vehicle_manager = MultiVehicleManager(
            self.world,
            self.config,
            self.output_dir
        )

        self.multi_vehicle_manager.ego_vehicles = self.ego_vehicles

        # 生成协同车辆
        num_coop_vehicles = self.config['cooperative'].get('num_coop_vehicles', 2)
        coop_vehicles = self.multi_vehicle_manager.spawn_cooperative_vehicles(num_coop_vehicles)
        Log.info(f"生成协同车辆: {len(coop_vehicles)} 辆")

        # 注册V2X节点
        if self.v2x_communication:
            for vehicle in coop_vehicles:
                location = vehicle.get_location()
                self.v2x_communication.register_node(
                    f'vehicle_{vehicle.id}',
                    (location.x, location.y, location.z),
                    {'type': 'vehicle', 'capabilities': ['bsm', 'rsm']}
                )

        # 初始化行人安全监控器
        self.safety_monitor = PedestrianSafetyMonitor(self.world, self.output_dir)

        # 初始化场景管理器
        self.scene_manager = SceneManager()

        # 等待场景稳定
        Log.info("等待场景稳定...")
        time.sleep(5.0)

        return True

    def setup_sensors(self):
        """设置传感器（增强版）"""
        for i, vehicle in enumerate(self.ego_vehicles):
            sensor_manager = SensorManager(self.world, self.config, self.output_dir)

            # 设置摄像头
            cameras = sensor_manager.setup_cameras(vehicle, self.scene_center, i + 1)
            if cameras == 0:
                Log.error(f"车辆 {i + 1} 没有摄像头安装成功")
                return False

            # 设置LiDAR
            lidars = sensor_manager.setup_lidar(vehicle, i + 1)

            Log.info(f"车辆 {i + 1} 传感器: {cameras}摄像头 + {lidars}LiDAR")

            self.sensor_managers[vehicle.id] = sensor_manager

        return True

    def collect_data(self):
        """收集数据（增强版）"""
        duration = self.config['scenario']['duration']
        Log.info(f"开始数据收集，时长: {duration}秒")

        self.start_time = time.time()
        self.is_running = True

        # 计时器
        last_update = time.time()
        last_v2x_update = time.time()
        last_perception_share = time.time()
        last_performance_sample = time.time()
        last_detailed_log = time.time()
        last_memory_check = time.time()
        last_safety_check = time.time()
        last_weather_change = time.time()

        # 状态标志
        memory_warning_issued = False
        early_stop_triggered = False

        # 内存阈值
        memory_warning_threshold = 350
        memory_critical_threshold = 400
        early_stop_threshold = 450

        try:
            while time.time() - self.start_time < duration and self.is_running:
                current_time = time.time()
                elapsed = current_time - self.start_time

                # 检查紧急停止
                if self.emergency_handler.check_emergency():
                    Log.critical("检测到紧急情况，执行紧急停止...")
                    self.emergency_handler.emergency_cleanup()
                    early_stop_triggered = True
                    break

                # 内存监控
                if current_time - last_memory_check >= 3.0:
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / (1024 * 1024)

                        if memory_mb > early_stop_threshold:
                            Log.critical(f"内存使用超过临界值({early_stop_threshold}MB): {memory_mb:.1f}MB")
                            early_stop_triggered = True
                            break
                        elif memory_mb > memory_critical_threshold:
                            if not memory_warning_issued:
                                Log.warning(f"内存使用严重过高: {memory_mb:.1f}MB")
                                self._force_memory_cleanup()
                                memory_warning_issued = True
                        elif memory_mb > memory_warning_threshold:
                            if not memory_warning_issued:
                                Log.warning(f"内存使用较高: {memory_mb:.1f}MB")
                                memory_warning_issued = True
                        elif memory_mb < memory_warning_threshold * 0.8:
                            memory_warning_issued = False

                    except Exception as e:
                        Log.debug(f"内存检查失败: {e}")

                    last_memory_check = current_time

                # 更新车辆状态
                if self.multi_vehicle_manager:
                    self.multi_vehicle_manager.update_vehicle_states()

                # V2X通信更新
                v2x_interval = self.config['v2x'].get('update_interval', 1.0)
                if self.v2x_communication and current_time - last_v2x_update >= v2x_interval:
                    self._update_v2x_communication()
                    last_v2x_update = current_time

                # 协同感知共享
                if (not memory_warning_issued and
                        self.config['cooperative'].get('enable_shared_perception', True) and
                        current_time - last_perception_share >= 2.0):
                    self._share_perception_data()
                    last_perception_share = current_time

                # 行人安全检查
                if current_time - last_safety_check >= 1.0 and self.safety_monitor:
                    safety_report = self.safety_monitor.check_pedestrian_safety()

                    # 高风险情况处理
                    if safety_report['risk_distribution']['high'] > 0:
                        Log.safety(f"行人安全警告: {safety_report['risk_distribution']['high']}个高风险情况")
                        self._broadcast_pedestrian_warnings(safety_report)

                    # 临界情况紧急处理
                    if safety_report['risk_distribution']['critical'] > 0:
                        Log.safety(f"行人安全紧急警告: {safety_report['risk_distribution']['critical']}个临界情况")
                        self._handle_critical_safety_situation(safety_report)

                    last_safety_check = current_time

                # 动态天气变化
                if current_time - last_weather_change >= 30.0:
                    self._update_dynamic_weather()
                    last_weather_change = current_time

                # 性能监控
                if current_time - last_performance_sample >= 10.0:
                    memory_info = self.performance_monitor.sample_memory()
                    cpu_info = self.performance_monitor.sample_cpu()

                    if current_time - last_detailed_log >= 60.0:
                        Log.performance("详细性能监控:")
                        Log.performance(f"  进程内存: {memory_info['process_mb']:.1f}MB")
                        Log.performance(f"  系统内存: {memory_info['system_used_percent']:.1f}%使用率")
                        Log.performance(f"  CPU: {cpu_info['total_percent']:.1f}% ({cpu_info['count']}核心)")
                        Log.performance(f"  总帧数: {self.collected_frames}")
                        last_detailed_log = current_time
                    else:
                        Log.performance(f"系统监控 - "
                                        f"内存: {memory_info['process_mb']:.1f}MB, "
                                        f"CPU: {cpu_info['total_percent']:.1f}%, "
                                        f"活跃线程: {threading.active_count()}, "
                                        f"帧数: {self.collected_frames}")

                    last_performance_sample = current_time

                    # 垃圾回收
                    gc.collect()

                # 进度更新
                if current_time - last_update >= 10.0:
                    total_frames = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())
                    self.collected_frames = total_frames
                    progress = (elapsed / duration) * 100

                    if total_frames > 0:
                        frames_per_second = total_frames / elapsed
                        remaining_time = (duration - elapsed)
                        eta_minutes = remaining_time / 60
                    else:
                        remaining_time = duration - elapsed
                        eta_minutes = remaining_time / 60

                    Log.info(f"进度: {elapsed:.0f}/{duration}秒 ({progress:.1f}%) | "
                             f"总帧数: {total_frames} | "
                             f"帧率: {frames_per_second:.1f} FPS | "
                             f"剩余: {eta_minutes:.1f}分钟")
                    last_update = current_time

                # 短暂休眠以减少CPU使用
                time.sleep(0.01)

        except KeyboardInterrupt:
            Log.info("数据收集被用户中断")
        except Exception as e:
            Log.error(f"数据收集错误: {e}")
            traceback.print_exc()
        finally:
            self.is_running = False
            elapsed = time.time() - self.start_time

            # 更新收集的帧数
            self.collected_frames = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())

            if self.collected_frames > 0:
                performance_summary = self.performance_monitor.get_performance_summary()

                if early_stop_triggered:
                    Log.warning("数据收集因内存过高而提前终止")
                else:
                    Log.info(f"收集完成: {self.collected_frames}帧, 用时: {elapsed:.1f}秒")

                fps = self.collected_frames / max(elapsed, 0.1)
                Log.info(f"平均帧率: {fps:.2f} FPS")
                Log.info(f"最大内存使用: {performance_summary['memory_statistics']['max_memory_mb']:.1f} MB")
                Log.info(f"平均CPU使用: {performance_summary['cpu_statistics']['average_cpu_percent']:.1f}%")

                # 生成行人安全报告
                if self.safety_monitor:
                    final_report = self.safety_monitor.generate_final_report()
                    Log.safety(f"行人安全最终报告:")
                    Log.safety(f"  高风险: {final_report['risk_distribution']['high']}次")
                    Log.safety(f"  临界风险: {final_report['risk_distribution']['critical']}次")
                    Log.safety(f"  安全评分: {final_report['safety_score']:.1f}/100")
            else:
                Log.warning("未收集到任何数据帧")

            # 保存元数据和生成摘要
            self._save_metadata()
            self._print_summary()

            # 数据格式转换
            if self.output_format != 'standard':
                self._convert_to_target_format()

    def _broadcast_pedestrian_warnings(self, safety_report):
        """广播行人警告（增强版）"""
        if not self.multi_vehicle_manager:
            return

        high_risk_count = safety_report.get('risk_distribution', {}).get('high', 0)
        critical_count = safety_report.get('risk_distribution', {}).get('critical', 0)

        if high_risk_count > 0 or critical_count > 0:
            # 获取所有车辆
            all_vehicles = self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles

            for vehicle in all_vehicles:
                if not hasattr(vehicle, 'is_alive') or not vehicle.is_alive:
                    continue

                try:
                    location = vehicle.get_location()
                    velocity = vehicle.get_velocity()
                    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6  # km/h

                    # 查找附近的潜在行人（模拟）
                    nearby_pedestrians = []

                    # 模拟检测：随机生成行人位置
                    for _ in range(random.randint(0, 3)):
                        angle = random.uniform(0, 2 * math.pi)
                        distance = random.uniform(5.0, 25.0)

                        pedestrian_location = (
                            location.x + distance * math.cos(angle),
                            location.y + distance * math.sin(angle),
                            location.z
                        )

                        # 计算距离和相对速度
                        actual_distance = math.sqrt(
                            (location.x - pedestrian_location[0]) ** 2 +
                            (location.y - pedestrian_location[1]) ** 2
                        )

                        if actual_distance < 30.0:  # 只广播近距离行人
                            # 评估风险
                            if actual_distance < 5.0:
                                risk_level = 'critical'
                            elif actual_distance < 10.0:
                                risk_level = 'high'
                            elif actual_distance < 20.0:
                                risk_level = 'medium'
                            else:
                                risk_level = 'low'

                            nearby_pedestrians.append({
                                'location': pedestrian_location,
                                'distance': actual_distance,
                                'risk_level': risk_level
                            })

                    # 广播警告
                    for pedestrian in nearby_pedestrians:
                        if pedestrian['risk_level'] in ['high', 'critical']:
                            self.multi_vehicle_manager.share_pedestrian_warning(
                                vehicle.id,
                                pedestrian['location'],
                                pedestrian['distance'],
                                speed,
                                pedestrian_id=random.randint(1000, 9999)
                            )

                except Exception as e:
                    Log.debug(f"广播行人警告失败: {e}")

    def _handle_critical_safety_situation(self, safety_report):
        """处理临界安全情况"""
        critical_count = safety_report.get('risk_distribution', {}).get('critical', 0)

        if critical_count > 0:
            Log.safety("⚠ 检测到临界安全情况，执行紧急措施")

            # 1. 向所有车辆发送紧急制动警告
            if self.v2x_communication:
                for vehicle in self.ego_vehicles:
                    try:
                        location = vehicle.get_location()
                        warning_data = {
                            'type': 'emergency_brake',
                            'location': (location.x, location.y, location.z),
                            'severity': 'critical',
                            'reason': 'pedestrian_critical_risk',
                            'timestamp': time.time()
                        }

                        self.v2x_communication.broadcast_roadside_safety_message(
                            f'emergency_system',
                            warning_data
                        )
                    except:
                        pass

            # 2. 降低车辆速度
            for vehicle in self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles:
                try:
                    if hasattr(vehicle, 'is_alive') and vehicle.is_alive:
                        control = carla.VehicleControl()
                        control.throttle = 0.0
                        control.brake = 0.5  # 轻度制动
                        vehicle.apply_control(control)
                except:
                    pass

            # 3. 记录紧急事件
            emergency_event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'pedestrian_critical_risk',
                'critical_count': critical_count,
                'safety_report': safety_report,
                'actions_taken': ['v2x_warning', 'speed_reduction']
            }

            emergency_dir = os.path.join(self.output_dir, "emergency_events")
            os.makedirs(emergency_dir, exist_ok=True)

            event_file = os.path.join(emergency_dir, f"emergency_{int(time.time())}.json")
            try:
                with open(event_file, 'w', encoding='utf-8') as f:
                    json.dump(emergency_event, f, indent=2)
            except:
                pass

    def _force_memory_cleanup(self):
        """强制内存清理"""
        Log.info("执行强制内存清理...")

        # 清理图像缓存
        for sensor_manager in self.sensor_managers.values():
            if hasattr(sensor_manager, 'image_processor'):
                if hasattr(sensor_manager.image_processor, 'image_cache'):
                    try:
                        old_size = len(sensor_manager.image_processor.image_cache)
                        sensor_manager.image_processor.image_cache.clear()
                        Log.debug(f"清理图像缓存: 释放了{old_size}个缓存项")
                    except:
                        pass

        # 刷新LiDAR批处理
        for sensor_manager in self.sensor_managers.values():
            if hasattr(sensor_manager, 'lidar_processor'):
                try:
                    sensor_manager.lidar_processor.flush_batch()
                    Log.debug("刷新LiDAR批处理数据")
                except:
                    pass

        # 清理融合缓存
        for sensor_manager in self.sensor_managers.values():
            if hasattr(sensor_manager, 'fusion_manager'):
                try:
                    sensor_manager.fusion_manager.cleanup()
                    Log.debug("清理融合缓存")
                except:
                    pass

        # 垃圾回收
        gc.collect()

        Log.info("强制内存清理完成")

    def _update_v2x_communication(self):
        """更新V2X通信"""
        if not self.v2x_communication:
            return

        # 更新车辆状态
        all_vehicles = self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles

        for vehicle in all_vehicles:
            if not hasattr(vehicle, 'is_alive') or not vehicle.is_alive:
                continue

            try:
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                transform = vehicle.get_transform()

                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6  # km/h

                vehicle_data = {
                    'position': (location.x, location.y, location.z),
                    'speed': speed,
                    'heading': transform.rotation.yaw,
                    'acceleration': (0, 0, 0),
                    'vehicle_id': vehicle.id,
                    'timestamp': time.time()
                }

                self.v2x_communication.broadcast_basic_safety_message(
                    f'vehicle_{vehicle.id}',
                    vehicle_data
                )

            except Exception as e:
                Log.debug(f"V2X状态更新失败: {e}")

        # 检查并处理接收到的消息
        for vehicle in self.ego_vehicles:
            messages = self.v2x_communication.get_messages_for_node(f'vehicle_{vehicle.id}')
            if messages and len(messages) > 0:
                Log.debug(f"车辆 {vehicle.id} 收到 {len(messages)} 条V2X消息")

                # 处理安全警告
                safety_messages = [m for m in messages if
                                   m['message']['message_type'] in ['warning', 'emergency_brake']]
                if safety_messages:
                    self._process_v2x_safety_warnings(vehicle, safety_messages)

    def _process_v2x_safety_warnings(self, vehicle, warnings):
        """处理V2X安全警告"""
        for warning in warnings:
            message = warning['message']
            data = message['data']

            if data.get('type') == 'pedestrian':
                Log.safety(f"车辆 {vehicle.id} 收到行人警告: {data.get('severity', 'unknown')}风险")

                # 根据风险等级采取行动
                if data.get('severity') in ['high', 'critical']:
                    try:
                        control = carla.VehicleControl()
                        control.throttle = 0.0
                        control.brake = 0.3
                        vehicle.apply_control(control)
                        Log.safety(f"车辆 {vehicle.id} 执行减速")
                    except:
                        pass

            elif data.get('type') == 'emergency_brake':
                Log.safety(f"车辆 {vehicle.id} 收到紧急制动警告")

                try:
                    control = carla.VehicleControl()
                    control.throttle = 0.0
                    control.brake = 1.0  # 紧急制动
                    vehicle.apply_control(control)
                except:
                    pass

    def _share_perception_data(self):
        """共享感知数据"""
        if not self.multi_vehicle_manager or not self.config['cooperative']['enable_shared_perception']:
            return

        for vehicle in self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles:
            if not hasattr(vehicle, 'is_alive') or not vehicle.is_alive:
                continue

            # 模拟物体检测
            detected_objects = self._simulate_object_detection(vehicle)

            if detected_objects:
                self.multi_vehicle_manager.share_perception_data(vehicle.id, detected_objects)

                # 保存共享感知数据
                frame_num = sum(mgr.get_frame_count() for mgr in self.sensor_managers.values())
                if frame_num % 10 == 0:  # 每10帧保存一次
                    self.multi_vehicle_manager.save_shared_perception(frame_num)

    def _simulate_object_detection(self, vehicle):
        """模拟物体检测"""
        detected_objects = []

        # 检测其他车辆
        all_vehicles = self.ego_vehicles + self.multi_vehicle_manager.cooperative_vehicles

        for other_vehicle in all_vehicles:
            if other_vehicle.id == vehicle.id or not hasattr(other_vehicle, 'is_alive') or not other_vehicle.is_alive:
                continue

            try:
                location = other_vehicle.get_location()
                velocity = other_vehicle.get_velocity()
                distance = vehicle.get_location().distance(location)

                if distance < 50.0:  # 检测范围50米
                    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6

                    obj_data = {
                        'class': 'vehicle',
                        'type': other_vehicle.type_id,
                        'position': {'x': location.x, 'y': location.y, 'z': location.z},
                        'velocity': {'x': velocity.x, 'y': velocity.y, 'z': velocity.z, 'speed_kmh': speed},
                        'distance': distance,
                        'confidence': max(0.7, 1.0 - distance / 50.0),
                        'size': {'width': 1.8, 'length': 4.5, 'height': 1.5},
                        'id': other_vehicle.id,
                        'timestamp': time.time()
                    }
                    detected_objects.append(obj_data)
            except:
                pass

        # 检测行人
        if self.traffic_manager and self.traffic_manager.pedestrians:
            for pedestrian in self.traffic_manager.pedestrians[:10]:  # 只检查前10个行人
                if not hasattr(pedestrian, 'is_alive') or not pedestrian.is_alive:
                    continue

                try:
                    location = pedestrian.get_location()
                    distance = vehicle.get_location().distance(location)

                    if distance < 30.0:  # 行人检测范围30米
                        obj_data = {
                            'class': 'pedestrian',
                            'type': pedestrian.type_id,
                            'position': {'x': location.x, 'y': location.y, 'z': location.z},
                            'distance': distance,
                            'confidence': max(0.6, 1.0 - distance / 30.0),
                            'size': {'width': 0.5, 'length': 0.5, 'height': 1.7},
                            'id': pedestrian.id,
                            'timestamp': time.time(),
                            'risk_level': 'high' if distance < 10.0 else 'medium' if distance < 20.0 else 'low'
                        }
                        detected_objects.append(obj_data)
                except:
                    pass

        return detected_objects

    def _update_dynamic_weather(self):
        """更新动态天气"""
        try:
            current_weather = self.world.get_weather()
            dynamic_weather = WeatherSystem.create_dynamic_weather(current_weather, variation=0.15)
            self.world.set_weather(dynamic_weather)

            Log.debug(f"天气更新: "
                      f"云量 {dynamic_weather.cloudiness:.0f}%, "
                      f"降水 {dynamic_weather.precipitation:.0f}%, "
                      f"雾密度 {dynamic_weather.fog_density:.0f}%")
        except:
            pass

    def _save_metadata(self):
        """保存元数据"""
        total_frames = self.collected_frames

        metadata = {
            'scenario': self.config['scenario'],
            'traffic': self.config['traffic'],
            'sensors': self.config['sensors'],
            'v2x': self.config['v2x'],
            'cooperative': self.config['cooperative'],
            'output_format': self.output_format,
            'output': self.config['output'],
            'collection': {
                'start_time': self.start_time,
                'duration': round(time.time() - self.start_time, 2),
                'total_frames': total_frames,
                'frame_rate': round(total_frames / max(time.time() - self.start_time, 0.1),
                                    2) if total_frames > 0 else 0,
                'completed': True,
                'early_stop': self.emergency_handler.emergency_stop
            }
        }

        if total_frames > 0:
            # 性能摘要
            performance_summary = self.performance_monitor.get_performance_summary()
            metadata['performance'] = performance_summary

            # 传感器摘要
            sensor_summaries = {}
            for vehicle_id, sensor_manager in self.sensor_managers.items():
                sensor_summaries[vehicle_id] = sensor_manager.generate_sensor_summary()
            metadata['sensor_summaries'] = sensor_summaries

            # V2X状态
            if self.v2x_communication:
                metadata['v2x_status'] = self.v2x_communication.get_network_status()

            # 协同摘要
            if self.multi_vehicle_manager:
                metadata['cooperative_summary'] = self.multi_vehicle_manager.generate_summary()

            # 安全报告
            if self.safety_monitor:
                metadata['safety_report'] = self.safety_monitor.generate_final_report()

            # 场景描述
            if self.scene_manager:
                scene_info = {
                    'center_location': {
                        'x': self.scene_center.x,
                        'y': self.scene_center.y,
                        'z': self.scene_center.z
                    } if self.scene_center else None,
                    'town': self.config['scenario']['town'],
                    'pedestrian_safety_mode': self.pedestrian_safety_mode
                }
                metadata['scene_info'] = scene_info

        meta_path = os.path.join(self.output_dir, "metadata", "collection_info.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        Log.info(f"元数据保存: {meta_path}")

    def _print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("数据收集摘要")
        print("=" * 60)

        # 统计文件
        raw_dirs = [d for d in os.listdir(os.path.join(self.output_dir, "raw"))
                    if os.path.isdir(os.path.join(self.output_dir, "raw", d))]

        total_raw_images = 0
        for raw_dir in raw_dirs:
            raw_path = os.path.join(self.output_dir, "raw", raw_dir)
            for root, dirs, files in os.walk(raw_path):
                total_raw_images += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])

        print(f"原始图像: {total_raw_images} 张")

        # LiDAR数据
        lidar_dir = os.path.join(self.output_dir, "lidar")
        if os.path.exists(lidar_dir):
            import glob
            bin_files = glob.glob(os.path.join(lidar_dir, "*.bin"))
            npy_files = glob.glob(os.path.join(lidar_dir, "*.npy"))
            pcd_files = glob.glob(os.path.join(lidar_dir, "*.pcd"))

            print(f"LiDAR数据: {len(bin_files)} .bin文件, "
                  f"{len(npy_files)} .npy文件, "
                  f"{len(pcd_files)} .pcd文件")

            # 计算总点数估计
            if bin_files:
                sample_files = bin_files[:min(5, len(bin_files))]
                total_points = 0
                for file in sample_files:
                    try:
                        file_size = os.path.getsize(file)
                        points_in_file = file_size // 16  # 每个点16字节
                        total_points += points_in_file
                    except:
                        pass

                if sample_files:
                    avg_points = total_points / len(sample_files)
                    estimated_total = avg_points * len(bin_files)
                    print(f"  估计总点数: {estimated_total:,.0f}")

        # 协同数据
        coop_dir = os.path.join(self.output_dir, "cooperative")
        if os.path.exists(coop_dir):
            v2x_files = []
            perception_files = []

            v2x_dir = os.path.join(coop_dir, "v2x_messages")
            if os.path.exists(v2x_dir):
                v2x_files = [f for f in os.listdir(v2x_dir) if f.endswith(('.json', '.gz'))]

            perception_dir = os.path.join(coop_dir, "shared_perception")
            if os.path.exists(perception_dir):
                perception_files = [f for f in os.listdir(perception_dir) if f.endswith('.json')]

            print(f"协同数据: {len(v2x_files)} V2X消息, {len(perception_files)} 共享感知文件")

        # 安全数据
        safety_dir = os.path.join(self.output_dir, "safety_reports")
        if os.path.exists(safety_dir):
            safety_files = len([f for f in os.listdir(safety_dir) if f.endswith('.json')])
            print(f"安全报告: {safety_files} 个")

        # 格式转换
        if self.output_format == 'v2xformer':
            v2x_dir = os.path.join(self.output_dir, "v2xformer_format")
            if os.path.exists(v2x_dir):
                print(f"V2XFormer格式: 已生成")

        if self.output_format == 'kitti':
            kitti_dir = os.path.join(self.output_dir, "kitti_format")
            if os.path.exists(kitti_dir):
                print(f"KITTI格式: 已生成")

        # 性能统计
        total_frames = self.collected_frames
        elapsed = time.time() - self.start_time

        print(f"\n性能统计:")
        if total_frames > 0:
            fps = total_frames / max(elapsed, 0.1)
            print(f"  平均帧率: {fps:.2f} FPS")

            performance = self.performance_monitor.get_performance_summary()
            print(f"  帧时统计:")
            print(f"    平均: {performance['frame_statistics']['average_frame_time'] * 1000:.1f}ms")
            print(f"    P50: {performance['frame_time_stats']['p50'] * 1000:.1f}ms")
            print(f"    P95: {performance['frame_time_stats']['p95'] * 1000:.1f}ms")
            print(f"    P99: {performance['frame_time_stats']['p99'] * 1000:.1f}ms")
            print(f"  内存使用:")
            print(f"    平均: {performance['memory_statistics']['average_memory_mb']:.1f} MB")
            print(f"    最大: {performance['memory_statistics']['max_memory_mb']:.1f} MB")
            print(f"  警告: {len(performance['warnings'])} 个")
            print(f"  总帧数: {total_frames}")
        else:
            print("  未收集到有效帧数据")

        # 行人安全统计
        if self.safety_monitor:
            try:
                final_report = self.safety_monitor.generate_final_report()
                print(f"\n行人安全统计:")
                print(f"  高风险交互: {final_report['risk_distribution']['high']}")
                print(f"  临界风险: {final_report['risk_distribution']['critical']}")
                print(f"  安全评分: {final_report['safety_score']:.1f}/100")
            except:
                pass

        print(f"\n输出目录: {self.output_dir}")
        print("=" * 60)

    def _convert_to_target_format(self):
        """转换为目标格式"""
        Log.info(f"转换为 {self.output_format} 格式...")

        if self.output_format == 'v2xformer':
            self._convert_to_v2xformer_format()
        elif self.output_format == 'kitti':
            self._convert_to_kitti_format()
        elif self.output_format == 'nuscenes':
            self._convert_to_nuscenes_format()

    def _convert_to_v2xformer_format(self):
        """转换为V2XFormer格式"""
        try:
            v2x_dir = os.path.join(self.output_dir, "v2xformer_format")

            # 创建标准目录结构
            splits = ['train', 'val', 'test']
            for split in splits:
                split_dir = os.path.join(v2x_dir, split)
                os.makedirs(split_dir, exist_ok=True)

                for subdir in ['image', 'point_cloud', 'calib', 'label']:
                    os.makedirs(os.path.join(split_dir, subdir), exist_ok=True)

            # 计算分割
            total_frames = self.collected_frames
            if total_frames == 0:
                Log.warning("没有数据帧可转换")
                return

            train_ratio = 0.7
            val_ratio = 0.2
            test_ratio = 0.1

            train_frames = int(total_frames * train_ratio)
            val_frames = int(total_frames * val_ratio)
            test_frames = total_frames - train_frames - val_frames

            splits_info = {
                'train': list(range(0, train_frames)),
                'val': list(range(train_frames, train_frames + val_frames)),
                'test': list(range(train_frames + val_frames, total_frames))
            }

            # 保存分割信息
            splits_file = os.path.join(v2x_dir, "splits.json")
            with open(splits_file, 'w', encoding='utf-8') as f:
                json.dump(splits_info, f, indent=2)

            # 创建配置文件
            config = {
                'dataset': {
                    'name': 'CVIPS_Pedestrian_Safety',
                    'version': '2.0',
                    'description': '行人安全增强数据集',
                    'splits': splits_info,
                    'total_frames': total_frames,
                    'created': datetime.now().isoformat()
                },
                'sensors': {
                    'cameras': self.config['sensors']['vehicle_cameras'],
                    'lidars': self.config['sensors']['lidar_sensors'],
                    'image_size': self.config['sensors']['image_size']
                },
                'classes': {
                    'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
                    'pedestrian': ['pedestrian'],
                    'traffic': ['traffic_light', 'traffic_sign']
                }
            }

            config_file = os.path.join(v2x_dir, "config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

            Log.info(f"V2XFormer格式转换完成: {v2x_dir}")

        except Exception as e:
            Log.error(f"V2XFormer格式转换失败: {e}")

    def _convert_to_kitti_format(self):
        """转换为KITTI格式"""
        try:
            kitti_dir = os.path.join(self.output_dir, "kitti_format")

            # 创建标准目录结构
            for subdir in ['training', 'testing']:
                full_dir = os.path.join(kitti_dir, subdir)
                os.makedirs(full_dir, exist_ok=True)

                for subsubdir in ['image_2', 'velodyne', 'calib', 'label_2']:
                    os.makedirs(os.path.join(full_dir, subsubdir), exist_ok=True)

            # 生成标定文件
            self._generate_kitti_calibration(kitti_dir)

            # 创建映射文件
            mapping = {
                'dataset_name': 'CVIPS_KITTI',
                'original_path': self.output_dir,
                'converted_path': kitti_dir,
                'total_frames': self.collected_frames,
                'conversion_time': datetime.now().isoformat()
            }

            mapping_file = os.path.join(kitti_dir, "mapping.json")
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2)

            Log.info(f"KITTI格式转换完成: {kitti_dir}")

        except Exception as e:
            Log.error(f"KITTI格式转换失败: {e}")

    def _convert_to_nuscenes_format(self):
        """转换为nuScenes格式"""
        try:
            nuscenes_dir = os.path.join(self.output_dir, "nuscenes_format")

            # 创建标准目录结构
            for subdir in ['samples', 'sweeps', 'maps']:
                os.makedirs(os.path.join(nuscenes_dir, subdir), exist_ok=True)

            # 创建基本配置文件
            config = {
                'dataset': {
                    'name': 'CVIPS_nuScenes',
                    'version': 'v1.0',
                    'description': '行人安全增强数据集 - nuScenes格式',
                    'created': datetime.now().isoformat()
                },
                'sensors': {
                    'cameras': 6,
                    'lidars': 1,
                    'radars': 0
                },
                'classes': [
                    'car', 'truck', 'bus', 'trailer',
                    'pedestrian', 'bicycle', 'motorcycle',
                    'traffic_cone', 'barrier'
                ]
            }

            config_file = os.path.join(nuscenes_dir, "config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

            Log.info(f"nuScenes格式转换开始: {nuscenes_dir}")

        except Exception as e:
            Log.error(f"nuScenes格式转换失败: {e}")

    def _generate_kitti_calibration(self, kitti_dir):
        """生成KITTI标定文件"""
        calib_template = """P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.341729e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 4.276802e-04 -9.999672e-01 -8.084491e-03 -1.198459e-02 -7.210626e-03 8.081198e-03 -9.999413e-01 -5.403984e-02 9.999738e-01 4.859485e-04 -7.206933e-03 -2.921968e-02
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"""

        # 为每帧生成标定文件
        for i in range(min(self.collected_frames, 1000)):  # 限制数量
            calib_file = os.path.join(kitti_dir, "training", "calib", f"{i:06d}.txt")
            try:
                with open(calib_file, 'w') as f:
                    f.write(calib_template)
            except:
                pass

    def run_validation(self):
        """运行数据验证"""
        if self.config['output'].get('validate_data', True) and self.collected_frames > 0:
            Log.info("运行数据验证...")

            detailed = self.config['output'].get('run_quality_check', True)
            validation_results = DataValidator.validate_dataset(self.output_dir, detailed)

            # 根据验证结果给出建议
            overall_score = validation_results.get('overall_score', 0)
            health_status = validation_results.get('health_status', 'UNKNOWN')

            Log.info(f"验证完成: 评分 {overall_score}/100, 状态 {health_status}")

            if overall_score < 60:
                Log.warning("数据集质量较低，建议检查采集过程")
            elif overall_score < 80:
                Log.info("数据集质量良好，部分项目需要优化")
            else:
                Log.info("数据集质量优秀")
        else:
            Log.info("跳过数据验证")

    def cleanup(self):
        """清理数据收集器"""
        Log.info("开始安全清理场景...")

        cleanup_start = time.time()

        try:
            self.is_running = False

            # 短暂等待确保所有处理完成
            time.sleep(0.5)

            # 停止V2X通信
            if self.v2x_communication:
                try:
                    Log.info("停止V2X通信...")
                    self.v2x_communication.stop()
                except:
                    pass

            # 保存行人安全数据
            if self.safety_monitor:
                try:
                    Log.info("保存行人安全数据...")
                    self.safety_monitor.save_data()
                except:
                    pass

            # 清理传感器管理器
            Log.info(f"清理 {len(self.sensor_managers)} 个传感器管理器...")
            for vehicle_id, sensor_manager in self.sensor_managers.items():
                try:
                    sensor_manager.cleanup()
                except:
                    pass

            self.sensor_managers.clear()
            time.sleep(0.2)

            # 清理多车辆管理器
            if self.multi_vehicle_manager:
                try:
                    Log.info("清理多车辆管理器...")
                    self.multi_vehicle_manager.cleanup()
                except:
                    pass

            # 清理交通管理器
            if self.traffic_manager:
                try:
                    Log.info("清理交通管理器...")
                    self.traffic_manager.cleanup()
                except:
                    pass

            # 清理主车
            Log.info(f"清理 {len(self.ego_vehicles)} 个主车...")
            for vehicle in self.ego_vehicles:
                try:
                    if hasattr(vehicle, 'destroy'):
                        vehicle.destroy()
                except:
                    pass
                time.sleep(0.01)

            self.ego_vehicles.clear()

            # 恢复默认天气
            try:
                if self.world:
                    default_weather = carla.WeatherParameters()
                    self.world.set_weather(default_weather)
            except:
                pass

            # 清理Python对象
            self.traffic_manager = None
            self.multi_vehicle_manager = None
            self.v2x_communication = None
            self.safety_monitor = None
            self.scene_manager = None
            self.scene_center = None

            # 垃圾回收
            gc.collect()

            Log.info(f"清理完成，总用时: {time.time() - cleanup_start:.2f}秒")

        except Exception as e:
            Log.error(f"清理过程中发生错误: {e}")
            try:
                # 尝试强制清理
                self.sensor_managers.clear()
                self.ego_vehicles.clear()
                gc.collect()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='CVIPS 行人安全增强数据收集系统')

    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--scenario', type=str, default='pedestrian_safety', help='场景名称')
    parser.add_argument('--town', type=str, default='Town10HD',
                        choices=['Town03', 'Town04', 'Town05', 'Town10HD'], help='地图')
    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'rainy', 'cloudy', 'foggy'], help='天气')
    parser.add_argument('--time-of-day', type=str, default='noon',
                        choices=['noon', 'sunset', 'night'], help='时间')

    parser.add_argument('--num-vehicles', type=int, default=8, help='背景车辆数')
    parser.add_argument('--num-pedestrians', type=int, default=12, help='行人数')
    parser.add_argument('--num-coop-vehicles', type=int, default=2, help='协同车辆数')

    parser.add_argument('--duration', type=int, default=60, help='收集时长(秒)')
    parser.add_argument('--capture-interval', type=float, default=2.0, help='捕捉间隔(秒)')
    parser.add_argument('--seed', type=int, help='随机种子')

    parser.add_argument('--batch-size', type=int, default=5, help='批处理大小')
    parser.add_argument('--enable-compression', action='store_true', help='启用数据压缩')
    parser.add_argument('--enable-downsampling', action='store_true', help='启用LiDAR下采样')

    parser.add_argument('--output-format', type=str, default='standard',
                        choices=['standard', 'v2xformer', 'kitti'], help='输出数据格式')

    parser.add_argument('--enable-lidar', action='store_true', help='启用LiDAR传感器')
    parser.add_argument('--enable-fusion', action='store_true', help='启用多传感器融合')
    parser.add_argument('--enable-v2x', action='store_true', help='启用V2X通信')
    parser.add_argument('--enable-cooperative', action='store_true', help='启用协同感知')
    parser.add_argument('--enable-enhancement', action='store_true', help='启用数据增强')
    parser.add_argument('--enable-annotations', action='store_true', help='启用自动标注')
    parser.add_argument('--enable-safety-monitor', action='store_true', default=True, help='启用行人安全监控')

    parser.add_argument('--run-analysis', action='store_true', help='运行数据集分析')
    parser.add_argument('--skip-validation', action='store_true', help='跳过数据验证')
    parser.add_argument('--skip-quality-check', action='store_true', help='跳过质量检查')

    args = parser.parse_args(remaining_argv)

    config = ConfigManager.load_config(args.config)
    config = ConfigManager.merge_args(config, args)

    config['performance']['batch_size'] = args.batch_size
    config['performance']['enable_compression'] = args.enable_compression
    config['performance']['enable_downsampling'] = args.enable_downsampling
    config['output']['output_format'] = args.output_format

    print("\n" + "=" * 60)
    print("CVIPS 行人安全增强数据收集系统")
    print("=" * 60)

    print(f"场景: {config['scenario']['name']}")
    print(f"地图: {config['scenario']['town']}")
    print(f"天气/时间: {config['scenario']['weather']}/{config['scenario']['time_of_day']}")
    print(f"时长: {config['scenario']['duration']}秒")
    print(f"交通: {config['traffic']['background_vehicles']}背景车辆 + {config['traffic']['pedestrians']}行人")
    print(f"协同: {config['cooperative']['num_coop_vehicles']} 协同车辆")
    print(f"输出格式: {config['output']['output_format']}")

    print(f"传感器:")
    print(
        f"  摄像头: {config['sensors']['vehicle_cameras']}车辆 + {config['sensors']['infrastructure_cameras']}基础设施")
    print(f"  LiDAR: {'启用' if config['sensors']['lidar_sensors'] > 0 else '禁用'}")
    print(f"  融合: {'启用' if config['output']['save_fusion'] else '禁用'}")
    print(f"  V2X: {'启用' if config['v2x']['enabled'] else '禁用'}")
    print(f"  协同: {'启用' if config['output']['save_cooperative'] else '禁用'}")
    print(f"  增强: {'启用' if config['enhancement']['enabled'] else '禁用'}")
    print(f"  安全监控: {'启用' if args.enable_safety_monitor else '禁用'}")

    print(f"性能:")
    print(f"  批处理大小: {config['performance']['batch_size']}")
    print(f"  压缩: {'启用' if config['performance']['enable_compression'] else '禁用'}")
    print(f"  下采样: {'启用' if config['performance']['enable_downsampling'] else '禁用'}")

    collector = DataCollector(config)

    try:
        if not collector.connect():
            print("连接CARLA服务器失败")
            return

        if not collector.setup_scene():
            print("场景设置失败")
            collector.cleanup()
            return

        if not collector.setup_sensors():
            print("传感器设置失败")
            collector.cleanup()
            return

        collector.collect_data()

        collector.run_validation()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n运行错误: {e}")
        traceback.print_exc()
    finally:
        collector.cleanup()
        print(f"\n数据集已保存到: {collector.output_dir}")


if __name__ == "__main__":
    main()