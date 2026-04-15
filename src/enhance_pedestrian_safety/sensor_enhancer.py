"""
传感器数据增强模块 - 提高数据质量和多样性（行人安全优化版）
"""

import numpy as np
import cv2
import random
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Callable
import concurrent.futures
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
from pathlib import Path


class WeatherType(Enum):
    """天气类型枚举"""
    CLEAR = "clear"
    RAINY = "rainy"
    FOGGY = "foggy"
    CLOUDY = "cloudy"
    NIGHT = "night"
    SUNSET = "sunset"


class EnhancementMethod(Enum):
    """增强方法枚举"""
    NORMALIZE = "normalize"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    SHARPNESS = "sharpness"
    GAMMA = "gamma"
    NOISE = "noise"
    BLUR = "blur"
    MOTION_BLUR = "motion_blur"
    RAIN = "rain"
    FOG = "fog"
    CLOUD = "cloud"
    VIGNETTE = "vignette"
    COLOR_TEMP = "color_temperature"
    JPEG_COMPRESSION = "jpeg_compression"
    COLOR_JITTER = "color_jitter"
    PEDESTRIAN_HIGHLIGHT = "pedestrian_highlight"  # 新增：行人高亮
    SAFETY_WARNING = "safety_warning"  # 新增：安全警告


@dataclass
class EnhancementConfig:
    """增强配置"""
    weather: WeatherType = WeatherType.CLEAR
    time_of_day: str = "noon"
    enabled_methods: List[EnhancementMethod] = None
    intensity_range: Tuple[float, float] = (0.5, 1.5)
    probability: float = 0.7
    max_methods_per_image: int = 5
    save_original: bool = True
    save_enhanced: bool = True
    output_format: str = "jpg"
    compression_quality: int = 90
    pedestrian_safety_mode: bool = True  # 新增：行人安全模式

    def __post_init__(self):
        if self.enabled_methods is None:
            self.enabled_methods = [
                EnhancementMethod.NORMALIZE,
                EnhancementMethod.CONTRAST,
                EnhancementMethod.BRIGHTNESS
            ]
        # 如果启用行人安全模式，添加相关方法
        if self.pedestrian_safety_mode:
            if EnhancementMethod.PEDESTRIAN_HIGHLIGHT not in self.enabled_methods:
                self.enabled_methods.append(EnhancementMethod.PEDESTRIAN_HIGHLIGHT)
            if EnhancementMethod.SAFETY_WARNING not in self.enabled_methods:
                self.enabled_methods.append(EnhancementMethod.SAFETY_WARNING)


class BatchEnhancer:
    """批量增强器"""

    def __init__(self, config: EnhancementConfig, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers
        self.enhancer = SensorDataEnhancer(config)
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'avg_time_per_image': 0
        }

    def process_batch(self, image_paths: List[str], output_dir: str) -> Dict:
        """批量处理图像"""
        start_time = time.time()
        results = []

        # 准备输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {}
            for img_path in image_paths:
                if os.path.exists(img_path):
                    output_path = self._get_output_path(img_path, output_dir)
                    future = executor.submit(
                        self._process_single_image,
                        img_path, output_path
                    )
                    future_to_path[future] = img_path

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.stats['successful'] += 1
                except Exception as e:
                    print(f"处理图像 {img_path} 失败: {e}")
                    self.stats['failed'] += 1
                    results.append({
                        'input_path': img_path,
                        'output_path': None,
                        'success': False,
                        'error': str(e)
                    })

        # 更新统计
        self.stats['total_processed'] += len(image_paths)
        total_time = time.time() - start_time
        self.stats['total_time'] += total_time

        if len(image_paths) > 0:
            self.stats['avg_time_per_image'] = self.stats['total_time'] / self.stats['total_processed']

        return {
            'results': results,
            'stats': self.stats.copy(),
            'batch_size': len(image_paths),
            'processing_time': total_time
        }

    def _process_single_image(self, input_path: str, output_path: str) -> Dict:
        """处理单张图像"""
        try:
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"无法读取图像: {input_path}")

            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 应用增强
            enhanced = self.enhancer.enhance_image(image_rgb)

            # 保存结果
            cv2.imwrite(output_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

            # 计算哈希值（用于去重）
            img_hash = hashlib.md5(enhanced.tobytes()).hexdigest()[:16]

            return {
                'input_path': input_path,
                'output_path': output_path,
                'success': True,
                'image_hash': img_hash,
                'original_size': os.path.getsize(input_path),
                'enhanced_size': os.path.getsize(output_path),
                'compression_ratio': os.path.getsize(output_path) / max(1, os.path.getsize(input_path))
            }
        except Exception as e:
            raise Exception(f"处理失败: {e}")

    def _get_output_path(self, input_path: str, output_dir: str) -> str:
        """生成输出路径"""
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)

        # 根据配置选择输出格式
        if self.config.output_format == "jpg":
            new_ext = ".jpg"
        elif self.config.output_format == "png":
            new_ext = ".png"
        else:
            new_ext = ext

        # 添加增强标记
        enhanced_name = f"{name}_enhanced{new_ext}"
        return os.path.join(output_dir, enhanced_name)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()


class SensorDataEnhancer:
    """传感器数据增强器（行人安全优化版）"""

    def __init__(self, config: Union[EnhancementConfig, Dict]):
        if isinstance(config, dict):
            self.config = EnhancementConfig(**config)
        else:
            self.config = config

        self.method_registry = self._setup_method_registry()
        self.weather_methods = self._setup_weather_methods()
        self.method_cache = {}
        self.perf_stats = {
            'calls': 0,
            'total_time': 0,
            'method_times': {}
        }

        # 行人检测模拟数据
        self.pedestrian_detections = []
        self.safety_warnings = []

    def _setup_method_registry(self) -> Dict[EnhancementMethod, Callable]:
        """设置方法注册表"""
        registry = {
            EnhancementMethod.NORMALIZE: self._normalize_image,
            EnhancementMethod.BRIGHTNESS: self._adjust_brightness,
            EnhancementMethod.CONTRAST: self._adjust_contrast,
            EnhancementMethod.SATURATION: self._adjust_saturation,
            EnhancementMethod.SHARPNESS: self._enhance_sharpness,
            EnhancementMethod.GAMMA: self._gamma_correction,
            EnhancementMethod.NOISE: self._add_noise,
            EnhancementMethod.BLUR: self._apply_gaussian_blur,
            EnhancementMethod.MOTION_BLUR: self._apply_motion_blur,
            EnhancementMethod.RAIN: self._add_rain_effect,
            EnhancementMethod.FOG: self._add_fog_effect,
            EnhancementMethod.CLOUD: self._add_cloud_effect,
            EnhancementMethod.VIGNETTE: self._add_vignette,
            EnhancementMethod.COLOR_TEMP: self._adjust_color_temperature,
            EnhancementMethod.JPEG_COMPRESSION: self._simulate_jpeg_compression,
            EnhancementMethod.COLOR_JITTER: self._color_jitter,
            EnhancementMethod.PEDESTRIAN_HIGHLIGHT: self._highlight_pedestrians,  # 新增
            EnhancementMethod.SAFETY_WARNING: self._add_safety_warnings  # 新增
        }
        return registry

    def _setup_weather_methods(self) -> Dict[WeatherType, List[EnhancementMethod]]:
        """设置天气相关方法"""
        return {
            WeatherType.CLEAR: [
                EnhancementMethod.NORMALIZE,
                EnhancementMethod.CONTRAST,
                EnhancementMethod.SHARPNESS
            ],
            WeatherType.RAINY: [
                EnhancementMethod.NORMALIZE,
                EnhancementMethod.RAIN,
                EnhancementMethod.CONTRAST,
                EnhancementMethod.MOTION_BLUR
            ],
            WeatherType.FOGGY: [
                EnhancementMethod.NORMALIZE,
                EnhancementMethod.FOG,
                EnhancementMethod.CONTRAST,
                EnhancementMethod.BLUR
            ],
            WeatherType.CLOUDY: [
                EnhancementMethod.NORMALIZE,
                EnhancementMethod.CLOUD,
                EnhancementMethod.COLOR_TEMP,
                EnhancementMethod.CONTRAST
            ],
            WeatherType.NIGHT: [
                EnhancementMethod.NORMALIZE,
                EnhancementMethod.BRIGHTNESS,
                EnhancementMethod.CONTRAST,
                EnhancementMethod.NOISE,
                EnhancementMethod.VIGNETTE,
                EnhancementMethod.PEDESTRIAN_HIGHLIGHT  # 夜间特别关注行人
            ],
            WeatherType.SUNSET: [
                EnhancementMethod.NORMALIZE,
                EnhancementMethod.COLOR_TEMP,
                EnhancementMethod.CONTRAST,
                EnhancementMethod.VIGNETTE
            ]
        }

    def enhance_image(self, image_data: np.ndarray,
                     sensor_type: str = 'camera',
                     return_methods: bool = False) -> Union[np.ndarray, Tuple]:
        """
        增强图像数据（行人安全优化版）

        Args:
            image_data: 原始图像数据 (H, W, C)
            sensor_type: 传感器类型 ('camera', 'depth', 'semantic')
            return_methods: 是否返回使用的增强方法列表

        Returns:
            增强后的图像数据（如果return_methods为True，则返回元组）
        """
        start_time = time.time()

        if sensor_type != 'camera':
            # 深度和语义分割图像使用不同的增强
            enhanced = self._enhance_non_rgb_image(image_data, sensor_type)
            self._update_perf_stats('non_rgb', time.time() - start_time)

            if return_methods:
                return enhanced, ['non_rgb_enhance']
            return enhanced

        # 获取增强方法
        methods = self._get_enhancement_methods()

        # 检查缓存
        cache_key = self._get_cache_key(image_data, methods)
        if cache_key in self.method_cache:
            self._update_perf_stats('cache_hit', time.time() - start_time)

            if return_methods:
                cached_result, cached_methods = self.method_cache[cache_key]
                return cached_result.copy(), cached_methods
            return self.method_cache[cache_key][0].copy()

        # 应用增强方法
        enhanced = image_data.copy()
        applied_methods = []

        for method in methods:
            try:
                method_start = time.time()
                enhanced = self.method_registry[method](enhanced)
                method_time = time.time() - method_start

                self._update_perf_stats(method.value, method_time)
                applied_methods.append(method.value)
            except Exception as e:
                print(f"增强方法 {method.value} 失败: {e}")
                continue

        # 确保图像在有效范围内
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        # 更新缓存
        self.method_cache[cache_key] = (enhanced.copy(), applied_methods)

        # 清理缓存（如果太大）
        if len(self.method_cache) > 100:
            self._cleanup_cache()

        total_time = time.time() - start_time
        self._update_perf_stats('total', total_time)

        if return_methods:
            return enhanced, applied_methods
        return enhanced

    def _get_enhancement_methods(self) -> List[EnhancementMethod]:
        """获取增强方法列表"""
        # 基础方法
        methods = list(self.config.enabled_methods)

        # 添加天气相关方法
        weather_methods = self.weather_methods.get(self.config.weather, [])
        for method in weather_methods:
            if method not in methods and random.random() < self.config.probability:
                methods.append(method)

        # 随机添加额外方法
        if random.random() < 0.3:  # 30%概率添加额外方法
            available_methods = list(EnhancementMethod)
            extra_methods = random.sample(
                [m for m in available_methods if m not in methods],
                k=random.randint(1, 2)
            )
            methods.extend(extra_methods)

        # 限制方法数量
        if len(methods) > self.config.max_methods_per_image:
            methods = random.sample(methods, self.config.max_methods_per_image)

        # 随机排序
        random.shuffle(methods)

        return methods

    def _get_cache_key(self, image_data: np.ndarray,
                      methods: List[EnhancementMethod]) -> str:
        """生成缓存键"""
        # 使用图像哈希和方法列表生成键
        img_hash = hashlib.md5(image_data.tobytes()).hexdigest()[:16]
        method_str = ','.join(sorted([m.value for m in methods]))
        config_str = f"{self.config.weather.value}_{self.config.time_of_day}"

        return f"{img_hash}_{method_str}_{config_str}"

    def _cleanup_cache(self):
        """清理缓存"""
        # 保留最近使用的50个条目
        if len(self.method_cache) > 50:
            keys_to_remove = list(self.method_cache.keys())[:-50]
            for key in keys_to_remove:
                del self.method_cache[key]

    def _update_perf_stats(self, method: str, duration: float):
        """更新性能统计"""
        self.perf_stats['calls'] += 1
        self.perf_stats['total_time'] += duration
        self.perf_stats['method_times'][method] = \
            self.perf_stats['method_times'].get(method, 0) + duration

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        stats = self.perf_stats.copy()
        if stats['calls'] > 0:
            stats['avg_time_per_call'] = stats['total_time'] / stats['calls']
        return stats

    # ========== 增强方法实现（行人安全优化版）==========

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """图像归一化（优化版）"""
        if image.dtype != np.uint8:
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            return normalized.astype(np.uint8)
        return image

    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """调整亮度（优化版）"""
        factor = random.uniform(*self.config.intensity_range)

        if factor != 1.0:
            img_float = image.astype(np.float32) * factor
            return np.clip(img_float, 0, 255).astype(np.uint8)
        return image

    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """调整对比度（优化版）"""
        factor = random.uniform(*self.config.intensity_range)

        if factor != 1.0:
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            contrasted = mean + factor * (image.astype(np.float32) - mean)
            return np.clip(contrasted, 0, 255).astype(np.uint8)
        return image

    def _adjust_saturation(self, image: np.ndarray) -> np.ndarray:
        """调整饱和度（优化版）"""
        factor = random.uniform(*self.config.intensity_range)

        if factor != 1.0:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return saturated
        return image

    def _enhance_sharpness(self, image: np.ndarray) -> np.ndarray:
        """增强锐度（优化版）"""
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """伽马校正（优化版）"""
        gamma = random.uniform(0.5, 2.0)

        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        corrected = cv2.LUT(image, table)
        return corrected

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """添加噪声（优化版）"""
        noise_type = random.choice(['gaussian', 'salt_pepper'])

        if noise_type == 'gaussian':
            mean = 0
            var = random.uniform(0.001, 0.005)
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, image.shape) * 255
            noisy = image.astype(np.float32) + gauss
            return np.clip(noisy, 0, 255).astype(np.uint8)

        else:  # salt_pepper
            amount = random.uniform(0.001, 0.005)
            s_vs_p = random.uniform(0.3, 0.7)

            noisy = image.copy()

            num_salt = np.ceil(amount * image.size * s_vs_p / 3)
            coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
            noisy[coords[0], coords[1], :] = 255

            num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p) / 3)
            coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
            noisy[coords[0], coords[1], :] = 0

            return noisy

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """应用高斯模糊（优化版）"""
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred

    def _apply_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """应用运动模糊（优化版）"""
        kernel_size = random.choice([7, 9, 11])
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])

        kernel = np.zeros((kernel_size, kernel_size))

        if direction == 'horizontal':
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
        elif direction == 'vertical':
            kernel[:, kernel_size // 2] = 1.0 / kernel_size
        else:  # diagonal
            for i in range(kernel_size):
                kernel[i, i] = 1.0 / kernel_size

        blurred = cv2.filter2D(image, -1, kernel)

        alpha = random.uniform(0.3, 0.7)
        result = cv2.addWeighted(image, 1 - alpha, blurred, alpha, 0)
        return result.astype(np.uint8)

    def _add_rain_effect(self, image: np.ndarray) -> np.ndarray:
        """添加雨滴效果（优化版）"""
        h, w = image.shape[:2]

        rain_layer = np.zeros((h, w), dtype=np.float32)

        num_drops = random.randint(200, 800)
        drop_length = random.randint(8, 15)

        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            brightness = random.uniform(0.3, 0.6)

            for i in range(drop_length):
                if y + i < h and x + i < w:
                    rain_layer[y + i, x + i] += brightness

        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

        rain_layer_3d = np.stack([rain_layer] * 3, axis=2)
        enhanced = image.astype(np.float32) * 0.9 + rain_layer_3d * 0.1 * 255
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _add_fog_effect(self, image: np.ndarray) -> np.ndarray:
        """添加雾效（优化版）"""
        h, w = image.shape[:2]

        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        distance = distance / np.max(distance)

        fog_intensity = random.uniform(0.2, 0.5)
        fog_color = random.choice([200, 210, 220])

        fog_strength = distance * fog_intensity
        fog_strength_3d = np.stack([fog_strength] * 3, axis=2)
        fog_layer = np.ones_like(image) * fog_color

        enhanced = image.astype(np.float32) * (1 - fog_strength_3d) + \
                  fog_layer * fog_strength_3d

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _add_cloud_effect(self, image: np.ndarray) -> np.ndarray:
        """添加云层效果（优化版）"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        hsv[:, :, 1] *= random.uniform(0.7, 0.9)

        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.8, 0.95)

        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)

        mean = np.mean(hsv[:, :, 2])
        hsv[:, :, 2] = mean + contrast_factor * (hsv[:, :, 2] - mean)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return enhanced

    def _add_vignette(self, image: np.ndarray) -> np.ndarray:
        """添加暗角效果（优化版）"""
        h, w = image.shape[:2]

        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]

        y_dist = (y_coords - center_y) / (h / 2)
        x_dist = (x_coords - center_x) / (w / 2)
        distance = np.sqrt(x_dist ** 2 + y_dist ** 2)

        vignette_intensity = random.uniform(0.2, 0.4)
        vignette = 1 - distance * vignette_intensity
        vignette = np.clip(vignette, 0.6, 1.0)

        vignette_3d = np.stack([vignette] * 3, axis=2)
        enhanced = image.astype(np.float32) * vignette_3d

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _adjust_color_temperature(self, image: np.ndarray) -> np.ndarray:
        """调整色温（优化版）"""
        temp_type = random.choice(['warm', 'cool'])

        img_float = image.astype(np.float32)

        if temp_type == 'warm':
            img_float[:, :, 0] *= random.uniform(1.0, 1.15)  # 红色
            img_float[:, :, 1] *= random.uniform(1.0, 1.1)   # 绿色
            img_float[:, :, 2] *= random.uniform(0.9, 1.0)   # 蓝色
        else:
            img_float[:, :, 0] *= random.uniform(0.9, 1.0)   # 红色
            img_float[:, :, 1] *= random.uniform(0.95, 1.0)  # 绿色
            img_float[:, :, 2] *= random.uniform(1.0, 1.15)  # 蓝色

        enhanced = np.clip(img_float, 0, 255)
        return enhanced.astype(np.uint8)

    def _simulate_jpeg_compression(self, image: np.ndarray) -> np.ndarray:
        """模拟JPEG压缩（优化版）"""
        quality = random.randint(70, 95)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)

        if result:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        return image

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """颜色抖动（优化版）"""
        transforms = []

        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            transforms.append(lambda img: self._adjust_brightness(img, brightness))

        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            transforms.append(lambda img: self._adjust_contrast(img, contrast))

        if random.random() > 0.5:
            saturation = random.uniform(0.8, 1.2)
            transforms.append(lambda img: self._adjust_saturation(img, saturation))

        if transforms:
            random.shuffle(transforms)
            for transform in transforms[:2]:
                image = transform(image)

        return image

    def _highlight_pedestrians(self, image: np.ndarray) -> np.ndarray:
        """高亮行人（新增方法）"""
        if not self.config.pedestrian_safety_mode:
            return image

        h, w = image.shape[:2]
        highlighted = image.copy()

        # 模拟行人检测（随机位置）
        num_pedestrians = random.randint(1, 3)

        for i in range(num_pedestrians):
            # 随机位置
            x = random.randint(50, w - 150)
            y = random.randint(50, h - 200)

            # 随机大小
            width = random.randint(30, 70)
            height = random.randint(80, 180)

            # 根据距离设置颜色（模拟风险评估）
            distance = random.uniform(5.0, 50.0)
            if distance < 10.0:
                color = (0, 0, 255)  # 红色：高风险
                thickness = 3
            elif distance < 20.0:
                color = (0, 165, 255)  # 橙色：中风险
                thickness = 2
            else:
                color = (0, 255, 0)  # 绿色：低风险
                thickness = 1

            # 绘制边界框
            cv2.rectangle(highlighted, (x, y), (x + width, y + height), color, thickness)

            # 添加标签
            label = f"Pedestrian {distance:.1f}m"
            cv2.putText(highlighted, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

            # 记录检测
            self.pedestrian_detections.append({
                'position': (x, y, width, height),
                'distance': distance,
                'risk_level': 'high' if distance < 10.0 else 'medium' if distance < 20.0 else 'low'
            })

        return highlighted

    def _add_safety_warnings(self, image: np.ndarray) -> np.ndarray:
        """添加安全警告（新增方法）"""
        if not self.config.pedestrian_safety_mode:
            return image

        warned = image.copy()

        # 检查是否有高风险行人
        high_risk = [d for d in self.pedestrian_detections if d['risk_level'] == 'high']

        if high_risk:
            # 在图像顶部添加警告条
            warning_height = 40
            warning_bar = np.zeros((warning_height, image.shape[1], 3), dtype=np.uint8)
            warning_bar[:] = (0, 0, 255)  # 红色背景

            # 添加警告文本
            warning_text = f"WARNING: {len(high_risk)} high-risk pedestrians detected!"
            cv2.putText(warning_bar, warning_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 将警告条添加到图像顶部
            warned = np.vstack([warning_bar, warned])

            # 记录警告
            self.safety_warnings.append({
                'timestamp': time.time(),
                'high_risk_count': len(high_risk),
                'message': warning_text
            })

        return warned

    def _enhance_non_rgb_image(self, image_data: np.ndarray, sensor_type: str) -> np.ndarray:
        """增强非RGB图像（深度、语义分割）"""
        if sensor_type == 'depth':
            if image_data.dtype != np.uint8:
                normalized = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
            else:
                normalized = image_data

            enhanced = cv2.medianBlur(normalized, 3)
            return enhanced.astype(np.uint8)

        elif sensor_type == 'semantic':
            kernel = np.ones((3, 3), np.uint8)
            enhanced = cv2.morphologyEx(image_data, cv2.MORPH_CLOSE, kernel)
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
            enhanced = np.round(enhanced).astype(image_data.dtype)
            return enhanced

        return image_data

    def save_enhanced_image(self, image_data: np.ndarray, output_path: str,
                           metadata: Optional[Dict] = None):
        """保存增强后的图像和元数据"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                cv2.imwrite(output_path, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(output_path + '.png', cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))

            if metadata:
                meta_path = output_path.rsplit('.', 1)[0] + '_meta.json'
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"保存增强图像失败: {e}")
            raise

    def generate_enhancement_report(self, output_dir: str) -> Dict:
        """生成增强报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'weather': self.config.weather.value,
                'time_of_day': self.config.time_of_day,
                'enabled_methods': [m.value for m in self.config.enabled_methods],
                'intensity_range': self.config.intensity_range,
                'probability': self.config.probability,
                'pedestrian_safety_mode': self.config.pedestrian_safety_mode
            },
            'performance_stats': self.get_performance_stats(),
            'method_usage': {
                method.value: self.perf_stats['method_times'].get(method.value, 0)
                for method in EnhancementMethod
            },
            'cache_info': {
                'cache_size': len(self.method_cache),
                'cache_hits': self.perf_stats.get('cache_hit_calls', 0)
            },
            'safety_statistics': {
                'pedestrian_detections': len(self.pedestrian_detections),
                'safety_warnings': len(self.safety_warnings),
                'high_risk_cases': len([d for d in self.pedestrian_detections if d['risk_level'] == 'high'])
            }
        }

        report_path = os.path.join(output_dir, 'enhancement_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report


# 兼容旧版本接口
def enhance_image(image_data, sensor_type='camera'):
    """兼容旧版本的增强函数"""
    config = EnhancementConfig()
    enhancer = SensorDataEnhancer(config)
    return enhancer.enhance_image(image_data, sensor_type)