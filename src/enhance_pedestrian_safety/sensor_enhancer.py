"""
传感器数据增强模块 - 提高数据质量和多样性
"""

import numpy as np
import cv2
import random
import os
import json
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import carla


class SensorDataEnhancer:
    """传感器数据增强器"""

    def __init__(self, config: Dict):
        self.config = config
        self.enhancement_methods = []
        self._setup_enhancement_methods()

    def _setup_enhancement_methods(self):
        """设置增强方法"""
        # 根据场景配置启用不同的增强方法
        weather = self.config.get('scenario', {}).get('weather', 'clear')
        time_of_day = self.config.get('scenario', {}).get('time_of_day', 'noon')

        # 基础增强方法
        self.enhancement_methods = ['normalize']

        # 根据天气和时间添加特定增强
        if weather == 'rainy':
            self.enhancement_methods.extend(['rain_effect', 'motion_blur', 'brightness_adjust'])
        elif weather == 'foggy':
            self.enhancement_methods.extend(['fog_effect', 'contrast_reduce'])
        elif weather == 'night':
            self.enhancement_methods.extend(['night_effect', 'noise_add', 'gamma_correction'])
        elif weather == 'cloudy':
            self.enhancement_methods.extend(['cloud_effect', 'color_temperature'])

        # 随机增强（可选）
        if self.config.get('enhancement', {}).get('enable_random', True):
            self.enhancement_methods.extend(self._get_random_enhancements())

    def _get_random_enhancements(self) -> List[str]:
        """获取随机增强方法"""
        random_methods = [
            'hue_shift', 'saturation_adjust', 'sharpness_enhance',
            'gaussian_blur', 'jpeg_compression', 'color_jitter'
        ]
        # 随机选择1-3个增强方法
        num_methods = random.randint(1, 3)
        return random.sample(random_methods, num_methods)

    def enhance_image(self, image_data: np.ndarray, sensor_type: str = 'camera') -> np.ndarray:
        """
        增强图像数据

        Args:
            image_data: 原始图像数据 (H, W, C)
            sensor_type: 传感器类型 ('camera', 'depth', 'semantic')

        Returns:
            增强后的图像数据
        """
        if sensor_type != 'camera':
            # 深度和语义分割图像使用不同的增强
            return self._enhance_non_rgb_image(image_data, sensor_type)

        # 转换为PIL图像进行处理
        pil_image = Image.fromarray(image_data)

        # 按顺序应用增强方法
        for method in self.enhancement_methods:
            pil_image = self._apply_enhancement_method(pil_image, method)

        # 确保图像在有效范围内
        enhanced_image = np.array(pil_image)
        enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

        return enhanced_image

    def _apply_enhancement_method(self, image: Image.Image, method: str) -> Image.Image:
        """应用单个增强方法"""
        try:
            if method == 'normalize':
                return self._normalize_image(image)
            elif method == 'rain_effect':
                return self._add_rain_effect(image)
            elif method == 'fog_effect':
                return self._add_fog_effect(image)
            elif method == 'night_effect':
                return self._apply_night_effect(image)
            elif method == 'motion_blur':
                return self._apply_motion_blur(image)
            elif method == 'brightness_adjust':
                return self._adjust_brightness(image)
            elif method == 'contrast_reduce':
                return self._reduce_contrast(image)
            elif method == 'noise_add':
                return self._add_noise(image)
            elif method == 'gamma_correction':
                return self._gamma_correction(image)
            elif method == 'cloud_effect':
                return self._add_cloud_effect(image)
            elif method == 'color_temperature':
                return self._adjust_color_temperature(image)
            elif method == 'hue_shift':
                return self._shift_hue(image)
            elif method == 'saturation_adjust':
                return self._adjust_saturation(image)
            elif method == 'sharpness_enhance':
                return self._enhance_sharpness(image)
            elif method == 'gaussian_blur':
                return self._apply_gaussian_blur(image)
            elif method == 'jpeg_compression':
                return self._simulate_jpeg_compression(image)
            elif method == 'color_jitter':
                return self._color_jitter(image)
            else:
                return image
        except Exception as e:
            print(f"增强方法 {method} 失败: {e}")
            return image

    def _enhance_non_rgb_image(self, image_data: np.ndarray, sensor_type: str) -> np.ndarray:
        """增强非RGB图像（深度、语义分割）"""
        if sensor_type == 'depth':
            # 深度图增强：归一化和噪声去除
            normalized = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
            # 应用轻微的高斯模糊去除噪声
            enhanced = cv2.GaussianBlur(normalized, (3, 3), 0)
            return enhanced.astype(np.uint8)

        elif sensor_type == 'semantic':
            # 语义分割图增强：保持类别不变，只做边界平滑
            kernel = np.ones((3, 3), np.uint8)
            # 形态学开运算去除小噪声
            enhanced = cv2.morphologyEx(image_data, cv2.MORPH_OPEN, kernel)
            return enhanced

        return image_data

    def _normalize_image(self, image: Image.Image) -> Image.Image:
        """图像归一化"""
        # 转换为numpy数组
        img_array = np.array(image)

        # 归一化到0-255
        if img_array.dtype != np.uint8:
            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
            img_array = img_array.astype(np.uint8)

        return Image.fromarray(img_array)

    def _add_rain_effect(self, image: Image.Image) -> Image.Image:
        """添加雨滴效果"""
        img_array = np.array(image)
        h, w, _ = img_array.shape

        # 创建雨滴层
        rain_layer = np.zeros((h, w), dtype=np.float32)

        # 生成随机雨滴位置
        num_drops = random.randint(100, 500)
        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(5, 20)
            thickness = random.randint(1, 2)
            brightness = random.uniform(0.7, 0.9)

            # 绘制雨滴（斜线）
            for i in range(length):
                if y + i < h and x + i < w:
                    rain_layer[y + i, x + i] += brightness
                    # 加粗雨滴
                    for j in range(thickness):
                        if x + i + j < w:
                            rain_layer[y + i, x + i + j] += brightness * 0.5

        # 模糊雨滴层
        rain_layer = cv2.GaussianBlur(rain_layer, (5, 5), 0)

        # 叠加到原图
        rain_layer_3d = np.stack([rain_layer] * 3, axis=2)
        enhanced = cv2.addWeighted(img_array.astype(np.float32), 0.8,
                                   rain_layer_3d * 255, 0.2, 0)

        return Image.fromarray(enhanced.astype(np.uint8))

    def _add_fog_effect(self, image: Image.Image) -> Image.Image:
        """添加雾效"""
        img_array = np.array(image)
        h, w, _ = img_array.shape

        # 创建雾效层
        fog_intensity = random.uniform(0.3, 0.6)
        fog_color = random.choice([200, 210, 220])  # 雾的颜色

        fog_layer = np.ones((h, w, 3), dtype=np.float32) * fog_color

        # 根据深度添加雾效（这里简化处理，实际应该使用深度图）
        # 创建简单的深度渐变（假设图像中心最近）
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        distance = distance / np.max(distance)  # 归一化

        # 雾效随距离增强
        fog_strength = distance * fog_intensity
        fog_strength_3d = np.stack([fog_strength] * 3, axis=2)

        # 混合原图和雾效
        enhanced = img_array.astype(np.float32) * (1 - fog_strength_3d) + \
                   fog_layer * fog_strength_3d

        return Image.fromarray(enhanced.astype(np.uint8))

    def _apply_night_effect(self, image: Image.Image) -> Image.Image:
        """应用夜间效果"""
        # 降低亮度
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.3, 0.6))

        # 降低对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.7, 0.9))

        # 添加暗角效果
        img_array = np.array(image)
        h, w, _ = img_array.shape

        # 创建暗角蒙版
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        distance = distance / np.max(distance)

        vignette = 1 - distance * 0.3  # 暗角强度
        vignette_3d = np.stack([vignette] * 3, axis=2)

        enhanced = img_array.astype(np.float32) * vignette_3d
        enhanced = np.clip(enhanced, 0, 255)

        return Image.fromarray(enhanced.astype(np.uint8))

    def _apply_motion_blur(self, image: Image.Image) -> Image.Image:
        """应用运动模糊"""
        img_array = np.array(image)

        # 随机选择模糊方向和强度
        kernel_size = random.choice([5, 7, 9])
        direction = random.choice(['horizontal', 'vertical'])

        if direction == 'horizontal':
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
        else:  # vertical
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:, kernel_size // 2] = 1.0 / kernel_size

        # 应用卷积
        blurred = cv2.filter2D(img_array, -1, kernel)

        # 混合原图和模糊图
        alpha = random.uniform(0.3, 0.7)
        enhanced = cv2.addWeighted(img_array, 1 - alpha, blurred, alpha, 0)

        return Image.fromarray(enhanced)

    def _adjust_brightness(self, image: Image.Image) -> Image.Image:
        """调整亮度"""
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _reduce_contrast(self, image: Image.Image) -> Image.Image:
        """降低对比度"""
        factor = random.uniform(0.7, 0.9)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def _add_noise(self, image: Image.Image) -> Image.Image:
        """添加噪声"""
        img_array = np.array(image)

        # 选择噪声类型
        noise_type = random.choice(['gaussian', 'salt_pepper'])

        if noise_type == 'gaussian':
            # 高斯噪声
            mean = 0
            var = random.uniform(0.001, 0.005)
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, img_array.shape)
            noisy = img_array + gauss * 255
            noisy = np.clip(noisy, 0, 255)

        else:  # salt_pepper
            # 椒盐噪声
            amount = random.uniform(0.001, 0.005)
            s_vs_p = random.uniform(0.3, 0.7)  # 盐 vs 椒的比例

            noisy = np.copy(img_array)

            # 盐噪声
            num_salt = np.ceil(amount * img_array.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
            noisy[coords[0], coords[1], :] = 255

            # 椒噪声
            num_pepper = np.ceil(amount * img_array.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
            noisy[coords[0], coords[1], :] = 0

        return Image.fromarray(noisy.astype(np.uint8))

    def _gamma_correction(self, image: Image.Image) -> Image.Image:
        """伽马校正"""
        img_array = np.array(image).astype(np.float32) / 255.0
        gamma = random.uniform(0.8, 1.2)

        corrected = np.power(img_array, gamma)
        corrected = (corrected * 255).astype(np.uint8)

        return Image.fromarray(corrected)

    def _add_cloud_effect(self, image: Image.Image) -> Image.Image:
        """添加云层效果"""
        img_array = np.array(image)

        # 轻微降低饱和度和对比度，模拟多云天气
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 0.9)  # 降低饱和度
        hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.9, 1.0)  # 轻微降低亮度

        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(enhanced)

    def _adjust_color_temperature(self, image: Image.Image) -> Image.Image:
        """调整色温"""
        img_array = np.array(image).astype(np.float32)

        # 随机选择冷色调或暖色调
        temp_type = random.choice(['warm', 'cool'])

        if temp_type == 'warm':
            # 暖色调：增加红色，减少蓝色
            img_array[:, :, 0] *= random.uniform(1.0, 1.1)  # 红色通道
            img_array[:, :, 2] *= random.uniform(0.9, 1.0)  # 蓝色通道
        else:
            # 冷色调：增加蓝色，减少红色
            img_array[:, :, 0] *= random.uniform(0.9, 1.0)  # 红色通道
            img_array[:, :, 2] *= random.uniform(1.0, 1.1)  # 蓝色通道

        img_array = np.clip(img_array, 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))

    def _shift_hue(self, image: Image.Image) -> Image.Image:
        """色调偏移"""
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # 随机偏移色调
        shift = random.randint(-10, 10)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(enhanced)

    def _adjust_saturation(self, image: Image.Image) -> Image.Image:
        """调整饱和度"""
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def _enhance_sharpness(self, image: Image.Image) -> Image.Image:
        """增强锐度"""
        factor = random.uniform(1.2, 1.5)
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def _apply_gaussian_blur(self, image: Image.Image) -> Image.Image:
        """应用高斯模糊"""
        kernel_size = random.choice([3, 5])
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size))

    def _simulate_jpeg_compression(self, image: Image.Image) -> Image.Image:
        """模拟JPEG压缩"""
        # 将图像保存为JPEG并重新加载以模拟压缩
        import io
        buffer = io.BytesIO()

        # 随机质量
        quality = random.randint(70, 95)
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        compressed = Image.open(buffer)
        return compressed

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        """颜色抖动"""
        # 应用随机颜色变换
        transforms = []

        # 随机亮度调整
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            transforms.append(lambda img: ImageEnhance.Brightness(img).enhance(brightness))

        # 随机对比度调整
        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            transforms.append(lambda img: ImageEnhance.Contrast(img).enhance(contrast))

        # 随机饱和度调整
        if random.random() > 0.5:
            saturation = random.uniform(0.8, 1.2)
            transforms.append(lambda img: ImageEnhance.Color(img).enhance(saturation))

        # 随机应用变换
        if transforms:
            random.shuffle(transforms)
            for transform in transforms[:2]:  # 最多应用2个变换
                image = transform(image)

        return image

    def save_enhanced_image(self, image_data: np.ndarray, output_path: str,
                            metadata: Optional[Dict] = None):
        """
        保存增强后的图像和元数据

        Args:
            image_data: 图像数据
            output_path: 输出路径
            metadata: 增强元数据
        """
        # 保存图像
        cv2.imwrite(output_path, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))

        # 保存元数据
        if metadata:
            meta_path = output_path.replace('.png', '_meta.json').replace('.jpg', '_meta.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    def generate_enhancement_report(self, output_dir: str):
        """生成增强报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'enhancement_methods': self.enhancement_methods,
            'config': self.config.get('enhancement', {}),
            'weather': self.config.get('scenario', {}).get('weather', 'clear'),
            'time_of_day': self.config.get('scenario', {}).get('time_of_day', 'noon'),
            'statistics': {
                'total_methods': len(self.enhancement_methods),
                'weather_specific_methods': [
                    m for m in self.enhancement_methods
                    if m in ['rain_effect', 'fog_effect', 'night_effect', 'cloud_effect']
                ],
                'quality_methods': [
                    m for m in self.enhancement_methods
                    if m in ['normalize', 'sharpness_enhance', 'gamma_correction']
                ],
                'random_methods': [
                    m for m in self.enhancement_methods
                    if m in ['hue_shift', 'saturation_adjust', 'color_jitter', 'jpeg_compression']
                ]
            }
        }

        report_path = os.path.join(output_dir, 'enhancement_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report


class SensorCalibrator:
    """传感器校准模块"""

    def __init__(self, config: Dict):
        self.config = config
        self.calibration_data = {}

    def generate_calibration_files(self, output_dir: str,
                                   vehicle_locations: List[Dict],
                                   camera_positions: List[Dict]):
        """
        生成传感器校准文件

        Args:
            output_dir: 输出目录
            vehicle_locations: 车辆位置信息
            camera_positions: 相机位置信息
        """
        calib_dir = os.path.join(output_dir, "calibration")
        os.makedirs(calib_dir, exist_ok=True)

        # 1. 生成相机内参
        self._generate_camera_intrinsics(calib_dir)

        # 2. 生成外参（相机到车辆）
        self._generate_extrinsics(calib_dir, vehicle_locations, camera_positions)

        # 3. 生成传感器间标定
        self._generate_sensor_calibration(calib_dir)

        # 4. 生成时间同步校准
        self._generate_temporal_calibration(calib_dir)

        print(f"校准文件已生成到: {calib_dir}")

    def _generate_camera_intrinsics(self, calib_dir: str):
        """生成相机内参"""
        image_size = self.config.get('sensors', {}).get('image_size', [1280, 720])
        width, height = image_size[0], image_size[1]

        # 内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        fx = width * 0.8  # 假设焦距
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        intrinsics = {
            'camera_matrix': [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ],
            'distortion_coefficients': [0.0, 0.0, 0.0, 0.0, 0.0],  # 无畸变
            'image_size': [width, height],
            'fov': 90.0,
            'sensor_type': 'pinhole'
        }

        # 为每个相机生成内参（简化处理，实际应该每个相机不同）
        for i in range(4):  # 假设4个车辆相机
            file_path = os.path.join(calib_dir, f'camera_{i + 1}_intrinsic.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(intrinsics, f, indent=2, ensure_ascii=False)

        # 基础设施相机内参（可能不同）
        for i in range(4):  # 假设4个基础设施相机
            file_path = os.path.join(calib_dir, f'infra_camera_{i + 1}_intrinsic.json')
            # 基础设施相机可能有不同的参数
            infra_intrinsics = intrinsics.copy()
            infra_intrinsics['fov'] = 120.0  # 更广的视角
            infra_intrinsics['sensor_type'] = 'fisheye'  # 鱼眼相机

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(infra_intrinsics, f, indent=2, ensure_ascii=False)

    def _generate_extrinsics(self, calib_dir: str,
                             vehicle_locations: List[Dict],
                             camera_positions: List[Dict]):
        """生成外参（相机到车辆坐标系）"""
        for i, (vehicle_loc, cam_pos) in enumerate(zip(vehicle_locations, camera_positions)):
            # 外参矩阵 [R|t]
            # 这里简化为单位矩阵，实际应根据安装位置计算
            extrinsics = {
                'vehicle_id': vehicle_loc.get('id', i + 1),
                'translation': cam_pos.get('translation', [0, 0, 0]),
                'rotation': cam_pos.get('rotation', [0, 0, 0]),
                'transform_matrix': [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ],
                'timestamp': datetime.now().isoformat()
            }

            file_path = os.path.join(calib_dir, f'vehicle_{i + 1}_extrinsic.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(extrinsics, f, indent=2, ensure_ascii=False)

    def _generate_sensor_calibration(self, calib_dir: str):
        """生成传感器间标定（相机到LiDAR等）"""
        # 相机到LiDAR标定
        cam_to_lidar = {
            'sensor_pair': ['camera_1', 'lidar_1'],
            'translation': [0.5, 0.0, -0.2],  # 相机相对于LiDAR的位置
            'rotation': [0, 0, 0],  # 旋转
            'calibration_method': 'manual',
            'accuracy': 0.01,  # 标定精度（米）
            'timestamp': datetime.now().isoformat()
        }

        file_path = os.path.join(calib_dir, 'camera_to_lidar_calib.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cam_to_lidar, f, indent=2, ensure_ascii=False)

        # 基础设施相机到全局坐标系标定
        infra_to_global = {
            'sensor_type': 'infrastructure_camera',
            'global_position': [0, 0, 12],  # 安装位置
            'orientation': [0, -25, 0],  # 俯仰角
            'calibration_method': 'gps_imu',
            'accuracy': 0.05,
            'timestamp': datetime.now().isoformat()
        }

        file_path = os.path.join(calib_dir, 'infrastructure_global_calib.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(infra_to_global, f, indent=2, ensure_ascii=False)

    def _generate_temporal_calibration(self, calib_dir: str):
        """生成时间同步校准"""
        temporal_calib = {
            'sensor_latencies': {
                'camera': 0.033,  # 33ms延迟
                'lidar': 0.010,  # 10ms延迟
                'gps': 0.001,  # 1ms延迟
                'imu': 0.005  # 5ms延迟
            },
            'sync_method': 'hardware_trigger',
            'sync_accuracy': 0.001,  # 1ms同步精度
            'master_clock': 'gps_time',
            'timestamp': datetime.now().isoformat()
        }

        file_path = os.path.join(calib_dir, 'temporal_calibration.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(temporal_calib, f, indent=2, ensure_ascii=False)


class DataQualityMonitor:
    """数据质量监控器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.quality_metrics = {
            'images': {'total': 0, 'valid': 0, 'issues': []},
            'lidar': {'total': 0, 'valid': 0, 'issues': []},
            'annotations': {'total': 0, 'valid': 0, 'issues': []},
            'calibration': {'total': 0, 'valid': 0, 'issues': []}
        }

    def check_image_quality(self, image_path: str) -> Dict:
        """检查图像质量"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'valid': False, 'error': '无法读取图像'}

            # 检查图像尺寸
            h, w, c = img.shape
            if h == 0 or w == 0:
                return {'valid': False, 'error': '图像尺寸无效'}

            # 检查图像是否全黑或全白
            mean_brightness = np.mean(img)
            if mean_brightness < 10 or mean_brightness > 245:
                return {'valid': False, 'warning': f'图像过暗或过亮: {mean_brightness}'}

            # 检查图像对比度
            contrast = np.std(img)
            if contrast < 20:
                return {'valid': True, 'warning': f'图像对比度过低: {contrast}'}

            return {'valid': True, 'dimensions': (w, h), 'brightness': mean_brightness, 'contrast': contrast}

        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def check_lidar_quality(self, lidar_path: str) -> Dict:
        """检查LiDAR数据质量"""
        try:
            # 检查文件是否存在和大小
            if not os.path.exists(lidar_path):
                return {'valid': False, 'error': '文件不存在'}

            file_size = os.path.getsize(lidar_path)
            if file_size == 0:
                return {'valid': False, 'error': '文件为空'}

            # 如果是.bin文件，检查点云数据
            if lidar_path.endswith('.bin'):
                points = np.fromfile(lidar_path, dtype=np.float32)
                num_points = len(points) // 4  # 假设每个点4个float

                if num_points < 100:
                    return {'valid': False, 'error': f'点云数量过少: {num_points}'}

                return {
                    'valid': True,
                    'num_points': num_points,
                    'file_size': file_size
                }

            return {'valid': True, 'file_size': file_size}

        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def update_metrics(self, data_type: str, check_result: Dict):
        """更新质量指标"""
        self.quality_metrics[data_type]['total'] += 1

        if check_result.get('valid', False):
            self.quality_metrics[data_type]['valid'] += 1
        else:
            self.quality_metrics[data_type]['issues'].append(check_result.get('error', '未知错误'))

    def generate_quality_report(self) -> Dict:
        """生成质量报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'issues_by_type': {},
            'quality_score': 0
        }

        total_valid = 0
        total_count = 0

        for data_type, metrics in self.quality_metrics.items():
            total = metrics['total']
            valid = metrics['valid']

            if total > 0:
                valid_ratio = valid / total
                report['summary'][data_type] = {
                    'total': total,
                    'valid': valid,
                    'valid_ratio': round(valid_ratio * 100, 2),
                    'issues_count': len(metrics['issues'])
                }

                total_valid += valid
                total_count += total

                # 记录问题
                if metrics['issues']:
                    report['issues_by_type'][data_type] = metrics['issues'][:10]  # 只显示前10个问题

        # 计算总体质量分数
        if total_count > 0:
            report['quality_score'] = round((total_valid / total_count) * 100, 2)

        # 保存报告
        report_path = os.path.join(self.output_dir, 'data_quality_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def print_quality_summary(self):
        """打印质量摘要"""
        print("\n" + "=" * 60)
        print("数据质量报告")
        print("=" * 60)

        report = self.generate_quality_report()

        for data_type, summary in report['summary'].items():
            print(f"\n{data_type.upper()}:")
            print(f"  总数: {summary['total']}")
            print(f"  有效: {summary['valid']}")
            print(f"  有效率: {summary['valid_ratio']}%")
            if summary['issues_count'] > 0:
                print(f"  问题数: {summary['issues_count']}")

        print(f"\n总体质量分数: {report['quality_score']}/100")

        if report['quality_score'] >= 90:
            print("✓ 数据质量优秀")
        elif report['quality_score'] >= 75:
            print("✓ 数据质量良好")
        elif report['quality_score'] >= 60:
            print("⚠ 数据质量一般")
        else:
            print("✗ 数据质量需要改进")

        print("=" * 60)