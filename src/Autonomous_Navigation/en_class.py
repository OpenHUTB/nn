#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
环境分类器模块 - 用于无人机视觉导航系统
基于手工特征（LAB颜色、纹理、颜色比例、空间网格）和规则判断，
无需机器学习库，仅依赖 OpenCV 和 NumPy。
"""

import cv2
import numpy as np
import random
from typing import Dict, Tuple, List, Optional


class EnvironmentClassifier:
    """增强版环境分类器（仅依赖OpenCV和numpy）"""

    # 环境类别列表（按优先级或常用顺序）
    ENVIRONMENTS: List[str] = [
        "Ruins", "Building", "Forest", "Road",
        "Sky", "Water", "Fire", "Animal", "Vehicle"
    ]

    # 基础权重（用于加权随机回退）
    BASE_WEIGHTS: Dict[str, float] = {
        "Ruins": 0.35, "Building": 0.20, "Forest": 0.15,
        "Road": 0.10, "Sky": 0.08, "Water": 0.05,
        "Fire": 0.02, "Animal": 0.03, "Vehicle": 0.02
    }

    # 规则判断阈值（集中管理，便于调优）
    THRESHOLDS: Dict[str, float] = {
        # 火灾
        'fire_red': 0.25,
        'fire_bright': 200,
        'fire_grad': 15,
        # 天空
        'sky_blue': 0.3,
        'sky_l_mean': 180,
        # 水域
        'water_blue': 0.25,
        'water_edges': 0.03,
        # 森林
        'forest_green': 0.3,
        'forest_grad': 20,
        # 废墟
        'ruins_edges': 0.07,
        'ruins_variance': 1200,
        'ruins_bright': 120,
        # 建筑
        'building_edges': 0.05,
        'building_bright': 100,
        'building_green': 0.1,
        # 道路
        'road_edges': 0.02,
        'road_bright_low': 100,
        'road_bright_high': 200,
        'road_color': 0.1,
        # 动物/车辆（简化）
        'animal_red': 0.15,
        'animal_edges': 0.04
    }

    def __init__(self) -> None:
        """初始化分类器：设置环境列表、权重、阈值"""
        self.environments = self.ENVIRONMENTS.copy()
        self.weights = self.BASE_WEIGHTS.copy()
        self.thresholds = self.THRESHOLDS.copy()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def classify(self, image: Optional[np.ndarray]) -> Tuple[str, float]:
        """
        对输入图像进行环境分类

        :param image: BGR 格式的图像（OpenCV 读取）
        :return: (环境名称, 置信度)
        """
        if image is None:
            return "Unknown", 0.0

        # 提取特征
        features = self._extract_features(image)

        # 优先使用规则判断
        env, conf = self._rule_based_classify(features)

        # 若规则无法判断，则使用加权随机回退
        if env == "Unknown":
            env, conf = self._weighted_random(features)

        return env, conf

    # ------------------------------------------------------------------
    # 特征提取（私有方法）
    # ------------------------------------------------------------------
    def _extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        从图像中提取丰富的特征向量

        包括：
        - LAB 颜色通道的均值和标准差 (6维)
        - 纹理特征：Sobel梯度均值/标准差、Laplacian方差、Canny边缘密度
        - 全局颜色比例：HSV空间下的蓝、绿、红色像素比例
        - 天空检测标志
        - 3×3 网格颜色比例 (27维)

        :param image: BGR 图像
        :return: 特征字典
        """
        h, w = image.shape[:2]

        # 转换颜色空间（一次性转换，避免重复计算）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        features = {}

        # ----- 1. LAB 颜色统计 -----
        l_mean, l_std = cv2.meanStdDev(lab[:, :, 0])
        a_mean, a_std = cv2.meanStdDev(lab[:, :, 1])
        b_mean, b_std = cv2.meanStdDev(lab[:, :, 2])
        features['l_mean'] = l_mean[0][0]
        features['l_std'] = l_std[0][0]
        features['a_mean'] = a_mean[0][0]
        features['a_std'] = a_std[0][0]
        features['b_mean'] = b_mean[0][0]
        features['b_std'] = b_std[0][0]

        # ----- 2. 纹理特征 -----
        # Sobel 梯度幅值
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        features['grad_mean'] = np.mean(mag)
        features['grad_std'] = np.std(mag)

        # Laplacian 方差（反映图像清晰度/纹理复杂度）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['lap_var'] = np.var(laplacian)

        # Canny 边缘密度
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (h * w)

        # ----- 3. 全局颜色比例（HSV 空间）-----
        # 蓝色范围
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        features['blue_ratio'] = np.sum(blue_mask > 0) / (h * w)

        # 绿色范围
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        features['green_ratio'] = np.sum(green_mask > 0) / (h * w)

        # 红色范围（HSV中红色分布在0~10和170~180两个区间）
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        features['red_ratio'] = np.sum(red_mask > 0) / (h * w)

        # 全局亮度与灰度方差
        features['brightness'] = np.mean(gray)
        features['gray_variance'] = np.var(gray)

        # ----- 4. 天空检测（基于顶部蓝色比例和LAB亮度）-----
        if h > 10:
            top_blue = np.sum(blue_mask[:h // 3, :] > 0) / (w * h // 3)
            bottom_blue = np.sum(blue_mask[2 * h // 3:, :] > 0) / (w * h // 3)
            features['is_sky'] = (
                top_blue > 0.25 and
                top_blue > bottom_blue * 1.8 and
                features['l_mean'] > 180
            )
        else:
            features['is_sky'] = False

        # ----- 5. 3×3 网格颜色比例（空间分布信息）-----
        grid_h, grid_w = h // 3, w // 3
        for i in range(3):
            for j in range(3):
                # 提取当前网格的 ROI
                roi = image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                if roi.size == 0:
                    continue

                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # 网格内蓝色比例
                blue_roi = cv2.inRange(roi_hsv, (100, 50, 50), (130, 255, 255))
                features[f'blue_grid_{i}_{j}'] = np.sum(blue_roi > 0) / (grid_h * grid_w)

                # 网格内绿色比例
                green_roi = cv2.inRange(roi_hsv, (40, 50, 50), (80, 255, 255))
                features[f'green_grid_{i}_{j}'] = np.sum(green_roi > 0) / (grid_h * grid_w)

                # 网格内红色比例
                red_roi1 = cv2.inRange(roi_hsv, (0, 50, 50), (10, 255, 255))
                red_roi2 = cv2.inRange(roi_hsv, (170, 50, 50), (180, 255, 255))
                red_roi = cv2.bitwise_or(red_roi1, red_roi2)
                features[f'red_grid_{i}_{j}'] = np.sum(red_roi > 0) / (grid_h * grid_w)

        return features

    # ------------------------------------------------------------------
    # 规则分类（私有方法）
    # ------------------------------------------------------------------
    def _rule_based_classify(self, f: Dict[str, float]) -> Tuple[str, float]:
        """
        基于预设规则进行环境分类（按优先级依次判断）

        :param f: 特征字典
        :return: (环境名称, 置信度) 或 ("Unknown", 0.0)
        """
        t = self.thresholds  # 本地别名，加速访问

        # 火灾：红色比例高 + 亮度高 + 纹理模糊（梯度均值低）
        if (f.get('red_ratio', 0) > t['fire_red'] and
                f.get('brightness', 0) > t['fire_bright'] and
                f.get('grad_mean', 0) < t['fire_grad']):
            return "Fire", 0.75

        # 天空：蓝色比例高 + 天空标志 + LAB L 均值高
        if (f.get('blue_ratio', 0) > t['sky_blue'] and
                f.get('is_sky', False) and
                f.get('l_mean', 0) > t['sky_l_mean']):
            return "Sky", 0.85

        # 水域：蓝色比例较高 + 非天空 + 边缘密度低
        if (f.get('blue_ratio', 0) > t['water_blue'] and
                not f.get('is_sky', False) and
                f.get('edge_density', 0) < t['water_edges']):
            return "Water", 0.70

        # 森林：绿色比例高 + 梯度均值高（纹理丰富）
        if (f.get('green_ratio', 0) > t['forest_green'] and
                f.get('grad_mean', 0) > t['forest_grad']):
            return "Forest", 0.80

        # 废墟：边缘密度高 + 灰度方差高 + 亮度低
        if (f.get('edge_density', 0) > t['ruins_edges'] and
                f.get('gray_variance', 0) > t['ruins_variance'] and
                f.get('brightness', 0) < t['ruins_bright']):
            return "Ruins", 0.85

        # 建筑：边缘密度较高 + 亮度低 + 绿色比例低
        if (f.get('edge_density', 0) > t['building_edges'] and
                f.get('brightness', 0) < t['building_bright'] and
                f.get('green_ratio', 0) < t['building_green']):
            return "Building", 0.75

        # 道路：边缘密度低 + 中等亮度 + 各颜色比例低
        if (f.get('edge_density', 0) < t['road_edges'] and
                t['road_bright_low'] < f.get('brightness', 0) < t['road_bright_high'] and
                f.get('blue_ratio', 0) < t['road_color'] and
                f.get('green_ratio', 0) < t['road_color'] and
                f.get('red_ratio', 0) < t['road_color']):
            return "Road", 0.70

        # 动物/车辆（简化，特征相似）：红色比例较高 + 边缘密度较高
        if (f.get('red_ratio', 0) > t['animal_red'] and
                f.get('edge_density', 0) > t['animal_edges']):
            return "Vehicle", 0.60

        # 无法确定
        return "Unknown", 0.0

    # ------------------------------------------------------------------
    # 加权随机回退（私有方法）
    # ------------------------------------------------------------------
    def _weighted_random(self, f: Dict[str, float]) -> Tuple[str, float]:
        """
        当规则无法确定时，根据图像特征调整基础权重，然后随机选择一个环境

        :param f: 特征字典
        :return: (环境名称, 置信度)
        """
        # 复制基础权重
        adj_weights = self.weights.copy()

        # 获取关键特征
        blue = f.get('blue_ratio', 0)
        green = f.get('green_ratio', 0)
        edges = f.get('edge_density', 0)

        # 根据特征调整权重
        if blue > 0.2:
            adj_weights["Sky"] *= 1.5
            if blue > 0.3:
                adj_weights["Water"] *= 0.5   # 减少水权重，避免误判

        if green > 0.15:
            adj_weights["Forest"] *= 2.0

        if edges > 0.05:
            adj_weights["Ruins"] *= 1.8

        # 归一化，计算概率分布
        total = sum(adj_weights.values())
        if total > 0:
            probs = [adj_weights[env] / total for env in self.environments]
        else:
            probs = [1.0 / len(self.environments)] * len(self.environments)

        # 随机选择环境
        env = random.choices(self.environments, weights=probs)[0]

        # 计算基础置信度，并根据特征微调
        conf = 0.6
        if env == "Ruins" and edges > 0.04:
            conf += 0.15
        elif env == "Forest" and green > 0.15:
            conf += 0.1
        elif env == "Sky" and blue > 0.2:
            conf += 0.1

        return env, min(conf, 0.9)   # 置信度不超过0.9