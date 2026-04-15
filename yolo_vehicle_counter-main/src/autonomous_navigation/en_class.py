# src/environment_classifier.py
import cv2
import numpy as np
import random


class EnvironmentClassifier:
    """增强版环境分类器（仅依赖OpenCV和numpy）"""

    def __init__(self):
        self.environments = [
            "Ruins", "Building", "Forest", "Road",
            "Sky", "Water", "Fire", "Animal", "Vehicle"
        ]
        self.weights = {
            "Ruins": 0.35, "Building": 0.20, "Forest": 0.15,
            "Road": 0.10, "Sky": 0.08, "Water": 0.05,
            "Fire": 0.02, "Animal": 0.03, "Vehicle": 0.02
        }
        self.thresholds = {
            'fire_red': 0.25, 'fire_bright': 200, 'fire_grad': 15,
            'sky_blue': 0.3, 'sky_l_mean': 180,
            'water_blue': 0.25, 'water_edges': 0.03,
            'forest_green': 0.3, 'forest_grad': 20,
            'ruins_edges': 0.07, 'ruins_variance': 1200, 'ruins_bright': 120,
            'building_edges': 0.05, 'building_bright': 100, 'building_green': 0.1,
            'road_edges': 0.02, 'road_bright_low': 100, 'road_bright_high': 200, 'road_color': 0.1,
            'animal_red': 0.15, 'animal_edges': 0.04
        }

    def classify(self, image):
        if image is None:
            return "Unknown", 0.0
        features = self._extract_features(image)
        env, conf = self._rule_based(features)
        if env == "Unknown":
            env, conf = self._weighted_random(features)
        return env, conf

    def _extract_features(self, image):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        f = {}
        # LAB
        l_mean, l_std = cv2.meanStdDev(lab[:, :, 0])
        a_mean, a_std = cv2.meanStdDev(lab[:, :, 1])
        b_mean, b_std = cv2.meanStdDev(lab[:, :, 2])
        f['l_mean'] = l_mean[0][0]
        f['l_std'] = l_std[0][0]
        f['a_mean'] = a_mean[0][0]
        f['a_std'] = a_std[0][0]
        f['b_mean'] = b_mean[0][0]
        f['b_std'] = b_std[0][0]
        # 纹理
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        f['grad_mean'] = np.mean(mag)
        f['grad_std'] = np.std(mag)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        f['lap_var'] = np.var(laplacian)
        edges = cv2.Canny(gray, 50, 150)
        f['edge_density'] = np.sum(edges > 0) / (h * w)
        # 颜色比例
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        f['blue_ratio'] = np.sum(blue_mask > 0) / (h * w)
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        f['green_ratio'] = np.sum(green_mask > 0) / (h * w)
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        f['red_ratio'] = np.sum(red_mask > 0) / (h * w)
        f['brightness'] = np.mean(gray)
        f['gray_variance'] = np.var(gray)
        # 天空检测
        if h > 10:
            top_blue = np.sum(blue_mask[:h // 3, :] > 0) / (w * h // 3)
            bottom_blue = np.sum(blue_mask[2 * h // 3:, :] > 0) / (w * h // 3)
            f['is_sky'] = (top_blue > 0.25) and (top_blue > bottom_blue * 1.8) and (f['l_mean'] > 180)
        else:
            f['is_sky'] = False
        # 3x3网格（可选，增加特征维度）
        grid_h, grid_w = h // 3, w // 3
        for i in range(3):
            for j in range(3):
                roi = image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                if roi.size == 0: continue
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                blue_roi = cv2.inRange(roi_hsv, (100, 50, 50), (130, 255, 255))
                f[f'blue_grid_{i}_{j}'] = np.sum(blue_roi > 0) / (grid_h * grid_w)
                green_roi = cv2.inRange(roi_hsv, (40, 50, 50), (80, 255, 255))
                f[f'green_grid_{i}_{j}'] = np.sum(green_roi > 0) / (grid_h * grid_w)
                red_roi1 = cv2.inRange(roi_hsv, (0, 50, 50), (10, 255, 255))
                red_roi2 = cv2.inRange(roi_hsv, (170, 50, 50), (180, 255, 255))
                red_roi = cv2.bitwise_or(red_roi1, red_roi2)
                f[f'red_grid_{i}_{j}'] = np.sum(red_roi > 0) / (grid_h * grid_w)
        return f

    def _rule_based(self, f):
        t = self.thresholds
        if (f.get('red_ratio', 0) > t['fire_red'] and f.get('brightness', 0) > t['fire_bright'] and f.get('grad_mean',
                                                                                                          0) < t[
            'fire_grad']):
            return "Fire", 0.75
        if (f.get('blue_ratio', 0) > t['sky_blue'] and f.get('is_sky', False) and f.get('l_mean', 0) > t['sky_l_mean']):
            return "Sky", 0.85
        if (f.get('blue_ratio', 0) > t['water_blue'] and not f.get('is_sky', False) and f.get('edge_density', 0) < t[
            'water_edges']):
            return "Water", 0.70
        if (f.get('green_ratio', 0) > t['forest_green'] and f.get('grad_mean', 0) > t['forest_grad']):
            return "Forest", 0.80
        if (f.get('edge_density', 0) > t['ruins_edges'] and f.get('gray_variance', 0) > t['ruins_variance'] and f.get(
                'brightness', 0) < t['ruins_bright']):
            return "Ruins", 0.85
        if (f.get('edge_density', 0) > t['building_edges'] and f.get('brightness', 0) < t['building_bright'] and f.get(
                'green_ratio', 0) < t['building_green']):
            return "Building", 0.75
        if (f.get('edge_density', 0) < t['road_edges'] and t['road_bright_low'] < f.get('brightness', 0) < t[
            'road_bright_high'] and
                f.get('blue_ratio', 0) < t['road_color'] and f.get('green_ratio', 0) < t['road_color'] and f.get(
                    'red_ratio', 0) < t['road_color']):
            return "Road", 0.70
        if (f.get('red_ratio', 0) > t['animal_red'] and f.get('edge_density', 0) > t['animal_edges']):
            return "Vehicle", 0.60
        return "Unknown", 0.0

    def _weighted_random(self, f):
        adj = self.weights.copy()
        blue = f.get('blue_ratio', 0)
        green = f.get('green_ratio', 0)
        edges = f.get('edge_density', 0)
        if blue > 0.2:
            adj["Sky"] *= 1.5
            if blue > 0.3:
                adj["Water"] *= 0.5
        if green > 0.15:
            adj["Forest"] *= 2.0
        if edges > 0.05:
            adj["Ruins"] *= 1.8
        total = sum(adj.values())
        probs = [adj[env] / total for env in self.environments] if total > 0 else [1 / len(self.environments)] * len(
            self.environments)
        env = random.choices(self.environments, weights=probs)[0]
        conf = 0.6
        if env == "Ruins" and edges > 0.04:
            conf += 0.15
        elif env == "Forest" and green > 0.15:
            conf += 0.1
        elif env == "Sky" and blue > 0.2:
            conf += 0.1
        return env, min(conf, 0.9)