"""
drone_perception.py
无人机感知模块 - 用于目标检测、深度估计和障碍物感知
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import time
import os

class DronePerception:
    """
    无人机感知模块主类
    提供目标检测、深度估计和障碍物感知功能
    """

    def __init__(self, camera_params: Optional[dict] = None):
        """
        初始化感知模块

        Args:
            camera_params: 相机内参矩阵和畸变系数
        """
        self.camera_params = camera_params or {
            'fx': 920.0, 'fy': 920.0,  # 焦距
            'cx': 640.0, 'cy': 360.0,  # 主点坐标
            'width': 1280, 'height': 720
        }

        # 初始化目标检测器
        self.target_detector = TargetDetector()

        # 初始化深度估计器
        self.depth_estimator = DepthEstimator()

        # 状态变量
        self.obstacles = []
        self.targets = []
        self.last_update_time = time.time()

        print("DronePerception模块初始化完成")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        处理单帧图像

        Args:
            frame: 输入图像帧 (BGR格式)

        Returns:
            包含感知结果的字典
        """
        if frame is None:
            return {"error": "空帧输入"}

        results = {
            'timestamp': time.time(),
            'frame_shape': frame.shape,
            'targets': [],
            'obstacles': [],
            'depth_map': None
        }

        try:
            # 1. 目标检测
            self.targets = self.target_detector.detect(frame)
            results['targets'] = self.targets

            # 2. 深度估计
            depth_map = self.depth_estimator.estimate(frame)
            results['depth_map'] = depth_map

            # 3. 障碍物检测
            self.obstacles = self.detect_obstacles(frame, depth_map)
            results['obstacles'] = self.obstacles

            # 4. 安全区域分析
            results['safety_score'] = self.analyze_safety(self.obstacles)

            self.last_update_time = time.time()

        except Exception as e:
            results['error'] = str(e)
            print(f"处理帧时出错: {e}")

        return results

    def detect_obstacles(self, frame: np.ndarray, depth_map: Optional[np.ndarray]) -> List[dict]:
        """
        检测障碍物

        Args:
            frame: 输入图像
            depth_map: 深度图

        Returns:
            障碍物列表
        """
        obstacles = []

        # 只检测非目标区域的障碍物（避免重复检测目标）
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        # 在目标区域创建掩码，避免将目标误检测为障碍物
        for target in self.targets:
            if 'bbox' in target:
                x1, y1, x2, y2 = map(int, target['bbox'])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        # 使用掩码后的图像进行障碍物检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

        # 使用自适应阈值而不是Canny，减少噪声
        blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)
        edges = cv2.adaptiveThreshold(blurred, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

        # 形态学操作去除小噪声
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:10]:  # 只处理前10个最大轮廓
            area = cv2.contourArea(contour)
            if 100 < area < 50000:  # 调整面积阈值范围
                x, y, w, h = cv2.boundingRect(contour)

                # 计算轮廓的中心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = x + w//2, y + h//2

                obstacle = {
                    'bbox': [x, y, x+w, y+h],
                    'area': area,
                    'center': (cX, cY),
                    'type': 'unknown',
                    'contour_points': len(contour)
                }

                # 如果有深度信息，估算距离
                if depth_map is not None and y < depth_map.shape[0] and x < depth_map.shape[1]:
                    # 获取ROI区域内的深度值
                    y_end = min(y+h, depth_map.shape[0])
                    x_end = min(x+w, depth_map.shape[1])
                    roi_depth = depth_map[y:y_end, x:x_end]

                    if roi_depth.size > 0:
                        # 使用中值而不是均值，减少异常值影响
                        median_depth = np.median(roi_depth)
                        obstacle['estimated_distance'] = float(median_depth)

                obstacles.append(obstacle)

        return obstacles

    def analyze_safety(self, obstacles: List[dict]) -> float:
        """
        分析当前帧的安全性

        Args:
            obstacles: 障碍物列表

        Returns:
            安全分数 (0-1)
        """
        if not obstacles:
            return 1.0

        danger_score = 0
        max_danger_per_obstacle = 0.3  # 每个障碍物最大危险分数

        for obs in obstacles:
            # 障碍物大小越大，危险分数越高（对数缩放）
            area_factor = min(np.log10(obs['area'] / 100 + 1), 1.0)

            # 距离因素
            if 'estimated_distance' in obs:
                # 距离越近危险越大，指数衰减
                dist = obs['estimated_distance']
                if dist <= 0.1:  # 非常近
                    dist_factor = 1.0
                else:
                    dist_factor = min(2.0 / (dist + 0.1), 1.0)
            else:
                dist_factor = 0.3  # 默认中等危险

            # 位置因素：靠近图像中心更危险
            h, w = self.camera_params['height'], self.camera_params['width']
            center_x, center_y = w // 2, h // 2
            obs_center_x, obs_center_y = obs['center']

            # 计算到图像中心的距离（归一化）
            dist_to_center = np.sqrt(((obs_center_x - center_x) / w)**2 +
                                    ((obs_center_y - center_y) / h)**2)
            center_factor = max(0, 1.0 - dist_to_center * 2)  # 越靠近中心值越大

            # 综合危险分数
            obstacle_danger = area_factor * dist_factor * center_factor
            danger_score += min(obstacle_danger, max_danger_per_obstacle)

        # 确保危险分数在0-1之间
        danger_score = min(danger_score, 1.0)
        safety_score = 1.0 - danger_score

        return round(safety_score, 2)

    def visualize(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """
        可视化感知结果

        Args:
            frame: 原始图像帧
            results: 感知结果

        Returns:
            可视化后的图像
        """
        vis_frame = frame.copy()

        # 绘制目标边界框
        for target in results.get('targets', []):
            if 'bbox' in target:
                x1, y1, x2, y2 = map(int, target['bbox'])
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if 'class_name' in target:
                    cv2.putText(vis_frame, f"Target: {target['class_name']}",
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 255, 0), 2)

        # 绘制障碍物边界框
        for obstacle in results.get('obstacles', []):
            if 'bbox' in obstacle:
                x1, y1, x2, y2 = map(int, obstacle['bbox'])
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

                # 显示估计距离
                if 'estimated_distance' in obstacle:
                    dist_text = f"{obstacle['estimated_distance']:.2f}"
                    cv2.putText(vis_frame, dist_text,
                               (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX,
                               0.4, (0, 0, 255), 1)

        # 显示安全分数
        safety_score = results.get('safety_score', 1.0)
        if safety_score > 0.7:
            safety_color = (0, 255, 0)  # 绿色
        elif safety_score > 0.4:
            safety_color = (0, 255, 255)  # 黄色
        else:
            safety_color = (0, 0, 255)  # 红色

        # 绘制安全分数背景框
        cv2.rectangle(vis_frame, (5, 5), (200, 85), (40, 40, 40), -1)
        cv2.rectangle(vis_frame, (5, 5), (200, 85), safety_color, 2)

        cv2.putText(vis_frame, f"Safety Score: {safety_score}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, safety_color, 2)

        # 显示帧信息
        cv2.putText(vis_frame, f"Targets: {len(results.get('targets', []))}",
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Obstacles: {len(results.get('obstacles', []))}",
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1)

        return vis_frame

    def get_status(self) -> dict:
        """
        获取模块状态

        Returns:
            状态字典
        """
        return {
            'active': True,
            'last_update': self.last_update_time,
            'camera_params': self.camera_params,
            'target_count': len(self.targets),
            'obstacle_count': len(self.obstacles)
        }

    def save_visualization(self, frame: np.ndarray, results: dict, filename: str = "perception_output.jpg"):
        """
        保存可视化结果到文件

        Args:
            frame: 原始图像帧
            results: 感知结果
            filename: 输出文件名
        """
        vis_frame = self.visualize(frame, results)
        cv2.imwrite(filename, vis_frame)
        print(f"可视化结果已保存到: {filename}")
        return filename


class TargetDetector:
    """目标检测器"""

    def __init__(self):
        # 这里可以加载YOLO等模型
        # 简化版本使用预定义的HSV颜色范围
        self.color_ranges = {
            'red': [
                ([0, 100, 100], [10, 255, 255]),    # 红色范围1
                ([160, 100, 100], [180, 255, 255])  # 红色范围2
            ],
            'green': ([40, 50, 50], [80, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255])
        }

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        检测目标

        Args:
            frame: 输入图像

        Returns:
            检测到的目标列表
        """
        targets = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, color_range in self.color_ranges.items():
            if color_name == 'red':
                # 红色有两个范围
                lower1 = np.array(color_range[0][0], dtype=np.uint8)
                upper1 = np.array(color_range[0][1], dtype=np.uint8)
                lower2 = np.array(color_range[1][0], dtype=np.uint8)
                upper2 = np.array(color_range[1][1], dtype=np.uint8)

                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower = np.array(color_range[0], dtype=np.uint8)
                upper = np.array(color_range[1], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)

            # 形态学操作
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 50000:  # 合理的面积范围
                    x, y, w, h = cv2.boundingRect(contour)

                    # 计算轮廓的圆度
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        circularity = 0

                    target = {
                        'bbox': [x, y, x+w, y+h],
                        'class_name': color_name,
                        'confidence': min(0.7 + circularity * 0.3, 0.95),  # 圆度影响置信度
                        'area': area,
                        'circularity': round(circularity, 2)
                    }
                    targets.append(target)

        return targets


class DepthEstimator:
    """深度估计器"""

    def __init__(self):
        # 这里可以加载MiDaS等深度估计模型
        pass

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        估计深度图

        Args:
            frame: 输入图像

        Returns:
            深度图 (归一化到0-1)
        """
        # 简化版本：使用图像强度作为深度代理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用双边滤波保留边缘
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # 计算局部对比度
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        contrast_map = np.abs(laplacian)

        # 归一化对比度图
        if contrast_map.max() > contrast_map.min():
            contrast_map = (contrast_map - contrast_map.min()) / (contrast_map.max() - contrast_map.min())

        # 计算强度图
        intensity_map = blurred.astype(float) / 255.0

        # 模拟距离：假设图像中心区域更近
        h, w = gray.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2

        # 创建中心距离权重图
        dist_from_center = np.sqrt(((x_coords - center_x) / (w//2))**2 +
                                  ((y_coords - center_y) / (h//2))**2)
        center_weight = np.clip(1.0 - dist_from_center, 0, 1)

        # 创建深度图：高强度（亮）+ 高对比度（边缘）+ 靠近中心 表示更近
        depth_map = 0.4 * intensity_map + 0.3 * (1 - contrast_map) + 0.3 * center_weight

        # 归一化到0-1
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

        # 应用高斯平滑
        depth_map = cv2.GaussianBlur(depth_map, (7, 7), 0)

        return depth_map


def test_perception():
    """测试函数"""
    print("=== 测试DronePerception模块 ===")

    # 初始化感知模块
    perception = DronePerception()

    # 创建测试图像
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加一些彩色方块模拟目标
    cv2.rectangle(test_frame, (100, 100), (200, 200), (0, 255, 0), -1)  # 绿色
    cv2.rectangle(test_frame, (400, 150), (500, 250), (0, 0, 255), -1)  # 红色
    cv2.rectangle(test_frame, (300, 300), (400, 400), (255, 0, 0), -1)  # 蓝色

    # 添加一些障碍物（灰色方块）
    cv2.rectangle(test_frame, (50, 50), (80, 200), (100, 100, 100), -1)  # 左边障碍物
    cv2.rectangle(test_frame, (550, 300), (600, 400), (150, 150, 150), -1)  # 右边障碍物

    # 添加一些随机噪声（模拟纹理）
    noise = np.random.randint(0, 30, test_frame.shape[:2], dtype=np.uint8)
    for i in range(3):
        test_frame[:,:,i] = cv2.add(test_frame[:,:,i], noise[:,:,np.newaxis])

    # 处理帧
    results = perception.process_frame(test_frame)

    # 打印详细结果
    print(f"\n检测到目标数量: {len(results['targets'])}")
    for i, target in enumerate(results['targets']):
        print(f"  目标{i+1}: {target.get('class_name', 'unknown')}, "
              f"位置: {target.get('bbox', [])}, "
              f"置信度: {target.get('confidence', 0):.2f}")

    print(f"\n检测到障碍物数量: {len(results['obstacles'])}")
    for i, obstacle in enumerate(results['obstacles'][:3]):  # 只显示前3个
        if 'estimated_distance' in obstacle:
            print(f"  障碍物{i+1}: 位置{obstacle.get('bbox', [])}, "
                  f"距离: {obstacle['estimated_distance']:.2f}")
        else:
            print(f"  障碍物{i+1}: 位置{obstacle.get('bbox', [])}")

    print(f"\n安全分数: {results['safety_score']}")

    if results.get('error'):
        print(f"错误信息: {results['error']}")

    # 保存可视化结果
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "perception_test_output.jpg")
    perception.save_visualization(test_frame, results, output_path)

    # 显示模块状态
    print("\n模块状态:")
    for key, value in perception.get_status().items():
        print(f"  {key}: {value}")

    # 保存深度图
    if results['depth_map'] is not None:
        depth_map = results['depth_map']
        depth_map_display = (depth_map * 255).astype(np.uint8)
        depth_map_colored = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_JET)
        depth_path = os.path.join(output_dir, "depth_map.jpg")
        cv2.imwrite(depth_path, depth_map_colored)
        print(f"深度图已保存到: {depth_path}")

    print(f"\n测试完成！结果已保存到 {output_dir} 目录")


def test_with_real_image(image_path: str = None):
    """使用真实图像测试"""
    print("=== 使用真实图像测试 ===")

    # 如果没有提供图像路径，创建一个简单的测试图像
    if image_path is None or not os.path.exists(image_path):
        print("未提供有效图像路径，创建测试图像...")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_frame, (320, 240), 100, (0, 255, 0), -1)  # 绿色圆形
        cv2.rectangle(test_frame, (100, 100), (200, 150), (255, 0, 0), -1)  # 蓝色矩形
        cv2.line(test_frame, (500, 100), (600, 400), (0, 0, 255), 5)  # 红色线
    else:
        test_frame = cv2.imread(image_path)
        if test_frame is None:
            print(f"无法读取图像: {image_path}")
            return

    # 初始化感知模块
    perception = DronePerception()

    # 处理帧
    results = perception.process_frame(test_frame)

    print(f"图像尺寸: {test_frame.shape}")
    print(f"检测到目标: {len(results['targets'])}")
    print(f"检测到障碍物: {len(results['obstacles'])}")
    print(f"安全分数: {results['safety_score']}")

    # 保存结果
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "real_image_test.jpg")
    perception.save_visualization(test_frame, results, output_path)

    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists("output"):
        os.makedirs("output")

    # 运行测试
    test_perception()

    # 如果需要测试真实图像，取消下面一行的注释并提供图像路径
    # test_with_real_image("your_image.jpg")