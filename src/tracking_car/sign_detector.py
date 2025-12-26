"""
sign_detector.py - 轻量级交通标志识别模块
最小化集成，不破坏现有结构
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional
import time

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class TrafficSignDetector:
    """轻量级交通标志检测器"""
    
    def __init__(self, config=None):
        """
        初始化
        Args:
            config: 可选配置，默认使用内置简单配置
        """
        # 默认配置
        self.config = config or {
            'enabled': True,
            'conf_threshold': 0.5,
            'show_signs': True,
            'enable_actions': False,  # 默认不触发动作，只显示
        }
        
        # 尝试加载YOLO模型，如果失败则使用简单的颜色检测
        self.model = None
        self.use_yolo = False
        
        try:
            from ultralytics import YOLO
            # 尝试加载预训练模型（可以用通用物体检测模型）
            self.model = YOLO('yolov8n.pt')  # 使用现有的YOLO模型
            self.use_yolo = True
            logger.info("✅ 使用YOLO进行标志检测")
        except Exception as e:
            logger.warning(f"无法加载YOLO模型，使用简单颜色检测: {e}")
            self._init_simple_detector()
        
        # 简单的标志颜色检测
        self.sign_colors = {
            'red': {  # 停车、禁止类标志
                'lower': np.array([0, 50, 50]),
                'upper': np.array([10, 255, 255])
            },
            'blue': {  # 指示类标志
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            'yellow': {  # 警告类标志
                'lower': np.array([20, 100, 100]),
                'upper': np.array([30, 255, 255])
            }
        }
        
        # 标志形状模板（可选）
        self.shape_templates = self._load_shape_templates()
        
        # 检测历史
        self.detected_signs_history = []
        
        logger.info("✅ 轻量级标志检测器初始化完成")
    
    def _init_simple_detector(self):
        """初始化简单检测器"""
        # 加载简单的形状模板
        self.templates = {
            'triangle': self._create_triangle_mask(),
            'circle': self._create_circle_mask(),
            'octagon': self._create_octagon_mask(),  # 停车标志
            'square': self._create_square_mask()
        }
    
    def detect(self, image: np.ndarray, ego_speed: float = 0.0) -> List[Dict]:
        """
        检测图像中的交通标志
        
        Args:
            image: 输入图像
            ego_speed: 自车速度（用于距离估计）
            
        Returns:
            List[Dict]: 检测到的标志列表
        """
        if not self.config.get('enabled', True):
            return []
        
        signs = []
        
        try:
            # 方法1：如果YOLO可用，使用YOLO检测
            if self.use_yolo and self.model is not None:
                signs = self._detect_with_yolo(image, ego_speed)
            
            # 方法2：否则使用简单的颜色+形状检测
            else:
                signs = self._detect_with_color(image, ego_speed)
            
            # 过滤重复的标志（简单的NMS）
            signs = self._non_max_suppression(signs)
            
            # 更新历史
            if signs:
                self.detected_signs_history = signs[:10]  # 保留最近10个
                
                # 简单的动作触发（可选的）
                if self.config.get('enable_actions', False):
                    self._trigger_actions(signs, ego_speed)
            
            return signs
            
        except Exception as e:
            logger.error(f"标志检测失败: {e}")
            return []
    
    def _detect_with_yolo(self, image: np.ndarray, ego_speed: float) -> List[Dict]:
        """使用YOLO检测"""
        results = self.model.predict(
            image,
            conf=self.config.get('conf_threshold', 0.5),
            verbose=False
        )
        
        signs = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # 过滤出可能是交通标志的类别（YOLO COCO数据集中的类别）
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # 只保留相关类别
                    if class_name in ['stop sign', 'traffic light', 'parking meter']:
                        bbox = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        # 推断标志类型
                        sign_type = self._infer_sign_type(class_name, bbox, image)
                        
                        sign_info = {
                            'bbox': bbox.tolist(),
                            'confidence': confidence,
                            'type': sign_type,
                            'class_id': class_id,
                            'class_name': class_name,
                            'timestamp': time.time()
                        }
                        
                        # 估算距离（基于边界框大小）
                        sign_info['distance'] = self._estimate_distance(sign_info['bbox'], ego_speed)
                        
                        signs.append(sign_info)
        
        return signs
    
    def _detect_with_color(self, image: np.ndarray, ego_speed: float) -> List[Dict]:
        """使用颜色检测"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        signs = []
        
        # 检测每种标志颜色
        for color_name, color_range in self.sign_colors.items():
            # 创建颜色掩码
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            # 形态学操作去除噪声
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 过滤太小的区域
                if area < 100:
                    continue
                
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算形状特征
                shape = self._detect_shape(contour)
                
                # 推断标志类型
                sign_type = self._infer_sign_type_from_color(color_name, shape)
                
                if sign_type:
                    sign_info = {
                        'bbox': [x, y, x + w, y + h],
                        'confidence': min(0.9, area / 1000.0),  # 简单置信度
                        'type': sign_type,
                        'color': color_name,
                        'shape': shape,
                        'timestamp': time.time()
                    }
                    
                    # 估算距离
                    sign_info['distance'] = self._estimate_distance(sign_info['bbox'], ego_speed)
                    
                    signs.append(sign_info)
        
        return signs
    
    def _infer_sign_type(self, class_name: str, bbox: np.ndarray, image: np.ndarray) -> str:
        """根据检测结果推断标志类型"""
        type_map = {
            'stop sign': 'stop',
            'traffic light': 'traffic_light',
            'parking meter': 'parking',
        }
        
        # 如果是stop sign，进一步确认（检查是否是八边形）
        if class_name == 'stop sign':
            # 提取ROI检查形状
            x1, y1, x2, y2 = map(int, bbox)
            roi = image[y1:y2, x1:x2]
            if self._is_octagon_shape(roi):
                return 'stop'
        
        return type_map.get(class_name, 'unknown')
    
    def _infer_sign_type_from_color(self, color: str, shape: str) -> Optional[str]:
        """根据颜色和形状推断标志类型"""
        # 简单的推断规则
        if color == 'red' and shape == 'octagon':
            return 'stop'
        elif color == 'red' and shape == 'circle':
            return 'no_entry'
        elif color == 'yellow' and shape == 'triangle':
            return 'warning'
        elif color == 'blue' and shape == 'circle':
            return 'mandatory'
        elif color == 'blue' and shape == 'square':
            return 'information'
        
        return None
    
    def _detect_shape(self, contour) -> str:
        """检测轮廓形状"""
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        num_sides = len(approx)
        
        if num_sides == 3:
            return 'triangle'
        elif num_sides == 4:
            # 判断是正方形还是长方形
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                return 'square'
            else:
                return 'rectangle'
        elif 8 <= num_sides <= 12:  # 八边形
            return 'octagon'
        else:
            # 计算圆形度
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                return 'circle'
        
        return 'unknown'
    
    def _estimate_distance(self, bbox: List[float], ego_speed: float) -> float:
        """估算标志距离（简化版本）"""
        # 基于边界框高度估算距离
        # 假设标志的标准高度为0.5米，焦距为1000像素
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        if bbox_height <= 0:
            return 100.0  # 默认远距离
        
        # 简化距离公式：距离 ∝ 1/高度
        distance = 500.0 / bbox_height
        
        # 根据速度调整（运动模糊）
        if ego_speed > 10:
            distance *= (1 + ego_speed / 100.0)
        
        return min(distance, 100.0)  # 最大100米
    
    def _non_max_suppression(self, signs: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """简单的非极大值抑制"""
        if len(signs) <= 1:
            return signs
        
        # 按置信度排序
        sorted_signs = sorted(signs, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        while sorted_signs:
            # 取置信度最高的
            best = sorted_signs.pop(0)
            keep.append(best)
            
            # 移除与best重叠度高的
            sorted_signs = [
                sign for sign in sorted_signs 
                if self._iou(best['bbox'], sign['bbox']) < iou_threshold
            ]
        
        return keep
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _trigger_actions(self, signs: List[Dict], ego_speed: float):
        """触发简单动作（可选）"""
        for sign in signs:
            sign_type = sign.get('type', '')
            distance = sign.get('distance', 100.0)
            
            if sign_type == 'stop' and distance < 20:
                logger.warning(f"🛑 前方 {distance:.1f}米有停车标志")
            
            elif 'speed_limit' in sign_type and distance < 30:
                try:
                    limit = int(sign_type.split('_')[-1])
                    if ego_speed * 3.6 > limit + 5:  # m/s转km/h
                        logger.warning(f"📏 前方限速{limit}km/h，当前{ego_speed*3.6:.0f}km/h")
                except:
                    pass
    
    def draw_signs(self, image: np.ndarray, signs: List[Dict]) -> np.ndarray:
        """在图像上绘制检测到的标志"""
        if not self.config.get('show_signs', True):
            return image
        
        result = image.copy()
        
        for sign in signs:
            bbox = sign['bbox']
            sign_type = sign.get('type', 'unknown')
            confidence = sign.get('confidence', 0.0)
            distance = sign.get('distance', 0.0)
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 根据标志类型选择颜色
            color_map = {
                'stop': (0, 0, 255),      # 红色
                'warning': (0, 165, 255),  # 橙色
                'traffic_light': (0, 255, 0),  # 绿色
                'speed_limit': (0, 255, 255),  # 黄色
                'unknown': (128, 128, 128)  # 灰色
            }
            
            color = color_map.get(sign_type, (128, 128, 128))
            
            # 绘制边界框
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{sign_type} {confidence:.2f}"
            if distance > 0:
                label += f" {distance:.1f}m"
            
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result
    
    def get_signs_info(self) -> Dict:
        """获取标志检测统计信息"""
        return {
            'total_signs': len(self.detected_signs_history),
            'recent_signs': self.detected_signs_history[-5:] if self.detected_signs_history else [],
            'enabled': self.config.get('enabled', True)
        }


def test_detector():
    """测试检测器"""
    detector = TrafficSignDetector()
    
    # 测试图片
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 模拟一个红色八边形（停车标志）
    center = (320, 240)
    radius = 50
    points = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((int(x), int(y)))
    
    cv2.fillPoly(test_image, [np.array(points)], (0, 0, 255))
    
    # 检测
    signs = detector.detect(test_image)
    
    print(f"检测到 {len(signs)} 个标志")
    for sign in signs:
        print(f"  类型: {sign.get('type')}, 置信度: {sign.get('confidence'):.2f}")
    
    # 绘制
    result = detector.draw_signs(test_image, signs)
    
    return len(signs) > 0


if __name__ == "__main__":
    success = test_detector()
    if success:
        print("✅ 标志检测器测试通过")
    else:
        print("⚠️  测试未检测到标志")