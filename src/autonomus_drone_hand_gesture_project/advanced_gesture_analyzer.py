"""
高级手势分析器模块
负责分析复杂手势和提供额外的手势识别功能
作者: xiaoshiyuan888
"""

import cv2
import numpy as np
import time
import math
from collections import deque


class AdvancedGestureAnalyzer:
    """高级手势分析器 - 识别复杂手势"""

    def __init__(self, speech_manager=None, config=None):
        self.speech_manager = speech_manager
        self.config = config

        # 手势轨迹追踪
        self.trajectory_history = deque(maxlen=30)
        self.fingertip_history = deque(maxlen=20)

        # 手势状态
        self.rotation_direction = None  # 旋转方向: 'cw' 顺时针, 'ccw' 逆时针
        self.grab_state = False  # 抓取状态
        self.photo_state = False  # 拍照状态

        # 颜色定义
        self.colors = {
            'rotation_cw': (0, 255, 255),  # 黄色 - 顺时针旋转
            'rotation_ccw': (255, 255, 0),  # 青色 - 逆时针旋转
            'grab': (255, 0, 255),  # 紫色 - 抓取
            'release': (0, 255, 0),  # 绿色 - 释放
            'photo': (255, 165, 0),  # 橙色 - 拍照
            'auto_flight': (128, 0, 128),  # 紫色 - 自动飞行
            'return_home': (0, 128, 128),  # 深青色 - 返航
        }

        # 语音消息映射
        self.messages = {
            'gesture_rotate_cw': "顺时针旋转",
            'gesture_rotate_ccw': "逆时针旋转",
            'gesture_grab': "抓取",
            'gesture_release': "释放",
            'gesture_photo': "拍照",
            'gesture_auto_flight': "自动飞行模式",
            'gesture_return_home': "返航",
        }

        # 初始化语音消息
        if self.speech_manager:
            self.speech_manager.messages.update(self.messages)

        print("✓ 高级手势分析器已初始化")

    def analyze(self, frame, hand_data):
        """分析手势数据，识别复杂手势"""
        if hand_data is None:
            return None

        result = {}

        # 分析旋转手势
        rotation_result = self.detect_rotation(hand_data)
        if rotation_result:
            result.update(rotation_result)

        # 分析抓取手势
        grab_result = self.detect_grab(hand_data)
        if grab_result:
            result.update(grab_result)

        # 分析拍照手势
        photo_result = self.detect_photo_gesture(hand_data)
        if photo_result:
            result.update(photo_result)

        # 分析复杂手势
        complex_result = self.detect_complex_gestures(hand_data)
        if complex_result:
            result.update(complex_result)

        return result if result else None

    def detect_rotation(self, hand_data):
        """检测旋转手势"""
        if 'fingertips' not in hand_data or len(hand_data['fingertips']) < 1:
            return None

        # 获取食指指尖
        index_fingertip = hand_data['fingertips'][0] if len(hand_data['fingertips']) > 0 else None

        if index_fingertip is None:
            return None

        # 添加指尖到历史
        self.fingertip_history.append(index_fingertip)

        if len(self.fingertip_history) < 15:
            return None

        # 分析指尖运动轨迹
        trajectory = list(self.fingertip_history)

        # 计算轨迹的中心点
        points = np.array(trajectory)
        center = np.mean(points, axis=0)

        # 计算每个点相对于中心的角度
        angles = []
        for point in trajectory:
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            angle = math.atan2(dy, dx)
            angles.append(angle)

        # 分析角度变化趋势
        if len(angles) >= 10:
            # 计算角度变化量
            angle_changes = []
            for i in range(1, len(angles)):
                change = angles[i] - angles[i-1]
                # 处理角度跨越2π边界的情况
                if change > math.pi:
                    change -= 2 * math.pi
                elif change < -math.pi:
                    change += 2 * math.pi
                angle_changes.append(change)

            avg_change = np.mean(angle_changes)

            # 判断旋转方向
            if abs(avg_change) > 0.05:  # 阈值
                if avg_change > 0:
                    self.rotation_direction = 'ccw'
                    return {
                        'gesture': 'RotateCCW',
                        'confidence': min(0.9, abs(avg_change) * 5),
                        'direction': 'ccw',
                        'angle_change': avg_change
                    }
                else:
                    self.rotation_direction = 'cw'
                    return {
                        'gesture': 'RotateCW',
                        'confidence': min(0.9, abs(avg_change) * 5),
                        'direction': 'cw',
                        'angle_change': avg_change
                    }

        return None

    def detect_grab(self, hand_data):
        """检测抓取手势（握拳）"""
        if 'contour' not in hand_data or 'bbox' not in hand_data:
            return None

        contour = hand_data['contour']
        bbox = hand_data['bbox']

        # 计算轮廓面积和边界框面积
        contour_area = cv2.contourArea(contour)
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)

        if bbox_area == 0:
            return None

        # 计算面积比（握拳时轮廓更接近边界框）
        area_ratio = contour_area / bbox_area

        # 获取手指数量
        finger_count = len(hand_data.get('fingers', []))

        # 握拳的特征：面积比高，手指数量少
        if area_ratio > 0.7 and finger_count <= 1:
            # 检查是否之前不是抓取状态
            if not self.grab_state:
                self.grab_state = True
                return {
                    'gesture': 'Grab',
                    'confidence': min(0.95, area_ratio * 1.2),
                    'state': 'grab'
                }
        else:
            if self.grab_state:
                self.grab_state = False
                return {
                    'gesture': 'Release',
                    'confidence': 0.8,
                    'state': 'release'
                }

        return None

    def detect_photo_gesture(self, hand_data):
        """检测拍照手势（OK手势）"""
        if 'fingers' not in hand_data or 'fingertips' not in hand_data:
            return None

        fingers = hand_data['fingers']
        fingertips = hand_data['fingertips']

        # OK手势的特征：拇指和食指形成圆圈，其他手指伸直
        if len(fingers) == 3 and len(fingertips) >= 2:
            # 检查拇指和食指的距离
            if len(fingertips) >= 2:
                thumb_tip = fingertips[0]
                index_tip = fingertips[1]

                distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)

                # 如果拇指和食指距离较近，可能是OK手势
                if distance < 50:  # 阈值
                    if not self.photo_state:
                        self.photo_state = True
                        return {
                            'gesture': 'TakePhoto',
                            'confidence': max(0.7, 1.0 - distance/100),
                            'distance': distance
                        }

        self.photo_state = False
        return None

    def detect_complex_gestures(self, hand_data):
        """检测复杂手势（返航、自动飞行等）"""
        if 'fingers' not in hand_data or 'fingertips' not in hand_data:
            return None

        finger_count = len(hand_data.get('fingers', []))
        fingertips = hand_data.get('fingertips', [])

        # 返航手势：五指张开然后缓慢握拳
        if finger_count == 5 and len(fingertips) == 5:
            # 检查是否保持这个姿势一段时间
            if hasattr(self, 'five_finger_start_time'):
                elapsed = time.time() - self.five_finger_start_time
                if elapsed > 2.0:  # 保持2秒
                    delattr(self, 'five_finger_start_time')
                    return {
                        'gesture': 'ReturnHome',
                        'confidence': 0.85,
                        'hold_time': elapsed
                    }
            else:
                self.five_finger_start_time = time.time()
        elif hasattr(self, 'five_finger_start_time'):
            delattr(self, 'five_finger_start_time')

        # 自动飞行手势：特定手指组合
        if finger_count == 4 and len(fingertips) == 4:
            # 检查是否为特定的手指组合（例如，缺少小指）
            return {
                'gesture': 'AutoFlight',
                'confidence': 0.75,
                'finger_pattern': 'four_fingers'
            }

        return None

    def draw_gesture_info(self, frame):
        """在帧上绘制手势信息"""
        h, w = frame.shape[:2]

        # 绘制旋转指示器
        if self.rotation_direction:
            center_x, center_y = w - 80, 100
            radius = 30

            if self.rotation_direction == 'cw':
                color = self.colors['rotation_cw']
                # 绘制顺时针箭头
                cv2.circle(frame, (center_x, center_y), radius, color, 2)
                cv2.putText(frame, "CW", (center_x - 15, center_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                color = self.colors['rotation_ccw']
                # 绘制逆时针箭头
                cv2.circle(frame, (center_x, center_y), radius, color, 2)
                cv2.putText(frame, "CCW", (center_x - 20, center_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制抓取状态
        if self.grab_state:
            grab_x, grab_y = w - 80, 150
            color = self.colors['grab']
            cv2.rectangle(frame, (grab_x - 20, grab_y - 20),
                         (grab_x + 20, grab_y + 20), color, 2)
            cv2.putText(frame, "抓取", (grab_x - 20, grab_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制指尖轨迹
        if len(self.fingertip_history) > 1:
            trajectory_color = (0, 255, 255)  # 黄色
            for i in range(1, len(self.fingertip_history)):
                pt1 = self.fingertip_history[i-1]
                pt2 = self.fingertip_history[i]
                cv2.line(frame, pt1, pt2, trajectory_color, 2)

            # 绘制当前指尖
            if self.fingertip_history:
                last_point = self.fingertip_history[-1]
                cv2.circle(frame, last_point, 5, (0, 0, 255), -1)

        return frame

    def get_gesture_info(self):
        """获取手势信息"""
        return {
            'rotation_direction': self.rotation_direction,
            'grab_state': self.grab_state,
            'photo_state': self.photo_state,
            'trajectory_length': len(self.fingertip_history)
        }