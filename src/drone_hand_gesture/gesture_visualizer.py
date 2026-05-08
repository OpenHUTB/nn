import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque

class GestureVisualizer:
    """手势识别可视化器"""
    
    def __init__(self):
        self.gesture_history = deque(maxlen=30)
        self.confidence_history = deque(maxlen=30)
        self.gesture_counts = {}
        
        # 手势颜色映射
        self.gesture_colors = {
            "open_palm": (0, 255, 0),
            "closed_fist": (0, 0, 255),
            "pointing_up": (255, 0, 0),
            "pointing_down": (255, 165, 0),
            "victory": (255, 0, 255),
            "thumb_up": (0, 255, 255),
            "thumb_down": (128, 0, 128),
            "ok_sign": (255, 255, 0),
            "rock": (192, 192, 192),
            "peace": (255, 192, 203),
            "hand_detected": (128, 128, 128),
            "no_hand": (0, 0, 0)
        }
        
        # 命令映射
        self.gesture_commands = {
            "open_palm": "起飞",
            "closed_fist": "降落",
            "pointing_up": "上升",
            "pointing_down": "下降",
            "victory": "前进",
            "thumb_up": "后退",
            "thumb_down": "停止",
            "ok_sign": "悬停",
            "rock": "左转",
            "peace": "右转"
        }
        
        # 创建图表
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('Gesture Recognition Analysis')
        
        # 实时置信度折线图
        self.line, = self.ax1.plot([], [], 'b-', linewidth=2)
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Confidence')
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True)
        
        # 手势分类柱状图
        self.bars = None
        self.ax2.set_xlabel('Gesture')
        self.ax2.set_ylabel('Count')
        self.ax2.set_ylim(0, 15)
        self.ax2.tick_params(axis='x', rotation=45)
        
        self.canvas = FigureCanvasAgg(self.fig)
    
    def update_history(self, gesture, confidence):
        """更新历史记录"""
        self.gesture_history.append(gesture)
        self.confidence_history.append(confidence)
        
        if gesture in self.gesture_counts:
            self.gesture_counts[gesture] += 1
        else:
            self.gesture_counts[gesture] = 1
    
    def draw_info_panel(self, image, gesture, confidence, command):
        """在图像上绘制信息面板"""
        height, width = image.shape[:2]
        
        # 创建半透明面板
        overlay = image.copy()
        cv2.rectangle(overlay, (width - 200, 0), (width, 150), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # 绘制手势信息
        color = self.gesture_colors.get(gesture, (128, 128, 128))
        
        cv2.putText(image, f"Gesture: {gesture}", (width - 190, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Confidence: {confidence:.2f}", (width - 190, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Command: {command}", (width - 190, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return image
    
    def draw_gesture_icon(self, image, gesture):
        """绘制手势图标"""
        icon_size = 60
        icon_x = 10
        icon_y = image.shape[0] - icon_size - 10
        
        # 创建图标背景
        cv2.rectangle(image, (icon_x, icon_y), (icon_x + icon_size, icon_y + icon_size),
                     self.gesture_colors.get(gesture, (128, 128, 128)), -1)
        
        # 绘制简单图标
        center_x = icon_x + icon_size // 2
        center_y = icon_y + icon_size // 2
        
        if gesture == "open_palm":
            # 手掌图标
            cv2.circle(image, (center_x, center_y), 20, (255, 255, 255), 2)
            for i in range(5):
                angle = -np.pi/2 + (i * np.pi/4)
                cv2.line(image, (center_x, center_y),
                        (int(center_x + 20 * np.cos(angle)),
                         int(center_y + 20 * np.sin(angle))), (255, 255, 255), 2)
        elif gesture == "closed_fist":
            # 拳头图标
            cv2.circle(image, (center_x, center_y), 15, (255, 255, 255), 2)
        elif gesture == "pointing_up":
            # 食指图标
            cv2.line(image, (center_x, center_y), (center_x, center_y - 25), (255, 255, 255), 3)
        
        return image
    
    def update_charts(self):
        """更新图表"""
        # 更新置信度折线图
        self.ax1.clear()
        self.ax1.plot(list(range(len(self.confidence_history))), 
                     list(self.confidence_history), 'b-', linewidth=2)
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Confidence')
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True)
        
        # 更新手势统计柱状图
        self.ax2.clear()
        if self.gesture_counts:
            gestures = list(self.gesture_counts.keys())
            counts = list(self.gesture_counts.values())
            colors = [self.gesture_colors.get(g, (128, 128, 128)) for g in gestures]
            colors = [(r/255, g/255, b/255) for r, g, b in colors]
            
            self.ax2.bar(gestures, counts, color=colors)
            self.ax2.set_xlabel('Gesture')
            self.ax2.set_ylabel('Count')
            self.ax2.set_ylim(0, max(counts) + 5 if counts else 15)
            self.ax2.tick_params(axis='x', rotation=45)
        
        self.canvas.draw()
        
        # 转换为OpenCV图像
        buf = self.canvas.buffer_rgba()
        img = np.asarray(buf)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        return img
    
    def draw_3d_landmarks(self, landmarks, image):
        """在图像角落绘制3D关键点投影"""
        if landmarks is None or len(landmarks) == 0:
            return image
        
        # 提取关键点
        points = np.array(landmarks).reshape(-1, 3)
        
        # 简化的3D到2D投影
        proj_size = 100
        proj_x = image.shape[1] - proj_size - 10
        proj_y = image.shape[0] - proj_size - 80
        
        # 绘制投影区域
        cv2.rectangle(image, (proj_x, proj_y), (proj_x + proj_size, proj_y + proj_size),
                     (100, 100, 100), 2)
        
        # 绘制关键点
        for i, (x, y, z) in enumerate(points):
            # 简单透视投影
            scale = 50 / (50 + z * 100)
            px = int(proj_x + (x * proj_size * scale) + proj_size // 2)
            py = int(proj_y + ((1 - y) * proj_size * scale) + proj_size // 2)
            
            color = (0, 255, 0) if i % 4 == 0 else (255, 0, 0)  # 指尖绿色，其他红色
            cv2.circle(image, (px, py), 3, color, -1)
        
        cv2.putText(image, "3D Projection", (proj_x, proj_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def show_statistics(self):
        """显示统计信息"""
        print("\n=== 手势识别统计 ===")
        print(f"总识别次数: {sum(self.gesture_counts.values())}")
        print("各类手势分布:")
        for gesture, count in sorted(self.gesture_counts.items(), key=lambda x: -x[1]):
            percentage = (count / sum(self.gesture_counts.values())) * 100
            print(f"  {gesture}: {count} ({percentage:.1f}%)")
    
    def reset(self):
        """重置历史记录"""
        self.gesture_history.clear()
        self.confidence_history.clear()
        self.gesture_counts = {}