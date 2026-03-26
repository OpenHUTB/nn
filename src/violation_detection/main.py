#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
违规行为识别系统 - 简化版（无需 dlib）
"""
import cv2
import time
import numpy as np
from detector import SimpleViolationDetector

class SimpleViolationSystem:
    """简化版违规行为识别系统"""
    
    def __init__(self):
        self.detector = SimpleViolationDetector()
        self.cap = None
        self.running = False
        
    def init_camera(self, camera_id=0):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print("❌ 无法打开摄像头")
            return False
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ 摄像头初始化成功")
        return True
    
    def detect_face_simple(self, frame):
        """
        使用 Haar 级联检测人脸
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用 OpenCV 的 Haar 级联分类器
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        
        if len(faces) > 0:
            # 返回最大的人脸
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = frame[y:y+h, x:x+w]
            return True, face_roi, (x, y, w, h)
        
        return False, None, None
    
    def detect_hands_simple(self, frame):
        """
        简化版手部检测（基于肤色）
        """
        # 转换到 HSV 颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 肤色范围
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 肤色掩码
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # 提取手部中心点
        hand_positions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:  # 过滤小区域
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hand_positions.append((cx, cy))
        
        return hand_positions
    
    def run(self):
        """运行主程序"""
        if not self.init_camera():
            return
        
        self.running = True
        
        print("\n" + "="*60)
        print("🚀 违规行为识别系统已启动（简化版）")
        print("="*60)
        print("检测规则：")
        print("  📱 玩手机：手部在脸部附近 > 3秒")
        print("  😴 睡觉：闭眼状态 > 5秒（基于眼睛亮度）")
        print("  🚶 离席：检测不到人脸 > 10秒")
        print("\n按 'q' 键退出程序")
        print("按 's' 键保存当前画面")
        print("="*60 + "\n")
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        while self.running:
            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            frame_count += 1
            
            # 计算 FPS
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            
            # 人脸检测
            face_detected, face_roi, face_rect = self.detect_face_simple(frame)
            
            # 手部检测
            hand_positions = self.detect_hands_simple(frame)
            
            # 更新检测状态
            violations = self.detector.update(
                face_detected, face_roi, hand_positions
            )
            
            # 绘制检测结果
            frame = self.draw_results(frame, face_detected, face_rect, 
                                     hand_positions, violations, fps)
            
            # 显示画面
            cv2.imshow("Violation Detection System", frame)
            
            # 按键检测
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q 或 ESC
                self.running = False
                break
            elif key == ord('s'):  # 保存截图
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 截图已保存: {filename}")
        
        self.cleanup()
    
    def draw_results(self, frame, face_detected, face_rect, 
                    hand_positions, violations, fps):
        """绘制检测结果"""
        
        # 绘制人脸框
        if face_detected and face_rect:
            x, y, w, h = face_rect
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # 显示人脸检测状态
            cv2.putText(frame, "Face Detected", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制手部位置
        for i, (hx, hy) in enumerate(hand_positions):
            cv2.circle(frame, (hx, hy), 10, (255, 0, 0), -1)
            cv2.putText(frame, f"Hand {i+1}", (hx-15, hy-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 显示违规行为
        y_offset = 80
        violation_colors = {
            "PHONE_USAGE": (0, 0, 255),
            "SLEEPING": (0, 0, 255),
            "ABSENCE": (0, 0, 255)
        }
        
        violation_texts = {
            "PHONE_USAGE": "⚠️ VIOLATION: Phone Usage Detected!",
            "SLEEPING": "😴 VIOLATION: Sleeping Detected!",
            "ABSENCE": "🚶 VIOLATION: Absence Detected!"
        }
        
        for violation in violations:
            if violation in violation_texts:
                text = violation_texts[violation]
                color = violation_colors[violation]
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30
        
        # 显示统计信息
        stats = self.detector.get_statistics()
        y_start = frame.shape[0] - 120
        
        cv2.putText(frame, "Statistics:", (10, y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Phone: {stats['phone']}", 
                   (10, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Sleep: {stats['sleep']}", 
                   (10, y_start + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Absence: {stats['absence']}", 
                   (10, y_start + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示 FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加标题
        cv2.putText(frame, "Violation Detection System", 
                   (frame.shape[1]//2 - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 添加操作提示
        cv2.putText(frame, "Press 'q' to quit | 's' to save", 
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("📊 最终统计:")
        stats = self.detector.get_statistics()
        print(f"  玩手机违规次数: {stats['phone']}")
        print(f"  睡觉违规次数: {stats['sleep']}")
        print(f"  离席违规次数: {stats['absence']}")
        print("="*60)
        print("👋 程序已退出")

def main():
    """主函数"""
    system = SimpleViolationSystem()
    try:
        system.run()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()