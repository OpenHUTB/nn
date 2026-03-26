#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
违规行为检测核心模块（简化版，无需 dlib）
"""
import time
import numpy as np
import cv2

class SimpleViolationDetector:
    """简化版违规行为检测器（不需要 dlib）"""
    
    def __init__(self):
        # 检测阈值配置
        self.phone_threshold = 3      # 玩手机检测阈值（秒）
        self.sleep_threshold = 5      # 睡觉检测阈值（秒）
        self.absence_threshold = 10   # 离席检测阈值（秒）
        
        # 状态记录
        self.phone_start_time = None
        self.sleep_start_time = None
        self.absence_start_time = None
        
        # 违规计数
        self.violations = {
            'phone': 0,
            'sleep': 0,
            'absence': 0
        }
        
        # 当前状态
        self.current_violations = []
        
        # 用于检测闭眼的帧计数
        self.eye_close_frames = 0
        
    def detect_phone_usage_simple(self, face_roi, hand_positions):
        """
        简化版玩手机检测
        判断手部是否在脸部区域
        """
        if face_roi is None or len(hand_positions) == 0:
            return False
        
        h, w = face_roi.shape[:2]
        face_center = (w//2, h//2)
        
        for hand_pos in hand_positions:
            # 计算手部到脸部中心的距离
            distance = np.sqrt((hand_pos[0] - face_center[0])**2 + 
                             (hand_pos[1] - face_center[1])**2)
            
            # 如果手在脸部附近（距离小于脸宽的一半）
            if distance < w/2:
                return True
        return False
    
    def detect_sleeping_simple(self, face_roi):
        """
        简化版睡觉检测
        通过检测眼睛区域的平均亮度判断（简单方法）
        """
        if face_roi is None:
            return False
        
        # 获取人脸的上半部分（眼睛大概区域）
        h, w = face_roi.shape[:2]
        eye_region = face_roi[int(h*0.2):int(h*0.4), int(w*0.2):int(w*0.8)]
        
        if eye_region.size == 0:
            return False
        
        # 转换为灰度图
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # 计算平均亮度
        avg_brightness = np.mean(gray)
        
        # 如果眼睛区域变暗，可能是闭眼
        # 阈值需要根据实际光照调整
        if avg_brightness < 80:
            self.eye_close_frames += 1
            return self.eye_close_frames > 15  # 连续15帧闭眼判定为睡觉
        else:
            self.eye_close_frames = 0
            return False
    
    def detect_absence_simple(self, face_detected):
        """检测离席"""
        return not face_detected
    
    def update(self, face_detected, face_roi, hand_positions):
        """
        更新检测状态
        """
        current_time = time.time()
        violations = []
        
        # 1. 检测玩手机
        phone_usage = self.detect_phone_usage_simple(face_roi, hand_positions)
        if phone_usage:
            if self.phone_start_time is None:
                self.phone_start_time = current_time
            elif current_time - self.phone_start_time > self.phone_threshold:
                if "phone" not in violations:
                    violations.append("PHONE_USAGE")
                    self.violations['phone'] += 1
                    print(f"⚠️ 检测到玩手机行为！")
        else:
            self.phone_start_time = None
        
        # 2. 检测睡觉
        sleeping = self.detect_sleeping_simple(face_roi)
        if sleeping:
            if self.sleep_start_time is None:
                self.sleep_start_time = current_time
            elif current_time - self.sleep_start_time > self.sleep_threshold:
                if "sleep" not in violations:
                    violations.append("SLEEPING")
                    self.violations['sleep'] += 1
                    print(f"😴 检测到睡觉行为！")
        else:
            self.sleep_start_time = None
        
        # 3. 检测离席
        absence = self.detect_absence_simple(face_detected)
        if absence:
            if self.absence_start_time is None:
                self.absence_start_time = current_time
            elif current_time - self.absence_start_time > self.absence_threshold:
                if "absence" not in violations:
                    violations.append("ABSENCE")
                    self.violations['absence'] += 1
                    print(f"🚶 检测到离席行为！")
        else:
            self.absence_start_time = None
        
        self.current_violations = violations
        return violations
    
    def get_statistics(self):
        """获取统计信息"""
        return self.violations