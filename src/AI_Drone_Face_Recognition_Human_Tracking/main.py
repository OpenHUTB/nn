import sys
import os
import threading
import time
from datetime import datetime

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.drone_controller import DroneController
from modules.face_detector import FaceDetector
from modules.person_detector import PersonDetector
from modules.face_recognizer import FaceRecognizer
from modules.ui_controller import UIController
from modules.voice_synthesizer import VoiceSynthesizer


class AIDroneSystem:
    def __init__(self):
        """初始化AI无人机系统"""
        print("🚀 正在初始化AI无人机系统...")

        # 初始化各个模块
        self.drone = DroneController()
        self.person_detector = PersonDetector()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.voice = VoiceSynthesizer()
        self.ui = UIController()

        # 状态变量
        self.running = False
        self.target_person = None
        self.current_target_bbox = None
        self.recognized_persons = {}

        # 线程锁
        self.lock = threading.Lock()

        print("✅ 系统初始化完成")

    def start(self):
        """启动系统"""
        self.running = True

        # 连接无人机
        if not self.drone.connect():
            print("❌ 无法连接无人机")
            return False

        # 启动UI界面
        ui_thread = threading.Thread(target=self.ui.start, args=(self,))
        ui_thread.daemon = True
        ui_thread.start()

        # 主循环
        self.main_loop()

        return True

    def main_loop(self):
        """主运行循环"""
        print("🔄 开始主循环...")

        while self.running:
            try:
                # 获取无人机图像
                frame = self.drone.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # 人物检测
                persons, person_frame = self.person_detector.detect(frame)

                # 如果有选中的目标，进行跟踪
                if self.target_person:
                    self.track_target(persons, person_frame)

                # 人脸检测与识别
                recognized_info = self.detect_and_recognize_faces(frame)

                # 更新UI显示
                self.ui.update_display({
                    'original_frame': frame,
                    'person_frame': person_frame,
                    'persons': persons,
                    'recognized_info': recognized_info,
                    'target': self.target_person
                })

                # 语音播报识别结果
                self.announce_recognition(recognized_info)

                time.sleep(0.05)  # 控制帧率

            except Exception as e:
                print(f"❌ 主循环错误: {e}")
                time.sleep(1)

    def detect_and_recognize_faces(self, frame):
        """检测并识别人脸"""
        # 检测人脸
        faces = self.face_detector.detect(frame)

        recognized_info = []

        for face in faces:
            # 提取人脸区域
            x, y, w, h = face
            face_img = frame[y:y + h, x:x + w]

            # 识别人脸
            identity = self.face_recognizer.recognize(face_img)

            if identity != "Unknown":
                recognized_info.append({
                    'bbox': (x, y, w, h),
                    'name': identity,
                    'confidence': 0.95  # 这里可以添加置信度
                })

        return recognized_info

    def track_target(self, persons, frame):
        """跟踪选定目标"""
        if not persons:
            return

        # 寻找最接近的目标
        target_bbox = None
        min_distance = float('inf')

        for person in persons:
            # 这里可以根据不同的策略选择目标
            # 例如：选择最大的、最接近中心的等
            distance = self.calculate_distance_to_center(person['bbox'], frame.shape)

            if distance < min_distance:
                min_distance = distance
                target_bbox = person['bbox']

        if target_bbox:
            self.current_target_bbox = target_bbox

            # 计算控制指令
            control_command = self.calculate_control_command(target_bbox, frame.shape)

            # 发送控制指令给无人机
            self.drone.move_to_target(control_command)

    def calculate_distance_to_center(self, bbox, frame_shape):
        """计算边界框中心到图像中心的距离"""
        x1, y1, x2, y2 = bbox
        bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)

        return ((bbox_center[0] - frame_center[0]) ** 2 +
                (bbox_center[1] - frame_center[1]) ** 2) ** 0.5

    def calculate_control_command(self, bbox, frame_shape):
        """根据目标位置计算无人机控制指令"""
        x1, y1, x2, y2 = bbox
        bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)

        # 计算偏移量（归一化到[-1, 1]）
        dx = (bbox_center[0] - frame_center[0]) / frame_shape[1]
        dy = (bbox_center[1] - frame_center[1]) / frame_shape[0]

        # 计算目标大小（用于调整距离）
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_shape[1] * frame_shape[0]
        area_ratio = bbox_area / frame_area

        # 生成控制指令
        command = {
            'forward': 0.0,
            'right': 0.0,
            'up': 0.0,
            'yaw': 0.0
        }

        # 调整无人机位置使目标居中
        if abs(dx) > 0.1:  # 如果水平偏移大于10%
            command['yaw'] = -dx * 0.5  # 旋转无人机

        if abs(dy) > 0.1:  # 如果垂直偏移大于10%
            command['up'] = dy * 0.5  # 上下移动

        # 根据目标大小调整距离
        if area_ratio < 0.2:  # 目标太小，需要靠近
            command['forward'] = 0.3
        elif area_ratio > 0.5:  # 目标太大，需要远离
            command['forward'] = -0.3

        return command

    def announce_recognition(self, recognized_info):
        """语音播报识别结果"""
        for info in recognized_info:
            name = info['name']
            if name not in self.recognized_persons:
                self.recognized_persons[name] = datetime.now()
                self.voice.speak(f"识别到 {name}")

    def select_target(self, bbox):
        """选择跟踪目标"""
        self.target_person = {
            'bbox': bbox,
            'selected_time': datetime.now()
        }
        print(f"🎯 已选择跟踪目标: {bbox}")

    def add_new_face(self, face_img, name):
        """添加新人脸到数据库"""
        success = self.face_recognizer.add_face(face_img, name)
        if success:
            print(f"✅ 成功添加人脸: {name}")
            self.voice.speak(f"已添加 {name} 到数据库")
        return success

    def stop(self):
        """停止系统"""
        print("🛑 正在停止系统...")
        self.running = False
        self.drone.disconnect()
        self.ui.stop()
        print("✅ 系统已停止")


def main():
    """主函数"""
    system = AIDroneSystem()

    try:
        system.start()
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    finally:
        system.stop()


if __name__ == "__main__":
    main()