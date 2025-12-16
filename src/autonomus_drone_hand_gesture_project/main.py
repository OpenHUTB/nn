# gesture_controller.py
"""
手势控制AirSim无人机的主程序
基础版本 - 使用摄像头手势控制无人机
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import sys

# 尝试导入TensorFlow Lite，如果失败则提示
try:
    import tflite_runtime.interpreter as tflite

    TFLITE_AVAILABLE = True
except ImportError:
    print("警告: tflite_runtime未安装，将使用模拟模式")
    print("安装命令: pip install tflite-runtime")
    TFLITE_AVAILABLE = False

# 尝试导入AirSim，如果失败则使用模拟模式
try:
    import airsim

    AIRSIM_AVAILABLE = True
except ImportError:
    print("警告: AirSim未安装，将使用模拟模式")
    print("安装命令: pip install airsim")
    AIRSIM_AVAILABLE = False


class GestureController:
    """手势控制器主类"""

    def __init__(self, use_airsim=True, camera_index=0):
        """
        初始化手势控制器

        参数:
            use_airsim: 是否使用AirSim控制
            camera_index: 摄像头索引
        """
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 摄像头
        self.cap = cv2.VideoCapture(camera_index)
        self.camera_width = 640
        self.camera_height = 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        # 手势控制参数
        self.current_gesture = 0  # 0:停止
        self.prev_gesture = 0
        self.gesture_names = {
            0: "停止",
            1: "向上",
            2: "向下",
            3: "向左",
            4: "向右",
            5: "向前",
            6: "向后",
            7: "指针(旋转)"
        }

        # 无人机控制参数
        self.use_airsim = use_airsim and AIRSIM_AVAILABLE
        self.drone_client = None
        self.drone_active = False
        self.control_speed = 1.0  # 控制速度
        self.yaw_speed = 30.0  # 偏航速度(度/秒)

        # 轨迹历史记录(用于旋转检测)
        self.point_history = []
        self.max_history_length = 16

        # 控制线程
        self.control_thread = None
        self.running = False

        # 加载模型(如果可用)
        self.keypoint_classifier = None
        self.point_history_classifier = None
        self.load_models()

        # 连接AirSim(如果可用)
        if self.use_airsim:
            self.connect_airsim()

    def load_models(self):
        """加载TFLite模型"""
        if not TFLITE_AVAILABLE:
            print("使用模拟手势识别")
            return

        try:
            # 加载关键点分类器
            keypoint_model_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
            self.keypoint_classifier = tflite.Interpreter(model_path=keypoint_model_path)
            self.keypoint_classifier.allocate_tensors()

            # 加载轨迹分类器
            point_history_model_path = 'model/point_history_classifier/point_history_classifier.tflite'
            self.point_history_classifier = tflite.Interpreter(model_path=point_history_model_path)
            self.point_history_classifier.allocate_tensors()

            print("模型加载成功!")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用模拟手势识别")

    def connect_airsim(self):
        """连接AirSim无人机"""
        if not AIRSIM_AVAILABLE:
            return

        try:
            self.drone_client = airsim.MultirotorClient()
            self.drone_client.confirmConnection()
            self.drone_client.enableApiControl(True)
            self.drone_client.armDisarm(True)
            print("AirSim连接成功!")
        except Exception as e:
            print(f"AirSim连接失败: {e}")
            self.use_airsim = False

    def pre_process_keypoint(self, landmarks):
        """
        预处理关键点数据

        参数:
            landmarks: MediaPipe手部关键点

        返回:
            预处理后的关键点数组
        """
        # 转换为相对坐标
        base_x, base_y = landmarks[0].x, landmarks[0].y

        keypoints = []
        for landmark in landmarks:
            # 相对于手腕的坐标
            x = landmark.x - base_x
            y = landmark.y - base_y
            keypoints.extend([x, y])

        # 转换为numpy数组并归一化
        keypoints = np.array(keypoints)
        max_value = max(abs(keypoints.max()), abs(keypoints.min()))

        if max_value > 0:
            keypoints = keypoints / max_value

        return keypoints.astype(np.float32)

    def pre_process_point_history(self, history):
        """
        预处理轨迹历史数据

        参数:
            history: 轨迹历史列表

        返回:
            预处理后的轨迹数组
        """
        if len(history) < 2:
            return None

        # 转换为相对坐标
        base_x, base_y = history[0][0], history[0][1]

        processed_history = []
        for point in history:
            x = point[0] - base_x
            y = point[1] - base_y
            processed_history.extend([x, y])

        # 填充或截断到固定长度
        target_length = self.max_history_length * 2  # 每个点有x,y
        current_length = len(processed_history)

        if current_length < target_length:
            # 填充零
            processed_history.extend([0] * (target_length - current_length))
        elif current_length > target_length:
            # 截断
            processed_history = processed_history[:target_length]

        return np.array(processed_history, dtype=np.float32)

    def predict_gesture(self, landmarks):
        """
        预测手势

        参数:
            landmarks: MediaPipe手部关键点

        返回:
            手势ID
        """
        # 如果模型不可用，使用模拟识别
        if self.keypoint_classifier is None:
            return self.simulate_gesture(landmarks)

        try:
            # 预处理关键点
            keypoints = self.pre_process_keypoint(landmarks)

            # 获取模型输入输出详情
            input_details = self.keypoint_classifier.get_input_details()
            output_details = self.keypoint_classifier.get_output_details()

            # 设置输入数据
            input_data = np.expand_dims(keypoints, axis=0).astype(np.float32)
            self.keypoint_classifier.set_tensor(input_details[0]['index'], input_data)

            # 运行推理
            self.keypoint_classifier.invoke()

            # 获取输出
            output_data = self.keypoint_classifier.get_tensor(output_details[0]['index'])
            gesture_id = np.argmax(output_data[0])

            return gesture_id

        except Exception as e:
            print(f"手势识别错误: {e}")
            return self.simulate_gesture(landmarks)

    def simulate_gesture(self, landmarks):
        """
        模拟手势识别(当模型不可用时使用)
        基于简单规则识别手势
        """
        if not landmarks:
            return 0

        # 获取关键点坐标
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        # 简单规则识别
        # 所有手指收起 -> 停止
        if index_tip.y < middle_tip.y and middle_tip.y < wrist.y:
            return 0
        # 食指伸出 -> 向前
        elif index_tip.y < wrist.y and thumb_tip.x > index_tip.x:
            return 5
        # 拇指向左 -> 向左
        elif thumb_tip.x < wrist.x - 0.1:
            return 3
        # 拇指向右 -> 向右
        elif thumb_tip.x > wrist.x + 0.1:
            return 4
        # 所有手指向上 -> 向上
        elif index_tip.y < wrist.y - 0.2:
            return 1
        # 所有手指向下 -> 向下
        elif index_tip.y > wrist.y + 0.2:
            return 2
        else:
            return 0

    def predict_rotation(self, point_history):
        """
        预测旋转方向

        参数:
            point_history: 轨迹历史

        返回:
            0: 顺时针, 1: 逆时针
        """
        if self.point_history_classifier is None or len(point_history) < 5:
            return 0  # 默认为顺时针

        try:
            # 预处理轨迹
            processed_history = self.pre_process_point_history(point_history)
            if processed_history is None:
                return 0

            # 获取模型输入输出详情
            input_details = self.point_history_classifier.get_input_details()
            output_details = self.point_history_classifier.get_output_details()

            # 设置输入数据
            input_data = np.expand_dims(processed_history, axis=0).astype(np.float32)
            self.point_history_classifier.set_tensor(input_details[0]['index'], input_data)

            # 运行推理
            self.point_history_classifier.invoke()

            # 获取输出
            output_data = self.point_history_classifier.get_tensor(output_details[0]['index'])
            rotation_id = np.argmax(output_data[0])

            return rotation_id

        except Exception as e:
            print(f"旋转识别错误: {e}")
            return 0

    def control_drone(self, gesture_id, rotation_id=0):
        """
        根据手势控制无人机

        参数:
            gesture_id: 手势ID
            rotation_id: 旋转方向(0:顺时针, 1:逆时针)
        """
        if not self.drone_active or not self.use_airsim:
            return

        try:
            if gesture_id == 0:  # 停止
                self.drone_client.hoverAsync().join()

            elif gesture_id == 1:  # 向上
                self.drone_client.moveByVelocityZAsync(
                    0, 0, -self.control_speed, 0.1,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, 0)
                )

            elif gesture_id == 2:  # 向下
                self.drone_client.moveByVelocityZAsync(
                    0, 0, self.control_speed, 0.1,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, 0)
                )

            elif gesture_id == 3:  # 向左
                self.drone_client.moveByVelocityAsync(
                    -self.control_speed, 0, 0, 0.1,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, 0)
                )

            elif gesture_id == 4:  # 向右
                self.drone_client.moveByVelocityAsync(
                    self.control_speed, 0, 0, 0.1,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, 0)
                )

            elif gesture_id == 5:  # 向前
                self.drone_client.moveByVelocityAsync(
                    0, -self.control_speed, 0, 0.1,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, 0)
                )

            elif gesture_id == 6:  # 向后
                self.drone_client.moveByVelocityAsync(
                    0, self.control_speed, 0, 0.1,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, 0)
                )

            elif gesture_id == 7:  # 旋转
                yaw_rate = self.yaw_speed if rotation_id == 0 else -self.yaw_speed
                self.drone_client.rotateByYawRateAsync(yaw_rate, 0.1)

        except Exception as e:
            print(f"无人机控制错误: {e}")

    def control_loop(self):
        """控制循环线程"""
        while self.running:
            if self.drone_active and self.current_gesture != self.prev_gesture:
                # 如果手势改变，执行控制
                rotation_id = 0

                # 如果是手势7，检测旋转方向
                if self.current_gesture == 7 and len(self.point_history) >= 5:
                    rotation_id = self.predict_rotation(self.point_history)

                # 控制无人机
                if self.use_airsim:
                    self.control_drone(self.current_gesture, rotation_id)
                else:
                    # 模拟控制，仅打印
                    rotation_text = "顺时针" if rotation_id == 0 else "逆时针"
                    gesture_text = self.gesture_names.get(self.current_gesture, "未知")
                    print(f"模拟控制: {gesture_text} {rotation_text if self.current_gesture == 7 else ''}")

                self.prev_gesture = self.current_gesture

            time.sleep(0.1)  # 控制频率10Hz

    def toggle_drone(self):
        """切换无人机激活状态"""
        if not self.use_airsim:
            print("AirSim不可用，无法控制无人机")
            return

        self.drone_active = not self.drone_active

        if self.drone_active:
            print("无人机已激活，准备起飞...")
            try:
                self.drone_client.takeoffAsync().join()
                print("无人机已起飞!")
            except Exception as e:
                print(f"起飞失败: {e}")
                self.drone_active = False
        else:
            print("无人机已停用，准备降落...")
            try:
                self.drone_client.landAsync().join()
                print("无人机已降落!")
            except Exception as e:
                print(f"降落失败: {e}")

    def run(self):
        """运行主循环"""
        print("=" * 50)
        print("手势控制无人机系统")
        print("=" * 50)
        print("快捷键:")
        print("  SPACE: 激活/停用无人机控制")
        print("  ESC: 退出程序")
        print("=" * 50)

        # 启动控制线程
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

        while self.cap.isOpened() and self.running:
            # 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头")
                break

            # 水平翻转图像（镜像）
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 处理图像
            results = self.hands.process(rgb_frame)

            # 重置当前手势（如果没有检测到手）
            self.current_gesture = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部关键点
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # 获取食指指尖位置（用于轨迹记录）
                    index_tip = hand_landmarks.landmark[8]
                    index_x = int(index_tip.x * self.camera_width)
                    index_y = int(index_tip.y * self.camera_height)

                    # 记录轨迹（用于旋转检测）
                    self.point_history.append((index_x, index_y))
                    if len(self.point_history) > self.max_history_length:
                        self.point_history.pop(0)

                    # 识别手势
                    self.current_gesture = self.predict_gesture(hand_landmarks.landmark)

                    # 在指尖绘制圆圈
                    cv2.circle(frame, (index_x, index_y), 8, (0, 255, 0), -1)

                    # 绘制轨迹
                    for i in range(1, len(self.point_history)):
                        cv2.line(frame, self.point_history[i - 1], self.point_history[i],
                                 (0, 255, 0), 2)

            # 显示手势信息
            gesture_text = self.gesture_names.get(self.current_gesture, "未知")
            status_color = (0, 255, 0) if self.drone_active else (0, 0, 255)
            status_text = "已连接" if self.use_airsim and self.drone_active else "未连接"

            cv2.putText(frame, f"手势: {gesture_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"状态: {status_text}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # 显示控制提示
            cv2.putText(frame, "SPACE: 切换控制  ESC: 退出", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 显示图像
            cv2.imshow('Gesture Control Drone', frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                self.toggle_drone()

        # 清理
        self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        self.running = False

        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        if self.use_airsim and self.drone_client:
            try:
                if self.drone_active:
                    self.drone_client.landAsync().join()
                self.drone_client.armDisarm(False)
                self.drone_client.enableApiControl(False)
                print("AirSim连接已关闭")
            except:
                pass

        print("程序已退出")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='手势控制无人机')
    parser.add_argument('--no-airsim', action='store_true', help='不使用AirSim（模拟模式）')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')

    args = parser.parse_args()

    # 创建控制器
    controller = GestureController(
        use_airsim=not args.no_airsim,
        camera_index=args.camera
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        controller.cleanup()
    except Exception as e:
        print(f"程序运行错误: {e}")
        controller.cleanup()


if __name__ == "__main__":
    main()