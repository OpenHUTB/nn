"""
手势轨迹记录器模块
负责录制、保存、加载和回放手势轨迹
作者: xiaoshiyuan888
"""

import os
import time
import pickle
import cv2
from datetime import datetime


class GestureTrajectoryRecorder:
    """手势轨迹记录器 - 记录、保存、加载和回放手势轨迹"""

    def __init__(self, speech_manager=None, config=None):
        self.speech_manager = speech_manager
        self.config = config
        self.trajectory_data = []
        self.is_recording = False
        self.is_playing = False
        self.playback_index = 0
        self.playback_paused = False
        self.max_trajectory_points = 1000
        self.recording_start_time = 0
        self.last_save_time = 0
        self.save_interval = 5

        # 轨迹文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.trajectory_dir = os.path.join(current_dir, 'trajectories')
        if not os.path.exists(self.trajectory_dir):
            os.makedirs(self.trajectory_dir)

        # 默认轨迹文件名
        self.default_filename = os.path.join(self.trajectory_dir,
                                             f'trajectory_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')

        # 轨迹可视化设置
        self.trajectory_colors = [
            (255, 0, 0),  # 红色 - 起点
            (0, 255, 0),  # 绿色 - 正常点
            (0, 0, 255),  # 蓝色 - 终点
            (255, 255, 0),  # 青色 - 特殊点
            (255, 0, 255)  # 紫色 - 特殊点
        ]

        self.show_trajectory = True
        self.trajectory_thickness = 2
        self.trajectory_max_length = 100

        print("✓ 手势轨迹记录器已初始化")

    def start_recording(self):
        """开始录制手势轨迹"""
        if self.is_recording:
            return False

        self.trajectory_data = []
        self.is_recording = True
        self.recording_start_time = time.time()
        self.last_save_time = time.time()

        print("🎬 开始录制手势轨迹")

        # 语音提示
        if self.speech_manager and self.speech_manager.enabled:
            self.speech_manager.speak('recording_start', immediate=True)

        return True

    def stop_recording(self):
        """停止录制手势轨迹"""
        if not self.is_recording:
            return False

        self.is_recording = False
        recording_duration = time.time() - self.recording_start_time

        print(f"⏹️ 停止录制手势轨迹")
        print(f"   录制时长: {recording_duration:.1f}秒")
        print(f"   轨迹点数: {len(self.trajectory_data)}")

        # 语音提示
        if self.speech_manager and self.speech_manager.enabled:
            self.speech_manager.speak('recording_stop', immediate=True)
            if len(self.trajectory_data) > 0:
                self.speech_manager.speak_direct(f"录制了{len(self.trajectory_data)}个轨迹点")

        return True

    def add_trajectory_point(self, hand_data, gesture, confidence, frame_shape):
        """添加轨迹点"""
        if not self.is_recording or len(self.trajectory_data) >= self.max_trajectory_points:
            return False

        if hand_data is None:
            return False

        # 创建轨迹点数据
        trajectory_point = {
            'timestamp': time.time(),
            'hand_position': hand_data['position'] if 'position' in hand_data else (0.5, 0.5),
            'hand_center': hand_data['center'] if 'center' in hand_data else (0, 0),
            'gesture': gesture,
            'confidence': confidence,
            'fingertips': hand_data.get('fingertips', []),
            'frame_shape': frame_shape
        }

        self.trajectory_data.append(trajectory_point)

        # 自动保存检查
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval and len(self.trajectory_data) > 10:
            self.auto_save()
            self.last_save_time = current_time

        return True

    def auto_save(self):
        """自动保存轨迹（临时文件）"""
        if len(self.trajectory_data) == 0:
            return

        temp_file = os.path.join(self.trajectory_dir, 'trajectory_temp.pkl')
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(self.trajectory_data, f)
            print(f"💾 自动保存轨迹到临时文件 ({len(self.trajectory_data)}个点)")
        except Exception as e:
            print(f"⚠ 自动保存轨迹失败: {e}")

    def save_trajectory(self, filename=None):
        """保存轨迹到文件"""
        if len(self.trajectory_data) == 0:
            print("⚠ 没有轨迹数据可保存")
            return False

        if filename is None:
            filename = self.default_filename

        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.trajectory_data, f)

            print(f"💾 轨迹已保存到: {filename}")
            print(f"   轨迹点数: {len(self.trajectory_data)}")

            # 语音提示
            if self.speech_manager and self.speech_manager.enabled:
                self.speech_manager.speak('recording_saved', immediate=True)
                self.speech_manager.speak_direct(f"保存了{len(self.trajectory_data)}个轨迹点")

            return True
        except Exception as e:
            print(f"❌ 保存轨迹失败: {e}")
            return False

    def load_trajectory(self, filename):
        """从文件加载轨迹"""
        try:
            if not os.path.exists(filename):
                print(f"❌ 轨迹文件不存在: {filename}")

                # 语音提示
                if self.speech_manager and self.speech_manager.enabled:
                    self.speech_manager.speak('recording_not_found', immediate=True)

                return False

            with open(filename, 'rb') as f:
                self.trajectory_data = pickle.load(f)

            print(f"📂 轨迹已从文件加载: {filename}")
            print(f"   轨迹点数: {len(self.trajectory_data)}")

            # 语音提示
            if self.speech_manager and self.speech_manager.enabled:
                self.speech_manager.speak('recording_loaded', immediate=True)
                self.speech_manager.speak_direct(f"加载了{len(self.trajectory_data)}个轨迹点")

            return True
        except Exception as e:
            print(f"❌ 加载轨迹失败: {e}")
            return False

    def start_playback(self):
        """开始回放轨迹"""
        if len(self.trajectory_data) == 0:
            print("⚠ 没有轨迹数据可回放")

            # 语音提示
            if self.speech_manager and self.speech_manager.enabled:
                self.speech_manager.speak('recording_not_found', immediate=True)

            return False

        self.is_playing = True
        self.playback_index = 0
        self.playback_paused = False

        print(f"▶️ 开始回放手势轨迹")
        print(f"   总帧数: {len(self.trajectory_data)}")

        # 语音提示
        if self.speech_manager and self.speech_manager.enabled:
            self.speech_manager.speak('recording_playback_start', immediate=True)

        return True

    def stop_playback(self):
        """停止回放轨迹"""
        if not self.is_playing:
            return False

        self.is_playing = False
        self.playback_paused = False

        print("⏹️ 停止回放手势轨迹")

        # 语音提示
        if self.speech_manager and self.speech_manager.enabled:
            self.speech_manager.speak('recording_playback_stop', immediate=True)

        return True

    def pause_playback(self):
        """暂停/继续回放"""
        self.playback_paused = not self.playback_paused

        status = "暂停" if self.playback_paused else "继续"
        print(f"⏸️ 回放已{status}")

        # 语音提示
        if self.speech_manager and self.speech_manager.enabled:
            if self.playback_paused:
                self.speech_manager.speak('recording_paused', immediate=True)
            else:
                self.speech_manager.speak('recording_resumed', immediate=True)

        return self.playback_paused

    def get_next_playback_point(self):
        """获取下一个回放点"""
        if not self.is_playing or self.playback_paused or len(self.trajectory_data) == 0:
            return None

        if self.playback_index >= len(self.trajectory_data):
            self.stop_playback()
            return None

        point = self.trajectory_data[self.playback_index]
        self.playback_index += 1

        # 检查是否到达末尾
        if self.playback_index >= len(self.trajectory_data):
            self.stop_playback()

        return point

    def clear_trajectory(self):
        """清除轨迹数据"""
        self.trajectory_data = []
        self.is_recording = False
        self.is_playing = False
        self.playback_index = 0

        print("🗑️ 轨迹数据已清除")

        # 语音提示
        if self.speech_manager and self.speech_manager.enabled:
            self.speech_manager.speak('recording_cleared', immediate=True)

        return True

    def draw_trajectory(self, frame):
        """在帧上绘制轨迹"""
        if not self.show_trajectory or len(self.trajectory_data) == 0:
            return frame

        h, w = frame.shape[:2]

        # 限制显示的轨迹点数
        display_points = min(len(self.trajectory_data), self.trajectory_max_length)
        start_idx = max(0, len(self.trajectory_data) - display_points)

        # 绘制轨迹线
        for i in range(start_idx, len(self.trajectory_data) - 1):
            point1 = self.trajectory_data[i]
            point2 = self.trajectory_data[i + 1]

            # 获取手部中心位置
            if 'hand_center' in point1 and 'hand_center' in point2:
                x1, y1 = point1['hand_center']
                x2, y2 = point2['hand_center']

                # 确保坐标在图像范围内
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                # 根据索引计算颜色
                color_idx = int((i - start_idx) / display_points * (len(self.trajectory_colors) - 1))
                color = self.trajectory_colors[color_idx]

                # 绘制线条
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         color, self.trajectory_thickness)

        # 绘制当前点
        if self.is_recording or self.is_playing:
            current_idx = len(self.trajectory_data) - 1 if self.is_recording else self.playback_index - 1
            if 0 <= current_idx < len(self.trajectory_data):
                point = self.trajectory_data[current_idx]
                if 'hand_center' in point:
                    x, y = point['hand_center']
                    x = max(0, min(w - 1, x))
                    y = max(0, min(h - 1, y))

                    # 绘制当前点
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 0), 2)

        return frame

    def get_status(self):
        """获取录制状态"""
        return {
            'is_recording': self.is_recording,
            'is_playing': self.is_playing,
            'playback_paused': self.playback_paused,
            'trajectory_points': len(self.trajectory_data),
            'playback_index': self.playback_index,
            'playback_total': len(self.trajectory_data),
            'recording_duration': time.time() - self.recording_start_time if self.is_recording else 0
        }

    def list_saved_trajectories(self):
        """列出保存的轨迹文件"""
        try:
            files = [f for f in os.listdir(self.trajectory_dir) if f.endswith('.pkl')]
            return sorted(files, reverse=True)
        except:
            return []