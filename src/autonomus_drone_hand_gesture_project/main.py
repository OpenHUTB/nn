# -*- coding: utf-8 -*-
"""
手势控制AirSim无人机 - 性能模式选择优化版
新增fast（最快）、balanced（平衡）、accurate（最准）三种性能模式
作者: xiaoshiyuan888
"""

import sys
import os
import time
import traceback
import json
import math
import threading
import tempfile
import pickle  # 新增：用于数据序列化
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from collections import deque, Counter

print("=" * 60)
print("Gesture Controlled Drone - Performance Mode Selection")
print("性能模式选择优化版!")
print("=" * 60)

# ========== 修复导入路径 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# ========== 核心模块导入 ==========
def safe_import():
    """安全导入所有模块"""
    modules_status = {}

    try:
        from PIL import Image, ImageDraw, ImageFont
        modules_status['PIL'] = True
        print("[PIL] ✓ 图像处理库就绪")
    except Exception as e:
        modules_status['PIL'] = False
        print(f"[PIL] ✗ 导入失败: {e}")
        return None, modules_status

    try:
        import cv2
        import numpy as np
        modules_status['OpenCV'] = True
        print("[OpenCV] ✓ 计算机视觉库就绪")
    except Exception as e:
        modules_status['OpenCV'] = False
        print(f"[OpenCV] ✗ 导入失败: {e}")
        return None, modules_status

    airsim_module = None
    try:
        airsim_module = __import__('airsim')
        modules_status['AirSim'] = True
        print(f"[AirSim] ✓ 成功导入")
    except ImportError:
        print("\n" + "!" * 60)
        print("⚠ AirSim库未找到!")
        print("!" * 60)
        print("安装AirSim:")
        print("1. 首先安装: pip install msgpack-rpc-python")
        print("2. 然后安装: pip install airsim")
        print("\n或从源码安装:")
        print("  pip install git+https://github.com/microsoft/AirSim.git")
        print("!" * 60)

        print("\n无AirSim继续运行? (y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            sys.exit(1)

    # 尝试导入语音合成库
    speech_module = None
    try:
        # 尝试导入pyttsx3（离线TTS）
        import pyttsx3
        speech_module = pyttsx3
        modules_status['Speech'] = True
        print("[Speech] ✓ pyttsx3语音库就绪 (离线)")
    except ImportError:
        print("\n" + "!" * 60)
        print("⚠ pyttsx3语音库未找到!")
        print("!" * 60)
        print("安装语音库 (使用清华大学镜像源):")
        print("1. 安装离线语音库: pip install pyttsx3 -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("2. 或者安装在线语音库: pip install gtts pygame -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("!" * 60)

        # 尝试其他语音库
        try:
            # 尝试使用gTTS（需要网络）
            from gtts import gTTS
            speech_module = {'gTTS': gTTS, 'type': 'gtts'}
            modules_status['Speech'] = True
            print("[Speech] ✓ gTTS语音库就绪 (需要网络连接)")

            # 尝试导入音频播放库
            try:
                import pygame
                pygame.mixer.init()
                speech_module['pygame'] = pygame
                print("[Speech] ✓ pygame音频播放库就绪")
            except ImportError:
                # 尝试其他播放方式
                try:
                    import pydub
                    from pydub import AudioSegment
                    from pydub.playback import play
                    speech_module['pydub'] = pydub
                    speech_module['AudioSegment'] = AudioSegment
                    speech_module['play'] = play
                    print("[Speech] ✓ pydub音频播放库就绪")
                except ImportError:
                    # 最后尝试使用系统命令
                    if os.name == 'nt':  # Windows
                        speech_module['play_method'] = 'windows'
                        print("[Speech] ✓ 使用Windows系统命令播放音频")
                    elif os.name == 'posix':  # Linux/Mac
                        speech_module['play_method'] = 'posix'
                        print("[Speech] ✓ 使用系统命令播放音频")
                    else:
                        print("[Speech] ✗ 所有音频播放库导入失败，语音功能将不可用")
                        speech_module = None
                        modules_status['Speech'] = False

        except ImportError:
            print("[Speech] ✗ 所有语音库导入失败，语音功能将不可用")
            speech_module = None
            modules_status['Speech'] = False

    return {
        'cv2': cv2,
        'np': np,
        'PIL': {'Image': Image, 'ImageDraw': ImageDraw, 'ImageFont': ImageFont},
        'airsim': airsim_module,
        'speech': speech_module
    }, modules_status


# 执行导入
libs, status = safe_import()
if not status.get('OpenCV', False) or not status.get('PIL', False):
    print("\n❌ 核心库缺失，无法启动。")
    input("按回车键退出...")
    sys.exit(1)

print("-" * 60)
print("✅ 环境检查通过，正在初始化...")
print("-" * 60)

# 解包库
cv2, np = libs['cv2'], libs['np']
Image, ImageDraw, ImageFont = libs['PIL']['Image'], libs['PIL']['ImageDraw'], libs['PIL']['ImageFont']


# ========== 增强语音反馈管理器 ==========
class EnhancedSpeechFeedbackManager:
    """增强的语音反馈管理器"""

    def __init__(self, speech_lib):
        self.speech_lib = speech_lib
        self.enabled = True
        self.volume = 1.0
        self.rate = 150
        self.voice_id = None
        self.last_speech_time = {}
        self.min_interval = 1.5  # 缩短相同语音的最小间隔（秒）

        # 语音队列，避免语音重叠
        self.speech_queue = []
        self.is_speaking = False
        self.queue_thread = None

        # 音频播放方法
        self.audio_method = None

        # 新增：手势状态追踪
        self.last_gesture_state = "none"  # 记录上次手势状态
        self.gesture_active_time = 0  # 手势持续活跃时间

        # 增强的语音消息映射
        self.messages = {
            # 连接相关
            'connecting': "正在连接无人机，请稍候",
            'connected': "无人机连接成功",
            'connection_failed': "无人机连接失败，进入模拟模式",

            # 飞行相关
            'taking_off': "无人机正在起飞",
            'takeoff_success': "起飞成功",
            'takeoff_failed': "起飞失败",
            'landing': "无人机正在降落",
            'land_success': "降落成功",
            'emergency_stop': "紧急停止，无人机已降落",
            'hovering': "无人机悬停中",

            # 手势相关 - 增强
            'gesture_detected': "手势识别就绪，请开始手势",
            'gesture_start': "开始识别手势",
            'gesture_end': "手势识别结束",
            'gesture_stop': "停止",
            'gesture_up': "向上",
            'gesture_down': "向下",
            'gesture_left': "向左",
            'gesture_right': "向右",
            'gesture_forward': "向前",
            'gesture_backward': "向后",
            'gesture_waiting': "等待手势",
            'gesture_error': "手势识别错误",
            'gesture_stable': "手势稳定",
            'gesture_change': "手势变化",
            'gesture_low_confidence': "手势识别置信度低",
            'gesture_good_confidence': "手势识别置信度高",
            'gesture_hover': "悬停",

            # 系统相关
            'program_start': "手势控制无人机系统已启动",
            'program_exit': "程序退出，感谢使用",
            'camera_error': "摄像头错误，请检查连接",
            'camera_ready': "摄像头就绪",
            'system_ready': "系统准备就绪",

            # 模式相关
            'simulation_mode': "进入模拟模式",
            'debug_mode_on': "调试模式已开启",
            'debug_mode_off': "调试模式已关闭",
            'display_mode_changed': "显示模式已切换",
            'help_toggled': "帮助信息已切换",
            'performance_mode_fast': "切换到最快性能模式",
            'performance_mode_balanced': "切换到平衡性能模式",
            'performance_mode_accurate': "切换到最准确性能模式",

            # 新增：性能相关
            'performance_good': "系统运行流畅",
            'performance_warning': "系统性能警告",

            # 新增：手势指导
            'move_closer': "请将手靠近摄像头",
            'move_away': "请将手移远一些",
            'good_position': "手部位置良好",
            'hand_detected': "手部已检测到",
            'hand_lost': "手部丢失，请重新放置",

            # 新增：录制相关
            'recording_start': "开始录制手势轨迹",
            'recording_stop': "停止录制",
            'recording_saved': "轨迹已保存",
            'recording_loaded': "轨迹已加载",
            'recording_playback_start': "开始回放手势轨迹",
            'recording_playback_stop': "回放结束",
            'recording_cleared': "轨迹已清除",
            'recording_paused': "回放已暂停",
            'recording_resumed': "回放继续",
            'recording_not_found': "未找到轨迹数据",
            'recording_frame_count': "轨迹帧数",
        }

        # 初始化语音引擎
        self.init_speech_engine()

    def init_speech_engine(self):
        """初始化语音引擎"""
        if self.speech_lib is None:
            print("⚠ 语音库未找到，语音功能禁用")
            self.enabled = False
            return

        try:
            if hasattr(self.speech_lib, 'init'):  # pyttsx3
                self.engine = self.speech_lib.init()
                self.audio_method = 'pyttsx3'

                # 设置语音参数
                voices = self.engine.getProperty('voices')

                # 尝试寻找中文语音
                for voice in voices:
                    # 检查语音名称是否包含中文相关标识
                    if 'chinese' in voice.name.lower() or 'zh' in voice.name.lower() or 'zh_CN' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        self.voice_id = voice.id
                        print(f"[Speech] 使用中文语音: {voice.name}")
                        break

                # 如果没找到中文语音，使用第一个可用语音
                if self.voice_id is None and len(voices) > 0:
                    self.engine.setProperty('voice', voices[0].id)
                    print(f"[Speech] 使用默认语音: {voices[0].name}")

                # 设置语速和音量
                self.engine.setProperty('rate', self.rate)
                self.engine.setProperty('volume', self.volume)

                print("✅ 语音引擎初始化成功 (pyttsx3)")

            elif isinstance(self.speech_lib, dict) and self.speech_lib.get('type') == 'gtts':
                print("✅ 语音引擎初始化成功 (gTTS，需要网络连接)")
                self.audio_method = 'gtts'

                # 确定播放方法
                if 'pygame' in self.speech_lib:
                    self.audio_method = 'gtts_pygame'
                    print("✅ 使用pygame播放音频")
                elif 'pydub' in self.speech_lib:
                    self.audio_method = 'gtts_pydub'
                    print("✅ 使用pydub播放音频")
                elif 'play_method' in self.speech_lib:
                    self.audio_method = f"gtts_{self.speech_lib['play_method']}"
                    print(f"✅ 使用系统命令播放音频")
                else:
                    self.audio_method = 'gtts_system'
                    print("✅ 使用默认系统播放器")

            else:
                print("⚠ 未知语音库类型，语音功能可能不正常")
                self.enabled = False

        except Exception as e:
            print(f"⚠ 语音引擎初始化失败: {e}")
            self.enabled = False

    def play_audio_file(self, audio_file):
        """播放音频文件（根据可用库选择方法）"""
        try:
            if self.audio_method == 'gtts_pygame' and 'pygame' in self.speech_lib:
                pygame = self.speech_lib['pygame']
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()

                # 等待播放完成
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

            elif self.audio_method == 'gtts_pydub' and 'pydub' in self.speech_lib:
                AudioSegment = self.speech_lib['AudioSegment']
                play = self.speech_lib['play']

                audio = AudioSegment.from_mp3(audio_file)
                play(audio)

            elif self.audio_method == 'gtts_windows':
                # Windows系统命令
                os.startfile(audio_file)
                # 等待播放完成（简单等待）
                time.sleep(1.5)

            elif self.audio_method == 'gtts_posix':
                # Linux/Mac系统命令
                import subprocess
                if sys.platform == 'darwin':  # macOS
                    subprocess.call(['afplay', audio_file])
                else:  # Linux
                    subprocess.call(['xdg-open', audio_file])

            else:
                # 通用方法：使用系统默认播放器
                import subprocess
                if sys.platform == 'win32':
                    os.startfile(audio_file)
                elif sys.platform == 'darwin':
                    subprocess.call(['open', audio_file])
                else:
                    subprocess.call(['xdg-open', audio_file])

            return True

        except Exception as e:
            print(f"⚠ 音频播放失败: {e}")
            return False

    def speak(self, message_key, force=False, immediate=False):
        """播放语音"""
        if not self.enabled:
            return

        # 检查是否在最小间隔内
        current_time = time.time()
        if not force and message_key in self.last_speech_time:
            if current_time - self.last_speech_time[message_key] < self.min_interval:
                return

        # 获取消息文本
        if message_key in self.messages:
            text = self.messages[message_key]
        else:
            text = message_key  # 直接使用传入的文本

        # 如果是立即播放，直接在新线程中播放
        if immediate:
            self.speak_direct(text)
        else:
            # 添加到语音队列
            self.speech_queue.append(text)

            # 如果没有在播放，启动播放线程
            if not self.is_speaking and self.queue_thread is None:
                self.queue_thread = threading.Thread(target=self._process_speech_queue)
                self.queue_thread.daemon = True
                self.queue_thread.start()

        # 更新时间戳
        self.last_speech_time[message_key] = current_time

    def _process_speech_queue(self):
        """处理语音队列"""
        while self.speech_queue and self.enabled:
            self.is_speaking = True

            text = self.speech_queue.pop(0)

            try:
                if self.audio_method == 'pyttsx3':
                    # 清除之前的语音
                    self.engine.stop()
                    # 播报新语音
                    self.engine.say(text)
                    self.engine.runAndWait()

                elif self.audio_method.startswith('gtts'):
                    # gTTS需要网络连接
                    tts = self.speech_lib['gTTS'](text=text, lang='zh-cn')

                    # 保存临时文件
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                        temp_file = f.name
                        tts.save(temp_file)

                    # 播放音频
                    self.play_audio_file(temp_file)

                    # 删除临时文件
                    try:
                        os.unlink(temp_file)
                    except:
                        pass  # 忽略删除错误

            except Exception as e:
                print(f"⚠ 语音播放失败: {e}")

            time.sleep(0.05)  # 减少等待时间，避免卡顿

        self.is_speaking = False
        self.queue_thread = None

    def speak_direct(self, text):
        """直接播放文本（不通过消息映射）"""
        if not self.enabled:
            return

        # 在新线程中播放
        thread = threading.Thread(target=self._speak_thread, args=(text,))
        thread.daemon = True
        thread.start()

    def _speak_thread(self, text):
        """语音播放线程"""
        try:
            if self.audio_method == 'pyttsx3':
                self.engine.say(text)
                self.engine.runAndWait()

            elif self.audio_method.startswith('gtts'):
                tts = self.speech_lib['gTTS'](text=text, lang='zh-cn')

                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    temp_file = f.name
                    tts.save(temp_file)

                self.play_audio_file(temp_file)

                try:
                    os.unlink(temp_file)
                except:
                    pass

        except Exception as e:
            print(f"⚠ 直接语音播放失败: {e}")

    def stop(self):
        """停止所有语音"""
        if hasattr(self, 'engine'):
            self.engine.stop()

        self.speech_queue.clear()
        self.is_speaking = False

    def set_enabled(self, enabled):
        """启用/禁用语音"""
        self.enabled = enabled
        if not enabled:
            self.stop()

    def toggle_enabled(self):
        """切换语音启用状态"""
        self.enabled = not self.enabled
        status = "启用" if self.enabled else "禁用"
        self.speak_direct(f"语音反馈已{status}")
        return self.enabled

    def get_status(self):
        """获取语音状态"""
        return {
            'enabled': self.enabled,
            'engine': 'pyttsx3' if self.audio_method == 'pyttsx3' else
            'gTTS' if self.audio_method.startswith('gtts') else
            'None',
            'queue_size': len(self.speech_queue),
            'is_speaking': self.is_speaking,
            'audio_method': self.audio_method
        }


# ========== 手势轨迹记录器 ==========
class GestureTrajectoryRecorder:
    """手势轨迹记录器 - 记录、保存、加载和回放手势轨迹"""

    def __init__(self, speech_manager=None):
        self.speech_manager = speech_manager
        self.trajectory_data = []
        self.is_recording = False
        self.is_playing = False
        self.playback_index = 0
        self.playback_paused = False
        self.max_trajectory_points = 1000  # 最大轨迹点数
        self.recording_start_time = 0
        self.last_save_time = 0
        self.save_interval = 5  # 自动保存间隔（秒）

        # 轨迹文件路径
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
        self.trajectory_max_length = 100  # 显示的最大轨迹长度

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

            # 获取手部中心位置（转换为图像坐标）
            if 'hand_center' in point1 and 'hand_center' in point2:
                x1, y1 = point1['hand_center']
                x2, y2 = point2['hand_center']

                # 确保坐标在图像范围内
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                # 根据索引计算颜色（渐变色）
                color_idx = int((i - start_idx) / display_points * (len(self.trajectory_colors) - 1))
                color = self.trajectory_colors[color_idx]

                # 绘制线条
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         color, self.trajectory_thickness)

        # 绘制当前点（如果正在录制或回放）
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


# ========== 配置管理器 ==========
class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self.config_file = os.path.join(current_dir, 'gesture_config.json')

        # 性能模式配置
        self.performance_modes = {
            'fast': {
                'name': '最快',
                'description': '性能优先，降低识别精度换取更高帧率',
                'detection_interval': 2,  # 每2帧检测一次
                'smooth_frames': 3,  # 平滑帧数
                'min_confidence': 0.5,  # 最小置信度阈值
                'resize_factor': 0.5,  # 图像缩放因子
                'skin_detection_enabled': True,
                'background_subtraction_enabled': False,
                'contour_simplify_epsilon': 0.03,
                'history_size': 10,
                'gesture_stability_threshold': 3,
                'color': (0, 255, 0),  # 绿色
            },
            'balanced': {
                'name': '平衡',
                'description': '平衡性能与精度，适用于大多数场景',
                'detection_interval': 1,  # 每帧检测
                'smooth_frames': 5,  # 平滑帧数
                'min_confidence': 0.6,  # 最小置信度阈值
                'resize_factor': 0.75,  # 图像缩放因子
                'skin_detection_enabled': True,
                'background_subtraction_enabled': True,
                'contour_simplify_epsilon': 0.02,
                'history_size': 15,
                'gesture_stability_threshold': 5,
                'color': (255, 165, 0),  # 橙色
            },
            'accurate': {
                'name': '最准',
                'description': '精度优先，提供最准确的手势识别',
                'detection_interval': 1,  # 每帧检测
                'smooth_frames': 7,  # 平滑帧数
                'min_confidence': 0.7,  # 最小置信度阈值
                'resize_factor': 1.0,  # 不缩放
                'skin_detection_enabled': True,
                'background_subtraction_enabled': True,
                'contour_simplify_epsilon': 0.01,
                'history_size': 20,
                'gesture_stability_threshold': 7,
                'color': (255, 0, 0),  # 红色
            }
        }

        self.default_config = {
            'camera': {
                'index': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'gesture': {
                'skin_lower_h': 0,
                'skin_upper_h': 25,
                'skin_lower_s': 30,
                'skin_upper_s': 255,
                'skin_lower_v': 60,
                'skin_upper_v': 255,
                'min_hand_area': 2000,
                'max_hand_area': 30000,
                'hand_ratio_threshold': 1.5,
                'defect_distance_threshold': 20,
                'palm_circle_radius_ratio': 0.3,
                'transition_threshold': 0.3,
                'position_stability_weight': 0.4,
                'gesture_cooldown': 0.5,
            },
            'drone': {
                'velocity': 2.5,
                'duration': 0.3,
                'altitude': -10.0,
                'control_interval': 0.3
            },
            'display': {
                'show_fps': True,
                'show_confidence': True,
                'show_help': True,
                'show_contours': True,
                'show_bbox': True,
                'show_fingertips': True,
                'show_palm_center': True,
                'show_hand_direction': True,
                'show_debug_info': False,
                'show_speech_status': True,
                'show_gesture_history': True,
                'show_stability_indicator': True,
                'show_trajectory': True,
                'show_recording_status': True,
                'show_performance_mode': True,  # 新增：显示性能模式
            },
            'performance': {
                'target_fps': 30,
                'resize_factor': 1.0,
                'enable_multiprocessing': False,
                'mode': 'balanced',  # 新增：性能模式选择
                'current_mode_index': 1,  # 当前模式索引
                'modes': ['fast', 'balanced', 'accurate'],  # 可用模式列表
            },
            'calibration': {
                'auto_calibrate_skin': True,
                'skin_calibration_frames': 30,
                'hand_size_calibration': True
            },
            'speech': {
                'enabled': True,
                'volume': 1.0,
                'rate': 150,
                'announce_gestures': True,
                'announce_connections': True,
                'announce_flight_events': True,
                'announce_gesture_changes': True,
                'announce_hand_status': True,
                'announce_performance': True,
                'announce_recording_events': True,
                'announce_performance_mode': True,  # 新增：播报性能模式切换
                'min_gesture_confidence': 0.7,
                'gesture_start_threshold': 3,
                'gesture_end_threshold': 10,
            },
            'recording': {
                'auto_save_interval': 5,
                'max_trajectory_points': 1000,
                'show_trajectory': True,
                'trajectory_thickness': 2,
                'trajectory_max_length': 100,
                'default_save_dir': 'trajectories',
            }
        }
        self.config = self.load_config()
        self.skin_calibration_data = []
        self.hand_size_calibration_done = False
        self.reference_hand_size = 0

    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    config = self.default_config.copy()
                    self._merge_config(config, loaded_config)
                    print("✓ 从文件加载配置")
                    return config
            except Exception as e:
                print(f"⚠ 加载配置失败: {e}, 使用默认配置")
                return self.default_config.copy()
        else:
            print("✓ 使用默认配置")
            return self.default_config.copy()

    def _merge_config(self, base, update):
        """递归合并配置"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print("✓ 配置已保存")
        except Exception as e:
            print(f"⚠ 保存配置失败: {e}")

    def get(self, *keys):
        """获取配置值"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def set(self, *keys, value):
        """设置配置值"""
        if len(keys) == 0:
            return

        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        self.save_config()

    def get_performance_mode_config(self, mode=None):
        """获取性能模式配置"""
        if mode is None:
            mode = self.get('performance', 'mode')

        if mode in self.performance_modes:
            return self.performance_modes[mode]
        else:
            return self.performance_modes['balanced']

    def get_current_performance_mode(self):
        """获取当前性能模式"""
        mode = self.get('performance', 'mode')
        if mode in self.performance_modes:
            return mode
        return 'balanced'

    def set_performance_mode(self, mode):
        """设置性能模式"""
        if mode in self.performance_modes:
            self.set('performance', 'mode', value=mode)

            # 更新当前模式索引
            modes = self.get('performance', 'modes')
            if modes and mode in modes:
                index = modes.index(mode)
                self.set('performance', 'current_mode_index', value=index)

            print(f"✓ 性能模式设置为: {self.performance_modes[mode]['name']}")
            return True
        return False

    def cycle_performance_mode(self):
        """循环切换性能模式"""
        modes = self.get('performance', 'modes')
        if not modes:
            modes = ['fast', 'balanced', 'accurate']

        current_index = self.get('performance', 'current_mode_index')
        if current_index is None:
            current_index = 0

        # 计算下一个模式索引
        next_index = (current_index + 1) % len(modes)
        next_mode = modes[next_index]

        # 设置新模式
        self.set('performance', 'current_mode_index', value=next_index)
        return self.set_performance_mode(next_mode)

    def calibrate_skin_color(self, frame, hand_mask):
        """自动校准肤色范围"""
        if not self.get('calibration', 'auto_calibrate_skin'):
            return

        if len(self.skin_calibration_data) < self.get('calibration', 'skin_calibration_frames'):
            # 转换到HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 获取肤色区域的HSV值
            skin_pixels = hsv[hand_mask > 0]

            if len(skin_pixels) > 100:  # 确保有足够的像素
                self.skin_calibration_data.append(skin_pixels)

        if len(self.skin_calibration_data) == self.get('calibration', 'skin_calibration_frames'):
            # 计算肤色范围
            all_skin_pixels = np.vstack(self.skin_calibration_data)

            h_min, h_max = np.percentile(all_skin_pixels[:, 0], [2, 98])
            s_min, s_max = np.percentile(all_skin_pixels[:, 1], [2, 98])
            v_min, v_max = np.percentile(all_skin_pixels[:, 2], [2, 98])

            # 更新配置
            self.set('gesture', 'skin_lower_h', value=int(max(0, h_min - 5)))
            self.set('gesture', 'skin_upper_h', value=int(min(180, h_max + 5)))
            self.set('gesture', 'skin_lower_s', value=int(max(0, s_min - 10)))
            self.set('gesture', 'skin_upper_s', value=int(min(255, s_max + 10)))
            self.set('gesture', 'skin_lower_v', value=int(max(0, v_min - 10)))
            self.set('gesture', 'skin_upper_v', value=int(min(255, v_max + 10)))

            print("✓ 肤色校准完成")
            print(f"  肤色范围: H[{self.get('gesture', 'skin_lower_h')}-{self.get('gesture', 'skin_upper_h')}], "
                  f"S[{self.get('gesture', 'skin_lower_s')}-{self.get('gesture', 'skin_upper_s')}], "
                  f"V[{self.get('gesture', 'skin_lower_v')}-{self.get('gesture', 'skin_upper_v')}]")

    def calibrate_hand_size(self, hand_area):
        """校准手部大小"""
        if not self.get('calibration', 'hand_size_calibration') or self.hand_size_calibration_done:
            return

        if hand_area > 0:
            self.reference_hand_size = hand_area
            self.hand_size_calibration_done = True
            print(f"✓ 手部大小校准完成: {self.reference_hand_size:.0f} 像素")


config = ConfigManager()


# ========== 改进的手势识别器（支持性能模式） ==========
class EnhancedGestureRecognizer:
    """增强的手势识别器 - 支持性能模式选择"""

    def __init__(self, speech_manager=None):
        self.speech_manager = speech_manager

        # 加载性能模式配置
        self.performance_mode = config.get_current_performance_mode()
        self.mode_config = config.get_performance_mode_config(self.performance_mode)

        # 根据性能模式初始化参数
        self.history_size = self.mode_config['history_size']
        self.detection_interval = self.mode_config['detection_interval']
        self.smooth_frames = self.mode_config['smooth_frames']
        self.min_confidence = self.mode_config['min_confidence']
        self.resize_factor = self.mode_config['resize_factor']

        # 增强的手势历史和平滑
        self.gesture_history = deque(maxlen=self.history_size)
        self.confidence_history = deque(maxlen=self.history_size)
        self.position_history = deque(maxlen=self.history_size)
        self.current_gesture = "Waiting"
        self.current_confidence = 0.0

        # 新增：手势状态追踪
        self.gesture_state = "none"
        self.gesture_stability_counter = 0
        self.last_stable_gesture = "Waiting"
        self.gesture_active_frames = 0
        self.last_gesture_change_time = 0

        # 记录上次播报的手势
        self.last_announced_gesture = None
        self.last_announced_time = 0
        self.last_hand_status_time = 0
        self.gesture_announce_interval = 2.0

        # 手部跟踪和状态
        self.last_hand_position = None
        self.hand_tracking = False
        self.track_window = None
        self.hand_states = deque(maxlen=15)
        self.hand_detected_frames = 0
        self.hand_lost_frames = 0

        # 性能统计
        self.process_times = deque(maxlen=30)
        self.frame_counter = 0
        self.last_performance_report = 0

        # 手势颜色映射
        self.gesture_colors = {
            "Stop": (0, 0, 255),
            "Forward": (0, 255, 0),
            "Up": (255, 255, 0),
            "Down": (255, 0, 255),
            "Left": (255, 165, 0),
            "Right": (0, 165, 255),
            "Waiting": (200, 200, 200),
            "Error": (255, 0, 0),
            "Hover": (255, 255, 255)
        }

        # 手势到语音的映射
        self.gesture_speech_map = {
            "Stop": "gesture_stop",
            "Forward": "gesture_forward",
            "Up": "gesture_up",
            "Down": "gesture_down",
            "Left": "gesture_left",
            "Right": "gesture_right",
            "Waiting": "gesture_waiting",
            "Error": "gesture_error",
            "Hover": "gesture_hover",
        }

        # 手势状态颜色
        self.state_colors = {
            "none": (100, 100, 100),
            "starting": (255, 165, 0),
            "active": (0, 255, 0),
            "ending": (255, 0, 0),
        }

        # 根据性能模式初始化背景减除器
        self.bg_subtractor = None
        if self.mode_config['background_subtraction_enabled']:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=25, detectShadows=True
            )

        # 形态学操作核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 性能监控
        self.avg_process_time = 0
        self.frame_rate = 0
        self.last_fps_check = time.time()

        # 存储手部数据用于轨迹记录
        self.last_hand_data = None

        # 性能模式信息
        self.performance_mode_color = self.mode_config['color']
        self.performance_mode_name = self.mode_config['name']

        print(f"✓ 增强的手势识别器已初始化 - 性能模式: {self.performance_mode_name}")

    def set_performance_mode(self, mode):
        """设置性能模式"""
        self.performance_mode = mode
        self.mode_config = config.get_performance_mode_config(mode)

        # 更新参数
        self.history_size = self.mode_config['history_size']
        self.detection_interval = self.mode_config['detection_interval']
        self.smooth_frames = self.mode_config['smooth_frames']
        self.min_confidence = self.mode_config['min_confidence']
        self.resize_factor = self.mode_config['resize_factor']

        # 更新背景减除器
        if self.mode_config['background_subtraction_enabled'] and self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=25, detectShadows=True
            )
        elif not self.mode_config['background_subtraction_enabled']:
            self.bg_subtractor = None

        # 更新队列大小
        self.gesture_history = deque(maxlen=self.history_size)
        self.confidence_history = deque(maxlen=self.history_size)
        self.position_history = deque(maxlen=self.history_size)

        # 更新显示信息
        self.performance_mode_color = self.mode_config['color']
        self.performance_mode_name = self.mode_config['name']

        print(f"✓ 切换到性能模式: {self.performance_mode_name}")

    def get_skin_mask(self, frame):
        """获取肤色掩码"""
        h_low = config.get('gesture', 'skin_lower_h')
        h_high = config.get('gesture', 'skin_upper_h')
        s_low = config.get('gesture', 'skin_lower_s')
        s_high = config.get('gesture', 'skin_upper_s')
        v_low = config.get('gesture', 'skin_lower_v')
        v_high = config.get('gesture', 'skin_upper_v')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper_skin = np.array([h_high, s_high, v_high], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        return skin_mask, hsv

    def enhance_skin_detection(self, frame, skin_mask):
        """增强肤色检测"""
        if not self.mode_config['skin_detection_enabled']:
            return skin_mask

        if self.bg_subtractor is not None:
            fg_mask = self.bg_subtractor.apply(frame)
            combined_mask = cv2.bitwise_and(skin_mask, fg_mask)
        else:
            combined_mask = skin_mask

        # 根据性能模式决定形态学操作次数
        if self.performance_mode == 'accurate':
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
            combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        elif self.performance_mode == 'balanced':
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
            combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 0)
        # fast模式不进行额外的形态学操作

        return combined_mask

    def preprocess_frame(self, frame):
        """预处理帧（根据性能模式调整大小）"""
        if self.resize_factor != 1.0:
            new_width = int(frame.shape[1] * self.resize_factor)
            new_height = int(frame.shape[0] * self.resize_factor)
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_frame
        return frame

    def find_best_hand_contour(self, mask, frame):
        """找到最佳的手部轮廓"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        best_contour = None
        best_score = 0.0
        min_area = config.get('gesture', 'min_hand_area') * (self.resize_factor ** 2)
        max_area = config.get('gesture', 'max_hand_area') * (self.resize_factor ** 2)

        # 根据性能模式调整检查的轮廓数量
        max_contours = 3 if self.performance_mode != 'fast' else 1

        for contour in contours[:max_contours]:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            score = self.rate_contour(contour, frame.shape)

            if score > best_score:
                best_score = score
                best_contour = contour

        return best_contour, best_score

    def rate_contour(self, contour, frame_shape):
        """评估轮廓作为手部的可能性"""
        score = 0.0
        area = cv2.contourArea(contour)
        min_area = config.get('gesture', 'min_hand_area') * (self.resize_factor ** 2)
        max_area = config.get('gesture', 'max_hand_area') * (self.resize_factor ** 2)

        if min_area < area < max_area:
            area_ratio = min(area / max_area, 1.0)
            score += area_ratio * 0.3

        perimeter = cv2.arcLength(contour, True)
        if perimeter > 100:
            score += 0.2

        if area > 0:
            compactness = perimeter ** 2 / area
            if 12 < compactness < 25:
                compactness_score = 1.0 - abs(compactness - 18) / 6
                score += compactness_score * 0.3

        x, y, w, h = cv2.boundingRect(contour)
        if h > 0:
            aspect_ratio = w / h
            if 0.4 < aspect_ratio < 2.5:
                aspect_score = 1.0 - abs(aspect_ratio - 1.0) / 1.5
                score += aspect_score * 0.2

        return score

    def analyze_hand_features(self, contour, frame_shape):
        """分析手部特征"""
        if contour is None:
            return None, 0.0

        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, 0.0

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        palm_center = (cx, cy)
        palm_radius = int(w * config.get('gesture', 'palm_circle_radius_ratio'))

        # 根据性能模式调整轮廓简化程度
        epsilon = self.mode_config['contour_simplify_epsilon'] * cv2.arcLength(contour, True)
        fingers, fingertips, defects = self.analyze_fingers(contour, palm_center, palm_radius, epsilon)

        direction = self.calculate_hand_direction(contour, cx, cy)
        h_img, w_img = frame_shape[:2]
        norm_x = cx / w_img
        norm_y = cy / h_img
        confidence = self.calculate_confidence(area, len(fingers), len(contour), bbox_area)

        result = {
            'contour': contour,
            'center': (cx, cy),
            'bbox': (x, y, x + w, y + h),
            'fingers': fingers,
            'fingertips': fingertips,
            'defects': defects,
            'palm_center': palm_center,
            'palm_radius': palm_radius,
            'direction': direction,
            'area': area,
            'position': (norm_x, norm_y),
            'bbox_size': (w, h),
            'confidence': confidence
        }

        return result, confidence

    def analyze_fingers(self, contour, palm_center, palm_radius, epsilon):
        """分析手指"""
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx, returnPoints=False)

        if hull is None or len(hull) < 3:
            return [], [], []

        defects = cv2.convexityDefects(approx, hull)
        fingers = []
        fingertips = []
        defect_points = []

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])

                start_dist = np.linalg.norm(np.array(start) - np.array(palm_center))
                end_dist = np.linalg.norm(np.array(end) - np.array(palm_center))
                far_dist = np.linalg.norm(np.array(far) - np.array(palm_center))

                if (start_dist > palm_radius * 1.2 and
                        end_dist > palm_radius * 1.2 and
                        d > config.get('gesture', 'defect_distance_threshold') * 256):

                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                    if b * c != 0:
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                        angle_degrees = math.degrees(angle)

                        if angle_degrees < 90:
                            finger = {
                                'start': start,
                                'end': end,
                                'far': far,
                                'depth': d,
                                'angle': angle_degrees
                            }
                            fingers.append(finger)

                            for point in [start, end]:
                                if point not in fingertips:
                                    point_dist = np.linalg.norm(np.array(point) - np.array(palm_center))
                                    if point_dist > palm_radius * 1.5:
                                        fingertips.append(point)

                            defect_points.append((start, end, far, d))

        if len(fingertips) == 0:
            hull_points = cv2.convexHull(approx, returnPoints=True)
            if len(hull_points) > 0:
                hull_points = hull_points.reshape(-1, 2)

                for point in hull_points:
                    point_tuple = tuple(point)
                    point_dist = np.linalg.norm(point - np.array(palm_center))
                    if point_dist > palm_radius * 1.5 and point_tuple not in fingertips:
                        fingertips.append(point_tuple)

        fingertips = fingertips[:5]

        return fingers, fingertips, defect_points

    def calculate_hand_direction(self, contour, cx, cy):
        """计算手部方向"""
        if len(contour) < 5:
            return 0.0

        points = contour.reshape(-1, 2).astype(np.float32)
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean)
        direction = math.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))

        return direction

    def calculate_confidence(self, area, finger_count, contour_length, bbox_area):
        """计算手势置信度"""
        confidence = 0.5
        min_area = config.get('gesture', 'min_hand_area') * (self.resize_factor ** 2)
        max_area = config.get('gesture', 'max_hand_area') * (self.resize_factor ** 2)

        if min_area < area < max_area:
            area_norm = (area - min_area) / (max_area - min_area)
            confidence += area_norm * 0.2

        if 0 <= finger_count <= 5:
            confidence += 0.2

        if contour_length > 200:
            confidence += 0.1

        if bbox_area > 0:
            fill_ratio = area / bbox_area
            if 0.2 < fill_ratio < 0.8:
                fill_score = 1.0 - abs(fill_ratio - 0.5) / 0.3
                confidence += fill_score * 0.1

        return min(confidence, 1.0)

    def recognize_gesture_improved(self, hand_data):
        """改进的手势识别逻辑"""
        if hand_data is None:
            return "Waiting", 0.3

        finger_count = len(hand_data.get('fingers', []))
        fingertips = hand_data.get('fingertips', [])
        norm_x, norm_y = hand_data['position']
        direction = hand_data.get('direction', 0.0)
        confidence = hand_data['confidence']
        w, h = hand_data['bbox_size']
        aspect_ratio = w / h if h > 0 else 1.0

        # 根据手指数量分类
        if finger_count == 0:
            if len(fingertips) == 0:
                return "Stop", confidence * 0.9
            else:
                return "Stop", confidence * 0.7

        elif finger_count == 1:
            if len(fingertips) >= 1:
                cx, cy = hand_data['center']
                fingertip = fingertips[0]
                dx = fingertip[0] - cx
                dy = fingertip[1] - cy

                if abs(dx) > abs(dy):
                    if dx > 0:
                        return "Right", confidence * 0.8
                    else:
                        return "Left", confidence * 0.8
                else:
                    if dy < 0:
                        return "Up", confidence * 0.8
                    else:
                        return "Forward", confidence * 0.8
            return "Forward", confidence * 0.7

        elif finger_count == 2:
            return "Forward", confidence * 0.7

        elif finger_count == 3:
            if -45 <= direction <= 45:
                if direction > 0:
                    return "Right", confidence * 0.7
                else:
                    return "Left", confidence * 0.7
            else:
                if direction > 0:
                    return "Down", confidence * 0.7
                else:
                    return "Up", confidence * 0.7

        elif finger_count >= 4:
            if norm_x < 0.4:
                return "Left", confidence * 0.8
            elif norm_x > 0.6:
                return "Right", confidence * 0.8
            elif norm_y < 0.4:
                return "Up", confidence * 0.8
            elif norm_y > 0.6:
                return "Down", confidence * 0.8
            else:
                if -45 <= direction <= 45:
                    return "Forward", confidence * 0.7
                else:
                    return "Stop", confidence * 0.7

        return "Waiting", confidence * 0.5

    def smooth_gesture_enhanced(self, new_gesture, new_confidence, hand_data):
        """增强的手势平滑处理"""
        current_time = time.time()

        # 检查手势冷却时间
        if current_time - self.last_gesture_change_time < config.get('gesture', 'gesture_cooldown'):
            return self.current_gesture, self.current_confidence

        # 添加到历史
        self.gesture_history.append(new_gesture)
        self.confidence_history.append(new_confidence)

        if hand_data is not None:
            self.position_history.append(hand_data['position'])

        # 计算手势稳定性
        if len(self.gesture_history) >= 3:
            # 检查最近N个手势是否一致
            recent_gestures = list(self.gesture_history)[-3:]
            gesture_counter = Counter(recent_gestures)
            most_common_gesture, most_common_count = gesture_counter.most_common(1)[0]

            # 计算位置稳定性
            position_stability = 1.0
            if len(self.position_history) >= 2 and hand_data is not None:
                current_pos = hand_data['position']
                prev_pos = self.position_history[-2] if len(self.position_history) >= 2 else current_pos
                position_diff = math.sqrt((current_pos[0] - prev_pos[0]) ** 2 + (current_pos[1] - prev_pos[1]) ** 2)
                position_stability = max(0, 1.0 - position_diff * 5.0)

            # 增强的稳定性检查
            stability_threshold = self.mode_config['gesture_stability_threshold']
            transition_threshold = config.get('gesture', 'transition_threshold')
            position_weight = config.get('gesture', 'position_stability_weight')

            # 计算综合稳定性得分
            gesture_stability = most_common_count / 3.0
            overall_stability = gesture_stability * (1.0 - position_weight) + position_stability * position_weight

            # 手势状态转换逻辑
            if overall_stability >= transition_threshold:
                if most_common_gesture != self.last_stable_gesture:
                    self.gesture_stability_counter += 1
                else:
                    self.gesture_stability_counter = max(0, self.gesture_stability_counter - 1)

                if self.gesture_stability_counter >= stability_threshold:
                    # 手势稳定，更新当前手势
                    self.current_gesture = most_common_gesture
                    self.current_confidence = np.mean(list(self.confidence_history)[-3:])
                    self.last_stable_gesture = most_common_gesture
                    self.gesture_stability_counter = 0
                    self.last_gesture_change_time = current_time
            else:
                # 手势不稳定，重置计数器
                self.gesture_stability_counter = max(0, self.gesture_stability_counter - 2)

        return self.current_gesture, self.current_confidence

    def update_gesture_state(self, hand_data, gesture, confidence):
        """更新手势状态"""
        current_time = time.time()

        if hand_data is None:
            # 手部丢失
            self.hand_lost_frames += 1
            self.hand_detected_frames = max(0, self.hand_detected_frames - 1)

            if self.hand_lost_frames > 10 and self.gesture_state != "none":
                self.gesture_state = "none"
                if (self.speech_manager and
                        config.get('speech', 'enabled') and
                        config.get('speech', 'announce_hand_status') and
                        current_time - self.last_hand_status_time > 3.0):
                    self.speech_manager.speak('hand_lost', immediate=True)
                    self.last_hand_status_time = current_time
            return

        # 手部检测到
        self.hand_detected_frames += 1
        self.hand_lost_frames = 0

        # 检查手部大小和位置
        hand_area = hand_data['area']
        min_area = config.get('gesture', 'min_hand_area') * (self.resize_factor ** 2)
        max_area = config.get('gesture', 'max_hand_area') * (self.resize_factor ** 2)

        # 提供手部位置反馈
        if (self.speech_manager and
                config.get('speech', 'enabled') and
                config.get('speech', 'announce_hand_status') and
                current_time - self.last_hand_status_time > 5.0):

            if hand_area < min_area * 0.8:
                self.speech_manager.speak('move_closer', immediate=True)
                self.last_hand_status_time = current_time
            elif hand_area > max_area * 1.2:
                self.speech_manager.speak('move_away', immediate=True)
                self.last_hand_status_time = current_time
            elif self.hand_detected_frames == 5:  # 首次稳定检测
                self.speech_manager.speak('hand_detected', immediate=True)
                self.last_hand_status_time = current_time

        # 手势状态转换
        if self.gesture_state == "none" and gesture != "Waiting" and confidence > 0.6:
            # 手势开始
            self.gesture_state = "starting"
            self.gesture_active_frames = 0
            if (self.speech_manager and
                    config.get('speech', 'enabled') and
                    config.get('speech', 'announce_gesture_changes')):
                self.speech_manager.speak('gesture_start', immediate=True)

        elif self.gesture_state == "starting":
            self.gesture_active_frames += 1
            if self.gesture_active_frames >= config.get('speech', 'gesture_start_threshold'):
                self.gesture_state = "active"
                if (self.speech_manager and
                        config.get('speech', 'enabled') and
                        config.get('speech', 'announce_gesture_changes')):
                    self.speech_manager.speak('gesture_stable', immediate=True)

        elif self.gesture_state == "active":
            if gesture == "Waiting" or confidence < 0.5:
                self.gesture_active_frames = max(0, self.gesture_active_frames - 2)
                if self.gesture_active_frames <= 0:
                    self.gesture_state = "ending"
            else:
                self.gesture_active_frames = min(20, self.gesture_active_frames + 1)

        elif self.gesture_state == "ending":
            self.gesture_active_frames -= 1
            if self.gesture_active_frames <= 0:
                self.gesture_state = "none"
                if (self.speech_manager and
                        config.get('speech', 'enabled') and
                        config.get('speech', 'announce_gesture_changes')):
                    self.speech_manager.speak('gesture_end', immediate=True)

    def visualize_detection(self, frame, hand_data, gesture, confidence):
        """可视化检测结果"""
        if hand_data is None:
            return frame

        show_contours = config.get('display', 'show_contours')
        show_bbox = config.get('display', 'show_bbox')
        show_fingertips = config.get('display', 'show_fingertips')
        show_palm_center = config.get('display', 'show_palm_center')
        show_hand_direction = config.get('display', 'show_hand_direction')
        show_debug_info = config.get('display', 'show_debug_info')
        show_gesture_history = config.get('display', 'show_gesture_history')
        show_stability_indicator = config.get('display', 'show_stability_indicator')

        # 绘制轮廓
        if show_contours and 'contour' in hand_data:
            cv2.drawContours(frame, [hand_data['contour']], -1, (0, 255, 0), 2)

        # 绘制边界框
        if show_bbox and 'bbox' in hand_data:
            x1, y1, x2, y2 = hand_data['bbox']
            color = self.gesture_colors.get(gesture, (255, 255, 255))

            # 根据手势状态调整边界框颜色
            state_color = self.state_colors.get(self.gesture_state, color)
            if self.gesture_state != "none":
                color = state_color

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 显示手势标签
            label = f"{gesture}"
            if config.get('display', 'show_confidence'):
                label += f" ({confidence:.0%})"

            # 显示手势状态
            if self.gesture_state != "none":
                state_text = {"starting": "开始", "active": "活跃", "ending": "结束"}
                label += f" [{state_text.get(self.gesture_state, '')}]"

            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # 绘制文本背景
            cv2.rectangle(frame,
                          (x1, y1 - text_height - 10),
                          (x1 + text_width, y1),
                          color, -1)

            # 绘制文本
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 绘制手掌中心
        if show_palm_center and 'palm_center' in hand_data:
            cx, cy = hand_data['palm_center']
            palm_radius = hand_data.get('palm_radius', 20)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), palm_radius, (0, 0, 255), 1)
            cv2.putText(frame, "Palm", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 绘制指尖
        if show_fingertips and 'fingertips' in hand_data:
            for i, point in enumerate(hand_data['fingertips']):
                cv2.circle(frame, point, 4, (255, 0, 0), -1)
                cv2.putText(frame, f"F{i + 1}", (point[0] + 5, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 绘制手部方向
        if show_hand_direction and 'direction' in hand_data and 'center' in hand_data:
            cx, cy = hand_data['center']
            direction = hand_data['direction']
            length = 50

            dx = length * math.cos(math.radians(direction))
            dy = length * math.sin(math.radians(direction))

            end_point = (int(cx + dx), int(cy + dy))
            cv2.arrowedLine(frame, (cx, cy), end_point, (255, 255, 0), 2)

            angle_text = f"Dir: {direction:.0f}°"
            cv2.putText(frame, angle_text, (cx, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 绘制稳定性指示器
        if show_stability_indicator:
            h, w = frame.shape[:2]
            indicator_x = w - 100
            indicator_y = 30

            # 绘制稳定性背景
            cv2.rectangle(frame, (indicator_x, indicator_y),
                          (indicator_x + 80, indicator_y + 15), (50, 50, 50), -1)

            # 计算稳定性指示条长度
            stability_level = min(1.0,
                                  self.gesture_stability_counter / self.mode_config['gesture_stability_threshold'])
            bar_length = int(70 * stability_level)

            # 根据稳定性级别选择颜色
            if stability_level > 0.7:
                bar_color = (0, 255, 0)  # 绿色
            elif stability_level > 0.4:
                bar_color = (255, 165, 0)  # 橙色
            else:
                bar_color = (255, 0, 0)  # 红色

            # 绘制稳定性指示条
            cv2.rectangle(frame, (indicator_x + 5, indicator_y + 5),
                          (indicator_x + 5 + bar_length, indicator_y + 10), bar_color, -1)

            # 绘制稳定性文本
            cv2.putText(frame, "稳定度", (indicator_x, indicator_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制手势历史（如果启用）
        if show_gesture_history and len(self.gesture_history) > 0:
            h, w = frame.shape[:2]
            history_y = h - 50

            # 绘制历史背景
            cv2.rectangle(frame, (10, history_y - 20), (200, history_y + 10), (0, 0, 0), -1)
            cv2.putText(frame, "手势历史:", (15, history_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示最近几个手势
            recent_gestures = list(self.gesture_history)[-5:] if len(self.gesture_history) >= 5 else list(
                self.gesture_history)
            for i, gest in enumerate(recent_gestures):
                color = self.gesture_colors.get(gest, (255, 255, 255))
                cv2.putText(frame, gest[0], (85 + i * 20, history_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示调试信息
        if show_debug_info:
            # 显示手指数量
            finger_count = len(hand_data.get('fingers', []))
            finger_text = f"Fingers: {finger_count}"
            cv2.putText(frame, finger_text, (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示手部位置
            pos_text = f"Pos: ({hand_data['position'][0]:.2f}, {hand_data['position'][1]:.2f})"
            cv2.putText(frame, pos_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示手势状态
            state_text = f"State: {self.gesture_state}"
            cv2.putText(frame, state_text, (150, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示稳定性计数器
            stability_text = f"Stability: {self.gesture_stability_counter}"
            cv2.putText(frame, stability_text, (150, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def recognize(self, frame):
        """识别手势"""
        start_time = time.time()

        try:
            # 预处理帧
            processed_frame = cv2.flip(frame, 1)

            # 根据性能模式调整图像大小
            original_frame = processed_frame.copy()
            if self.resize_factor != 1.0:
                processed_frame = self.preprocess_frame(processed_frame)

            # 每隔几帧检测一次以提高性能
            if self.frame_counter % self.detection_interval != 0:
                self.frame_counter += 1
                return self.current_gesture, self.current_confidence, original_frame

            # 获取肤色掩码
            skin_mask, hsv = self.get_skin_mask(processed_frame)

            # 增强肤色检测
            enhanced_mask = self.enhance_skin_detection(processed_frame, skin_mask)

            # 找到最佳的手部轮廓
            hand_contour, contour_score = self.find_best_hand_contour(enhanced_mask, processed_frame)

            # 分析手部特征
            hand_data, confidence = self.analyze_hand_features(hand_contour, processed_frame.shape)

            # 保存手部数据用于轨迹记录
            self.last_hand_data = hand_data

            # 识别手势
            if hand_data is not None:
                # 校准肤色（如果需要）
                config.calibrate_skin_color(processed_frame, enhanced_mask)

                # 校准手部大小（如果需要）
                config.calibrate_hand_size(hand_data['area'])

                # 识别手势
                gesture, raw_confidence = self.recognize_gesture_improved(hand_data)
                confidence = max(confidence, raw_confidence)

                # 更新手势状态
                self.update_gesture_state(hand_data, gesture, confidence)

                # 增强的手势平滑
                final_gesture, final_confidence = self.smooth_gesture_enhanced(gesture, confidence, hand_data)
            else:
                gesture, confidence = "Waiting", 0.3
                self.update_gesture_state(None, gesture, confidence)
                final_gesture, final_confidence = gesture, confidence

            # 手势语音播报
            if (self.speech_manager and
                    config.get('speech', 'enabled') and
                    config.get('speech', 'announce_gestures')):

                current_time = time.time()

                # 根据置信度提供反馈
                if confidence >= 0.8 and current_time - self.last_hand_status_time > 5.0:
                    self.speech_manager.speak('gesture_good_confidence', immediate=True)
                    self.last_hand_status_time = current_time
                elif confidence < 0.5 and current_time - self.last_hand_status_time > 5.0:
                    self.speech_manager.speak('gesture_low_confidence', immediate=True)
                    self.last_hand_status_time = current_time

                # 手势语音播报
                if (final_gesture != "Waiting" and
                        final_gesture != "Error" and
                        final_gesture != "摄像头错误" and
                        final_confidence >= self.min_confidence and
                        current_time - self.last_announced_time > self.gesture_announce_interval):

                    if final_gesture in self.gesture_speech_map:
                        speech_key = self.gesture_speech_map[final_gesture]
                        self.speech_manager.speak(speech_key)
                    else:
                        self.speech_manager.speak_direct(f"手势{final_gesture}")

                    self.last_announced_gesture = final_gesture
                    self.last_announced_time = current_time

            # 性能报告
            current_time = time.time()
            if current_time - self.last_performance_report > 30.0:
                if (self.speech_manager and
                        config.get('speech', 'enabled') and
                        config.get('speech', 'announce_performance')):

                    if self.avg_process_time < 20:
                        self.speech_manager.speak('performance_good', immediate=True)
                    elif self.avg_process_time > 50:
                        self.speech_manager.speak('performance_warning', immediate=True)

                    self.last_performance_report = current_time

            # 可视化结果（在原图上）
            if hand_data is not None:
                # 需要将坐标转换回原始图像大小
                if self.resize_factor != 1.0:
                    # 缩放坐标
                    scale_factor = 1.0 / self.resize_factor
                    if 'center' in hand_data:
                        hand_data['center'] = (int(hand_data['center'][0] * scale_factor),
                                               int(hand_data['center'][1] * scale_factor))
                    if 'bbox' in hand_data:
                        x1, y1, x2, y2 = hand_data['bbox']
                        hand_data['bbox'] = (int(x1 * scale_factor), int(y1 * scale_factor),
                                             int(x2 * scale_factor), int(y2 * scale_factor))
                    if 'fingertips' in hand_data:
                        hand_data['fingertips'] = [(int(x * scale_factor), int(y * scale_factor))
                                                   for (x, y) in hand_data['fingertips']]
                    if 'palm_center' in hand_data:
                        hand_data['palm_center'] = (int(hand_data['palm_center'][0] * scale_factor),
                                                    int(hand_data['palm_center'][1] * scale_factor))

                original_frame = self.visualize_detection(
                    original_frame, hand_data, final_gesture, final_confidence
                )
            else:
                original_frame = self.visualize_detection(
                    original_frame, None, final_gesture, final_confidence
                )

            # 更新计数器
            self.frame_counter += 1

            # 计算处理时间
            process_time = (time.time() - start_time) * 1000
            self.process_times.append(process_time)

            # 更新平均处理时间
            if len(self.process_times) > 0:
                self.avg_process_time = np.mean(list(self.process_times))

            # 更新帧率
            current_time = time.time()
            if current_time - self.last_fps_check >= 1.0:
                self.frame_rate = self.frame_counter
                self.frame_counter = 0
                self.last_fps_check = current_time

            return final_gesture, final_confidence, original_frame

        except Exception as e:
            print(f"⚠ 手势识别错误: {e}")
            return "Error", 0.0, frame

    def get_performance_stats(self):
        """获取性能统计"""
        if len(self.process_times) == 0:
            return 0.0, self.frame_rate

        return np.mean(list(self.process_times)), self.frame_rate

    def get_performance_mode_info(self):
        """获取性能模式信息"""
        return {
            'name': self.performance_mode_name,
            'mode': self.performance_mode,
            'color': self.performance_mode_color,
            'detection_interval': self.detection_interval,
            'resize_factor': self.resize_factor,
            'smooth_frames': self.smooth_frames,
            'min_confidence': self.min_confidence
        }

    def set_simulated_gesture(self, gesture):
        """设置模拟的手势"""
        self.current_gesture = gesture
        self.current_confidence = 0.9

        # 模拟手势也触发语音提示
        if (self.speech_manager and
                config.get('speech', 'enabled') and
                config.get('speech', 'announce_gestures')):

            if gesture in self.gesture_speech_map:
                self.speech_manager.speak(self.gesture_speech_map[gesture])
            else:
                self.speech_manager.speak_direct(f"手势{gesture}")

            self.last_announced_gesture = gesture
            self.last_announced_time = time.time()


# ========== 简单的无人机控制器 ==========
class SimpleDroneController:
    """简单的无人机控制器"""

    def __init__(self, airsim_module, speech_manager=None):
        self.airsim = airsim_module
        self.client = None
        self.connected = False
        self.flying = False
        self.speech_manager = speech_manager

        # 控制参数
        self.velocity = config.get('drone', 'velocity')
        self.duration = config.get('drone', 'duration')
        self.altitude = config.get('drone', 'altitude')
        self.control_interval = config.get('drone', 'control_interval')

        # 控制状态
        self.last_control_time = 0
        self.last_gesture = None

        # 上次语音提示状态
        self.last_connection_announced = False
        self.last_takeoff_announced = False
        self.last_land_announced = False

        print("✓ 简单的无人机控制器已初始化")

    def connect(self):
        """连接AirSim无人机"""
        if self.connected:
            return True

        # 语音提示：正在连接
        if (self.speech_manager and
                config.get('speech', 'enabled') and
                config.get('speech', 'announce_connections')):
            self.speech_manager.speak('connecting')

        if self.airsim is None:
            print("⚠ AirSim不可用，使用模拟模式")

            # 语音提示：模拟模式
            if (self.speech_manager and
                    config.get('speech', 'enabled') and
                    config.get('speech', 'announce_connections')):
                self.speech_manager.speak('simulation_mode')

            self.connected = True
            return True

        print("连接AirSim...")

        try:
            self.client = self.airsim.MultirotorClient()
            self.client.confirmConnection()
            print("✅ 已连接AirSim!")

            # 语音提示：连接成功
            if (self.speech_manager and
                    config.get('speech', 'enabled') and
                    config.get('speech', 'announce_connections')):
                self.speech_manager.speak('connected')

            self.client.enableApiControl(True)
            print("✅ API控制已启用")

            self.client.armDisarm(True)
            print("✅ 无人机已武装")

            self.connected = True
            return True

        except Exception as e:
            print(f"❌ 连接失败: {e}")

            # 语音提示：连接失败
            if (self.speech_manager and
                    config.get('speech', 'enabled') and
                    config.get('speech', 'announce_connections')):
                self.speech_manager.speak('connection_failed')

            print("\n使用模拟模式继续? (y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                self.connected = True
                print("✅ 使用模拟模式")

                # 语音提示：模拟模式
                if (self.speech_manager and
                        config.get('speech', 'enabled') and
                        config.get('speech', 'announce_connections')):
                    self.speech_manager.speak('simulation_mode')

                return True

            return False

    def takeoff(self):
        """起飞"""
        if not self.connected:
            return False

        # 语音提示：正在起飞
        if (self.speech_manager and
                config.get('speech', 'enabled') and
                config.get('speech', 'announce_flight_events') and
                not self.last_takeoff_announced):
            self.speech_manager.speak('taking_off')
            self.last_takeoff_announced = True
            self.last_land_announced = False

        try:
            if self.airsim is None or self.client is None:
                print("✅ 模拟起飞")
                self.flying = True

                # 语音提示：起飞成功
                if (self.speech_manager and
                        config.get('speech', 'enabled') and
                        config.get('speech', 'announce_flight_events')):
                    self.speech_manager.speak('takeoff_success')

                return True

            print("起飞中...")
            self.client.takeoffAsync().join()
            time.sleep(1)

            # 上升到指定高度
            self.client.moveToZAsync(self.altitude, 3).join()

            self.flying = True
            print("✅ 无人机成功起飞")

            # 语音提示：起飞成功
            if (self.speech_manager and
                    config.get('speech', 'enabled') and
                    config.get('speech', 'announce_flight_events')):
                self.speech_manager.speak('takeoff_success')

            return True
        except Exception as e:
            print(f"❌ 起飞失败: {e}")

            # 语音提示：起飞失败
            if (self.speech_manager and
                    config.get('speech', 'enabled') and
                    config.get('speech', 'announce_flight_events')):
                self.speech_manager.speak('takeoff_failed')

            return False

    def land(self):
        """降落"""
        if not self.connected:
            return False

        # 语音提示：正在降落
        if (self.speech_manager and
                config.get('speech', 'enabled') and
                config.get('speech', 'announce_flight_events') and
                not self.last_land_announced):
            self.speech_manager.speak('landing')
            self.last_land_announced = True
            self.last_takeoff_announced = False

        try:
            if self.airsim is None or self.client is None:
                print("✅ 模拟降落")
                self.flying = False

                # 语音提示：降落成功
                if (self.speech_manager and
                        config.get('speech', 'enabled') and
                        config.get('speech', 'announce_flight_events')):
                    self.speech_manager.speak('land_success')

                return True

            print("降落中...")
            self.client.landAsync().join()
            self.flying = False
            print("✅ 无人机已降落")

            # 语音提示：降落成功
            if (self.speech_manager and
                    config.get('speech', 'enabled') and
                    config.get('speech', 'announce_flight_events')):
                self.speech_manager.speak('land_success')

            return True
        except Exception as e:
            print(f"降落失败: {e}")
            return False

    def move_by_gesture(self, gesture, confidence):
        """根据手势移动"""
        if not self.connected or not self.flying:
            return False

        # 检查控制间隔
        current_time = time.time()
        if current_time - self.last_control_time < self.control_interval:
            return False

        # 检查置信度阈值
        min_confidence = config.get('gesture', 'min_confidence')
        if confidence < min_confidence:
            # 低置信度语音提示
            if (self.speech_manager and
                    config.get('speech', 'enabled') and
                    config.get('speech', 'announce_gestures') and
                    confidence < min_confidence * 0.8):
                self.speech_manager.speak('gesture_low_confidence')
            return False

        try:
            if self.airsim is None or self.client is None:
                print(f"模拟移动: {gesture}")
                self.last_control_time = current_time
                self.last_gesture = gesture
                return True

            success = False

            if gesture == "Up":
                self.client.moveByVelocityZAsync(0, 0, -self.velocity, self.duration)
                success = True
            elif gesture == "Down":
                self.client.moveByVelocityZAsync(0, 0, self.velocity, self.duration)
                success = True
            elif gesture == "Left":
                self.client.moveByVelocityAsync(-self.velocity, 0, 0, self.duration)
                success = True
            elif gesture == "Right":
                self.client.moveByVelocityAsync(self.velocity, 0, 0, self.duration)
                success = True
            elif gesture == "Forward":
                self.client.moveByVelocityAsync(0, -self.velocity, 0, self.duration)
                success = True
            elif gesture == "Stop":
                self.client.hoverAsync()
                success = True
                # 悬停语音提示
                if (self.speech_manager and
                        config.get('speech', 'enabled') and
                        config.get('speech', 'announce_flight_events')):
                    self.speech_manager.speak('hovering')
            elif gesture == "Hover":
                self.client.hoverAsync()
                success = True
                if (self.speech_manager and
                        config.get('speech', 'enabled') and
                        config.get('speech', 'announce_flight_events')):
                    self.speech_manager.speak('hovering')

            if success:
                self.last_control_time = current_time
                self.last_gesture = gesture

            return success
        except Exception as e:
            print(f"控制命令失败: {e}")
            return False

    def emergency_stop(self):
        """紧急停止"""
        if self.connected:
            try:
                if self.flying and self.client is not None:
                    print("紧急降落...")

                    # 语音提示：紧急停止
                    if (self.speech_manager and
                            config.get('speech', 'enabled') and
                            config.get('speech', 'announce_flight_events')):
                        self.speech_manager.speak('emergency_stop')

                    self.land()
                if self.client is not None:
                    self.client.armDisarm(False)
                    self.client.enableApiControl(False)
                    print("✅ 紧急停止完成")
            except:
                pass

        self.connected = False
        self.flying = False


# ========== 中文UI渲染器 ==========
class ChineseUIRenderer:
    """中文UI渲染器"""

    def __init__(self, speech_manager=None):
        self.fonts = {}
        self.speech_manager = speech_manager
        self.load_fonts()

        # 颜色定义
        self.colors = {
            'title': (0, 255, 255),
            'connected': (0, 255, 0),
            'disconnected': (0, 0, 255),
            'flying': (0, 255, 0),
            'landed': (255, 165, 0),
            'warning': (0, 165, 255),
            'info': (255, 255, 255),
            'help': (255, 200, 100),
            'speech_enabled': (0, 255, 0),
            'speech_disabled': (255, 0, 0),
            'performance_good': (0, 255, 0),
            'performance_warning': (255, 165, 0),
            'performance_bad': (255, 0, 0),
            'recording': (255, 50, 50),
            'playback': (50, 50, 255),
            'paused': (255, 255, 0),
            'performance_fast': (0, 255, 0),  # 绿色
            'performance_balanced': (255, 165, 0),  # 橙色
            'performance_accurate': (255, 0, 0),  # 红色
        }

        print("✓ 中文UI渲染器已初始化")

    def load_fonts(self):
        """加载字体"""
        font_paths = [
            'simhei.ttf',
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]

        for path in font_paths:
            try:
                self.fonts[14] = ImageFont.truetype(path, 14)
                self.fonts[16] = ImageFont.truetype(path, 16)
                self.fonts[18] = ImageFont.truetype(path, 18)
                self.fonts[20] = ImageFont.truetype(path, 20)
                self.fonts[24] = ImageFont.truetype(path, 24)
                print(f"✓ 字体已加载: {path}")
                return
            except:
                continue

        print("⚠ 未找到字体，使用默认")

    def draw_text(self, frame, text, pos, size=16, color=(255, 255, 255)):
        """在图像上绘制文本"""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            font = self.fonts.get(size, self.fonts.get(16))

            # 绘制阴影
            shadow_color = (0, 0, 0)
            shadow_pos = (pos[0] + 1, pos[1] + 1)
            draw.text(shadow_pos, text, font=font, fill=shadow_color)

            # 绘制文字
            rgb_color = color[::-1]  # BGR to RGB
            draw.text(pos, text, font=font, fill=rgb_color)

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            # 备用方案：使用OpenCV绘制英文
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 25, color, 1)
            return frame

    def draw_status_bar(self, frame, drone_controller, gesture, confidence, fps, process_time, trajectory_recorder=None,
                        gesture_recognizer=None):
        """绘制状态栏"""
        h, w = frame.shape[:2]

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # 标题
        title = "手势控制无人机系统 - 性能模式版"
        frame = self.draw_text(frame, title, (10, 10), size=20, color=self.colors['title'])

        # 连接状态
        status_color = self.colors['connected'] if drone_controller.connected else self.colors['disconnected']
        status_text = f"无人机: {'已连接' if drone_controller.connected else '未连接'}"
        frame = self.draw_text(frame, status_text, (10, 40), size=16, color=status_color)

        # 飞行状态
        flight_color = self.colors['flying'] if drone_controller.flying else self.colors['landed']
        flight_text = f"飞行状态: {'飞行中' if drone_controller.flying else '已降落'}"
        frame = self.draw_text(frame, flight_text, (10, 65), size=16, color=flight_color)

        # 手势信息
        if confidence > 0.7:
            gesture_color = (0, 255, 0)
        elif confidence > 0.5:
            gesture_color = (255, 165, 0)
        else:
            gesture_color = (200, 200, 200)

        gesture_text = f"当前手势: {gesture}"
        if config.get('display', 'show_confidence'):
            gesture_text += f" ({confidence:.0%})"

        frame = self.draw_text(frame, gesture_text, (w // 2, 40), size=16, color=gesture_color)

        # 性能模式显示
        if gesture_recognizer and config.get('display', 'show_performance_mode'):
            mode_info = gesture_recognizer.get_performance_mode_info()

            # 根据模式选择颜色
            if mode_info['mode'] == 'fast':
                mode_color = self.colors['performance_fast']
            elif mode_info['mode'] == 'balanced':
                mode_color = self.colors['performance_balanced']
            else:  # accurate
                mode_color = self.colors['performance_accurate']

            mode_text = f"性能模式: {mode_info['name']}"
            frame = self.draw_text(frame, mode_text, (w // 2, 65), size=14, color=mode_color)

        # 录制/回放状态
        elif trajectory_recorder and config.get('display', 'show_recording_status'):
            recorder_status = trajectory_recorder.get_status()

            if recorder_status['is_recording']:
                status_color = self.colors['recording']
                status_text = f"录制中: {recorder_status['trajectory_points']}点"
                frame = self.draw_text(frame, status_text, (w // 2, 65), size=14, color=status_color)
            elif recorder_status['is_playing']:
                if recorder_status['playback_paused']:
                    status_color = self.colors['paused']
                    status_text = f"回放暂停: {recorder_status['playback_index']}/{recorder_status['playback_total']}"
                else:
                    status_color = self.colors['playback']
                    status_text = f"回放中: {recorder_status['playback_index']}/{recorder_status['playback_total']}"
                frame = self.draw_text(frame, status_text, (w // 2, 65), size=14, color=status_color)
            else:
                status_text = f"轨迹点: {recorder_status['trajectory_points']}"
                frame = self.draw_text(frame, status_text, (w // 2, 65), size=12, color=self.colors['info'])

        # 语音状态
        if config.get('display', 'show_speech_status') and self.speech_manager:
            speech_status = self.speech_manager.get_status()
            speech_color = self.colors['speech_enabled'] if speech_status['enabled'] else self.colors['speech_disabled']
            speech_text = f"语音: {'启用' if speech_status['enabled'] else '禁用'}"
            frame = self.draw_text(frame, speech_text, (w // 2, 90), size=12, color=speech_color)

        # 性能信息
        if config.get('display', 'show_fps'):
            perf_text = f"帧率: {fps:.1f}"
            if process_time > 0:
                perf_text += f" | 延迟: {process_time:.1f}ms"

                # 根据处理时间选择颜色
                if process_time < 20:
                    perf_color = self.colors['performance_good']
                elif process_time < 50:
                    perf_color = self.colors['performance_warning']
                else:
                    perf_color = self.colors['performance_bad']
            else:
                perf_color = self.colors['info']

            frame = self.draw_text(frame, perf_text, (w - 200, 65), size=12, color=perf_color)

        # 控制提示
        control_text = "提示: 确保手部完全进入画面，保持稳定手势"
        frame = self.draw_text(frame, control_text, (10, 90), size=12, color=self.colors['info'])

        return frame

    def draw_help_bar(self, frame):
        """绘制帮助栏"""
        if not config.get('display', 'show_help'):
            return frame

        h, w = frame.shape[:2]

        # 绘制底部帮助栏
        cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)

        # 帮助文本
        help_lines = [
            "C:连接  空格:起飞/降落  ESC:退出  W/A/S/D/F/X:键盘控制",
            "H:切换帮助  R:重置识别  T:切换显示模式  D:调试信息",
            "V:切换语音反馈  M:测试语音  P:性能报告  O:切换性能模式",
            "1:开始录制 2:停止录制 3:保存轨迹 4:回放轨迹 5:清除轨迹 6:暂停/继续"
        ]

        for i, line in enumerate(help_lines):
            y_pos = h - 85 + i * 20
            frame = self.draw_text(frame, line, (10, y_pos), size=14, color=self.colors['help'])

        return frame

    def draw_warning(self, frame, message):
        """绘制警告信息"""
        h, w = frame.shape[:2]

        # 在顶部绘制警告
        warning_bg = np.zeros((40, w, 3), dtype=np.uint8)
        warning_bg[:, :] = (0, 69, 255)

        frame[120:160, 0:w] = cv2.addWeighted(
            frame[120:160, 0:w], 0.3,
            warning_bg, 0.7, 0
        )

        # 绘制警告文本
        frame = self.draw_text(frame, message, (10, 135),
                               size=16, color=self.colors['warning'])

        return frame


# ========== 性能监控器 ==========
class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.frame_times = deque(maxlen=60)
        self.last_update = time.time()
        self.fps = 0
        self.frame_count = 0

    def update(self):
        """更新性能数据"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_count += 1

        # 每秒计算一次FPS
        if current_time - self.last_update >= 1.0:
            if len(self.frame_times) > 1:
                time_diff = self.frame_times[-1] - self.frame_times[0]
                if time_diff > 0:
                    self.fps = len(self.frame_times) / time_diff
                else:
                    self.fps = 0
            self.last_update = current_time

    def get_stats(self):
        """获取性能统计"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count
        }


# ========== 主程序 ==========
def main():
    """主函数"""
    # 初始化语音管理器
    print("初始化语音反馈系统...")
    speech_manager = EnhancedSpeechFeedbackManager(libs['speech'])

    # 程序启动语音提示
    if speech_manager.enabled:
        speech_manager.speak('program_start', force=True, immediate=True)
        speech_manager.speak('system_ready', immediate=True)

    # 初始化组件
    print("初始化组件...")

    gesture_recognizer = EnhancedGestureRecognizer(speech_manager)
    drone_controller = SimpleDroneController(libs['airsim'], speech_manager)
    ui_renderer = ChineseUIRenderer(speech_manager)
    performance_monitor = PerformanceMonitor()

    # 手势轨迹记录器
    print("初始化手势轨迹记录器...")
    trajectory_recorder = GestureTrajectoryRecorder(speech_manager)

    # 初始化摄像头
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('camera', 'width'))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('camera', 'height'))
            cap.set(cv2.CAP_PROP_FPS, config.get('camera', 'fps'))

            # 获取实际参数
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

            print(f"✓ 摄像头已初始化")
            print(f"  分辨率: {actual_width}x{actual_height}")
            print(f"  帧率: {actual_fps}")

            # 摄像头就绪语音提示
            if speech_manager.enabled:
                speech_manager.speak('camera_ready', immediate=True)
        else:
            print("❌ 摄像头不可用，使用模拟模式")

            # 摄像头错误语音提示
            if speech_manager.enabled:
                speech_manager.speak('camera_error', immediate=True)

            cap = None
    except Exception as e:
        print(f"⚠ 摄像头初始化失败: {e}")

        # 摄像头错误语音提示
        if speech_manager.enabled:
            speech_manager.speak('camera_error', immediate=True)

        cap = None

    # 显示欢迎信息
    print("\n" + "=" * 60)
    print("手势控制无人机系统 - 性能模式版")
    print("=" * 60)
    print("系统状态:")
    print(f"  摄像头: {'已连接' if cap else '模拟模式'}")
    print(f"  手势识别: 增强的平滑处理算法")
    print(f"  语音反馈: {'已启用' if speech_manager.enabled else '已禁用'}")
    print(f"  性能模式: {gesture_recognizer.performance_mode_name}")
    print(f"  轨迹记录: 支持录制/回放功能")
    print(f"  AirSim: {'可用' if libs['airsim'] else '模拟模式'}")
    print("=" * 60)

    # 显示操作说明
    print("\n操作说明:")
    print("1. 按 [C] 连接无人机 (AirSim模拟器)")
    print("2. 按 [空格键] 起飞/降落")
    print("3. 性能模式选择:")
    print("   - 按 [O] 键循环切换性能模式: 最快(fast) → 平衡(balanced) → 最准(accurate)")
    print("   - 最快模式: 性能优先，帧率最高，识别精度较低")
    print("   - 平衡模式: 性能与精度平衡，适用于大多数场景")
    print("   - 最准模式: 精度优先，提供最准确的手势识别")
    print("4. 手势控制:")
    print("   - 手势识别后会有语音提示: 向上、向下、向左、向右、向前、停止")
    print("   - 手势稳定性越高，识别越准确")
    print("   - 手部距离摄像头适中时效果最佳")
    print("   * 手势识别置信度 > 60% 时才会执行")
    print("5. 轨迹记录功能:")
    print("   [1]开始录制 [2]停止录制 [3]保存轨迹 [4]回放轨迹 [5]清除轨迹 [6]暂停/继续")
    print("   - 录制时系统会记录手部位置和手势")
    print("   - 回放时可以查看录制的轨迹")
    print("   - 轨迹数据自动保存到 trajectories/ 目录")
    print("6. 键盘控制:")
    print("   [W]向上 [S]向下 [A]向左 [D]向右 [F]向前 [X]停止")
    print("7. 调试功能:")
    print("   [H]切换帮助显示 [R]重置手势识别 [T]切换显示模式 [D]调试信息")
    print("8. 语音控制:")
    print("   [V]切换语音反馈 [M]测试语音 [P]性能报告")
    print("9. 按 [ESC] 安全退出")
    print("=" * 60)
    print("程序启动成功!")
    print("-" * 60)

    # 键盘手势映射
    key_to_gesture = {
        ord('w'): "Up", ord('W'): "Up",
        ord('s'): "Down", ord('S'): "Down",
        ord('a'): "Left", ord('A'): "Left",
        ord('d'): "Right", ord('D'): "Right",
        ord('f'): "Forward", ord('F'): "Forward",
        ord('x'): "Stop", ord('X'): "Stop",
        ord('h'): "Hover", ord('H'): "Hover",
    }

    # 显示模式
    display_modes = ['normal', 'detailed', 'minimal']
    current_display_mode = 0

    # 主循环
    print("\n进入主循环，按ESC退出...")

    try:
        while True:
            # 更新性能监控
            performance_monitor.update()

            # 读取摄像头帧
            if cap:
                ret, frame = cap.read()
                if not ret:
                    # 创建空白帧
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    gesture, confidence = "摄像头错误", 0.0
                else:
                    # 手势识别
                    gesture, confidence, frame = gesture_recognizer.recognize(frame)
            else:
                # 模拟模式
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                gesture, confidence = gesture_recognizer.current_gesture, gesture_recognizer.current_confidence

            # 获取性能统计
            perf_stats = performance_monitor.get_stats()
            process_time, frame_rate = gesture_recognizer.get_performance_stats()

            # 根据显示模式调整显示选项
            if display_modes[current_display_mode] == 'normal':
                config.set('display', 'show_contours', value=True)
                config.set('display', 'show_bbox', value=True)
                config.set('display', 'show_fingertips', value=True)
                config.set('display', 'show_gesture_history', value=True)
                config.set('display', 'show_stability_indicator', value=True)
                config.set('display', 'show_trajectory', value=True)
                config.set('display', 'show_recording_status', value=True)
                config.set('display', 'show_performance_mode', value=True)
                config.set('display', 'show_debug_info', value=False)
            elif display_modes[current_display_mode] == 'detailed':
                config.set('display', 'show_contours', value=True)
                config.set('display', 'show_bbox', value=True)
                config.set('display', 'show_fingertips', value=True)
                config.set('display', 'show_palm_center', value=True)
                config.set('display', 'show_hand_direction', value=True)
                config.set('display', 'show_gesture_history', value=True)
                config.set('display', 'show_stability_indicator', value=True)
                config.set('display', 'show_trajectory', value=True)
                config.set('display', 'show_recording_status', value=True)
                config.set('display', 'show_performance_mode', value=True)
                config.set('display', 'show_debug_info', value=True)
            elif display_modes[current_display_mode] == 'minimal':
                config.set('display', 'show_contours', value=False)
                config.set('display', 'show_bbox', value=True)
                config.set('display', 'show_fingertips', value=False)
                config.set('display', 'show_gesture_history', value=False)
                config.set('display', 'show_stability_indicator', value=False)
                config.set('display', 'show_trajectory', value=True)
                config.set('display', 'show_recording_status', value=True)
                config.set('display', 'show_performance_mode', value=True)
                config.set('display', 'show_debug_info', value=False)

            # 绘制轨迹（如果启用）
            if config.get('display', 'show_trajectory'):
                frame = trajectory_recorder.draw_trajectory(frame)

            # 绘制UI
            frame = ui_renderer.draw_status_bar(
                frame, drone_controller, gesture, confidence,
                perf_stats['fps'], process_time, trajectory_recorder, gesture_recognizer
            )

            frame = ui_renderer.draw_help_bar(frame)

            # 显示连接提示
            if not drone_controller.connected:
                warning_msg = "⚠ 按C键连接无人机，或使用模拟模式"
                frame = ui_renderer.draw_warning(frame, warning_msg)

            # 显示图像（窗口标题用英文）
            cv2.imshow('Gesture Controlled Drone - Performance Mode', frame)

            # ========== 键盘控制 ==========
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC键
                print("\n退出程序...")
                break

            elif key == ord('c') or key == ord('C'):
                if not drone_controller.connected:
                    drone_controller.connect()

            elif key == 32:  # 空格键
                if drone_controller.connected:
                    if drone_controller.flying:
                        drone_controller.land()
                    else:
                        drone_controller.takeoff()
                    time.sleep(0.5)

            elif key == ord('h') or key == ord('H'):
                # 切换帮助显示
                current = config.get('display', 'show_help')
                config.set('display', 'show_help', value=not current)
                print(f"帮助显示: {'开启' if not current else '关闭'}")

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak('help_toggled', immediate=True)

            elif key == ord('r') or key == ord('R'):
                # 重置手势识别
                print("重置手势识别...")
                gesture_recognizer = EnhancedGestureRecognizer(speech_manager)
                print("✓ 手势识别已重置")

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak_direct("手势识别已重置")

            elif key == ord('t') or key == ord('T'):
                # 切换显示模式
                current_display_mode = (current_display_mode + 1) % len(display_modes)
                mode_name = display_modes[current_display_mode]
                print(f"显示模式: {mode_name}")

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak('display_mode_changed', immediate=True)

            elif key == ord('d') or key == ord('D'):
                # 切换调试信息
                current = config.get('display', 'show_debug_info')
                config.set('display', 'show_debug_info', value=not current)
                status = '开启' if not current else '关闭'
                print(f"调试信息: {status}")

                # 语音提示
                if speech_manager.enabled:
                    if not current:
                        speech_manager.speak('debug_mode_on', immediate=True)
                    else:
                        speech_manager.speak('debug_mode_off', immediate=True)

            elif key == ord('v') or key == ord('V'):
                # 切换语音反馈
                new_status = speech_manager.toggle_enabled()
                status = '启用' if new_status else '禁用'
                print(f"语音反馈: {status}")
                config.set('speech', 'enabled', value=new_status)

            elif key == ord('m') or key == ord('M'):
                # 测试语音
                if speech_manager.enabled:
                    print("测试语音...")
                    speech_manager.speak_direct("语音反馈测试，系统运行正常")
                else:
                    print("语音反馈已禁用，按V键启用")

            elif key == ord('p') or key == ord('P'):
                # 性能报告
                if speech_manager.enabled:
                    print("生成性能报告...")
                    if process_time < 20:
                        speech_manager.speak_direct("系统性能优秀，运行流畅")
                    elif process_time < 50:
                        speech_manager.speak_direct("系统性能良好")
                    else:
                        speech_manager.speak_direct("系统性能警告，请检查")

            elif key == ord('o') or key == ord('O'):
                # 切换性能模式
                if config.cycle_performance_mode():
                    new_mode = config.get_current_performance_mode()
                    gesture_recognizer.set_performance_mode(new_mode)

                    # 语音提示
                    if speech_manager.enabled:
                        if new_mode == 'fast':
                            speech_manager.speak('performance_mode_fast', immediate=True)
                        elif new_mode == 'balanced':
                            speech_manager.speak('performance_mode_balanced', immediate=True)
                        else:  # accurate
                            speech_manager.speak('performance_mode_accurate', immediate=True)

                    print(f"✓ 已切换到性能模式: {gesture_recognizer.performance_mode_name}")

            # 新增：轨迹记录控制
            elif key == ord('1'):
                # 开始录制轨迹
                if trajectory_recorder.start_recording():
                    print("✅ 开始录制手势轨迹")
                else:
                    print("⚠ 已经在录制中")

            elif key == ord('2'):
                # 停止录制轨迹
                if trajectory_recorder.stop_recording():
                    print("✅ 停止录制手势轨迹")
                else:
                    print("⚠ 当前没有在录制")

            elif key == ord('3'):
                # 保存轨迹
                if trajectory_recorder.save_trajectory():
                    print("✅ 轨迹保存成功")
                else:
                    print("⚠ 没有轨迹数据可保存")

            elif key == ord('4'):
                # 回放轨迹
                if trajectory_recorder.start_playback():
                    print("✅ 开始回放手势轨迹")
                else:
                    print("⚠ 没有轨迹数据可回放")

            elif key == ord('5'):
                # 清除轨迹
                if trajectory_recorder.clear_trajectory():
                    print("✅ 轨迹数据已清除")
                else:
                    print("⚠ 清除轨迹失败")

            elif key == ord('6'):
                # 暂停/继续回放
                if trajectory_recorder.pause_playback():
                    print("✅ 切换回放暂停状态")
                else:
                    print("⚠ 当前没有在回放")

            elif key in key_to_gesture:
                # 键盘控制
                simulated_gesture = key_to_gesture[key]
                gesture_recognizer.set_simulated_gesture(simulated_gesture)
                gesture = simulated_gesture
                confidence = 0.9
                if drone_controller.connected and drone_controller.flying:
                    drone_controller.move_by_gesture(gesture, confidence)

            # 真实手势控制
            current_time = time.time()
            if (gesture and gesture != "Waiting" and
                    gesture != "摄像头错误" and gesture != "Error" and
                    drone_controller.connected and drone_controller.flying):
                drone_controller.move_by_gesture(gesture, confidence)

            # 处理轨迹记录
            if cap and ret:
                # 如果正在录制，添加轨迹点
                if trajectory_recorder.is_recording:
                    # 获取手势识别的手部数据
                    if hasattr(gesture_recognizer, 'last_hand_data'):
                        hand_data = gesture_recognizer.last_hand_data
                        trajectory_recorder.add_trajectory_point(
                            hand_data, gesture, confidence, frame.shape
                        )

                # 如果正在回放，获取回放点
                if trajectory_recorder.is_playing and not trajectory_recorder.playback_paused:
                    playback_point = trajectory_recorder.get_next_playback_point()
                    if playback_point:
                        # 这里可以添加回放点的可视化或处理
                        pass

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序错误: {e}")
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n清理资源...")
        if cap:
            cap.release()
        cv2.destroyAllWindows()

        # 程序退出语音提示
        if speech_manager.enabled:
            speech_manager.speak('program_exit', force=True, immediate=True)
            time.sleep(1)

        drone_controller.emergency_stop()

        # 自动保存轨迹
        if trajectory_recorder and len(trajectory_recorder.trajectory_data) > 0:
            print("自动保存轨迹数据...")
            trajectory_recorder.save_trajectory()

        config.save_config()

        print("程序安全退出")
        print("=" * 60)
        print("\n感谢使用手势控制无人机系统!")
        input("按回车键退出...")


# ========== 程序入口 ==========
if __name__ == "__main__":
    main()