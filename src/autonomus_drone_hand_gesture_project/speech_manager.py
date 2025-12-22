"""
语音反馈管理器模块
负责语音提示和音频播放
作者: xiaoshiyuan888
"""

import threading
import time
import os
import sys
import tempfile


class EnhancedSpeechFeedbackManager:
    """增强的语音反馈管理器"""

    def __init__(self, speech_lib, config=None):
        self.speech_lib = speech_lib
        self.config = config
        self.enabled = True
        self.volume = 1.0
        self.rate = 150
        self.voice_id = None
        self.last_speech_time = {}
        self.min_interval = 1.5

        # 语音队列
        self.speech_queue = []
        self.is_speaking = False
        self.queue_thread = None

        # 音频播放方法
        self.audio_method = None

        # 手势状态追踪
        self.last_gesture_state = "none"
        self.gesture_active_time = 0

        # 语音消息映射
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

            # 高级飞行模式
            'returning_home': "正在返航",
            'return_home_success': "返航成功",
            'auto_flight_start': "开始自动飞行模式",
            'auto_flight_complete': "自动飞行完成",
            'circle_flight_start': "开始圆形盘旋",
            'circle_flight_complete': "圆形盘旋完成",
            'eight_flight_start': "开始8字形飞行",
            'eight_flight_complete': "8字形飞行完成",
            'altitude_increasing': "正在增加高度",
            'altitude_decreasing': "正在降低高度",

            # 手势相关
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
            'gesture_grab': "抓取",  # 新增
            'gesture_release': "释放",  # 新增
            'gesture_rotate_cw': "顺时针旋转",  # 新增
            'gesture_rotate_ccw': "逆时针旋转",  # 新增
            'gesture_photo': "拍照",  # 新增
            'gesture_auto_flight': "自动飞行模式",  # 新增
            'gesture_return_home': "返航",  # 新增
            'gesture_circle_flight': "圆形盘旋",  # 新增
            'gesture_eight_flight': "8字形飞行",  # 新增
            'gesture_square_flight': "方形轨迹",  # 新增
            'gesture_increase_altitude': "增加高度",  # 新增
            'gesture_decrease_altitude': "降低高度",  # 新增
            'gesture_set_altitude': "设置高度",  # 新增

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

            # 性能相关
            'performance_good': "系统运行流畅",
            'performance_warning': "系统性能警告",
            'performance_critical': "系统性能严重警告",
            'performance_report': "性能报告生成完成",
            'performance_snapshot': "性能快照已保存",
            'performance_log_exported': "性能日志已导出",

            # 手势指导
            'move_closer': "请将手靠近摄像头",
            'move_away': "请将手移远一些",
            'good_position': "手部位置良好",
            'hand_detected': "手部已检测到",
            'hand_lost': "手部丢失，请重新放置",

            # 录制相关
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
        """播放音频文件"""
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
                time.sleep(1.5)

            elif self.audio_method == 'gtts_posix':
                # Linux/Mac系统命令
                import subprocess
                if sys.platform == 'darwin':
                    subprocess.call(['afplay', audio_file])
                else:
                    subprocess.call(['xdg-open', audio_file])

            else:
                # 通用方法
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
            text = message_key

        # 立即播放或加入队列
        if immediate:
            self.speak_direct(text)
        else:
            self.speech_queue.append(text)

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
                    self.engine.stop()
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
                print(f"⚠ 语音播放失败: {e}")

            time.sleep(0.05)

        self.is_speaking = False
        self.queue_thread = None

    def speak_direct(self, text):
        """直接播放文本"""
        if not self.enabled:
            return

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