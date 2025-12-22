# modules/voice_synthesizer.py
import threading
import queue
import time
import os


class VoiceSynthesizer:
    def __init__(self, enabled=True, lang="zh-CN"):
        self.enabled = enabled
        self.lang = lang
        self.message_queue = queue.Queue()
        self.currently_speaking = False
        self.voice_engine = None
        self.worker_thread = None
        self.running = False

        if self.enabled:
            self._init_voice_engine()
            self.start()

    def _init_voice_engine(self):
        """初始化语音引擎"""
        try:
            import pyttsx3
            self.voice_engine = pyttsx3.init()

            # 设置语音参数
            self.voice_engine.setProperty('rate', 150)  # 语速
            self.voice_engine.setProperty('volume', 0.8)  # 音量

            # 尝试设置中文语音
            voices = self.voice_engine.getProperty('voices')
            for voice in voices:
                if "zh" in voice.language or "chinese" in voice.name.lower():
                    self.voice_engine.setProperty('voice', voice.id)
                    break

            print("✅ 语音引擎初始化成功")
            return True
        except ImportError:
            print("⚠️  pyttsx3未安装，语音功能不可用")
            print("   请运行: pip install pyttsx3")
        except Exception as e:
            print(f"⚠️  语音引擎初始化失败: {e}")

        self.enabled = False
        return False

    def speak(self, text, priority=1):
        """语音播报"""
        if not self.enabled or not text:
            return False

        try:
            self.message_queue.put((text, priority, time.time()))
            return True
        except:
            return False

    def _voice_worker(self):
        """语音工作线程"""
        while self.running:
            try:
                # 从队列获取消息（最多等待1秒）
                text, priority, timestamp = self.message_queue.get(timeout=1)

                if self.voice_engine:
                    self.currently_speaking = True
                    self.voice_engine.say(text)
                    self.voice_engine.runAndWait()
                    self.currently_speaking = False

                # 标记任务完成
                self.message_queue.task_done()

                # 短暂暂停，避免连续播报
                time.sleep(0.5)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"语音播报错误: {e}")
                self.currently_speaking = False
                time.sleep(1)

    def start(self):
        """启动语音服务"""
        if not self.enabled or self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.worker_thread.start()
        print("✅ 语音服务已启动")

    def stop(self):
        """停止语音服务"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        if self.voice_engine:
            self.voice_engine.stop()
        print("✅ 语音服务已停止")

    def speak_detection_result(self, face_count, person_count):
        """播报检测结果"""
        if face_count == 0 and person_count == 0:
            text = "未检测到目标"
        else:
            text = f"检测到{face_count}个人脸，{person_count}个行人"
        return self.speak(text)

    def speak_recognition_result(self, name):
        """播报识别结果"""
        if name == "Unknown":
            text = "未识别到已知人员"
        else:
            text = f"识别到{name}"
        return self.speak(text)

    def speak_drone_status(self, status, is_flying=False):
        """播报无人机状态"""
        if is_flying:
            text = f"无人机正在飞行，状态：{status}"
        else:
            text = f"无人机状态：{status}"
        return self.speak(text)

    def speak_tracking_status(self, tracking_mode, target_count=0):
        """播报追踪状态"""
        if tracking_mode == "追踪":
            text = f"追踪模式已开启，正在追踪{target_count}个目标"
        else:
            text = "手动控制模式"
        return self.speak(text)

    def speak_system_status(self, status):
        """播报系统状态"""
        text = f"系统状态：{status}"
        return self.speak(text)

    def is_speaking(self):
        """是否正在播报"""
        return self.currently_speaking

    def get_queue_size(self):
        """获取消息队列大小"""
        return self.message_queue.qsize()

    def clear_queue(self):
        """清空消息队列"""
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except:
                break