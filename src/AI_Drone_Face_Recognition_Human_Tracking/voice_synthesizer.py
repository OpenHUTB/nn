# 语音合成模块
import os
import threading
import time
import queue
import sys
import subprocess


class VoiceSynthesizer:
    def __init__(self, voice_language='zh-cn', volume=1.0, rate=150):
        """初始化语音合成器"""
        print("🔊 初始化语音合成模块...")

        self.voice_language = voice_language
        self.volume = volume
        self.rate = rate
        self.enabled = True
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.worker_thread = None
        self.audio_files = []

        # 引擎优先级：系统 > pyttsx3 > edge_tts > gtts（网络依赖越高的优先级越低）
        self.engine_priority = ['system', 'pyttsx3', 'edge_tts', 'gtts']

        # 检查可用的语音引擎
        self.available_engines = self._detect_engines()

        if not self.available_engines:
            print("⚠️  未找到语音合成引擎，语音功能将不可用")
            self.enabled = False
            return

        print(f"✅ 语音合成器初始化完成，可用引擎: {', '.join(self.available_engines.keys())}")

        # 启动语音工作线程
        self._start_worker()

    @staticmethod
    def _detect_engines():
        """检测可用的语音合成引擎"""
        engines = {}

        # 1. 首先检查系统语音（最高优先级，不需要网络）
        try:
            import platform
            system = platform.system()

            if system == 'Windows':
                try:
                    import win32com.client
                    win32com.client.Dispatch("SAPI.SpVoice")
                    engines['system'] = {
                        'name': 'Windows 系统语音',
                        'module': 'win32com',
                        'supported_langs': ['zh', 'en'],
                        'network_required': False,
                        'priority': 1
                    }
                    print("✅ 检测到 Windows 系统语音引擎")
                except ImportError:
                    print("⚠️  win32com 未安装，Windows 系统语音不可用")

            elif system == 'Darwin':
                # 检查 macOS 的 say 命令
                try:
                    result = subprocess.run(['which', 'say'], capture_output=True, text=True)
                    if result.returncode == 0:
                        engines['system'] = {
                            'name': 'macOS 系统语音',
                            'module': 'say',
                            'supported_langs': ['zh', 'en'],
                            'network_required': False,
                            'priority': 1
                        }
                        print("✅ 检测到 macOS 系统语音引擎")
                except Exception:
                    print("⚠️  macOS 系统语音检查失败")

            elif system == 'Linux':
                # 检查 Linux 的 espeak 命令
                try:
                    result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
                    if result.returncode == 0:
                        engines['system'] = {
                            'name': 'Linux 系统语音',
                            'module': 'espeak',
                            'supported_langs': ['zh', 'en'],
                            'network_required': False,
                            'priority': 1
                        }
                        print("✅ 检测到 Linux 系统语音引擎")
                except Exception:
                    print("⚠️  Linux 系统语音检查失败")

        except Exception as e:
            print(f"⚠️  系统语音检测错误: {e}")

        # 2. 尝试pyttsx3 (离线引擎，不需要网络)
        try:
            import pyttsx3
            # 测试是否能初始化
            engine = pyttsx3.init()
            engine.stop()
            engines['pyttsx3'] = {
                'name': 'pyttsx3 (离线)',
                'module': 'pyttsx3',
                'supported_langs': ['zh', 'en'],
                'network_required': False,
                'priority': 2
            }
            print("✅ 检测到 pyttsx3 引擎 (离线)")
        except ImportError:
            print("⚠️  pyttsx3 未安装")
        except Exception as e:
            print(f"⚠️  pyttsx3 初始化失败: {e}")

        # 3. 尝试edge-tts (需要网络)
        try:
            import edge_tts
            engines['edge_tts'] = {
                'name': 'Edge TTS',
                'module': 'edge_tts',
                'supported_langs': ['zh-CN', 'en-US', 'ja-JP'],
                'network_required': True,
                'priority': 3
            }
            print("✅ 检测到 Edge TTS 引擎 (需要网络)")
        except ImportError:
            print("⚠️  edge-tts 未安装")

        # 4. 最后尝试gTTS (需要网络，且容易连接失败)
        try:
            from gtts import gTTS
            engines['gtts'] = {
                'name': 'Google TTS',
                'module': 'gtts',
                'supported_langs': ['zh-cn', 'en', 'ja', 'ko', 'fr', 'de', 'es'],
                'network_required': True,
                'priority': 4  # 最低优先级，因为网络问题最多
            }
            print("✅ 检测到 gTTS 引擎 (需要网络)")
        except ImportError:
            print("⚠️  gTTS 未安装")

        # 按优先级排序引擎
        sorted_engines = {}
        for engine_type in sorted(engines.keys(), key=lambda x: engines[x].get('priority', 99)):
            sorted_engines[engine_type] = engines[engine_type]

        return sorted_engines

    def _start_worker(self):
        """启动语音工作线程"""
        if not self.enabled:
            return

        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
        print("✅ 语音工作线程已启动")

    def _speech_worker(self):
        """语音工作线程"""
        while self.enabled:
            try:
                # 从队列获取语音任务
                text, engine_type = self.speech_queue.get(timeout=1)

                if text:
                    self.is_speaking = True
                    print(f"🔊 正在播放: {text}")

                    # 根据选择的引擎播放语音
                    success = self._speak_with_engine(text, engine_type)

                    if success:
                        print("✅ 语音播放完成")
                    else:
                        print("❌ 语音播放失败，尝试备用引擎...")
                        # 失败后尝试其他可用引擎
                        backup_success = self._try_all_engines(text, exclude=engine_type)
                        if backup_success:
                            print("✅ 备用引擎播放成功")

                    self.is_speaking = False
                    self.speech_queue.task_done()

            except queue.Empty:
                continue
            except (OSError, RuntimeError, ImportError) as e:
                print(f"❌ 语音工作线程错误: {e}")
                self.is_speaking = False
            except Exception as e:
                print(f"❌ 语音工作线程未知错误: {e}")
                self.is_speaking = False

    def _speak_with_engine(self, text, engine_type='auto'):
        """使用指定的引擎播放语音"""
        if not self.enabled or not text:
            return False

        if engine_type == 'auto':
            # 自动选择：按优先级尝试可用引擎
            return self._try_all_engines(text)
        else:
            # 使用指定引擎
            return self._try_speak_with_engine(text, engine_type)

    def _try_all_engines(self, text, exclude=None):
        """尝试所有可用引擎（按优先级顺序）"""
        for engine_type in self.available_engines.keys():
            if exclude and engine_type == exclude:
                continue
            if self._try_speak_with_engine(text, engine_type):
                print(f"✅ 使用 {engine_type} 引擎成功")
                return True
        print("❌ 所有引擎尝试失败")
        return False

    def _try_speak_with_engine(self, text, engine_type):
        """尝试使用特定引擎播放语音"""
        try:
            if engine_type == 'gtts' and 'gtts' in self.available_engines:
                return self._speak_gtts(text)

            elif engine_type == 'pyttsx3' and 'pyttsx3' in self.available_engines:
                return self._speak_pyttsx3(text)

            elif engine_type == 'edge_tts' and 'edge_tts' in self.available_engines:
                return self._speak_edge_tts(text)

            elif engine_type == 'system' and 'system' in self.available_engines:
                return self._speak_system(text)

            else:
                print(f"❌ 引擎 {engine_type} 不可用")
                return False

        except ImportError as import_err:
            print(f"❌ {engine_type} 引擎导入错误: {import_err}")
            return False
        except (OSError, IOError) as io_err:
            print(f"❌ {engine_type} 引擎IO错误: {io_err}")
            return False
        except Exception as other_err:
            print(f"❌ {engine_type} 引擎错误: {other_err}")
            return False

    def _speak_gtts(self, text):
        """使用gTTS播放语音"""
        try:
            from gtts import gTTS
            import tempfile

            # 检查网络连接
            try:
                import socket
                socket.create_connection(("www.google.com", 80), timeout=3)
            except (socket.timeout, socket.error):
                print("⚠️  gTTS: 网络连接不可用，跳过此引擎")
                return False

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_file = f.name

            try:
                # 生成语音文件（增加超时时间）
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry

                # 配置重试策略
                session = requests.Session()
                retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
                adapter = HTTPAdapter(max_retries=retry)
                session.mount('http://', adapter)
                session.mount('https://', adapter)

                # 保存原有requests会话
                import gtts.tokenizer
                original_session = gtts.tokenizer.requests_session
                gtts.tokenizer.requests_session = session

                tts = gTTS(text=text, lang=self.voice_language, slow=False, timeout=10)
                tts.save(temp_file)

                # 恢复原有会话
                gtts.tokenizer.requests_session = original_session

                # 播放语音
                return self._play_audio_file(temp_file, engine='gtts')

            except Exception as tts_err:
                print(f"❌ gTTS生成错误: {tts_err}")
                # 清理临时文件
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return False

        except ImportError as import_err:
            print(f"❌ gTTS导入错误: {import_err}")
            return False
        except Exception as other_err:
            print(f"❌ gTTS未知错误: {other_err}")
            return False

    def _speak_pyttsx3(self, text):
        """使用pyttsx3播放语音"""
        try:
            import pyttsx3

            # 初始化引擎
            engine = pyttsx3.init()

            # 设置属性
            engine.setProperty('rate', self.rate)  # 语速
            engine.setProperty('volume', self.volume)  # 音量

            # 设置语言
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
                elif 'english' in voice.name.lower() or 'en' in voice.id.lower():
                    engine.setProperty('voice', voice.id)

            # 播放语音
            engine.say(text)
            engine.runAndWait()
            engine.stop()

            return True

        except ImportError as import_err:
            print(f"❌ pyttsx3导入错误: {import_err}")
            return False
        except RuntimeError as runtime_err:
            print(f"❌ pyttsx3运行时错误: {runtime_err}")
            return False
        except Exception as other_err:
            print(f"❌ pyttsx3未知错误: {other_err}")
            return False

    def _speak_edge_tts(self, text):
        """使用edge-tts播放语音"""
        try:
            import edge_tts
            import asyncio
            import tempfile

            # 检查网络连接
            try:
                import socket
                socket.create_connection(("www.microsoft.com", 80), timeout=3)
            except (socket.timeout, socket.error):
                print("⚠️  edge-tts: 网络连接不可用，跳过此引擎")
                return False

            # 选择语音
            if self.voice_language == 'zh-cn':
                voice = 'zh-CN-XiaoxiaoNeural'
            elif self.voice_language == 'en':
                voice = 'en-US-AriaNeural'
            else:
                voice = 'zh-CN-XiaoxiaoNeural'

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_file = f.name

            try:
                # 异步生成语音（增加超时）
                async def generate_speech():
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(temp_file)

                # 设置超时
                try:
                    asyncio.run(asyncio.wait_for(generate_speech(), timeout=15))
                except asyncio.TimeoutError:
                    print("❌ edge-tts: 请求超时")
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                    return False

                # 播放语音
                return self._play_audio_file(temp_file, engine='edge_tts')

            except Exception as edge_err:
                print(f"❌ edge-tts生成错误: {edge_err}")
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return False

        except ImportError as import_err:
            print(f"❌ edge-tts导入错误: {import_err}")
            return False
        except Exception as other_err:
            print(f"❌ edge-tts未知错误: {other_err}")
            return False

    @staticmethod
    def _speak_system(text):
        """使用系统语音"""
        try:
            import platform
            import subprocess

            system = platform.system()

            if system == 'Windows':
                # Windows
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Speak(text)
                return True

            elif system == 'Darwin':
                # macOS
                result = subprocess.run(['say', text], capture_output=True, text=True)
                return result.returncode == 0

            elif system == 'Linux':
                # Linux
                result = subprocess.run(['espeak', text], capture_output=True, text=True)
                return result.returncode == 0

            else:
                print(f"❌ 不支持的系统: {system}")
                return False

        except ImportError as import_err:
            print(f"❌ 系统语音模块导入错误: {import_err}")
            return False
        except (OSError, subprocess.SubprocessError) as proc_err:
            print(f"❌ 系统语音执行错误: {proc_err}")
            return False
        except Exception as other_err:
            print(f"❌ 系统语音未知错误: {other_err}")
            return False

    def _play_audio_file(self, audio_file, engine=''):
        """播放音频文件（通用方法）"""
        try:
            # 尝试多个播放方法
            playback_methods = [
                self._play_with_playsound,
                self._play_with_pydub,
                self._play_with_system
            ]

            for method in playback_methods:
                if method(audio_file):
                    # 记录文件（用于调试）
                    self.audio_files.append(audio_file)
                    if len(self.audio_files) > 10:
                        old_file = self.audio_files.pop(0)
                        if os.path.exists(old_file):
                            os.unlink(old_file)
                    return True

            print(f"❌ {engine}: 所有播放方法都失败")
            return False

        except Exception as e:
            print(f"❌ {engine}: 播放错误: {e}")
            return False
        finally:
            # 清理临时文件（如果还在）
            if os.path.exists(audio_file) and audio_file not in self.audio_files:
                try:
                    os.unlink(audio_file)
                except:
                    pass

    def _play_with_playsound(self, audio_file):
        """使用playsound播放"""
        try:
            from playsound import playsound
            playsound(audio_file)
            return True
        except ImportError:
            print("⚠️  playsound 未安装")
            return False
        except Exception as e:
            print(f"⚠️  playsound 播放失败: {e}")
            return False

    def _play_with_pydub(self, audio_file):
        """使用pydub播放"""
        try:
            from pydub import AudioSegment
            from pydub.playback import play

            audio = AudioSegment.from_file(audio_file)
            play(audio)
            return True
        except ImportError:
            print("⚠️  pydub 未安装")
            return False
        except Exception as e:
            print(f"⚠️  pydub 播放失败: {e}")
            return False

    def _play_with_system(self, audio_file):
        """使用系统命令播放"""
        try:
            import platform
            import subprocess

            system = platform.system()

            if system == 'Windows':
                os.startfile(audio_file)
                return True
            elif system == 'Darwin':
                subprocess.call(['afplay', audio_file])
                return True
            elif system == 'Linux':
                subprocess.call(['aplay', audio_file])
                return True
            else:
                return False

        except Exception as e:
            print(f"⚠️  系统命令播放失败: {e}")
            return False

    def speak(self, text, engine_type='auto', blocking=False):
        """播放语音

        Args:
            text: 要播放的文本
            engine_type: 语音引擎类型 ('auto', 'gtts', 'pyttsx3', 'edge_tts', 'system')
            blocking: 是否阻塞直到播放完成
        """
        if not self.enabled or not text:
            print("⚠️  语音合成器未启用或文本为空")
            return False

        if not self.available_engines:
            print("⚠️  没有可用的语音引擎")
            return False

        print(f"🔊 准备播放: {text}")

        if blocking:
            # 阻塞模式，直接播放
            return self._speak_with_engine(text, engine_type)
        else:
            # 非阻塞模式，加入队列
            self.speech_queue.put((text, engine_type))
            return True

    def speak_async(self, text, engine_type='auto'):
        """异步播放语音（非阻塞）"""
        return self.speak(text, engine_type, blocking=False)

    def speak_sync(self, text, engine_type='auto'):
        """同步播放语音（阻塞直到完成）"""
        return self.speak(text, engine_type, blocking=True)

    def speak_detection_result(self, count, person_type='person'):
        """播放检测结果"""
        if person_type == 'person':
            if count == 0:
                text = "未检测到人物"
            elif count == 1:
                text = "检测到一个人物"
            else:
                text = f"检测到{count}个人物"
        else:
            if count == 0:
                text = "未检测到人脸"
            elif count == 1:
                text = "检测到一张人脸"
            else:
                text = f"检测到{count}张人脸"

        return self.speak_async(text)

    def speak_recognition_result(self, name):
        """播放识别结果"""
        if name == "Unknown":
            text = "未识别到人脸"
        else:
            text = f"识别到 {name}"

        return self.speak_async(text)

    def speak_drone_status(self, status, action=None):
        """播放无人机状态"""
        if action == 'takeoff':
            text = "无人机起飞"
        elif action == 'land':
            text = "无人机降落"
        elif action == 'hover':
            text = "无人机悬停"
        elif action == 'connected':
            text = "无人机已连接"
        elif action == 'disconnected':
            text = "无人机已断开连接"
        elif status == 'tracking':
            text = "开始跟踪目标"
        elif status == 'lost':
            text = "丢失跟踪目标"
        else:
            text = f"无人机状态: {status}"

        return self.speak_async(text)

    def stop(self):
        """停止语音合成器"""
        print("🛑 停止语音合成器...")
        self.enabled = False

        # 清空队列
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break

        # 清理临时文件
        for audio_file in self.audio_files:
            try:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
            except:
                pass
        self.audio_files.clear()

        # 等待工作线程结束
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)

        print("✅ 语音合成器已停止")

    def set_language(self, language):
        """设置语音语言"""
        self.voice_language = language
        print(f"🔤 语音语言设置为: {language}")

    def set_volume(self, volume):
        """设置音量 (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        print(f"🔊 音量设置为: {self.volume}")

    def set_rate(self, rate):
        """设置语速 (words per minute)"""
        self.rate = max(50, min(300, rate))
        print(f"⚡ 语速设置为: {self.rate}")

    def get_status(self):
        """获取语音合成器状态"""
        engine_details = {}
        for name, info in self.available_engines.items():
            engine_details[name] = {
                'name': info['name'],
                'network_required': info.get('network_required', True)
            }

        return {
            'enabled': self.enabled,
            'is_speaking': self.is_speaking,
            'queue_size': self.speech_queue.qsize(),
            'available_engines': engine_details,
            'language': self.voice_language,
            'volume': self.volume,
            'rate': self.rate,
            'recommended_engine': self.get_recommended_engine()
        }

    def get_recommended_engine(self):
        """获取推荐的引擎（优先不需要网络的）"""
        for engine_type, info in self.available_engines.items():
            if not info.get('network_required', True):
                return engine_type
        # 如果没有离线引擎，返回第一个可用的
        return list(self.available_engines.keys())[0] if self.available_engines else None


def check_dependencies():
    """检查依赖"""
    print("🔍 检查语音合成依赖...")

    dependencies = [
        ('pyttsx3', '离线语音合成 (推荐)'),
        ('gtts', 'Google TTS (需要网络)'),
        ('playsound', '音频播放'),
        ('edge-tts', '微软Edge TTS (需要网络)'),
        ('pydub', '备用音频播放')
    ]

    missing = []
    recommended = []

    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} ({module}): 已安装")
            if module == 'pyttsx3':
                recommended.append('pyttsx3 (离线，推荐)')
        except ImportError:
            print(f"❌ {name} ({module}): 未安装")
            missing.append(module)

    if missing:
        print(f"\n💡 安装缺失的依赖:")
        print(f"   pip install {' '.join(missing)}")

    if recommended:
        print(f"\n💡 推荐使用的引擎: {', '.join(recommended)}")
    else:
        print("\n⚠️  没有找到离线引擎，需要网络连接才能使用语音功能")

    return len(missing) == 0


# 测试函数 - 修复版
def test_voice_synthesizer():
    """测试语音合成器"""
    print("🔊 测试语音合成模块")
    print("=" * 50)

    # 创建语音合成器
    print("🔄 创建语音合成器...")
    voice = VoiceSynthesizer(voice_language='zh-cn')

    if not voice.enabled:
        print("❌ 语音合成器不可用，检查依赖安装")
        print("💡 请运行以下命令安装推荐依赖:")
        print("   pip install pyttsx3")
        return

    # 显示状态
    status = voice.get_status()
    print(f"✅ 语音合成器状态:")
    print(f"   可用引擎: {len(status['available_engines'])} 个")
    for eng_name, eng_info in status['available_engines'].items():
        network = "需要网络" if eng_info['network_required'] else "离线可用"
        print(f"     - {eng_info['name']} ({network})")

    print(f"   推荐引擎: {status['recommended_engine']}")
    print(f"   语言: {status['language']}")
    print(f"   音量: {status['volume']}")
    print(f"   语速: {status['rate']}")

    # 测试语音
    test_phrases = [
        "你好，欢迎使用AI无人机系统",
        "系统初始化完成",
        "检测到三个人物",
        "识别到张三",
        "无人机已连接",
        "开始跟踪目标"
    ]

    print("\n🎤 测试语音播放:")
    print("   按 Ctrl+C 中断测试")
    print("-" * 40)

    try:
        # 1. 测试推荐引擎
        print(f"1. 使用推荐引擎 ({status['recommended_engine']})...")
        for i, phrase in enumerate(test_phrases[:2], 1):
            print(f"   {i}. {phrase}")
            success = voice.speak_sync(phrase, engine_type=status['recommended_engine'])
            if success:
                print("      ✅ 播放成功")
            else:
                print("      ❌ 播放失败")
            time.sleep(1)

        # 2. 测试自动选择
        print("\n2. 测试自动引擎选择...")
        for i, phrase in enumerate(test_phrases[2:4], 1):
            print(f"   {i}. {phrase}")
            success = voice.speak_sync(phrase, engine_type='auto')
            if success:
                print(f"      ✅ 播放成功")
            else:
                print(f"      ❌ 播放失败，请检查网络连接")
            time.sleep(1)

        # 3. 测试专用函数
        print("\n3. 测试专用语音函数...")
        print("   • 检测结果...")
        voice.speak_detection_result(3)
        time.sleep(2)

        print("   • 识别结果...")
        voice.speak_recognition_result("李四")
        time.sleep(2)

        print("   • 无人机状态...")
        voice.speak_drone_status('tracking')
        time.sleep(2)

        # 4. 测试所有可用引擎
        print("\n4. 测试所有可用引擎...")
        available_engines = list(status['available_engines'].keys())
        for engine in available_engines:
            eng_info = status['available_engines'][engine]
            print(f"   • 测试 {eng_info['name']}...")
            phrase = f"这是{engine}引擎测试"
            success = voice.speak_sync(phrase, engine_type=engine)
            if success:
                print(f"      ✅ 播放成功")
            else:
                print(f"      ❌ 播放失败")
            time.sleep(1)

        print("\n✅ 语音测试完成!")

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        voice.stop()
        print("✅ 语音合成器已清理")


# 简单使用示例
def quick_example():
    """快速使用示例"""
    print("🚀 语音合成器快速示例")

    # 创建语音合成器
    voice = VoiceSynthesizer()

    if voice.enabled:
        # 获取推荐引擎
        status = voice.get_status()
        recommended = status['recommended_engine']
        print(f"💡 使用推荐引擎: {recommended}")

        # 播放欢迎语音
        voice.speak_sync("AI无人机系统已就绪", engine_type=recommended)

        # 播放检测结果
        voice.speak_detection_result(2)

        # 播放识别结果
        voice.speak_recognition_result("张三")

        # 清理
        voice.stop()
    else:
        print("❌ 语音功能不可用，请安装依赖")
        print("💡 推荐安装: pip install pyttsx3")


if __name__ == "__main__":
    print("🎤 语音合成模块测试")
    print("=" * 50)

    print("\n选择测试模式:")
    print("1. 🔧 完整测试（推荐）")
    print("2. 🚀 快速示例")
    print("3. 📋 检查依赖")
    print("4. ❌ 退出")

    try:
        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == "1":
            test_voice_synthesizer()
        elif choice == "2":
            quick_example()
        elif choice == "3":
            check_dependencies()
        elif choice == "4":
            print("👋 退出")
        else:
            print("⚠️  无效选择，运行完整测试")
            test_voice_synthesizer()

    except KeyboardInterrupt:
        print("\n👋 用户中断")
    except Exception as e:
        print(f"❌ 运行出错: {e}")