import pyttsx3
import platform


class VoiceAssistant:
    """语音助手类（增强兼容性+详细日志）"""

    def __init__(self):
        self.engine = None
        self._init_engine()  # 初始化语音引擎

    def _init_engine(self):
        """初始化语音引擎，兼容不同系统"""
        try:
            self.engine = pyttsx3.init()
            # 基础配置
            self.engine.setProperty("rate", 150)  # 语速（100-200为宜）
            self.engine.setProperty("volume", 1.0)  # 音量（0.0-1.0）

            # 系统适配：Windows优先使用中文语音包
            system = platform.system()
            if system == "Windows":
                self._set_chinese_voice()

            print("✅ 语音引擎初始化成功！")
        except Exception as e:
            print(f"❌ 语音引擎初始化失败: {str(e)}")
            self.engine = None

    def _set_chinese_voice(self):
        """Windows系统设置中文语音包（需提前安装）"""
        try:
            voices = self.engine.getProperty('voices')
            # 遍历语音包，选择中文语音
            for voice in voices:
                if "zh-CN" in voice.id or "中文" in voice.name:
                    self.engine.setProperty('voice', voice.id)
                    print(f"✅ 已切换为中文语音包: {voice.name}")
                    break
            else:
                print("⚠️ 未找到中文语音包，使用默认语音（可能为英文）")
        except Exception as e:
            print(f"⚠️ 设置中文语音包失败: {str(e)}")

    def speak(self, text):
        """文本转语音（增加容错）"""
        if not self.engine:
            print("❌ 语音引擎未初始化，无法合成语音")
            return

        if not text.strip():
            print("❌ 合成文本不能为空！")
            return

        try:
            print(f"🔊 正在合成语音: {text}")
            self.engine.say(text)
            self.engine.runAndWait()  # 等待语音播放完成
            print("✅ 语音合成完成！")
        except Exception as e:
            print(f"❌ 语音合成失败: {str(e)}")

    def list_voices(self):
        """列出当前系统可用的语音包（调试用）"""
        if not self.engine:
            print("❌ 语音引擎未初始化，无法获取语音包")
            return []

        try:
            voices = self.engine.getProperty('voices')
            print("\n📋 系统可用语音包列表：")
            for idx, voice in enumerate(voices):
                print(f"  [{idx}] 名称: {voice.name} | ID: {voice.id} | 语言: {voice.languages}")
            return voices
        except Exception as e:
            print(f"❌ 获取语音包列表失败: {str(e)}")
            return []

    def adjust_settings(self, rate=None, volume=None):
        """调整语速/音量（调试用）"""
        if not self.engine:
            print("❌ 语音引擎未初始化，无法调整设置")
            return

        if rate is not None:
            if 50 <= rate <= 300:
                self.engine.setProperty("rate", rate)
                print(f"✅ 语速已调整为: {rate}")
            else:
                print("⚠️ 语速范围需在50-300之间！")

        if volume is not None:
            if 0.0 <= volume <= 1.0:
                self.engine.setProperty("volume", volume)
                print(f"✅ 音量已调整为: {volume}")
            else:
                print("⚠️ 音量范围需在0.0-1.0之间！")

    def __del__(self):
        """析构函数，释放引擎资源"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass


# ===================== 独立运行测试逻辑 =====================
if __name__ == "__main__":
    # 初始化语音助手
    voice_assist = VoiceAssistant()

    # 打印系统信息
    print("\n" + "=" * 50)
    print("🎯 语音助手测试工具（独立运行模式）")
    print(f"💻 当前系统: {platform.system()}")
    print("=" * 50)

    # 列出可用语音包
    voice_assist.list_voices()

    # 交互菜单
    print("\n📢 操作菜单：")
    print("  1 → 播放测试文本（中文）")
    print("  2 → 播放测试文本（英文）")
    print("  3 → 自定义文本播放")
    print("  4 → 调整语速（默认150）")
    print("  5 → 调整音量（默认1.0）")
    print("  q → 退出程序")
    print("-" * 30)

    while True:
        choice = input("\n请输入操作编号: ").strip()

        if choice == "q":
            print("🔚 退出语音助手测试程序...")
            break

        elif choice == "1":
            # 中文测试
            voice_assist.speak("你好，我是AI无人机语音助手，测试语音合成功能正常！")

        elif choice == "2":
            # 英文测试
            voice_assist.speak("Hello, I am the AI drone voice assistant, test speech synthesis function is normal!")

        elif choice == "3":
            # 自定义文本
            custom_text = input("请输入要合成的文本: ").strip()
            if custom_text:
                voice_assist.speak(custom_text)
            else:
                print("❌ 自定义文本不能为空！")

        elif choice == "4":
            # 调整语速
            try:
                new_rate = int(input("请输入新语速（50-300）: ").strip())
                voice_assist.adjust_settings(rate=new_rate)
            except ValueError:
                print("❌ 请输入有效的数字！")

        elif choice == "5":
            # 调整音量
            try:
                new_volume = float(input("请输入新音量（0.0-1.0）: ").strip())
                voice_assist.adjust_settings(volume=new_volume)
            except ValueError:
                print("❌ 请输入有效的数字（如0.5）！")

        else:
            print("❌ 无效的操作编号，请重新输入！")