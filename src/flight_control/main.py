import airsim
import time
import os
import signal
import threading
import sys
from pynput import keyboard

class FlightControl:
    def __init__(self):
        self.client = None
        self.SPEED = 2.0
        self.HEIGHT = -3.0
        self.is_flying = True
        self.use_gui = len(sys.argv) > 1 and sys.argv[1] == "--gui"
        self.gui = None
        self.gui_started = False
        self.current_velocity = (0, 0, 0)
        
    def setup(self):
        """设置飞行控制系统"""
        self.print_startup_info()
        self.import_gui()
        self.connect_drone()
        self.takeoff()
        self.print_control_instructions()
    
    def print_startup_info(self):
        """打印启动信息"""
        print("脚本开始运行...")
        print(f"Python 版本: {sys.version}")
        print(f"启用 GUI: {self.use_gui}")
    
    def import_gui(self):
        """导入 GUI 模块"""
        if self.use_gui:
            try:
                from gui import FlightControlGUI
                self.FlightControlGUI = FlightControlGUI
                print("成功导入 GUI 模块")
            except Exception as e:
                print(f"导入 GUI 模块失败: {e}")
                self.use_gui = False
    
    def connect_drone(self):
        """连接到无人机"""
        print("正在连接到 AirSim 模拟器...")
        try:
            self.client = airsim.MultirotorClient()
            print("已创建客户端实例")
            self.client.confirmConnection()
            print("成功连接到 AirSim 模拟器")
            self.client.enableApiControl(True)
            print("已启用 API 控制")
            self.client.armDisarm(True)
            print("已武装无人机")
        except Exception as e:
            print(f"连接失败: {e}")
            print("请确保 AirSim 模拟器已启动")
            exit(1)
    
    def takeoff(self):
        """起飞并到达指定高度"""
        print("已连接无人机")
        print("起飞中...")
        try:
            self.client.takeoffAsync().join()
            self.client.moveToZAsync(self.HEIGHT, 1.5).join()
            time.sleep(0.5)
            print("起飞完成，已到达指定高度")
        except Exception as e:
            print(f"起飞失败: {e}")
    
    def print_control_instructions(self):
        """打印控制指令"""
        print("="*60)
        print("手动控制")
        print("W 前  S 后  A 左  D 右")
        print("Z 上升  X 下降  H 悬停  B 返航")
        print("ESC 退出并降落")
        print("="*60)
    
    def on_press(self, key):
        """处理按键按下事件"""
        try:
            # 退出程序
            if key == keyboard.Key.esc:
                self.exit_program()
                return False

            # 悬停
            if hasattr(key, 'char') and key.char == 'h':
                print("悬停")
                try:
                    self.client.hoverAsync().join()
                    self.current_velocity = (0, 0, 0)
                except Exception as e:
                    print(f"悬停操作错误: {e}")

            # 返航
            if hasattr(key, 'char') and key.char == 'b':
                print("返航原点")
                try:
                    self.client.moveToPositionAsync(0, 0, self.HEIGHT, 2).join()
                    self.current_velocity = (0, 0, 0)
                except Exception as e:
                    print(f"返航操作错误: {e}")

            # 实时移动
            if hasattr(key, 'char') and key.char == 'w':
                try:
                    self.client.moveByVelocityBodyFrameAsync(self.SPEED, 0, 0, 0.05)
                    self.current_velocity = (self.SPEED, 0, 0)
                except Exception as e:
                    print(f"前进操作错误: {e}")
            if hasattr(key, 'char') and key.char == 's':
                try:
                    self.client.moveByVelocityBodyFrameAsync(-self.SPEED*0.7, 0, 0, 0.05)
                    self.current_velocity = (-self.SPEED*0.7, 0, 0)
                except Exception as e:
                    print(f"后退操作错误: {e}")
            if hasattr(key, 'char') and key.char == 'a':
                try:
                    self.client.moveByVelocityBodyFrameAsync(0, -self.SPEED, 0, 0.05)
                    self.current_velocity = (0, -self.SPEED, 0)
                except Exception as e:
                    print(f"向左操作错误: {e}")
            if hasattr(key, 'char') and key.char == 'd':
                try:
                    self.client.moveByVelocityBodyFrameAsync(0, self.SPEED, 0, 0.05)
                    self.current_velocity = (0, self.SPEED, 0)
                except Exception as e:
                    print(f"向右操作错误: {e}")

            # 高度
            if hasattr(key, 'char') and key.char == 'z':
                try:
                    self.HEIGHT -= 0.5
                    self.client.moveToZAsync(self.HEIGHT, 0.8)
                    print(f"设置高度: {abs(self.HEIGHT):.1f}m")
                except Exception as e:
                    print(f"上升操作错误: {e}")
            if hasattr(key, 'char') and key.char == 'x':
                try:
                    self.HEIGHT += 0.5
                    self.client.moveToZAsync(self.HEIGHT, 0.8)
                    print(f"设置高度: {abs(self.HEIGHT):.1f}m")
                except Exception as e:
                    print(f"下降操作错误: {e}")

        except Exception as e:
            print(f"按键处理错误: {e}")
    
    def on_release(self, key):
        """处理按键释放事件"""
        if hasattr(key, 'char') and key.char in ['w', 's', 'a', 'd']:
            try:
                self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.05)
                self.current_velocity = (0, 0, 0)
            except Exception as e:
                print(f"按键释放操作错误: {e}")
    
    def run_gui(self):
        """运行 GUI 界面"""
        try:
            print("启动可视化控制面板...")
            self.gui = self.FlightControlGUI(self.client)
            if self.gui.running:
                self.gui_started = True
                print("可视化控制面板已启动")
                self.gui.run()
            else:
                print("可视化控制面板启动失败，回退到命令行控制")
        except Exception as e:
            print(f"GUI 运行失败: {e}")
            print("回退到命令行控制")
    
    def start_gui(self):
        """启动 GUI 线程"""
        if self.use_gui:
            gui_thread = threading.Thread(target=self.run_gui)
            gui_thread.daemon = False  # 设置为非守护线程，确保 GUI 能够正常运行
            gui_thread.start()
            
            # 等待 GUI 启动
            time.sleep(2)
            if not self.gui_started:
                print("GUI 启动失败，继续使用命令行控制")
        else:
            print("未启用可视化控制面板，仅使用命令行控制")
    
    def exit_program(self):
        """安全退出程序"""
        print("\n安全降落...")
        try:
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("无人机已安全降落")
        except Exception as e:
            print(f"降落过程中发生错误: {e}")
        finally:
            if self.gui:
                self.gui.stop()
            os.kill(os.getpid(), signal.SIGTERM)
    
    def run(self):
        """运行主程序"""
        # 启动 GUI
        self.start_gui()
        
        # 启动键盘监听
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        # 保持程序运行
        print("程序已启动，按 ESC 键退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n收到中断信号，正在降落...")
            self.exit_program()
        finally:
            listener.join()

if __name__ == "__main__":
    flight_control = FlightControl()
    flight_control.setup()
    flight_control.run()