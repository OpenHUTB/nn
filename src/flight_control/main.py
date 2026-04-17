import airsim
import time
import os
import signal
import threading
import sys
import json
from pynput import keyboard

class FlightControl:
    def __init__(self):
        self.client = None
        self.SPEED = 2.0
        self.HEIGHT = -3.0
        self.is_flying = True
        # 更灵活的命令行参数解析
        self.use_gui = "--gui" in sys.argv
        self.use_map = "--map" in sys.argv
        self.gui = None
        self.map_display = None
        self.gui_started = False
        self.map_started = False
        self.current_velocity = (0, 0, 0)
        self.waypoints = []
        self.is_cruising = False
        # 加载配置文件
        self.load_config()
        
    def setup(self):
        """设置飞行控制系统"""
        self.print_startup_info()
        self.import_gui()
        self.import_map()
        connected = self.connect_drone()
        if connected:
            self.takeoff()
        self.print_control_instructions()
    
    def load_config(self):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 加载基本配置
                if 'speed' in config:
                    self.SPEED = config['speed']
                if 'default_height' in config:
                    self.HEIGHT = config['default_height']
                # 加载巡航配置
                if 'cruise' in config:
                    cruise_config = config['cruise']
                    if 'speed' in cruise_config:
                        self.SPEED = cruise_config['speed']
                    if 'waypoints' in cruise_config:
                        self.waypoints = [tuple(wp) for wp in cruise_config['waypoints']]
                print(f"成功加载配置文件: {config_path}")
                print(f"默认飞行速度: {self.SPEED} m/s")
                print(f"默认飞行高度: {abs(self.HEIGHT)} m")
                if self.waypoints:
                    print(f"加载了 {len(self.waypoints)} 个默认航点")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
        else:
            print(f"配置文件不存在: {config_path}")
    
    def print_startup_info(self):
        """打印启动信息"""
        print("脚本开始运行...")
        print(f"Python 版本: {sys.version}")
        print(f"启用 GUI: {self.use_gui}")
        print(f"启用地图显示: {self.use_map}")
        print("使用说明:")
        print("  python main.py              - 仅使用命令行控制")
        print("  python main.py --gui        - 启用 GUI 控制")
        print("  python main.py --map        - 启用地图显示")
        print("  python main.py --gui --map  - 同时启用 GUI 和地图显示")
    
    def import_gui(self):
        """导入 GUI 模块"""
        if self.use_gui:
            try:
                print("开始导入 GUI 模块...")
                from .gui import FlightControlGUI
                self.FlightControlGUI = FlightControlGUI
                print("成功导入 GUI 模块")
            except Exception as e:
                print(f"相对导入失败: {e}")
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(__file__))
                    print(f"添加路径: {os.path.dirname(__file__)}")
                    from gui import FlightControlGUI
                    self.FlightControlGUI = FlightControlGUI
                    print("成功导入 GUI 模块（直接导入）")
                except Exception as e2:
                    print(f"导入 GUI 模块失败: {e2}")
                    import traceback
                    traceback.print_exc()
                    self.use_gui = False
    
    def import_map(self):
        """导入地图显示模块"""
        if self.use_map:
            try:
                print("开始导入地图显示模块...")
                # 尝试相对导入
                from .map_display import MapDisplay
                self.MapDisplay = MapDisplay
                print("成功导入地图显示模块（相对导入）")
            except Exception as e:
                print(f"相对导入失败: {e}")
                try:
                    # 尝试直接导入
                    import sys
                    import os
                    # 添加src目录到Python路径
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
                    print(f"添加路径: {os.path.join(os.path.dirname(__file__), '..', '..')}")
                    from src.flight_control.map_display import MapDisplay
                    self.MapDisplay = MapDisplay
                    print("成功导入地图显示模块（直接导入）")
                except Exception as e2:
                    print(f"导入地图显示模块失败: {e2}")
                    import traceback
                    traceback.print_exc()
                    self.use_map = False
    
    def import_map(self):
        """导入地图显示模块"""
        if self.use_map:
            try:
                # 尝试相对导入
                from .map_display import MapDisplay
                self.MapDisplay = MapDisplay
                print("成功导入地图显示模块")
            except Exception as e:
                print(f"相对导入失败: {e}")
                try:
                    # 尝试直接导入
                    import sys
                    import os
                    # 添加src目录到Python路径
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
                    from src.flight_control.map_display import MapDisplay
                    self.MapDisplay = MapDisplay
                    print("成功导入地图显示模块")
                except Exception as e2:
                    print(f"导入地图显示模块失败: {e2}")
                    self.use_map = False
    
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
            print("如果没有 AirSim 模拟器，程序将以演示模式运行")
            # 不退出，继续运行以测试GUI功能
            return False
        return True
    
    def takeoff(self):
        """起飞并到达指定高度"""
        if not self.client:
            print("未连接到无人机，跳过起飞")
            return
        
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
            self.gui = self.FlightControlGUI(self.client, self)
            if self.gui.running:
                self.gui_started = True
                print("可视化控制面板已启动")
                self.gui.run()
            else:
                print("可视化控制面板启动失败，回退到命令行控制")
        except Exception as e:
            print(f"GUI 运行失败: {e}")
            print("回退到命令行控制")
    
    def run_map(self):
        """运行地图显示"""
        try:
            print("启动地图显示...")
            self.map_display = self.MapDisplay()
            if self.map_display.running:
                self.map_started = True
                print("地图显示已启动")
                self.map_display.run()
            else:
                print("地图显示启动失败")
        except Exception as e:
            print(f"地图显示运行失败: {e}")
    
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
    
    def start_map(self):
        """启动地图显示线程"""
        if self.use_map:
            map_thread = threading.Thread(target=self.run_map)
            map_thread.daemon = False  # 设置为非守护线程，确保地图显示能够正常运行
            map_thread.start()
            
            # 等待地图显示启动
            time.sleep(2)
            if not self.map_started:
                print("地图显示启动失败")
        else:
            print("未启用地图显示")
    
    def update_position(self):
        """更新无人机位置"""
        # 模拟位置数据，用于测试地图显示
        x, y = 0, 0
        direction = 1
        while True:
            try:
                if self.map_display:
                    if self.client:
                        # 获取真实无人机状态
                        state = self.client.getMultirotorState()
                        pos = state.kinematics_estimated.position
                        x, y = pos.x_val, pos.y_val
                    else:
                        # 模拟位置数据
                        x += 0.1 * direction
                        if abs(x) > 10:
                            direction *= -1
                    # 更新地图显示
                    self.map_display.update_position(x, y)
            except Exception as e:
                pass
            time.sleep(0.5)
    
    def exit_program(self):
        """安全退出程序"""
        print("\n安全降落...")
        try:
            if self.client:
                self.client.landAsync().join()
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
                print("无人机已安全降落")
        except Exception as e:
            print(f"降落过程中发生错误: {e}")
        finally:
            if self.gui:
                self.gui.stop()
            if self.map_display:
                self.map_display.stop()
            os.kill(os.getpid(), signal.SIGTERM)
    
    def run(self):
        """运行主程序"""
        try:
            print("开始运行主程序...")
            # 启动 GUI
            self.start_gui()
            # 启动地图显示
            self.start_map()
            
            # 启动键盘监听
            print("启动键盘监听...")
            listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            listener.start()
            
            # 启动位置更新线程
            print("启动位置更新线程...")
            position_thread = threading.Thread(target=self.update_position)
            position_thread.daemon = True
            position_thread.start()
            
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
        except Exception as e:
            print(f"主程序运行失败: {e}")
            import traceback
            traceback.print_exc()
            self.exit_program()

if __name__ == "__main__":
    flight_control = FlightControl()
    flight_control.setup()
    flight_control.run()