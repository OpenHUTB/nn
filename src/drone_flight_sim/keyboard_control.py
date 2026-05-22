# keyboard_control.py
"""键盘控制模块

使用 pynput 库实现键盘监听，支持无人机的手动键盘控制。
按 WASD 移动，QE 升降，空格悬停，P 拍照，ESC 退出。
"""

# 导入 pynput 库的键盘监听模块
from pynput import keyboard
# 导入线程模块，用于持续移动
import threading
# 导入时间模块
import time

# 全局变量，用于控制监听循环
listener_running = True

# 速度档位配置
SPEED_LEVELS = [1, 2, 3, 5, 8]  # 5个速度档位: 慢/中/快/很快/极速


class KeyboardController:
    """键盘控制器类

    监听键盘输入，根据按键执行对应的无人机控制命令。
    """

    def __init__(self, drone):
        """初始化键盘控制器

        参数:
            drone: DroneController 实例，用于执行控制命令
        """
        self.drone = drone
        self.active_keys = set()  # 记录当前按下的按键
        self.current_movement = None  # 当前移动状态: 'forward', 'backward', 'left', 'right', 'up', 'down'
        self.last_position = None  # 记录按下时的起始位置
        self.photo_mode = False  # 拍照模式标志
        self._movement_thread = None  # 持续移动线程
        self._movement_running = False  # 移动线程运行标志
        self._lock = threading.Lock()  # 线程锁
        # 每次移动的距离（米）
        self.move_step = 3

        # 速度档位
        self.speed_level = 2  # 默认第2档(中速)
        self.drone.velocity = SPEED_LEVELS[self.speed_level]

        # 起飞点位置
        self.home_position = None

    def _get_direction_key(self, key):
        """获取按键对应的移动方向"""
        key_char = key.char if hasattr(key, 'char') else None
        key_str = str(key)

        # W / 上方向键：前进
        if key_char == 'w' or key_char == 'W' or key_str == 'Key.up':
            return 'forward'
        # S / 下方向键：后退
        elif key_char == 's' or key_char == 'S' or key_str == 'Key.down':
            return 'backward'
        # A：向左
        elif key_char == 'a' or key_char == 'A':
            return 'left'
        # D：向右
        elif key_char == 'd' or key_char == 'D':
            return 'right'
        # Q：上升
        elif key_char == 'q' or key_char == 'Q':
            return 'up'
        # E：下降
        elif key_char == 'e' or key_char == 'E':
            return 'down'
        return None

    def _start_movement(self, direction):
        """开始移动，只在状态改变时输出信息"""
        with self._lock:
            # 检查状态是否改变
            if self.current_movement == direction:
                return

            # 获取当前无人机位置作为起始位置
            self.last_position = self.drone.get_position()
            self.current_movement = direction

        # 停止之前的移动线程
        self._stop_movement_thread()

        # 启动新的持续移动线程
        self._movement_running = True
        self._movement_thread = threading.Thread(
            target=self._continuous_move_loop,
            args=(direction,),
            daemon=True
        )
        self._movement_thread.start()

        # 根据方向打印开始信息
        direction_names = {
            'forward': '前进',
            'backward': '后退',
            'left': '向左',
            'right': '向右',
            'up': '上升',
            'down': '下降'
        }
        # 高度锁定时不允许上下移动
        if self.height_locked and (direction == 'up' or direction == 'down'):
            print("⚠️ 高度已锁定，无法上下移动")
            return

        print(f"🚀 {direction_names.get(direction, direction)} (速度: {self.drone.velocity} m/s)")

    def _continuous_move_loop(self, direction):
        """持续移动循环，在独立线程中运行"""
        move_methods = {
            'forward': self.drone.go_forward_continuous,
            'backward': self.drone.go_backward_continuous,
            'left': self.drone.go_left_continuous,
            'right': self.drone.go_right_continuous,
            'up': self.drone.go_up_continuous,
            'down': self.drone.go_down_continuous
        }
        move_func = move_methods.get(direction)

        while self._movement_running:
            if move_func:
                move_func()
            time.sleep(0.05)  # 每50ms调用一次，保持移动

    def _stop_movement_thread(self):
        """停止移动线程"""
        self._movement_running = False
        if self._movement_thread and self._movement_thread.is_alive():
            self._movement_thread.join(timeout=0.2)
        self._movement_thread = None

    def _stop_movement(self, direction):
        """停止移动，显示移动距离"""
        with self._lock:
            if self.current_movement != direction:
                return

            # 停止移动线程
            self._stop_movement_thread()

            # 计算移动距离
            start_pos = self.last_position
            self.current_movement = None
            self.last_position = None

        if start_pos is not None:
            current_pos = self.drone.get_position()
            self.drone.hover()

            # 计算并打印移动距离
            if direction == 'forward':
                distance = current_pos.x_val - start_pos.x_val
                print(f"✅ 前进完成，移动了 {distance:.1f}m")
            elif direction == 'backward':
                distance = start_pos.x_val - current_pos.x_val
                print(f"✅ 后退完成，移动了 {distance:.1f}m")
            elif direction == 'left':
                distance = abs(current_pos.y_val - start_pos.y_val)
                print(f"✅ 向左完成，移动了 {distance:.1f}m")
            elif direction == 'right':
                distance = abs(current_pos.y_val - start_pos.y_val)
                print(f"✅ 向右完成，移动了 {distance:.1f}m")
            elif direction == 'up':
                distance = abs(start_pos.z_val) - abs(current_pos.z_val)
                print(f"✅ 上升完成，移动了 {distance:.1f}m")
            elif direction == 'down':
                distance = abs(current_pos.z_val) - abs(start_pos.z_val)
                print(f"✅ 下降完成，移动了 {distance:.1f}m")

    def on_press(self, key):
        """按键按下事件处理

        当键盘按键被按下时调用，根据按键执行相应操作。

        参数:
            key: 按键对象
        """
        global listener_running

        try:
            # ESC 键：退出程序
            if key == keyboard.Key.esc:
                print("\n🚨 收到退出指令，正在停止无人机...")
                self.drone.emergency_stop()
                listener_running = False
                return False

            # 空格键：悬停（停止当前移动并总结）
            if key == keyboard.Key.space:
                if self.current_movement:
                    self._stop_movement(self.current_movement)
                else:
                    self.drone.hover()
                    print("🛸 悬停")
                return

            # 获取按键字符
            key_char = key.char if hasattr(key, 'char') else None

            # 拍照模式下的特殊按键
            if self.photo_mode:
                if key_char == 'p' or key_char == 'P':
                    self.drone.capture_image()
                    print("📸 拍照完成！")
                elif key_char == 'b' or key_char == 'B':
                    self.photo_mode = False
                    print("📷 退出拍照预览模式")
                return

            # 速度档位切换 1-5
            if key_char in ['1', '2', '3', '4', '5']:
                self.speed_level = int(key_char) - 1
                self.drone.velocity = SPEED_LEVELS[self.speed_level]
                names = ['慢', '中', '快', '很快', '极速']
                print(f"🎚️ 速度档位: {key_char} ({names[self.speed_level]}) - {SPEED_LEVELS[self.speed_level]} m/s")
                return

            # R 键：一键返航
            if key_char == 'r' or key_char == 'R':
                if self.home_position:
                    print("🏠 开始返航...")
                    self.drone.fly_to_position(
                        self.home_position.x_val,
                        self.home_position.y_val,
                        self.home_position.z_val,
                        velocity=5
                    )
                    self.drone.safe_land()
                else:
                    print("⚠️ 未设置返航点")
                return
            
            # O 键：一键环绕
            if key_char == 'o' or key_char == 'O':
                if self.current_movement:
                    self._stop_movement(self.current_movement)
                print("\n🔄 开始环绕飞行...")
                self._circle_flight()
                return
            
            # P 键：拍照
            if key_char == 'p' or key_char == 'P':
                self.drone.capture_image()
                return

            # T 键：拍摄所有类型图像
            if key_char == 't' or key_char == 'T':
                self.drone.capture_all_cameras()
                return

            # N 键：拍摄深度图像
            if key_char == 'n' or key_char == 'N':
                self.drone.capture_depth_image()
                return

            # B 键：拍照预览模式
            if key_char == 'b' or key_char == 'B':
                self.photo_mode = True
                print("📷 进入拍照预览模式 (P拍照, B退出)")
                return

            # L 键：降落
            if key_char == 'l' or key_char == 'L':
                if self.current_movement:
                    self._stop_movement(self.current_movement)
                print("\n🛬 收到降落指令...")
                self.drone.safe_land()
                listener_running = False
                return False

            # 移动按键：WASD, QE, 方向键上下
            direction = self._get_direction_key(key)
            if direction:
                self._start_movement(direction)

        except AttributeError:
            # 处理特殊按键（如功能键）
            pass

    def on_release(self, key):
        """按键释放事件处理

        当键盘按键被释放时调用，停止持续移动并显示移动距离。

        参数:
            key: 按键对象
        """
        try:
            # 拍照预览模式下松开 B 退出预览
            if self.photo_mode:
                key_char = key.char if hasattr(key, 'char') else None
                if key_char == 'b' or key_char == 'B':
                    self.photo_mode = False
                    print("📷 退出拍照预览模式")
                return

            # 获取释放的按键对应的方向
            direction = self._get_direction_key(key)
            if direction:
                # 如果释放的是当前移动方向的按键，停止移动
                if self.current_movement == direction:
                    self._stop_movement(direction)

        except AttributeError:
            pass
    def _circle_flight(self):
        """一键环绕飞行：绕当前物体飞矩形轨迹"""
        # 获取当前位置
        pos = self.drone.get_position()
        x, y, z = pos.x_val, pos.y_val, pos.z_val
        
        # 环绕半径（米）
        radius = 5
        
        print(f"📍 起始位置: ({x:.1f}, {y:.1f})")
        print("   矩形环绕轨迹: 右 → 前 → 左 → 后")
        
        # 矩形环绕的4个点
        waypoints = [
            (x + radius, y, z),      # 右边
            (x + radius, y + radius, z),  # 前边
            (x, y + radius, z),      # 左边
            (x, y, z),               # 回到起点
        ]
        
        # 依次飞过每个点
        for i, (wx, wy, wz) in enumerate(waypoints, 1):
            print(f"   航点{i}: ({wx:.1f}, {wy:.1f})")
            self.drone.fly_to_position(wx, wy, wz, velocity=3)
            import time
            time.sleep(0.5)
        
        # 悬停收尾
        self.drone.hover()
        print("✅ 环绕完成")

    def start(self):
        """启动键盘监听

        开始监听键盘输入，直到收到退出指令。
        """
        global listener_running
        listener_running = True

        # 创建键盘监听器
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as listener:
            listener.join()


def print_control_help():
    """打印控制说明"""
    print("\n" + "=" * 50)
    print("         🕹️  无人机键盘控制说明")
    print("=" * 50)
    print("""
  📍 移动控制:
     W         : 前进
     S         : 后退
     A         : 向左横移
     D         : 向右横移
     Q         : 上升
     E         : 下降

  🎮 功能键:
     空格      : 悬停（停止移动）
     1-5       : 切换速度档位（慢/中/快/很快/极速）
     R         : 一键返航
     P         : 拍照
     T         : 拍摄所有图像(RGB+深度+分割)
     N         : 拍摄深度图像
     O         : 一键环绕（飞矩形轨迹）

  🛬 安全控制:
     L         : 执行降落
     ESC       : 紧急停止并退出

  ⚠️  注意:
     - 释放移动键后会自动显示移动距离并悬停
    """)
    print("=" * 50 + "\n")
