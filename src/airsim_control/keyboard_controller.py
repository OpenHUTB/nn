import time

try:
    import keyboard
except ImportError:
    # 提供一个模拟的keyboard实现，用于测试
    class MockKeyboard:
        def is_pressed(self, key):
            # 模拟键盘输入，默认不按键
            return False
    
    keyboard = MockKeyboard()

from agents.agent import Agent


class KeyboardController(Agent):
    def __init__(self, client, move_type):
        super(KeyboardController, self).__init__(client, move_type)
        self.client.start()
        self.speed = 5  # 移动速度
        self.turn_speed = 2  # 转向速度
        self.height_adjust_speed = 3  # 高度调整速度
        self.print_controls()

    def print_controls(self):
        print("\n控制说明:")
        print("键盘控制:")
        print("W: 向前移动")
        print("S: 向后移动")
        print("A: 向左转向")
        print("D: 向右转向")
        print("Q: 向上移动")
        print("E: 向下移动")
        print("空格键: 停止移动")
        print("ESC: 退出控制\n")

    def get_state(self):
        state = self.client.get_state()
        position = state.kinematics_estimated.position
        return {
            'position': (position.x_val, position.y_val, position.z_val),
            'velocity': state.kinematics_estimated.linear_velocity
        }

    def handle_keyboard_input(self):
        vx, vy, vz = 0, 0, 0

        # 处理键盘输入
        if keyboard.is_pressed('w'):
            vx = self.speed
        elif keyboard.is_pressed('s'):
            vx = -self.speed

        if keyboard.is_pressed('a'):
            vy = -self.turn_speed
        elif keyboard.is_pressed('d'):
            vy = self.turn_speed

        if keyboard.is_pressed('q'):
            vz = self.height_adjust_speed
        elif keyboard.is_pressed('e'):
            vz = -self.height_adjust_speed

        # 空格键停止移动
        if keyboard.is_pressed('space'):
            vx, vy, vz = 0, 0, 0

        return vx, vy, vz

    def act(self):
        # 先获取键盘输入
        vx, vy, vz = self.handle_keyboard_input()
        
        # 检查碰撞信息
        collision_info = self.client.get_collision_info()
        if collision_info.has_collided:
            print("碰撞发生！")
            # 碰撞后停止移动并开始掉落
            vx, vy = 0, 0
            vz = 2  # 在NED坐标系中，正值表示向下移动，让无人机开始掉落
            print("无人机开始下降...")
            # 尝试获取更详细的碰撞信息
            try:
                # 检查是否有碰撞位置信息
                if hasattr(collision_info, 'position'):
                    pos = collision_info.position
                    print(f"碰撞位置: ({pos.x_val}, {pos.y_val}, {pos.z_val})")
                # 检查是否有碰撞物体信息
                if hasattr(collision_info, 'object_name'):
                    print(f"碰撞物体: {collision_info.object_name}")
                # 检查是否有碰撞法线信息
                if hasattr(collision_info, 'normal'):
                    normal = collision_info.normal
                    print(f"碰撞法线: ({normal.x_val}, {normal.y_val}, {normal.z_val})")
                # 尝试获取无人机当前状态信息
                state = self.client.get_state()
                if state:
                    pos = state.kinematics_estimated.position
                    print(f"当前位置: ({pos.x_val}, {pos.y_val}, {pos.z_val})")
                    vel = state.kinematics_estimated.linear_velocity
                    print(f"当前速度: ({vel.x_val}, {vel.y_val}, {vel.z_val})")
            except Exception as e:
                # 如果获取详细信息失败，忽略错误
                pass
        
        # 移动无人机
        self.client.move(self.move_type, vx, vy, vz)

        # 检查是否按下ESC键退出
        if keyboard.is_pressed('esc'):
            print("退出控制")
            return False
        
        return True



    def run(self, loop_cnt=100):
        for _ in range(loop_cnt):
            if not self.act():
                break
            time.sleep(0.1)
