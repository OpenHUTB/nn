import threading
import time
import sys
import numpy as np

class KeyboardInputHandler(threading.Thread):
    """键盘输入监听线程（独立模块，仅处理按键指令）"""
    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True

    def run(self):
        """打印控制说明并启动监听"""
        self._print_help()
        while self.running:
            try:
                key = self._get_key_non_blocking()
                if key:
                    self._handle_key(key)
                time.sleep(0.01)
            except:
                continue

    def _print_help(self):
        """打印控制指令说明"""
        print("\n===== 控制指令说明 =====")
        print("w: 开始行走 | s: 停止行走 | e: 紧急停止 | r: 恢复站立")
        print("a: 左转 | d: 右转 | 空格: 原地转向 | z: 减速 | x: 加速")
        print("m: 传感器模拟开关 | p: 打印传感器数据")
        print("1: 慢走 | 2: 正常走 | 3: 小跑 | 4: 原地踏步")
        print("========================\n")

    def _get_key_non_blocking(self):
        """非阻塞获取键盘输入（跨平台）"""
        if sys.platform == "win32":
            import msvcrt
            if msvcrt.kbhit():
                return msvcrt.getch().decode('utf-8').lower()
        else:
            import select
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
        return None

    def _handle_key(self, key):
        """处理按键指令"""
        if key == 'w':
            current_gait = self.stabilizer.gait_mode
            self.stabilizer.set_state("WALK")
            self.stabilizer.set_gait_mode(current_gait)
            print(f"[指令] 行走 | 步态: {current_gait} | 速度: {self.stabilizer.walk_speed:.2f}")
        elif key == 's':
            self.stabilizer.set_state("STOP")
            print("[指令] 停止行走")
        elif key == 'e':
            self.stabilizer.set_state("EMERGENCY")
            print("[指令] 紧急停止")
        elif key == 'r':
            self.stabilizer.set_state("STAND")
            print("[指令] 恢复站立")
        elif key == 'a':
            self.stabilizer.set_turn_angle(self.stabilizer.turn_angle + 0.05)
            print(f"[指令] 左转 | 转向: {self.stabilizer.turn_angle:.2f}rad")
        elif key == 'd':
            self.stabilizer.set_turn_angle(self.stabilizer.turn_angle - 0.05)
            print(f"[指令] 右转 | 转向: {self.stabilizer.turn_angle:.2f}rad")
        elif key == ' ':
            self.stabilizer.set_turn_angle(0.2 if self.stabilizer.turn_angle <= 0 else -0.2)
            print(f"[指令] 原地转向 | 转向: {self.stabilizer.turn_angle:.2f}rad")
        elif key == 'z':
            self.stabilizer.set_walk_speed(self.stabilizer.walk_speed - 0.1)
            print(f"[指令] 减速 | 速度: {self.stabilizer.walk_speed:.2f}")
        elif key == 'x':
            self.stabilizer.set_walk_speed(self.stabilizer.walk_speed + 0.1)
            print(f"[指令] 加速 | 速度: {self.stabilizer.walk_speed:.2f}")
        elif key == 'm':
            self.stabilizer.enable_sensor_simulation = not self.stabilizer.enable_sensor_simulation
            print(f"[指令] 传感器模拟{'开启' if self.stabilizer.enable_sensor_simulation else '关闭'}")
        elif key == 'p':
            self.stabilizer.print_sensor_data()
        elif key in ['1', '2', '3', '4']:
            gait_map = {'1': 'SLOW', '2': 'NORMAL', '3': 'TROT', '4': 'STEP_IN_PLACE'}
            self.stabilizer.set_gait_mode(gait_map[key])
            print(f"[指令] 切换为{self.stabilizer.gait_mode}模式")
