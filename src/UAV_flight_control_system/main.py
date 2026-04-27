import airsim
import time
import os
import signal
import math
import threading
from pynput import keyboard

# ======================= 连接无人机 =======================
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# ======================= 核心参数 =======================
SPEED = 1.5
HEIGHT = -3
is_flying = True

# ========== 飞行更加流畅 ==========
smooth = 0.5
speed_level = 2
speed_ratio = [0.6, 1.0, 1.6]

# ======================= 起飞 =======================
print("已连接无人机")
print("起飞中...")
try:
    client.takeoffAsync()
    time.sleep(2)
    client.moveToZAsync(HEIGHT, 1).join()
except:
    pass

print("="*60)
print("W 前  S 后  A 左  D 右")
print("Z 上升  X 下降  H 悬停  B 返航")
print("O 环绕   M 方形  P 调速   ESC 退出")
print("="*60)

# ===================== 环绕飞行 =======================
def orbit_mode():
    radius = 6
    speed = 1
    angle = 0
    while is_flying:
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        client.moveToPositionAsync(x, y, HEIGHT, speed)
        angle += 0.05
        time.sleep(0.1)

def start_orbit():
    threading.Thread(target=orbit_mode, daemon=True).start()

# ======================= 绕方形轨迹飞行 =======================
def square_mode():
    length = 3.5
    speed = 1.2
    while is_flying:
        client.moveToPositionAsync(length, 0, HEIGHT, speed).join()
        client.moveToPositionAsync(length, length, HEIGHT, speed).join()
        client.moveToPositionAsync(0, length, HEIGHT, speed).join()
        client.moveToPositionAsync(0, 0, HEIGHT, speed).join()

def start_square():
    threading.Thread(target=square_mode, daemon=True).start()

# ======================= 原地旋转功能 =======================
def rotate_mode():
    print("开启原地旋转模式")
    while is_flying:
        # 原地旋转（机身自转）
        client.rotateByYawRateAsync(20, 0.1)
        time.sleep(0.1)

def start_rotate():
    threading.Thread(target=rotate_mode, daemon=True).start()

# ======================= 螺旋上升飞行 =======================
def spiral_mode():
    print("开启螺旋上升模式")
    angle = 0
    radius = 4
    current_z = HEIGHT
    while is_flying:
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        current_z -= 0.08  # 缓慢升高
        client.moveToPositionAsync(x, y, current_z, 1)
        angle += 0.15
        time.sleep(0.1)

def start_spiral():
    threading.Thread(target=spiral_mode, daemon=True).start()

# ======================= 键盘 =======================
def on_press(key):
    global speed_level
    try:
        if key == keyboard.Key.esc:
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
            return False

 #  =======================速度档位切换 =======================
        if key.char == 'p':
            speed_level = speed_level % 3 + 1
            tips = ["当前：低速模式", "当前：标准模式", "当前：高速模式"]
            print(tips[speed_level - 1])

        if key.char == 'h':
            client.hoverAsync().join()
        if key.char == 'b':
            client.moveToPositionAsync(0,0,HEIGHT,1.5).join()

        now_speed = SPEED * speed_ratio[speed_level - 1] * smooth
        if key.char == 'w': client.moveByVelocityBodyFrameAsync(now_speed,0,0,0.1)
        if key.char == 's': client.moveByVelocityBodyFrameAsync(-now_speed,0,0,0.1)
        if key.char == 'a': client.moveByVelocityBodyFrameAsync(0,-now_speed,0,0.1)
        if key.char == 'd': client.moveByVelocityBodyFrameAsync(0,now_speed,0,0.1)
        if key.char == 'z': client.moveToZAsync(HEIGHT-0.5, 1)
        if key.char == 'x': client.moveToZAsync(HEIGHT+0.5, 1)

        if key.char == 'o': start_orbit()
        if key.char == 'm': start_square()
        if key.char == 'n': start_rotate()
        if key.char == 'l': start_spiral()
        
        
    except:
        pass

def on_release(key):
    try:
        client.moveByVelocityBodyFrameAsync(0,0,0, 0.1)
    except:
        pass

# ======================= 启动 =======================
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while is_flying:
    time.sleep(0.1)
