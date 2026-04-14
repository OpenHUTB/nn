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
print("O 环绕   M 方形   ESC 退出")
print("="*60)

# ======================= 环绕飞行 =======================
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

# ======================= 方形飞行 =======================
def square_mode():
    length = 3
    speed = 1.2
    while is_flying:
        client.moveToPositionAsync(length, 0, HEIGHT, speed).join()
        client.moveToPositionAsync(length, length, HEIGHT, speed).join()
        client.moveToPositionAsync(0, length, HEIGHT, speed).join()
        client.moveToPositionAsync(0, 0, HEIGHT, speed).join()

def start_square():
    threading.Thread(target=square_mode, daemon=True).start()

# ======================= 键盘 =======================
def on_press(key):
    try:
        if key == keyboard.Key.esc:
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
            return False

        if key.char == 'h':
            client.hoverAsync().join()
        if key.char == 'b':
            client.moveToPositionAsync(0,0,HEIGHT,1.5).join()

        if key.char == 'w': client.moveByVelocityBodyFrameAsync(SPEED,0,0,0.1)
        if key.char == 's': client.moveByVelocityBodyFrameAsync(-SPEED,0,0,0.1)
        if key.char == 'a': client.moveByVelocityBodyFrameAsync(0,-SPEED,0,0.1)
        if key.char == 'd': client.moveByVelocityBodyFrameAsync(0,SPEED,0,0.1)
        if key.char == 'z': client.moveToZAsync(HEIGHT-0.5, 1)
        if key.char == 'x': client.moveToZAsync(HEIGHT+0.5, 1)

        if key.char == 'o': start_orbit()
        if key.char == 'm': start_square()

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