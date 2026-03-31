import airsim
import time
import os
import signal
from pynput import keyboard

# ======================= 连接无人机 =======================
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# ======================= 核心参数 =======================
SPEED = 2.0
HEIGHT = -3.0
is_flying = True

# ======================= 起飞 =======================
print("已连接无人机")
print("起飞中...")
client.takeoffAsync().join()
client.moveToZAsync(HEIGHT, 1.5).join()
time.sleep(0.5)

print("="*60)
print("手动控制")
print("W 前  S 后  A 左  D 右")
print("Z 上升  X 下降  H 悬停  B 返航")
print("ESC 退出并降落")
print("="*60)

# ======================= 实时键盘 =======================
def on_press(key):
    try:
        # ========== 绝杀退出：按ESC直接结束程序 ==========
        if key == keyboard.Key.esc:
            print("\n安全降落...")
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
            
            os.kill(os.getpid(), signal.SIGTERM)
            return False

        # 悬停
        if key.char == 'h':
            print("悬停")
            client.hoverAsync().join()

        # 返航
        if key.char == 'b':
            print("返航原点")
            client.moveToPositionAsync(0, 0, HEIGHT, 2).join()

        # 实时移动
        if key.char == 'w':
            client.moveByVelocityBodyFrameAsync(SPEED, 0, 0, 0.05)
        if key.char == 's':
            client.moveByVelocityBodyFrameAsync(-SPEED*0.7, 0, 0, 0.05)
        if key.char == 'a':
            client.moveByVelocityBodyFrameAsync(0, -SPEED, 0, 0.05)
        if key.char == 'd':
            client.moveByVelocityBodyFrameAsync(0, SPEED, 0, 0.05)

        # 高度
        if key.char == 'z':
            client.moveToZAsync(HEIGHT - 0.5, 0.8)
        if key.char == 'x':
            client.moveToZAsync(HEIGHT + 0.5, 0.8)

    except:
        pass

def on_release(key):
    client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.05)

# ======================= 启动键盘监听 =======================
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while is_flying:
    time.sleep(0.01)

listener.join()