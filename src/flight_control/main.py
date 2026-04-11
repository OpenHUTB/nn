import airsim
import time
from pynput import keyboard

print("脚本开始运行...")

# 连接无人机
print("正在连接到 AirSim 模拟器...")
try:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("成功连接到 AirSim 模拟器")
    client.enableApiControl(True)
    client.armDisarm(True)
except Exception as e:
    print(f"连接失败: {e}")
    print("请确保 AirSim 模拟器已启动")
    exit(1)

# 核心参数
SPEED = 2.0
HEIGHT = -3.0

# 起飞
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

def on_press(key):
    # 退出
    if key == keyboard.Key.esc:
        print("\n安全降落...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        return False

    # 检查是否是字符键
    if not hasattr(key, 'char'):
        return

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

def on_release(key):
    client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.05)

# 键盘监听
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# 保持程序运行
print("程序已启动，按 ESC 键退出...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n收到中断信号，正在降落...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("程序已退出")

listener.join()