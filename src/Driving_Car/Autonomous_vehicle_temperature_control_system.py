import random
import time
import sys

# 模拟温度传感器：生成-50~50℃的随机温度
def get_temperature():
    """生成-50~50℃的随机温度，也可手动设置固定值测试"""
    return random.uniform(-50.0, 50.0)
    # 测试固定温度：return 26.0  # 温度适宜
    # return -10.0  # 温度过低
    # return 35.0  # 温度过高

# 温度判断与显示（控制台彩色输出+弹窗可选）
def display_temperature_info(temp):
    """根据温度显示对应信息，支持控制台彩色输出"""
    # 控制台彩色字体（Windows/Linux通用的ANSI转义码）
    class Color:
        BLUE = '\033[94m'    # 蓝色
        GREEN = '\033[92m'   # 绿色
        RED = '\033[91m'     # 红色
        RESET = '\033[0m'    # 重置颜色

    if -50.0 <= temp <= 25.0:
        # 温度过低：蓝色文本
        text = "温度过低，是否需要打开空调？"
        print(f"{Color.BLUE}{text}{Color.RESET}")
    elif temp == 26.0:
        # 温度适宜：绿色文本
        text = "温度适宜"
        print(f"{Color.GREEN}{text}{Color.RESET}")
    elif 27.0 <= temp <= 50.0:
        # 温度过高：红色文本
        text = "温度过高，是否需要打开空调？"
        print(f"{Color.RED}{text}{Color.RESET}")
    else:
        text = "温度超出测量范围"
        print(text)

# 主循环
try:
    while True:
        current_temp = get_temperature()
        print(f"\n当前温度: {current_temp:.1f}℃", end=" ")
        display_temperature_info(current_temp)
        time.sleep(0.5)  # 控制帧率
except KeyboardInterrupt:
    print("\n程序手动终止")