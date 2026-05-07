import random
import time
import sys


def get_fuel_color_code(fuel):
    """返回ANSI颜色码，终端支持时生效"""
    # ANSI颜色码：绿色、黄色、红色、默认
    if fuel >= 70:
        return "\033[32m"  # 绿色
    elif fuel >= 20:
        return "\033[33m"  # 黄色
    elif fuel >= 3:
        return "\033[31m"  # 红色
    else:
        return "\033[31m"  # 红色


RESET_COLOR = "\033[0m"  # 重置颜色


def display_fuel_gauge(fuel):
    """在控制台显示油表信息"""
    # 清屏（跨平台）
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

    # 输出油表标题
    print("=" * 40)
    print("        Carla Autonomous Vehicle Fuel Gauge")
    print("=" * 40)

    # 处理油量显示逻辑
    if fuel >= 3:
        fuel_text = f"{fuel}"
        color_code = get_fuel_color_code(fuel)
        print(f"\n        Current Fuel Level: {color_code}{fuel_text}{RESET_COLOR}")
    else:
        print(f"\n        {get_fuel_color_code(fuel)}油量不足{RESET_COLOR}")

    # 绘制简易进度条
    print("\n        [", end="")
    progress = min(int(fuel), 100)
    for i in range(100):
        if i < progress:
            print(get_fuel_color_code(fuel) + "■" + RESET_COLOR, end="")
        else:
            print(" ", end="")
    print("]")
    print("\n        (Press Ctrl+C to exit)")


def get_carla_fuel_data():
    """模拟从Carla获取油量数据"""
    global current_fuel
    # 模拟油量缓动变化
    current_fuel += random.randint(-2, 2)
    current_fuel = max(0, min(100, current_fuel))
    return current_fuel


# 初始化油量
current_fuel = 80

if __name__ == "__main__":
    try:
        while True:
            fuel_level = get_carla_fuel_data()
            display_fuel_gauge(fuel_level)
            time.sleep(0.5)  # 控制刷新频率
    except KeyboardInterrupt:
        print("\n\n        Exiting Fuel Gauge...")
        sys.exit(0)