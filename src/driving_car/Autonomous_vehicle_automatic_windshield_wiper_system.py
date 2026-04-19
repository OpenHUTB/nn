import time
import random
from collections import deque

def get_raindrop_count():
    """模拟雨滴识别器，返回当前挡风玻璃的雨滴数量（0~150）"""
    return random.randint(0, 150)

def get_stable_rain_count(history, current_count, window_size=5):
    """
    滑动窗口平均法处理雨滴数量，避免瞬时波动导致档位频繁切换（防抖逻辑）
    :param history: 存储历史雨滴数量的队列
    :param current_count: 当前检测的雨滴数量
    :param window_size: 滑动窗口大小
    :return: 稳定后的雨滴数量平均值
    """
    history.append(current_count)
    if len(history) > window_size:
        history.popleft()
    return sum(history) / len(history)

def judge_wiper_gear(raindrop_num):
    """
    根据雨滴数量判断雨刮器档位
    :param raindrop_num: 雨滴数量
    :return: 雨刮器状态、档位参数值
    """
    if raindrop_num == 0:
        return "暂停", 0.0
    elif 0 < raindrop_num <= 50:
        return "小档", 0.33
    elif 50 < raindrop_num <= 100:
        return "中档", 0.66
    else:
        return "高档", 1.0

def main():
    # 初始化滑动窗口队列，用于存储历史雨滴数量（防抖）
    rain_history = deque(maxlen=5)
    print("自动雨刮器系统已启动（按Ctrl+C退出）...\n")
    try:
        while True:
            # 1. 获取当前雨滴数量（模拟传感器数据）
            current_rain = get_raindrop_count()
            # 2. 处理雨滴数量，得到稳定值
            stable_rain = get_stable_rain_count(rain_history, current_rain)
            # 3. 判断雨刮器档位
            wiper_status, wiper_value = judge_wiper_gear(round(stable_rain))
            # 4. 输出状态信息
            print(f"当前检测雨滴数量：{current_rain} | 稳定后数量：{round(stable_rain)} | 雨刮器状态：{wiper_status}（参数值：{wiper_value}）")
            # 5. 控制雨刮器（此处为模拟，实际场景中替换为硬件/仿真接口调用）
            # control.wiper = wiper_value  # CARLA环境下的控制逻辑
            # vehicle.apply_control(control)
            time.sleep(1)  # 每秒更新一次状态
    except KeyboardInterrupt:
        print("\n\n自动雨刮器系统已停止运行")

if __name__ == '__main__':
    main()