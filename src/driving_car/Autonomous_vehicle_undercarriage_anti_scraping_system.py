import random
import time

def get_chassis_ground_distance():
    """自动模拟底盘距离地面的距离（0~15cm 随机波动）"""
    return round(random.uniform(0, 15), 1)

def get_safety_status(distance_cm):
    """根据距离返回状态文本和颜色标识"""
    if distance_cm > 10.0:
        return "安全", "绿色"
    elif 5.0 <= distance_cm <= 10.0:
        return "注意安全", "黄色"
    else:
        return "危险", "红色"

def print_color_text(text, color):
    """终端带颜色输出文字（Windows/Linux/Mac通用）"""
    color_codes = {
        "绿色": "\033[32m",
        "黄色": "\033[33m",
        "红色": "\033[31m",
        "重置": "\033[0m"
    }
    print(f"{color_codes[color]}{text}{color_codes['重置']}")

def main():
    """主函数：自动循环测试防刮车底系统"""
    print("===== 无人车防刮车底系统（自动测试模式）=====")
    print("系统已启动，每秒自动更新距离和状态 | 按 Ctrl+C 退出\n")

    try:
        while True:
            distance = get_chassis_ground_distance()
            status_text, status_color = get_safety_status(distance)
            # 输出实时数据
            print(f"当前底盘离地距离：{distance} cm | 状态：", end="")
            print_color_text(status_text, status_color)
            # 每秒更新一次
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n自动测试结束，系统退出！")

if __name__ == "__main__":
    main()