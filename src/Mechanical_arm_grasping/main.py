# 代码1：超声波传感器目标位置检测（简单版）
import time

def ultrasonic_sensor_simulation():
    """模拟超声波传感器，返回目标与机械臂的距离（单位：cm）"""
    # 随机生成合理距离（模拟真实传感器检测结果）
    import random
    target_distance = random.uniform(5, 50)  # 目标距离5-50cm
    print(f"【超声波传感器检测】目标距离：{target_distance:.2f} cm")
    return target_distance

def judge_target_position(distance):
    """根据距离判断目标位置状态"""
    if distance < 15:
        return "近距离（可抓取）"
    elif 15 <= distance <= 30:
        return "中距离（需伸展）"
    else:
        return "远距离（超出范围）"

# 主程序
if __name__ == "__main__":
    print("=== 机械臂目标检测（简单版）===")
    for i in range(3):  # 检测3次
        distance = ultrasonic_sensor_simulation()
        position_state = judge_target_position(distance)
        print(f"第{i+1}次检测结果：{position_state}\n")
        time.sleep(1)  # 间隔1秒检测