import random
import time
import sys

# -------------------------- 防追尾预警系统核心逻辑 --------------------------
class RearCollisionWarning:
    def __init__(self):
        # 初始化后方障碍物距离（模拟200米外无障碍物）
        self.rear_distance = float('inf')

    def simulate_distance_detection(self):
        """模拟距离识别器：随机生成0-200米的障碍物距离（可改为手动输入）"""
        # 随机生成距离，模拟实时检测（也可注释此行为固定值测试）
        self.rear_distance = random.uniform(0, 200)
        # 测试固定距离示例：取消下面注释，注释上面随机数即可
        # self.rear_distance = 5  # 测试0-10米场景
        # self.rear_distance = 30  # 测试10-49米场景
        # self.rear_distance = 75  # 测试50-99米场景
        # self.rear_distance = 150 # 测试100-200米场景

    def get_warning_info(self):
        """根据距离返回预警文本和颜色标识（控制台用字符/文字表示颜色）"""
        distance = self.rear_distance

        if 100 < distance <= 200:
            return "后方安全", "绿色"
        elif 50 < distance <= 99:
            return "后方有障碍物，谨慎驾驶！", "黄色"
        elif 10 < distance <= 49:
            return "后方注意安全", "红色"
        elif 0 <= distance <= 10:
            return "小心追尾", "红色"
        else:
            # 超出200米，默认显示后方安全
            return "后方安全", "绿色"

# -------------------------- 控制台输出美化（可选） --------------------------
def print_warning(text, color, distance):
    """在控制台打印带颜色标识的预警信息（Windows/Linux通用）"""
    # 清空控制台（跨平台方式）
    print("\033c" if sys.platform != "win32" else "\n" * 50, end="")
    # 打印标题
    print("=" * 50)
    print("        无人车防追尾预警系统")
    print("=" * 50)
    # 打印距离和预警信息
    print(f"当前后方障碍物距离：{distance:.1f} 米")
    print(f"预警信息（{color}）：{text}")
    print("=" * 50)
    print("按 Ctrl+C 退出程序")

# -------------------------- 主循环 --------------------------
def main():
    warning_system = RearCollisionWarning()
    try:
        while True:
            # 更新障碍物距离
            warning_system.simulate_distance_detection()
            # 获取预警信息
            text, color = warning_system.get_warning_info()
            # 控制台输出
            print_warning(text, color, warning_system.rear_distance)
            # 每秒更新一次（模拟实时检测频率）
            time.sleep(1)
    except KeyboardInterrupt:
        # 捕获Ctrl+C，优雅退出
        print("\n程序已退出")
        sys.exit(0)

if __name__ == '__main__':
    main()