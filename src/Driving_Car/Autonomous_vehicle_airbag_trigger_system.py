import tkinter as tk
import random
import time

# -------------------------- 配置参数 --------------------------
# 压力阈值（单位：牛）
LOW_PRESSURE_THRESHOLD = 40000   # 4万牛
MAX_PRESSURE = 2000000           # 200万牛
UPDATE_INTERVAL = 300  # 界面更新间隔（毫秒），约30FPS

# -------------------------- 主窗口初始化 --------------------------
root = tk.Tk()
root.title("Autonomous Vehicle Airbag Trigger System")
root.geometry("800x600")  # 窗口大小

# 用于显示文字的标签（居中、大字体）
text_label = tk.Label(
    root,
    font=("SimHei", 36),  # 黑体，36号字（支持中文）
    bg="black"  # 黑色背景
)
text_label.pack(expand=True, fill="both")  # 标签占满整个窗口

# 存储当前压力值
current_pressure = 0.0

# -------------------------- 核心函数 --------------------------
def generate_pressure():
    """生成0~200万牛的模拟压力数据"""
    global current_pressure
    # 5%概率触发高压力（碰撞），否则低压力
    if random.random() < 0.05:
        # 4万~200万牛：随机值
        current_pressure = random.uniform(LOW_PRESSURE_THRESHOLD, MAX_PRESSURE)
    else:
        # 0~4万牛：随机值
        current_pressure = random.uniform(0, LOW_PRESSURE_THRESHOLD)
    return current_pressure

def update_ui():
    """更新界面文字和颜色"""
    # 生成新的压力数据
    pressure = generate_pressure()
    # 判断压力区间，设置文字和颜色
    if 0 <= pressure <= LOW_PRESSURE_THRESHOLD:
        # 黄色：车辆碰撞
        text_label.config(text="车辆碰撞", fg="#FFFF00")
    elif LOW_PRESSURE_THRESHOLD < pressure < MAX_PRESSURE:
        # 红色：危险，安全气囊已打开
        text_label.config(text="危险，安全气囊已打开", fg="#FF0000")
    else:
        # 灰色：无碰撞（超出范围的情况）
        text_label.config(text="无碰撞", fg="#808080")
    # 打印压力值到控制台（调试用）
    print(f"当前压力：{pressure:.2f} N")
    # 定时更新（递归调用，实现循环）
    root.after(UPDATE_INTERVAL, update_ui)

# -------------------------- 启动程序 --------------------------
if __name__ == "__main__":
    # 首次调用更新函数
    update_ui()
    # 启动主循环
    root.mainloop()