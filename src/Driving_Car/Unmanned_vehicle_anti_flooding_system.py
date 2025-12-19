import tkinter as tk
import random

# ------------------- 主窗口配置 -------------------
root = tk.Tk()
root.title("无人车防泡水系统")
root.geometry("800x600")  # 窗口大小
root.resizable(False, False)  # 禁止调整窗口大小

# ------------------- 全局变量 -------------------
current_depth = tk.DoubleVar(value=0.0)  # 存储当前水深，初始值0.0
status_text = tk.StringVar(value="安全")  # 存储状态文字，初始值“安全”
status_color = tk.StringVar(value="green")  # 存储状态文字颜色，初始值绿色


# ------------------- 水深传感器模拟函数 -------------------
def update_water_depth():
    """模拟底盘水深传感器数据（0-10cm随机波动），每秒更新一次"""
    # 生成0-10cm的随机水深
    depth = random.uniform(0.0, 10.0)
    current_depth.set(depth)

    # 根据水深判断状态
    if 0.0 <= depth <= 5.0:
        status_text.set("安全")
        status_color.set("green")  # 绿色
    else:
        status_text.set("注意安全")
        status_color.set("red")  # 红色

    # 每秒调用一次自身，实现实时更新
    root.after(1000, update_water_depth)


# ------------------- UI布局 -------------------
# 1. 状态文字标签（居中、大号字体）
status_label = tk.Label(
    root,
    textvariable=status_text,
    font=("SimHei", 80),  # 中文黑体，80号字体
    fg=status_color.get()
)
status_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)  # 居中放置

# 2. 水深数值标签（居中、中号字体）
depth_label = tk.Label(
    root,
    text=f"当前底盘水深：{current_depth.get():.1f} cm",
    font=("SimHei", 40),  # 中文黑体，40号字体
    fg="black"
)
depth_label.place(relx=0.5, rely=0.6, anchor=tk.CENTER)


# 实时更新标签内容的函数
def update_labels():
    # 更新水深数值
    depth_text = f"当前底盘水深：{current_depth.get():.1f} cm"
    depth_label.config(text=depth_text)
    # 更新状态文字颜色
    status_label.config(fg=status_color.get())
    # 每100ms更新一次，保证流畅
    root.after(100, update_labels)


# ------------------- 启动更新 -------------------
update_water_depth()  # 启动水深更新
update_labels()  # 启动标签更新

# ------------------- 主循环 -------------------
root.mainloop()