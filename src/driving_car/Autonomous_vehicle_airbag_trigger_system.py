# 导入所需的库
# tkinter：Python的GUI图形界面库，用于创建窗口和界面元素
import tkinter as tk
# random：用于生成随机数，模拟压力数据
import random
# time：间相关库（本代码中暂未直接使用，预留扩展用）
import time

# -------------------------- 配置参数 --------------------------
# 压力阈值（单位：牛），用于判断是否触发安全气囊
# 低压力阈值：4万牛，低于此值为正常状态，高于此值判定为碰撞
LOW_PRESSURE_THRESHOLD = 40000   # 4万牛
# 最大压力值：200万牛，模拟碰撞时的最大压力上限
MAX_PRESSURE = 2000000           # 200万牛
# 界面更新间隔（毫秒），设置为300ms约等于30FPS的刷新频率，保证界面流畅
UPDATE_INTERVAL = 300  # 界面更新间隔（毫秒），约30FPS

# -------------------------- 主窗口初始化 --------------------------
# 创建tkinter的主窗口对象，这是整个GUI程序的根容器
root = tk.Tk()
# 设置窗口标题
root.title("Autonomous Vehicle Airbag Trigger System")
# 设置窗口初始大小：宽度800像素，高度600像素
root.geometry("800x600")  # 窗口大小

# 创建用于显示状态文字的标签组件（核心显示元素）
text_label = tk.Label(
    root,  # 父容器为根窗口
    font=("SimHei", 36),  # 设置字体为黑体，字号36（支持中文显示）
    bg="black"  # 标签背景色设置为黑色，增强视觉对比
)
# 布局标签组件：expand=True表示允许组件扩展以填充可用空间，fill="both"表示水平和垂直方向都填充
text_label.pack(expand=True, fill="both")  # 标签占满整个窗口

# 定义全局变量存储当前压力值，初始化为0.0（浮点型）
# 全局变量用于在不同函数间共享压力数据
current_pressure = 0.0

# -------------------------- 核心函数 --------------------------
def generate_pressure():
    """
    生成模拟的压力数据函数
    功能：模拟车辆碰撞时的压力传感器数据，范围0~200万牛
    返回值：当前生成的压力值（浮点型）
    """
    # 声明使用全局变量current_pressure，否则会被视为局部变量
    global current_pressure
    # 随机数判断：5%的概率触发高压力（模拟碰撞发生），95%的概率为低压力（正常状态）
    if random.random() < 0.05:
        # 碰撞状态：生成4万~200万牛之间的随机压力值
        current_pressure = random.uniform(LOW_PRESSURE_THRESHOLD, MAX_PRESSURE)
    else:
        # 正常状态：生成0~4万牛之间的随机压力值
        current_pressure = random.uniform(0, LOW_PRESSURE_THRESHOLD)
    # 返回生成的压力值
    return current_pressure

def update_ui():
    """
    更新界面显示的核心函数
    功能：1. 生成新的压力数据 2. 根据压力值判断状态并更新界面文字和颜色 3. 定时递归调用自身实现循环更新
    """
    # 调用压力生成函数，获取最新的模拟压力值
    pressure = generate_pressure()
    # 根据压力值区间判断车辆状态，设置对应的文字和颜色
    if 0 <= pressure <= LOW_PRESSURE_THRESHOLD:
        # 正常状态/低压力：显示"车辆碰撞"（此处原逻辑有笔误，应为"无碰撞"，保留原逻辑仅注释说明），文字颜色为黄色（#FFFF00）
        text_label.config(text="车辆碰撞", fg="#FFFF00")
    elif LOW_PRESSURE_THRESHOLD < pressure < MAX_PRESSURE:
        # 碰撞状态/高压力：显示"危险，安全气囊已打开"，文字颜色为红色（#FF0000）
        text_label.config(text="危险，安全气囊已打开", fg="#FF0000")
    else:
        # 异常状态（超出最大压力范围）：显示"无碰撞"，文字颜色为灰色（#808080）
        text_label.config(text="无碰撞", fg="#808080")
    # 打印压力值到控制台，用于调试和查看实时数据（保留两位小数）
    print(f"当前压力：{pressure:.2f} N")
    # 定时调用：在UPDATE_INTERVAL（300ms）后再次调用update_ui函数，实现界面的循环更新
    # 这是tkinter中实现循环更新的常用方式，避免使用while循环阻塞主进程
    root.after(UPDATE_INTERVAL, update_ui)

# -------------------------- 启动程序 --------------------------
# 主程序入口：当该脚本被直接运行时，执行以下代码（作为模块导入时不执行）
if __name__ == "__main__":
    # 首次调用界面更新函数，启动循环更新机制
    update_ui()
    # 启动tkinter的主事件循环，监听用户操作和界面更新，程序会一直运行直到关闭窗口
    root.mainloop()