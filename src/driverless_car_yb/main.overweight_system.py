import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.widgets import Button

# -------------------------- 初始化参数配置 --------------------------
MAX_LOAD = 100.0  # 无人车载重上限（单位：kg）
INITIAL_WEIGHT = 30.0  # 初始重量（空车重量）
LOAD_SPEED = 5.0  # 加载速度（kg/帧）
UNLOAD_SPEED = 8.0  # 卸载速度（kg/帧）
OVERLOAD_THRESHOLD = 1.1  # 超重阈值（110%载重上限）
FRAME_INTERVAL = 200  # 动画帧间隔（ms）
TOTAL_FRAMES = 120  # 总动画帧数

# -------------------------- 创建画布和子图 --------------------------
fig, (ax_weight, ax_progress) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('无人车超重检测系统', fontsize=16, fontweight='bold')

# 调整子图间距和按钮位置
plt.subplots_adjust(bottom=0.15, hspace=0.3)
ax_pause = plt.axes([0.4, 0.05, 0.1, 0.04])  # 暂停/继续按钮
ax_reset = plt.axes([0.5, 0.05, 0.1, 0.04])  # 重置按钮

# -------------------------- 初始化可视化元素 --------------------------
# 1. 重量显示图
ax_weight.set_xlim(0, TOTAL_FRAMES)
ax_weight.set_ylim(0, MAX_LOAD * 1.3)  # 预留超重显示空间
ax_weight.set_xlabel('时间（帧）')
ax_weight.set_ylabel('重量（kg）')
ax_weight.grid(True, alpha=0.3)

# 绘制载重上限线和超重阈值线
ax_weight.axhline(y=MAX_LOAD, color='green', linestyle='--', linewidth=2, label=f'载重上限: {MAX_LOAD}kg')
ax_weight.axhline(y=MAX_LOAD * OVERLOAD_THRESHOLD, color='red', linestyle='--', linewidth=2,
                  label=f'超重阈值: {MAX_LOAD * OVERLOAD_THRESHOLD:.1f}kg')

# 实时重量曲线
weight_line, = ax_weight.plot([], [], color='blue', linewidth=3, marker='o', markersize=4, label='当前重量')
ax_weight.legend(loc='upper left')

# 2. 载重进度条
progress_bar = ax_progress.barh(y=0, width=0, height=0.6, color='green', alpha=0.8)
ax_progress.set_xlim(0, 100)  # 进度条按百分比显示
ax_progress.set_ylim(-1, 1)
ax_progress.set_xlabel('载重占比（%）')
ax_progress.set_yticks([])
ax_progress.grid(True, alpha=0.3, axis='x')

# 进度条百分比文本
progress_text = ax_progress.text(50, 0, '0%', ha='center', va='center', fontsize=14, fontweight='bold')

# 3. 状态提示文本（超重报警）
status_text = fig.text(0.5, 0.85, '状态：正常 - 等待加载', ha='center', fontsize=14, fontweight='bold', color='green')

# -------------------------- 全局变量 --------------------------
current_weight = INITIAL_WEIGHT
current_frame = 0
is_paused = False
is_overloaded = False
weight_history = [current_weight]  # 重量历史记录


# -------------------------- 按钮回调函数 --------------------------
def pause_resume(event):
    global is_paused
    is_paused = not is_paused
    if is_paused:
        pause_btn.label.set_text('继续')
        status_text.set_text('状态：暂停')
        status_text.set_color('orange')
    else:
        pause_btn.label.set_text('暂停')
        status_text.set_text('状态：正常 - 加载中' if not is_overloaded else '状态：超重 - 卸载中')
        status_text.set_color('green' if not is_overloaded else 'red')


def reset_system(event):
    global current_weight, current_frame, is_paused, is_overloaded, weight_history
    current_weight = INITIAL_WEIGHT
    current_frame = 0
    is_paused = False
    is_overloaded = False
    weight_history = [current_weight]
    pause_btn.label.set_text('暂停')
    status_text.set_text('状态：正常 - 等待加载')
    status_text.set_color('green')
    # 重置可视化元素
    weight_line.set_data([], [])
    progress_bar[0].set_width(0)
    progress_text.set_text(f'{current_weight / MAX_LOAD * 100:.0f}%')
    progress_bar[0].set_color('green')


# 创建按钮
pause_btn = Button(ax_pause, '暂停', color='lightblue', hovercolor='lightgreen')
reset_btn = Button(ax_reset, '重置', color='lightcoral', hovercolor='lightpink')
pause_btn.on_clicked(pause_resume)
reset_btn.on_clicked(reset_system)


# -------------------------- 动画更新函数 --------------------------
def update_animation(frame):
    global current_weight, current_frame, is_overloaded

    if is_paused:
        return weight_line, progress_bar, progress_text, status_text

    current_frame += 1
    load_percent = current_weight / MAX_LOAD * 100  # 计算载重占比

    # -------------------------- 重量变化逻辑 --------------------------
    if current_frame < 30:  # 前30帧：加载货物
        current_weight += LOAD_SPEED
        status_text.set_text(f'状态：正常 - 加载中（{current_weight:.1f}kg）')
        status_text.set_color('green')
    elif 30 <= current_frame < 50:  # 30-50帧：继续加载至超重
        current_weight += LOAD_SPEED
        if current_weight >= MAX_LOAD * OVERLOAD_THRESHOLD:
            is_overloaded = True
            status_text.set_text(f'状态：超重报警！（{current_weight:.1f}kg）')
            status_text.set_color('red')
            progress_bar[0].set_color('red')
    elif 50 <= current_frame < 90:  # 50-90帧：超重后卸载
        current_weight -= UNLOAD_SPEED
        if current_weight <= MAX_LOAD:
            is_overloaded = False
            status_text.set_text(f'状态：正常 - 卸载中（{current_weight:.1f}kg）')
            status_text.set_color('green')
            progress_bar[0].set_color('green')
    else:  # 90帧后：保持正常重量
        current_weight = MAX_LOAD * 0.8  # 稳定在80%载重
        status_text.set_text(f'状态：正常 - 稳定运行（{current_weight:.1f}kg）')
        status_text.set_color('green')

    # 防止重量为负或超过最大显示范围
    current_weight = max(INITIAL_WEIGHT, min(current_weight, MAX_LOAD * 1.3))
    weight_history.append(current_weight)

    # -------------------------- 更新可视化元素 --------------------------
    # 1. 重量曲线
    x_data = list(range(len(weight_history)))
    weight_line.set_data(x_data, weight_history)
    ax_weight.set_xlim(0, max(len(weight_history), TOTAL_FRAMES))  # 自适应x轴范围

    # 2. 进度条
    load_percent = current_weight / MAX_LOAD * 100
    progress_bar[0].set_width(load_percent)
    progress_text.set_text(f'{load_percent:.0f}%')

    # 3. 超重闪烁效果（每2帧切换透明度）
    if is_overloaded:
        alpha = 0.5 if frame % 4 < 2 else 1.0
        progress_bar[0].set_alpha(alpha)
        status_text.set_alpha(alpha)
    else:
        progress_bar[0].set_alpha(0.8)
        status_text.set_alpha(1.0)

    return weight_line, progress_bar[0], progress_text, status_text


# -------------------------- 启动动画 --------------------------
ani = animation.FuncAnimation(
    fig, update_animation,
    frames=TOTAL_FRAMES,
    interval=FRAME_INTERVAL,
    blit=False,  # 此处设为False以支持文本更新
    repeat=False  # 不重复播放
)

# 可选：保存动画为GIF（需要安装pillow库）
# ani.save('无人车超重检测.gif', writer='pillow', dpi=150, fps=5)

plt.show()