import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

# -------------------------- 初始化参数设置 --------------------------
# 道路参数
road_width = 6  # 道路总宽度（m）
lane_width = 3  # 单车道宽度（m）
road_length = 100  # 道路长度（m）

# 车辆参数
car_width = 1.8  # 车宽（m）
car_length = 4.5  # 车长（m）
car_speed = 15  # 行驶速度（m/s）
lane_change_duration = 2.0  # 变道总时长（s）
blinker_duration = 0.5  # 转向灯闪烁周期（s）

# 初始状态
start_lane = 1  # 起始车道（1=左车道，2=右车道）
target_lane = 2  # 目标车道（1=左车道，2=右车道）
initial_x = 10  # 初始x坐标（道路起始位置）
initial_y = (start_lane - 1) * lane_width + lane_width / 2  # 初始y坐标（车道中心线）
target_y = (target_lane - 1) * lane_width + lane_width / 2  # 目标车道中心线y坐标

# 安全距离参数
safety_distance_front = 20  # 前车安全距离（m）
safety_distance_back = 15  # 后车安全距离（m）

# 动画参数
fps = 30  # 帧率
total_time = 5  # 动画总时长（s）
num_frames = int(fps * total_time)  # 总帧数

# -------------------------- 创建画布和坐标轴 --------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, road_length)
ax.set_ylim(-1, road_width + 1)
ax.set_xlabel('距离 (m)')
ax.set_ylabel('道路宽度 (m)')
ax.set_title('无人车自动变道模拟')
ax.grid(True, alpha=0.3)

# 绘制道路（灰色背景）
road = Rectangle((0, 0), road_length, road_width, facecolor='#f0f0f0', edgecolor='black', linewidth=2)
ax.add_patch(road)

# 绘制车道线（白色虚线）
dashed_x = np.arange(0, road_length, 5)  # 虚线分段x坐标
for y in [lane_width]:  # 中间车道线
    for x in dashed_x:
        lane_line = Rectangle((x, y - 0.1), 3, 0.2, facecolor='white', linewidth=0)
        ax.add_patch(lane_line)

# -------------------------- 初始化车辆和元素 --------------------------
# 无人车（蓝色车身）
car = Rectangle((initial_x - car_length / 2, initial_y - car_width / 2),
                car_length, car_width, facecolor='royalblue', edgecolor='black', linewidth=2)
ax.add_patch(car)

# 转向灯（左右各一个，初始透明）
left_blinker = Circle((initial_x - car_length / 2 + 0.5, initial_y), 0.3, facecolor='yellow', alpha=0)
right_blinker = Circle((initial_x + car_length / 2 - 0.5, initial_y), 0.3, facecolor='yellow', alpha=0)
ax.add_patch(left_blinker)
ax.add_patch(right_blinker)

# 前车（灰色，用于安全距离检测演示）
front_car = Rectangle((initial_x + safety_distance_front + car_length, target_y - car_width / 2),
                      car_length, car_width, facecolor='gray', edgecolor='black', linewidth=2)
ax.add_patch(front_car)

# 后车（灰色，用于安全距离检测演示）
back_car = Rectangle((initial_x - safety_distance_back - car_length, target_y - car_width / 2),
                     car_length, car_width, facecolor='gray', edgecolor='black', linewidth=2)
ax.add_patch(back_car)

# 状态文本（显示变道状态）
status_text = ax.text(road_length - 30, road_width - 1, '状态：正常行驶',
                      fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# -------------------------- 动画更新函数 --------------------------
def update(frame):
    # 计算当前时间
    t = frame / fps

    # 基础行驶：x方向匀速移动
    current_x = initial_x + car_speed * t

    # 变道逻辑（0.5s准备→1.5s变道→2.5s后结束）
    if t < 0.5:
        # 阶段1：检测安全距离，准备变道（转向灯闪烁）
        current_y = initial_y
        blinker_alpha = 1 if (t % blinker_duration) < blinker_duration / 2 else 0
        if start_lane < target_lane:  # 向右变道→右转向灯
            right_blinker.set_alpha(blinker_alpha)
            left_blinker.set_alpha(0)
        else:  # 向左变道→左转向灯
            left_blinker.set_alpha(blinker_alpha)
            right_blinker.set_alpha(0)
        status_text.set_text('状态：检测安全距离')

    elif 0.5 <= t < 0.5 + lane_change_duration:
        # 阶段2：平稳变道（正弦曲线平滑过渡y坐标）
        lane_change_t = t - 0.5  # 变道时间（0~2s）
        # 正弦函数实现平滑变道（0→π，y从初始→目标）
        y_ratio = (1 - np.cos(np.pi * lane_change_t / lane_change_duration)) / 2
        current_y = initial_y + (target_y - initial_y) * y_ratio
        # 转向灯常亮
        if start_lane < target_lane:
            right_blinker.set_alpha(1)
        else:
            left_blinker.set_alpha(1)
        status_text.set_text('状态：正在变道')

    else:
        # 阶段3：变道完成，恢复正常行驶
        current_y = target_y
        left_blinker.set_alpha(0)
        right_blinker.set_alpha(0)
        status_text.set_text('状态：变道完成')

    # 更新车辆位置
    car.set_xy((current_x - car_length / 2, current_y - car_width / 2))

    # 更新前后车位置（随道路同步移动，模拟相对静止）
    front_car.set_xy((current_x + safety_distance_front + car_length / 2, target_y - car_width / 2))
    back_car.set_xy((current_x - safety_distance_back - 3 * car_length / 2, target_y - car_width / 2))

    # 边界处理：车辆驶出画面后重置x坐标
    if current_x > road_length + car_length / 2:
        car.set_xy((-car_length / 2, current_y - car_width / 2))

    return car, left_blinker, right_blinker, front_car, back_car, status_text


# -------------------------- 生成动画 --------------------------
# 创建动画（blit=True加速渲染）
ani = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=1000 / fps, blit=True, repeat=True
)

# 显示动画
plt.tight_layout()
plt.show()

# （可选）保存动画为GIF（需要安装ffmpeg或pillow）
# ani.save('autonomous_lane_change.gif', writer='pillow', fps=fps, dpi=150)s