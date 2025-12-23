import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

# -------------------------- 初始化参数设置 --------------------------
# 道路参数：定义道路物理尺寸
road_width = 6  # 道路总宽度（单位：米）
lane_width = 3  # 单车道宽度（单位：米），双车道总宽=3*2=6m，与road_width一致
road_length = 100  # 道路总长度（单位：米），即动画横向范围

# 车辆参数：定义无人车物理属性和运动特性
car_width = 1.8  # 车身宽度（单位：米），小于车道宽度保证安全
car_length = 4.5  # 车身长度（单位：米），符合常见轿车尺寸
car_speed = 15  # 恒定行驶速度（单位：米/秒），约54km/h，城市道路常见速度
lane_change_duration = 2.0  # 完成一次变道的总时间（单位：秒），保证平稳性
blinker_duration = 0.5  # 转向灯闪烁周期（单位：秒），0.5秒亮/0.5秒灭

# 初始状态：定义车辆起始和目标状态
start_lane = 1  # 起始车道（1=左车道，2=右车道）
target_lane = 2  # 目标车道（1=左车道，2=右车道），此处为左→右变道
initial_x = 10  # 初始x坐标（道路起始方向位置），距离道路左端10m
# 初始y坐标：计算起始车道中心线位置（车道索引*车道宽 - 车道宽/2）
initial_y = (start_lane - 1) * lane_width + lane_width / 2
# 目标y坐标：计算目标车道中心线位置，与初始y坐标相差一个车道宽（3m）
target_y = (target_lane - 1) * lane_width + lane_width / 2

# 安全距离参数：变道时与前后车的安全间隔
safety_distance_front = 20  # 与前车的最小安全距离（单位：米）
safety_distance_back = 15  # 与后车的最小安全距离（单位：米）

# 动画参数：控制动画播放效果
fps = 30  # 帧率（单位：帧/秒），30fps为流畅动画标准
total_time = 5  # 动画总时长（单位：秒），包含变道全流程
num_frames = int(fps * total_time)  # 总帧数 = 帧率*时长，用于动画迭代

# -------------------------- 创建画布和坐标轴 --------------------------
# 创建绘图画布和子图，设置尺寸为12x6英寸（宽x高）
fig, ax = plt.subplots(figsize=(12, 6))
# 设置x轴范围（道路长度），y轴范围（略大于道路宽度，留出边界）
ax.set_xlim(0, road_length)
ax.set_ylim(-1, road_width + 1)
# 设置坐标轴标签和标题，明确图表含义
ax.set_xlabel('距离 (m)')
ax.set_ylabel('道路宽度 (m)')
ax.set_title('无人车自动变道模拟')
ax.grid(True, alpha=0.3)  # 显示网格线，透明度0.3避免遮挡

# 绘制道路背景：灰色矩形表示路面
road = Rectangle(
    (0, 0),  # 左下角坐标（x=0, y=0）
    road_length, road_width,  # 宽=道路长度，高=道路宽度
    facecolor='#f0f0f0',  # 路面颜色（浅灰色）
    edgecolor='black', linewidth=2  # 道路边界线（黑色，线宽2）
)
ax.add_patch(road)  # 将道路图形添加到子图

# 绘制车道线：白色虚线分隔双车道
dashed_x = np.arange(0, road_length, 5)  # 虚线分段x坐标（每5m一段）
for y in [lane_width]:  # 中间车道线的y坐标（3m处，即两车道分界）
    for x in dashed_x:
        # 每个虚线段：长3m、宽0.2m的白色矩形
        lane_line = Rectangle(
            (x, y - 0.1),  # 左下角坐标（x, 2.9m）
            3, 0.2,  # 长3m，宽0.2m（覆盖3m±0.1m范围）
            facecolor='white', linewidth=0  # 白色填充，无边框
        )
        ax.add_patch(lane_line)  # 添加虚线到子图

# -------------------------- 初始化车辆和元素 --------------------------
# 无人车主车身：蓝色矩形（代表本车）
car = Rectangle(
    (initial_x - car_length / 2, initial_y - car_width / 2),  # 左下角坐标（基于中心偏移）
    car_length, car_width,  # 长=车身长，宽=车身宽
    facecolor='royalblue',  # 车身颜色（宝蓝色）
    edgecolor='black', linewidth=2  # 车身边框（黑色，线宽2）
)
ax.add_patch(car)  # 添加车身到子图

# 转向灯：左右各一个黄色圆形，初始透明（alpha=0）
# 左转向灯：位于车身左上角（x=车头左前+0.5m，y=车身中心）
left_blinker = Circle(
    (initial_x - car_length / 2 + 0.5, initial_y),  # 圆心坐标
    0.3,  # 半径（0.3m，符合转向灯实际尺寸）
    facecolor='yellow', alpha=0  # 黄色填充，初始透明
)
# 右转向灯：位于车身右上角（x=车尾右后-0.5m，y=车身中心）
right_blinker = Circle(
    (initial_x + car_length / 2 - 0.5, initial_y),  # 圆心坐标
    0.3,  # 半径
    facecolor='yellow', alpha=0  # 黄色填充，初始透明
)
ax.add_patch(left_blinker)  # 添加左转向灯
ax.add_patch(right_blinker)  # 添加右转向灯

# 前车：灰色矩形（用于演示变道时的前车安全距离）
front_car = Rectangle(
    (initial_x + safety_distance_front + car_length, target_y - car_width / 2),  # 初始位置（目标车道前车）
    car_length, car_width,  # 尺寸与本车一致
    facecolor='gray', edgecolor='black', linewidth=2  # 灰色车身，黑色边框
)
ax.add_patch(front_car)  # 添加前车到子图

# 后车：灰色矩形（用于演示变道时的后车安全距离）
back_car = Rectangle(
    (initial_x - safety_distance_back - car_length, target_y - car_width / 2),  # 初始位置（目标车道后车）
    car_length, car_width,  # 尺寸与本车一致
    facecolor='gray', edgecolor='black', linewidth=2  # 灰色车身，黑色边框
)
ax.add_patch(back_car)  # 添加后车到子图

# 状态文本：显示当前变道阶段，位于画面右上角
status_text = ax.text(
    road_length - 30, road_width - 1,  # 文本位置（x=70m，y=5m）
    '状态：正常行驶',  # 初始文本
    fontsize=12,  # 字体大小
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)  # 白色圆角背景，透明度0.8
)

# -------------------------- 动画更新函数 --------------------------
def update(frame):
    """
    动画每一帧的更新逻辑：根据当前帧数计算车辆状态并更新图形
    参数frame：当前帧数（从0到num_frames-1）
    返回：所有需要更新的图形元素（用于blit优化）
    """
    # 计算当前时间t：帧数/帧率（单位：秒）
    t = frame / fps

    # 基础行驶逻辑：x方向匀速移动（v*t，初始位置+速度*时间）
    current_x = initial_x + car_speed * t

    # 变道三阶段逻辑：准备→变道→完成
    if t < 0.5:
        # 阶段1：安全距离检测（前0.5秒）- 准备变道
        current_y = initial_y  # y坐标保持初始车道（不变道）
        # 转向灯闪烁：根据时间取模控制亮灭（0~0.25s亮，0.25~0.5s灭）
        blinker_alpha = 1 if (t % blinker_duration) < blinker_duration / 2 else 0
        # 根据变道方向激活对应转向灯（左→右变道激活右转向灯）
        if start_lane < target_lane:  # 起始车道 < 目标车道 → 向右变道
            right_blinker.set_alpha(blinker_alpha)  # 右转向灯闪烁
            left_blinker.set_alpha(0)  # 左转向灯关闭
        else:  # 起始车道 > 目标车道 → 向左变道
            left_blinker.set_alpha(blinker_alpha)  # 左转向灯闪烁
            right_blinker.set_alpha(0)  # 右转向灯关闭
        status_text.set_text('状态：检测安全距离')  # 更新状态文本

    elif 0.5 <= t < 0.5 + lane_change_duration:
        # 阶段2：平稳变道（0.5~2.5秒，共2秒变道时长）
        lane_change_t = t - 0.5  # 变道时间偏移（从0开始计算变道时长）
        # 正弦曲线平滑过渡y坐标：实现"慢-快-慢"的平稳变道
        # 原理：cos(0)=1 → cos(π)=-1，因此y_ratio从0→1线性过渡（平滑）
        y_ratio = (1 - np.cos(np.pi * lane_change_t / lane_change_duration)) / 2
        # 当前y坐标 = 初始y + 变道幅度*过渡比例（从初始车道→目标车道）
        current_y = initial_y + (target_y - initial_y) * y_ratio
        # 转向灯常亮（变道过程中持续提示）
        if start_lane < target_lane:
            right_blinker.set_alpha(1)  # 右转向灯常亮
        else:
            left_blinker.set_alpha(1)  # 左转向灯常亮
        status_text.set_text('状态：正在变道')  # 更新状态文本

    else:
        # 阶段3：变道完成（2.5秒后）- 正常行驶
        current_y = target_y  # y坐标固定在目标车道中心线
        left_blinker.set_alpha(0)  # 关闭左转向灯
        right_blinker.set_alpha(0)  # 关闭右转向灯
        status_text.set_text('状态：变道完成')  # 更新状态文本

    # 更新本车位置：根据当前x、y坐标调整矩形左下角位置（基于中心偏移）
    car.set_xy((current_x - car_length / 2, current_y - car_width / 2))

    # 更新前后车位置：模拟相对静止（随本车同步移动，保持安全距离）
    # 前车位置：本车x + 安全距离 + 车身长度/2（保证前车尾部与本车头部距离）
    front_car.set_xy((current_x + safety_distance_front + car_length / 2, target_y - car_width / 2))
    # 后车位置：本车x - 安全距离 - 3*车身长度/2（保证后车头部与本车尾部距离）
    back_car.set_xy((current_x - safety_distance_back - 3 * car_length / 2, target_y - car_width / 2))

    # 边界处理：当本车完全驶出道路右端（x>100+2.25m）时，重置到道路左端
    if current_x > road_length + car_length / 2:
        car.set_xy((-car_length / 2, current_y - car_width / 2))  # 左端重置位置

    # 返回所有需要更新的图形元素（blit=True时只重绘这些元素，加速渲染）
    return car, left_blinker, right_blinker, front_car, back_car, status_text

# -------------------------- 生成动画 --------------------------
# 创建动画对象：基于FuncAnimation（逐帧更新）
ani = animation.FuncAnimation(
    fig,  # 绘图画布
    update,  # 每一帧的更新函数
    frames=num_frames,  # 总帧数（0到num_frames-1）
    interval=1000 / fps,  # 帧间隔（单位：毫秒）=1000ms/帧率
    blit=True,  # 开启blit优化（只重绘变化的元素，提升流畅度）
    repeat=True  # 动画循环播放
)

# 调整子图布局（避免元素重叠）并显示动画
plt.tight_layout()
plt.show()

# （可选）保存动画为GIF文件（需要安装pillow库）
# ani.save('autonomous_lane_change.gif', writer='pillow', fps=fps, dpi=150)