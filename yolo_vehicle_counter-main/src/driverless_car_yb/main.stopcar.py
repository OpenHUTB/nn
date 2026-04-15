import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# -------------------------- 1. 系统参数配置 --------------------------
# 车辆参数（简化 Ackermann 模型）
CAR_LENGTH = 4.5  # 车长(m)
CAR_WIDTH = 1.8  # 车宽(m)
WHEELBASE = 2.8  # 轴距(m)
MAX_STEER = np.radians(30)  # 最大转向角(弧度)

# 泊车场景参数
PARKING_SPOT_LENGTH = 5.0  # 车位长度(m)
PARKING_SPOT_WIDTH = 2.5  # 车位宽度(m)
PARKING_POS = np.array([10, 5])  # 车位中心坐标(x,y)
INIT_CAR_POS = np.array([2, 5])  # 车辆初始位置(x,y)
INIT_CAR_YAW = 0.0  # 车辆初始航向角(0=水平向右)

# 控制参数
STEP_TIME = 0.1  # 仿真步长(s)
MAX_ITER = 200  # 最大仿真步数（防止无限循环）
TARGET_POS_TOL = 0.3  # 位置误差容忍度(m)
TARGET_YAW_TOL = np.radians(5)  # 航向角误差容忍度(弧度)


# -------------------------- 2. 路径规划（几何泊车路径） --------------------------
def plan_parking_path(init_pos, init_yaw, target_pos, target_yaw):
    """
    规划自动泊车路径（两段圆弧+直线的组合路径）
    :return: 路径点列表 [(x1,y1,yaw1), (x2,y2,yaw2), ...]
    """
    path = []
    current_pos = init_pos.copy()
    current_yaw = init_yaw
    path.append((current_pos[0], current_pos[1], current_yaw))

    # 阶段1：第一段圆弧（向右转向，靠近车位）
    steer_angle = MAX_STEER * 0.8  # 转向角（小于最大转向角）
    curve_radius = WHEELBASE / np.tan(steer_angle)  # 转弯半径
    step_num = 30  # 该阶段步数
    for _ in range(step_num):
        # 车辆运动学更新
        current_yaw += (np.tan(steer_angle) / WHEELBASE) * STEP_TIME * 10  # 角速度=v*tan(δ)/L（v=10m/s简化）
        current_pos[0] += 10 * np.cos(current_yaw) * STEP_TIME
        current_pos[1] += 10 * np.sin(current_yaw) * STEP_TIME
        path.append((current_pos[0], current_pos[1], current_yaw))

    # 阶段2：直线行驶（调整横向位置）
    steer_angle = 0.0  # 回正方向盘
    step_num = 20
    for _ in range(step_num):
        current_pos[0] += 10 * np.cos(current_yaw) * STEP_TIME
        current_pos[1] += 10 * np.sin(current_yaw) * STEP_TIME
        path.append((current_pos[0], current_pos[1], current_yaw))

    # 阶段3：第二段圆弧（向左转向，对准车位）
    steer_angle = -MAX_STEER * 0.8  # 反向转向
    step_num = 35
    for _ in range(step_num):
        current_yaw += (np.tan(steer_angle) / WHEELBASE) * STEP_TIME * 10
        current_pos[0] += 10 * np.cos(current_yaw) * STEP_TIME
        current_pos[1] += 10 * np.sin(current_yaw) * STEP_TIME
        path.append((current_pos[0], current_pos[1], current_yaw))

    # 阶段4：直线入库（精准停车）
    steer_angle = 0.0
    step_num = 15
    for _ in range(step_num):
        current_pos[0] += 5 * np.cos(current_yaw) * STEP_TIME  # 减速
        current_pos[1] += 5 * np.sin(current_yaw) * STEP_TIME
        path.append((current_pos[0], current_pos[1], current_yaw))

    return path


# -------------------------- 3. 车辆绘制函数 --------------------------
def draw_car(ax, x, y, yaw, color='blue'):
    """在指定坐标轴绘制车辆（矩形+航向箭头）"""
    # 车辆四个顶点（基于中心坐标和航向角旋转）
    car_corners = np.array([
        [CAR_LENGTH / 2, CAR_WIDTH / 2],
        [CAR_LENGTH / 2, -CAR_WIDTH / 2],
        [-CAR_LENGTH / 2, -CAR_WIDTH / 2],
        [-CAR_LENGTH / 2, CAR_WIDTH / 2]
    ])
    # 旋转矩阵
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    rotated_corners = (rot_mat @ car_corners.T).T + np.array([x, y])

    # 绘制车身
    ax.fill(rotated_corners[:, 0], rotated_corners[:, 1], color=color, alpha=0.7)
    # 绘制航向箭头（车头方向）
    arrow_len = CAR_LENGTH / 2
    ax.arrow(x, y, arrow_len * np.cos(yaw), arrow_len * np.sin(yaw),
             head_width=0.3, head_length=0.5, fc='red', ec='red')


# -------------------------- 4. 动画主函数 --------------------------
def main():
    # 1. 规划泊车路径
    target_yaw = np.radians(0)  # 车位航向角（与初始方向一致）
    parking_path = plan_parking_path(INIT_CAR_POS, INIT_CAR_YAW, PARKING_POS, target_yaw)

    # 2. 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(2, 8)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Autonomous Parking Simulation (Self-Driving Car)')
    ax.grid(True)
    ax.axis('equal')

    # 3. 绘制车位（灰色矩形）
    parking_corner = PARKING_POS - np.array([PARKING_SPOT_LENGTH / 2, PARKING_SPOT_WIDTH / 2])
    parking_rect = patches.Rectangle(parking_corner, PARKING_SPOT_LENGTH, PARKING_SPOT_WIDTH,
                                     linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5)
    ax.add_patch(parking_rect)

    # 4. 初始化车辆图形对象
    car_plot = ax.fill([], [], color='blue', alpha=0.7)[0]
    arrow_plot = ax.arrow(0, 0, 0, 0, head_width=0.3, head_length=0.5, fc='red', ec='red')
    path_line, = ax.plot([], [], 'orange', linewidth=2, alpha=0.6, label='Parking Path')  # 路径轨迹线

    # 5. 动画更新函数
    def update(frame):
        if frame >= len(parking_path):
            frame = len(parking_path) - 1  # 最后一帧保持静止

        # 获取当前帧的车辆状态
        x, y, yaw = parking_path[frame]

        # 更新车辆绘制
        car_corners = np.array([
            [CAR_LENGTH / 2, CAR_WIDTH / 2],
            [CAR_LENGTH / 2, -CAR_WIDTH / 2],
            [-CAR_LENGTH / 2, -CAR_WIDTH / 2],
            [-CAR_LENGTH / 2, CAR_WIDTH / 2]
        ])
        rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rotated_corners = (rot_mat @ car_corners.T).T + np.array([x, y])
        car_plot.set_xy(rotated_corners)

        # 更新航向箭头
        arrow_len = CAR_LENGTH / 2
        arrow_plot.set_data(x=x, y=y, dx=arrow_len * np.cos(yaw), dy=arrow_len * np.sin(yaw))

        # 更新路径轨迹（绘制到当前帧）
        path_x = [p[0] for p in parking_path[:frame + 1]]
        path_y = [p[1] for p in parking_path[:frame + 1]]
        path_line.set_data(path_x, path_y)

        # 显示当前状态信息
        ax.set_title(f'Autonomous Parking Simulation (Frame: {frame}/{len(parking_path) - 1})')

        return car_plot, arrow_plot, path_line

    # 6. 创建并运行动画
    ani = FuncAnimation(
        fig, update,
        frames=len(parking_path),
        interval=STEP_TIME * 1000,  # 与仿真步长同步（毫秒）
        blit=True,
        repeat=False  # 只播放一次
    )

    # 显示图例
    ax.legend(loc='upper right')

    # 保存动画（可选，需要安装 ffmpeg，或改为保存为GIF）
    # ani.save('autonomous_parking.gif', writer='pillow', fps=10)

    # 显示动画
    plt.show()


if __name__ == '__main__':
    main()