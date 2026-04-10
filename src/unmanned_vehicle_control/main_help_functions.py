import numpy as np

from src.config import N, V_REF, A
from src.config import CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS
def update_reference_point(x0, y0, current_idx, x_traj, y_traj, min_distance=5.0):
    # 校验轨迹数组长度一致性与非空
    if len(x_traj) != len(y_traj) or len(x_traj) == 0:
        raise ValueError("x_traj和y_traj长度必须一致且非空")
    # 校验current_idx的合法性，若超出范围则修正为有效索引
    if current_idx < 0 or current_idx >= len(x_traj):
        current_idx = 0  # 超出范围时重置为0索引
    distance_to_current_point = np.sqrt((x_traj[current_idx] - x0) ** 2 + (y_traj[current_idx] - y0) ** 2)

    if distance_to_current_point < min_distance:
        current_idx = (current_idx + 1) % len(x_traj)

    return current_idx

def get_eight_trajectory(x_init, y_init, total_points=100):
    t = 2 * np.pi

    t_values = np.linspace(0, t, total_points)
    x_traj = x_init + A * np.sin(t_values)
    y_traj = y_init + A * np.sin(t_values) * np.cos(t_values)

    v_ref = [V_REF for _ in range(total_points)]

    cos_angle = np.cos(-np.pi / 4)
    sin_angle = np.sin(-np.pi / 4)

    x_traj_rotated = x_init + cos_angle * (x_traj - x_init) - sin_angle * (y_traj - y_init)
    y_traj_rotated = y_init + sin_angle * (x_traj - x_init) + cos_angle * (y_traj - y_init)

    theta_ref = []
    for i in range(total_points - 1):
        dx = x_traj_rotated[i + 1] - x_traj_rotated[i]
        dy = y_traj_rotated[i + 1] - y_traj_rotated[i]
        theta = np.arctan2(dy, dx)
        theta_ref.append(theta)

    theta_ref.append(theta_ref[-1])

    return x_traj_rotated, y_traj_rotated, v_ref, theta_ref
def get_circle_trajectory(
    x_center=CIRCLE_CENTER_X,  # 从config.py读取默认圆心X
    y_center=CIRCLE_CENTER_Y,  # 从config.py读取默认圆心Y
  #  radius=CIRCLE_RADIUS,     # 从config.py读取默认半径
    radius=20,                 # 手动调节可适用无错误的半径
    total_points=200
):
    """生成圆形轨迹（默认参数从配置文件读取）"""
    t_values = np.linspace(0, 2 * np.pi, total_points)
    x_traj = x_center + radius * np.cos(t_values)
    y_traj = y_center + radius * np.sin(t_values)

    v_ref = [V_REF for _ in range(total_points)]
    # 计算参考角度（切线方向）
    theta_ref = []
    for i in range(total_points - 1):
        dx = x_traj[i + 1] - x_traj[i]
        dy = y_traj[i + 1] - y_traj[i]
        theta = np.arctan2(dy, dx)
        theta_ref.append(theta)
    theta_ref.append(theta_ref[-1])  # 最后一个点保持与前一个相同

    return x_traj, y_traj, v_ref, theta_ref

def get_spiral_trajectory(x_init, y_init, total_points=200, turns=2, scale=2):
    """生成螺旋线轨迹"""
    t_values = np.linspace(0, 2 * np.pi * turns, total_points)
    # 极坐标方程：r = scale * t（半径随角度线性增加）
    r = scale * t_values
    x_traj = x_init + r * np.cos(t_values)
    y_traj = y_init + r * np.sin(t_values)
    v_ref = [V_REF for _ in range(total_points)]
    # 计算参考角度（切线方向）
    theta_ref = []
    for i in range(total_points - 1):
        dx = x_traj[i + 1] - x_traj[i]
        dy = y_traj[i + 1] - y_traj[i]
        theta = np.arctan2(dy, dx)
        theta_ref.append(theta)
    theta_ref.append(theta_ref[-1])  # 最后一个点保持与前一个相同

    return x_traj, y_traj, v_ref, theta_ref


def get_square_trajectory(x_init, y_init, side_length=23, total_points=92):
    """生成方形轨迹"""
    # 计算方形四个顶点
    points = [
        (x_init, y_init),  # 起点
        (x_init + side_length, y_init),  # 右顶点
        (x_init + side_length, y_init + side_length),  # 右上顶点
        (x_init, y_init + side_length),  # 上顶点
        (x_init, y_init)  # 回到起点（闭合）
    ]

    # 计算每条边的点数（平均分配）
    points_per_side = total_points // 4
    t_values = np.linspace(0, 1, points_per_side, endpoint=False)

    x_traj = []
    y_traj = []

    # 生成四条边的轨迹点
    for i in range(4):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        x_segment = x0 + (x1 - x0) * t_values
        y_segment = y0 + (y1 - y0) * t_values
        x_traj.extend(x_segment)
        y_traj.extend(y_segment)

    # 确保总点数正确
    x_traj = np.array(x_traj[:total_points])
    y_traj = np.array(y_traj[:total_points])

    # 参考速度
    v_ref = [V_REF for _ in range(total_points)]

    # 计算参考角度（切线方向）
    theta_ref = []
    for i in range(total_points - 1):
        dx = x_traj[i + 1] - x_traj[i]
        dy = y_traj[i + 1] - y_traj[i]
        theta = np.arctan2(dy, dx)
        theta_ref.append(theta)
    theta_ref.append(theta_ref[-1])  # 最后一个点保持与前一个相同

    return x_traj, y_traj, v_ref, theta_ref

def get_ref_trajectory(x_traj, y_traj, theta_traj, current_idx):
    if current_idx + N < len(x_traj):
        x_ref = x_traj[current_idx:current_idx + N]
        y_ref = y_traj[current_idx:current_idx + N]
        theta_ref = theta_traj[current_idx:current_idx + N]
    else:
        x_ref = np.concatenate((x_traj[current_idx:], x_traj[:N - (len(x_traj) - current_idx)]))
        y_ref = np.concatenate((y_traj[current_idx:], y_traj[:N - (len(x_traj) - current_idx)]))
        theta_ref = np.concatenate((theta_traj[current_idx:], theta_traj[:N - (len(x_traj) - current_idx)]))
    return x_ref, y_ref, theta_ref

def get_straight_trajectory(x_init, y_init, distance=3000, total_points=1000):
    x_traj = np.linspace(x_init + 5, x_init + distance, total_points)
    y_traj = np.full(total_points, y_init + 3)

    v_ref = [V_REF for _ in range(total_points)]

    return x_traj, y_traj, v_ref

def calculate_lateral_deviation(x, y, x_ref1, y_ref1, x_ref2, y_ref2):
    num = (y_ref2 - y_ref1) * x - (x_ref2 - x_ref1) * y + x_ref2 * y_ref1 - y_ref2 * x_ref1
    denom = np.sqrt((y_ref2 - y_ref1) ** 2 + (x_ref2 - x_ref1) ** 2)
    if denom > 0.01:
        return num / denom
    else:
        return 0
