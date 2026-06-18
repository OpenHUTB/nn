import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# ==========================================
# 局部路径规划 Demo (Python 版本)
# ==========================================

# 单个周期代价函数
def my_costfun(Initialpose_car, pose_car, Nobs_obsnocross, XY_obsnocross):
    """
    Initialpose_car 汽车初始位置，用于前进势场绘制
    pose_car 车辆预测位置，由mpc计算出
    Nobs_obsnocross 不可跨越障碍物数量
    XY_obsnocross 不可跨越障碍物数量坐标
    """
    x0, y0 = Initialpose_car
    x, y = pose_car
    X_obsnocross = XY_obsnocross[:, 0]
    Y_obsnocross = XY_obsnocross[:, 1]
    
    # 道路环境势场参数
    A_road = 2
    A_lane = 1
    sigma_lane = 0.8
    # 前进方向势场参数
    A_dir = 0.05
    # 障碍物势场参数
    A_obsnocross = 4
    sigmax_obsnocross = 10
    sigmay_obsnocross = 0.5

    # 代价计算
    # 左道路边界、右道路边界、虚线车道
    if y > 3.5 or y < -3.5: # 出界，代价无限大
        U_road = 1e9
    else: # 不出界，正常计算。使用 max 避免除0警告
        y_diff_1 = max(abs(y - 3.5), 1e-5)
        y_diff_2 = max(abs(y + 3.5), 1e-5)
        U_road = A_road * (1 / y_diff_1)**2 + A_road * (1 / y_diff_2)**2
        
    U_lane = A_lane * np.exp(-(y - 0)**2 / (2 * sigma_lane**2))
    
    # 目标方向
    U_direction = -A_dir * (x - x0)

    # 不可跨越障碍物 (边长为3.5的矩形)
    U_obsnocross = np.zeros(Nobs_obsnocross)
    for j in range(Nobs_obsnocross):
        if (y > (Y_obsnocross[j] - 1.75) and y < (Y_obsnocross[j] + 1.75) and 
            x >= (X_obsnocross[j] - 40) and x <= (X_obsnocross[j] + 40)):
            # y轴碰到障碍物 且 距离障碍物40米以内时
            dx_obsnocross = abs(x - X_obsnocross[j])
            dy_obsnocross = abs(y - Y_obsnocross[j])
            dx_safe = max(dx_obsnocross, 1e-5) # 避免除0
            U_obsnocross[j] = (1 / dx_safe)**2 + A_obsnocross * (
                np.exp(-(dx_obsnocross)**2 / (2 * sigmax_obsnocross) 
                       - (dy_obsnocross)**2 / (2 * sigmay_obsnocross))
            ) # 高斯函数+指数函数
        else:
            U_obsnocross[j] = 0

    # 代价之和
    cost = U_road + U_lane + U_direction + np.sum(U_obsnocross)
    return cost

# 完整Np周期总代价函数
def my_costallfun(delta_u, Np, Nc, State_Initial, Q, R1, R2, Nobs_obsnocross, XY_obsnocross):
    T = 0.2 # 预测间隔
    
    # 赋予初值
    y_dot = State_Initial[0] # 状态量
    x_dot = State_Initial[1]
    phi   = State_Initial[2]
    Y     = State_Initial[3]
    X     = State_Initial[4]
    Ay    = State_Initial[5] # 控制量
    
    # 预测状态量
    y_dot_predict = np.zeros(Np)
    x_dot_predict = np.zeros(Np)
    phi_predict   = np.zeros(Np)
    Y_predict     = np.zeros(Np)
    X_predict     = np.zeros(Np)

    cost1 = np.zeros(Np) # 存储每一个预测时域的代价和
    ay = np.zeros(Np)    # 控制量矩阵
    delta_ay = np.zeros(Np) # 控制增量矩阵，Nc之后都是相同的
    
    # 开始计算各个周期代价 (注意: Python 索引从 0 开始)
    for i in range(Np):
        if i == 0: # 第一个周期
            delta_ay[i] = delta_u[0]
            ay[i] = Ay + delta_ay[i]
            
            # 状态量更新
            y_dot_predict[i] = y_dot + T * ay[i]
            x_dot_predict[i] = x_dot
            phi_predict[i]   = phi + T * ay[i] / x_dot
            Y_predict[i]     = Y + T * (x_dot * np.sin(phi) + y_dot * np.cos(phi))
            X_predict[i]     = X + T * (x_dot * np.cos(phi) - y_dot * np.sin(phi))
            
            # 计算代价
            Initialpose_car = [X, Y]
            pose_car = [X_predict[i], Y_predict[i]]
            cost1[i] = my_costfun(Initialpose_car, pose_car, Nobs_obsnocross, XY_obsnocross)
            
        elif i > 0 and i < Nc: # Nc内周期
            delta_ay[i] = delta_u[i]
            ay[i] = ay[i-1] + delta_ay[i]
            
            y_dot_predict[i] = y_dot_predict[i-1] + T * ay[i]
            x_dot_predict[i] = x_dot # 视为不变
            phi_predict[i]   = phi_predict[i-1] + T * ay[i] / x_dot_predict[i-1]
            Y_predict[i]     = Y_predict[i-1] + T * (x_dot * np.sin(phi_predict[i-1]) + y_dot_predict[i-1] * np.cos(phi_predict[i-1]))
            X_predict[i]     = X_predict[i-1] + T * (x_dot * np.cos(phi_predict[i-1]) - y_dot_predict[i-1] * np.sin(phi_predict[i-1]))
            
            Initialpose_car = [X, Y]
            pose_car = [X_predict[i], Y_predict[i]]
            cost1[i] = my_costfun(Initialpose_car, pose_car, Nobs_obsnocross, XY_obsnocross)
            
        else: # Nc至Np内周期
            delta_ay[i] = delta_u[Nc-1]
            ay[i] = ay[i-1] + delta_ay[i]
            
            y_dot_predict[i] = y_dot_predict[i-1] + T * ay[i]
            x_dot_predict[i] = x_dot # 视为不变
            phi_predict[i]   = phi_predict[i-1] + T * ay[i] / x_dot_predict[i-1]
            Y_predict[i]     = Y_predict[i-1] + T * (x_dot * np.sin(phi_predict[i-1]) + y_dot_predict[i-1] * np.cos(phi_predict[i-1]))
            X_predict[i]     = X_predict[i-1] + T * (x_dot * np.cos(phi_predict[i-1]) - y_dot_predict[i-1] * np.sin(phi_predict[i-1]))
            
            Initialpose_car = [X, Y]
            pose_car = [X_predict[i], Y_predict[i]]
            cost1[i] = my_costfun(Initialpose_car, pose_car, Nobs_obsnocross, XY_obsnocross)

    # 计算总代价 (@ 为矩阵乘法)
    costall = ay.T @ R1 @ ay + delta_ay[:Nc].T @ R2 @ delta_ay[:Nc] + Q * np.sum(cost1)
    return costall

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    consume_time = np.zeros(30)

    # 障碍物信息设置 (单向双车道，每条车道3.5米宽，左车道中心1.75，右车道中心-1.75)
    Nobs_obsnocross = 2
    XY_obsnocross = np.array([[30, 1.75], [60, -1.75]])

    # 参数设置
    T = 0.2
    Np = 8
    Nc = 2
    Q = 100 # 势场权重
    R1 = np.eye(Np) # 控制量权重
    R2 = np.eye(Nc) # 控制增量权重

    # 绘制代价地图
    mapSize_x = [0, 80] # m
    mapSize_y = [-5, 5] # m
    resolution = 0.1    # 栅格分辨率
    
    grid_x = np.linspace(mapSize_x[0], mapSize_x[1], int((mapSize_x[1]-mapSize_x[0])/resolution) + 1)
    grid_y = np.linspace(mapSize_y[0], mapSize_y[1], int((mapSize_y[1]-mapSize_y[0])/resolution) + 1)
    grid_size_x = len(grid_x)
    grid_size_y = len(grid_y)

    costMap = np.zeros((grid_size_y, grid_size_x))
    
    print("Generating cost map...")
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            pose_car = [grid_x[i], grid_y[j]]
            costMap[j, i] = my_costfun([0, 0], pose_car, Nobs_obsnocross, XY_obsnocross)

    costMap = np.minimum(costMap, 5)

    # 绘图代价地图
    plt.ion() # 开启交互模式以便动态刷新轨迹
    fig_map = plt.figure(figsize=(12, 5))
    
    # 3D 绘图
    ax1 = fig_map.add_subplot(121, projection='3d')
    X_mesh, Y_mesh = np.meshgrid(grid_x, grid_y)
    surf = ax1.plot_surface(X_mesh, Y_mesh, costMap, cmap='viridis')
    fig_map.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_title('3D Cost Map')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('cost')

    # 2D 绘图
    ax2 = fig_map.add_subplot(122)
    img = ax2.imshow(costMap, extent=[mapSize_x[0], mapSize_x[1], mapSize_y[0], mapSize_y[1]], 
                     origin='lower', aspect='auto', cmap='viridis')
    fig_map.colorbar(img, ax=ax2)
    ax2.set_title('2D Cost Map')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    plt.pause(0.5)

    # 预测轨迹动态绘图设置
    fig_traj, ax_traj = plt.subplots(figsize=(10, 4))
    ax_traj.set_xlim(0, 80)
    ax_traj.set_ylim(-5, 5)
    ax_traj.set_title('Local Path Planning Trajectory')
    ax_traj.set_xlabel('X')
    ax_traj.set_ylabel('Y')
    ax_traj.grid(True)
    
    # 画出车道中心线及边界作参考
    ax_traj.axhline(0, color='gray', linestyle='--')
    ax_traj.axhline(3.5, color='black', linewidth=2)
    ax_traj.axhline(-3.5, color='black', linewidth=2)
    
    # 画出障碍物
    for obs in XY_obsnocross:
        ax_traj.plot(obs[0], obs[1], 'sk', markersize=10, label='Obstacle')

    # 预测状态缓存初始化
    y_dot_predict = np.zeros(Np)
    x_dot_predict = np.zeros(Np)
    phi_predict = np.zeros(Np)
    Y_predict = np.zeros(Np)
    X_predict = np.zeros(Np)
    ay = np.zeros(Np)

    print("Starting optimization loop...")
    # 进入循环 + 绘图
    for j in range(30):
        if j == 0:
            # 初始状态vy、vx、phi、Y、X、ay
            State_Initial = np.array([0, 15, 0, 1.75, 0, 0]) + 0.000001
        else:
            # 初始状态为上一周期的第一个预测步
            State_Initial = np.array([y_dot_predict[0], x_dot_predict[0], phi_predict[0], 
                                      Y_predict[0], X_predict[0], ay[0]]) + 0.000001

        # 设置约束
        bounds = [(-2.0, 2.0) for _ in range(Nc)]
        
        # 记录耗时并开始求解
        start_time = time.time()
        
        # 使用 scipy.optimize.minimize 替代 fmincon (SLSQP方法适用于有约束优化)
        res = minimize(
            my_costallfun, 
            np.zeros(Nc), 
            args=(Np, Nc, State_Initial, Q, R1, R2, Nobs_obsnocross, XY_obsnocross), 
            method='SLSQP', 
            bounds=bounds, 
            options={'maxiter': 1500, 'ftol': 1e-6, 'disp': False}
        )
        AA = res.x
        
        consume_time[j] = time.time() - start_time

        # 根据优化出来的增量重新计算输出状态以更新并画图
        delta_ay = np.zeros(Np)
        for i in range(Np):
            if i == 0:
                delta_ay[i] = AA[0]
                ay[i] = State_Initial[5] + delta_ay[i]
                y_dot_predict[i] = State_Initial[0] + T * ay[i]
                x_dot_predict[i] = State_Initial[1]
                phi_predict[i] = State_Initial[2] + T * ay[i] / State_Initial[1]
                Y_predict[i] = State_Initial[3] + T * (State_Initial[1] * np.sin(State_Initial[2]) + State_Initial[0] * np.cos(State_Initial[2]))
                X_predict[i] = State_Initial[4] + T * (State_Initial[1] * np.cos(State_Initial[2]) - State_Initial[0] * np.sin(State_Initial[2]))
            elif i < Nc:
                delta_ay[i] = AA[i]
                ay[i] = ay[i-1] + delta_ay[i]
                y_dot_predict[i] = y_dot_predict[i-1] + T * ay[i]
                x_dot_predict[i] = State_Initial[1]
                phi_predict[i] = phi_predict[i-1] + T * ay[i] / x_dot_predict[i-1]
                Y_predict[i] = Y_predict[i-1] + T * (State_Initial[1] * np.sin(phi_predict[i-1]) + y_dot_predict[i-1] * np.cos(phi_predict[i-1]))
                X_predict[i] = X_predict[i-1] + T * (State_Initial[1] * np.cos(phi_predict[i-1]) - y_dot_predict[i-1] * np.sin(phi_predict[i-1]))
            else:
                delta_ay[i] = AA[Nc-1]
                ay[i] = ay[i-1] + delta_ay[i]
                y_dot_predict[i] = y_dot_predict[i-1] + T * ay[i]
                x_dot_predict[i] = State_Initial[1]
                phi_predict[i] = phi_predict[i-1] + T * ay[i] / x_dot_predict[i-1]
                Y_predict[i] = Y_predict[i-1] + T * (State_Initial[1] * np.sin(phi_predict[i-1]) + y_dot_predict[i-1] * np.cos(phi_predict[i-1]))
                X_predict[i] = X_predict[i-1] + T * (State_Initial[1] * np.cos(phi_predict[i-1]) - y_dot_predict[i-1] * np.sin(phi_predict[i-1]))

        # 绘制规划结果曲线
        ax_traj.plot(State_Initial[4], State_Initial[3], '*y', markersize=8) # 绘制当前起点
        
        # 将起点插入预测点最前方以便画出连贯路径
        traj_X = np.insert(X_predict, 0, State_Initial[4])
        traj_Y = np.insert(Y_predict, 0, State_Initial[3])
        ax_traj.plot(traj_X, traj_Y, '-r', linewidth=1) # 绘制路线
        
        plt.pause(0.1) # 暂停以触发重绘

    ave_time = np.mean(consume_time)
    print(f'平均耗时: {ave_time:.6f} seconds\n')
    
    # 取消交互模式并保持最终图片显示
    plt.ioff()
    plt.show()