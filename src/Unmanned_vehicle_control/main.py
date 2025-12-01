# 导入时间模块，用于时间相关操作
import time
# 导入线程模块，用于创建多线程
import threading

# 导入CARLA模拟器核心类，用于与CARLA仿真环境交互
from src.carla_simulator import CarlaSimulator
# 从配置文件导入初始化参数：初始位置坐标、MPC预测时域、时间步长、参考速度、目标圈数
from src.config import X_INIT_M, Y_INIT_M, N, dt, V_REF, LAPS
# 导入轨迹相关函数：生成8字形轨迹、获取参考轨迹段、更新参考点索引
from src.help_functions import get_eight_trajectory, get_ref_trajectory, update_reference_point
# 导入日志记录类，用于记录和可视化控制数据
from src.logger import Logger
# 导入MPC控制器类，实现模型预测控制算法
from src.mpc_controller import MpcController

def draw_trajectory_in_thread(carla, x_traj, y_traj, dt):
    """
    在独立线程中持续绘制全局参考轨迹
    Args:
        carla: CarlaSimulator实例，用于调用绘图接口
        x_traj: 轨迹点x坐标列表
        y_traj: 轨迹点y坐标列表
        dt: 时间步长，控制绘制频率
    """
    while True:
        # 调用CARLA接口绘制轨迹，绿色、高度0.2米、生命周期2*dt
        carla.draw_trajectory(x_traj, y_traj, height=0.2, green=255, life_time=dt * 2)
        # 按时间步长休眠，控制绘制频率
        time.sleep(dt)

# ========== 初始化CARLA仿真环境 ==========
# 创建CARLA模拟器实例
carla = CarlaSimulator()
# 加载指定的仿真地图（Town02_Opt为优化版城镇地图）
carla.load_world('Town02_Opt')
# 在指定初始位置生成自动驾驶车辆（特斯拉Model3）
carla.spawn_ego_vehicle('vehicle.tesla.model3', x=X_INIT_M, y=Y_INIT_M, z=0.1)
# 打印车辆的动力学特性参数（如轴距、最大转向角等）
carla.print_ego_vehicle_characteristics()
# 设置 spectator 视角：初始位置上空50米，俯视角度-90度
carla.set_spectator(X_INIT_M, Y_INIT_M, z=50, pitch=-90)

# ========== 初始化日志记录器 ==========
logger = Logger()

# ========== 生成参考轨迹 ==========
# 生成8字形参考轨迹，返回位置坐标、参考速度、参考航向角序列
x_traj, y_traj, v_ref, theta_traj = get_eight_trajectory(X_INIT_M, Y_INIT_M)
# 初始化当前参考点索引
current_idx = 0
# 初始化已完成圈数
laps = 0

# ========== 启动轨迹绘制线程 ==========
# 创建绘制轨迹的后台线程，避免阻塞主控制循环
trajectory_thread = threading.Thread(target=draw_trajectory_in_thread, args=(carla, x_traj, y_traj, dt))
# 设置为守护线程，主程序退出时自动终止
trajectory_thread.daemon = True
# 启动线程
trajectory_thread.start()

# ========== 初始化MPC控制器 ==========
# 创建MPC控制器实例，指定预测时域N和时间步长dt
mpc_controller = MpcController(horizon=N, dt=dt)

try:
    # ========== 主控制循环 ==========
    while True:
        # 记录当前循环开始时间，用于计算MPC求解耗时
        start_time = time.time()

        # 重置MPC求解器，清除上一次迭代的状态
        mpc_controller.reset_solver()

        # 获取车辆当前状态：x坐标、y坐标、航向角、速度
        x0, y0, theta0, v0 = carla.get_main_ego_vehicle_state()
        # 将车辆当前状态设置为MPC控制器的初始状态
        mpc_controller.set_init_vehicle_state(x0, y0, theta0, v0)

        # 根据当前参考点索引获取MPC预测时域内的参考轨迹段
        x_ref, y_ref, theta_ref = get_ref_trajectory(x_traj, y_traj, theta_traj, current_idx)
        # 绘制红色的局部参考轨迹（MPC预测段），高度1.0米
        carla.draw_trajectory(x_ref, y_ref, height=1.0, red=255, life_time=dt * 2)

        # 记录控制器输入数据：车辆状态和参考状态
        logger.log_controller_input(x0, y0, v0, theta0, x_ref[0], y_ref[0], V_REF, theta_ref[0])
        # 更新MPC的代价函数参数，传入参考轨迹和参考速度
        mpc_controller.update_cost_function(x_ref, y_ref, theta_ref, v_ref)

        # 求解MPC优化问题，得到最优控制序列
        mpc_controller.solve()

        # 获取求解得到的控制量：前轮转角（弧度）、纵向加速度（m/s²）
        wheel_angle_rad, acceleration_m_s_2 = mpc_controller.get_controls_value()

        # 将MPC输出转换为车辆执行器指令：油门、刹车、转向（归一化到[-1,1]）
        throttle, brake, steer = CarlaSimulator.process_control_inputs(wheel_angle_rad, acceleration_m_s_2)
        # 记录控制器输出数据
        logger.log_controller_output(steer, throttle, brake)
        # 将控制指令应用到车辆
        carla.apply_control(steer, throttle, brake)

        # ========== 更新参考点索引 ==========
        # 保存更新前的参考点索引
        prev_current_idx = current_idx
        # 根据车辆当前位置更新参考点索引（投影到最近轨迹点）
        current_idx = update_reference_point(x0, y0, current_idx, x_traj, y_traj)
        # 判断是否完成一圈：参考点索引从最后一个点回到第一个点
        if prev_current_idx == len(x_traj) - 1 and current_idx == 0:
            laps += 1  # 完成圈数加1

        # ========== 控制循环频率 ==========
        # 计算MPC求解总耗时
        end_time = time.time()
        mpc_calculation_time = end_time - start_time
        # 打印MPC计算耗时
        print(f"Calculation time of MPC controller: {mpc_calculation_time:.6f} seconds")

        # 休眠剩余时间，保证总循环周期为dt
        time.sleep(max(dt - mpc_calculation_time, 0))

        # 判断是否完成目标圈数，满足则退出主循环
        if laps == LAPS:
            break

finally:
    # ========== 资源清理 ==========
    # 清理CARLA仿真环境资源（销毁车辆、关闭连接等）
    carla.clean()
    # 显示日志数据的可视化图表（如轨迹跟踪误差、控制输入曲线等）
    logger.show_plots()