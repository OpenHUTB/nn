import carla
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from typing import List, Tuple

# ====================== 配置区（统一管理，方便修改） ======================
# 仿真连接参数
HOST = '127.0.0.1'
PORT = 2000
TIMEOUT = 20.0
PROJECT_ROOT = r"D:\github\nn"
SAVE_DIR = os.path.join(PROJECT_ROOT, "src", "path_tracking")

# 路径生成参数
PATH_POINTS = 80
PATH_STEP = 3.0

# 车辆控制参数
LOOKAHEAD_DIST = 8.0
WHEELBASE = 2.5
THROTTLE = 0.3
RUN_DURATION = 46
CONTROL_FREQ = 0.05
STEER_LIMIT_MAX = 0.5
STEER_LIMIT_MIN = -0.5
STEER_SCALE = 0.8

# 绘图输出参数
PLOT_FIGSIZE = (8, 6)
PLOT_DPI = 300
# ====================== 工具函数 ======================
def ensure_dir(path: str) -> None:
    """
    校验并创建文件夹
    :param path: 文件夹路径
    """
    os.makedirs(path, exist_ok=True)

def cleanup_vehicles(world: carla.World) -> None:
    """
    清空场景内所有车辆模型
    :param world: CARLA世界实例
    """
    vehicles = world.get_actors().filter('vehicle.*')
    for veh in vehicles:
        if veh.is_alive:
            veh.destroy()

def generate_ref_path(
    world: carla.World,
    start_wp: carla.Waypoint,
    num_points: int,
    step: float = 3.0
) -> List[Tuple[float, float]]:
    """
    沿车道中心线生成参考行驶路径
    :param world: CARLA世界实例
    :param start_wp: 起始路点
    :param num_points: 路径采样点数
    :param step: 路点步进距离
    :return: 坐标点列表
    """
    path = []
    current = start_wp
    for _ in range(num_points):
        path.append((current.transform.location.x, current.transform.location.y))
        next_wps = current.next(step)
        if not next_wps:
            break
        current = next_wps[0]
    return path

def draw_path(world: carla.World, path: List[Tuple[float, float]]) -> None:
    """
    在仿真界面绘制参考路径线条
    :param world: CARLA世界实例
    :param path: 参考路径坐标集合
    """
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        world.debug.draw_line(
            carla.Location(x1, y1, 0.5),
            carla.Location(x2, y2, 0.5),
            thickness=0.1,
            color=carla.Color(0, 255, 0),
            life_time=0.5
        )

def draw_goal(world: carla.World, goal: Tuple[float, float]) -> None:
    """
    绘制路径预瞄目标点
    :param world: CARLA世界实例
    :param goal: 目标点坐标
    """
    x, y = goal
    world.debug.draw_point(
        carla.Location(x, y, 1.0),
        size=0.2,
        color=carla.Color(0, 0, 255),
        life_time=0.5
    )
# ====================== 纯追踪控制器（类优化） ======================
class PurePursuit:
    def __init__(self, path: List[Tuple[float, float]], lookahead: float, wheelbase: float):
        self.path = np.array(path)
        self.Ld = lookahead
        self.wheelbase = wheelbase

    def get_goal_point(self, veh_loc: carla.Location) -> Tuple[float, float]:
        """找到最近点 + 偏移，返回目标点"""
        dx = self.path[:, 0] - veh_loc.x
        dy = self.path[:, 1] - veh_loc.y
        dists = np.hypot(dx, dy)
        idx = np.argmin(dists)
        # 边界防护，防止超出路径范围
        offset = 3
        target_idx = idx + offset
        if target_idx >= len(self.path):
            target_idx = len(self.path) - 1
        return tuple(self.path[target_idx])

    def calculate_steer(self, trans: carla.Transform, goal: Tuple[float, float]) -> float:
        """计算转向角（带安全限幅）"""
        vx, vy = trans.location.x, trans.location.y
        yaw = np.radians(trans.rotation.yaw)
        gx, gy = goal

        dx = gx - vx
        dy = gy - vy

        # 坐标变换：世界系 → 车身系
        lx = dx * np.cos(yaw) + dy * np.sin(yaw)
        ly = -dx * np.sin(yaw) + dy * np.cos(yaw)

        if lx == 0:
            return 0.0

        alpha = np.arctan2(ly, lx)
        steer = np.arctan2(2 * self.wheelbase * np.sin(alpha), self.Ld)
        return np.clip(steer / STEER_SCALE, STEER_LIMIT_MIN, STEER_LIMIT_MAX)

# ====================== 主流程 ======================
def main():
    ensure_dir(SAVE_DIR)

    # 连接 CARLA
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)
    world = client.get_world()
    print("✅ 已连接 CARLA 模拟器")

    # 清理旧车辆
    cleanup_vehicles(world)

    # 生成车辆
    bp_lib = world.get_blueprint_library()
    veh_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(veh_bp, spawn_point)
    vehicle.set_simulate_physics(True)
    print("✅ 车辆生成完成")

    # 生成参考路径
    start_wp = world.get_map().get_waypoint(spawn_point.location)
    ref_path = generate_ref_path(world, start_wp, PATH_POINTS, step=PATH_STEP)
    print(f"✅ 参考路径已生成，共 {len(ref_path)} 个点")

    # 初始化控制器
    controller = PurePursuit(ref_path, LOOKAHEAD_DIST, WHEELBASE)

    # 轨迹记录
    actual_x, actual_y = [], []
    error_list = []
    try:
        print(f"\n🚗 开始路径跟踪，总时长：{RUN_DURATION} 秒...")
        start_time = time.time()

        while time.time() - start_time < RUN_DURATION:
            loop_start = time.time()
            trans = vehicle.get_transform()
            loc = trans.location

            # 记录轨迹
            actual_x.append(loc.x)
            actual_y.append(loc.y)

            # 计算跟踪误差
            dx = ref_path[min(len(ref_path)-1, len(actual_x)-1)][0] - loc.x
            dy = ref_path[min(len(ref_path)-1, len(actual_x)-1)][1] - loc.y
            error_list.append(np.hypot(dx, dy))

            # 绘制可视化
            draw_path(world, ref_path)
            goal = controller.get_goal_point(loc)
            draw_goal(world, goal)

            # 计算控制量
            steer = controller.calculate_steer(trans, goal)
            control = carla.VehicleControl(throttle=THROTTLE, steer=steer, brake=0.0)
            vehicle.apply_control(control)

            # 精准控制频率
            elapsed = time.time() - loop_start
            if elapsed < CONTROL_FREQ:
                time.sleep(CONTROL_FREQ - elapsed)

    finally:
        # 安全销毁车辆
        if vehicle.is_alive:
            vehicle.destroy()
            print("\n✅ 车辆已安全销毁")

        # 输出跟踪误差
        print(f"\n📊 平均跟踪误差: {np.mean(error_list):.2f} m")

        # 保存轨迹图
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=PLOT_FIGSIZE)
        plt.plot([p[0] for p in ref_path], [p[1] for p in ref_path], 'g-', linewidth=3, label='参考路径')
        plt.plot(actual_x, actual_y, 'r.', markersize=3, label='实际轨迹')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.title('无人车路径跟踪结果')
        plt.xlabel('X 坐标 (m)')
        plt.ylabel('Y 坐标 (m)')
        img_path = os.path.join(SAVE_DIR, "result.png")
        plt.savefig(img_path, bbox_inches='tight', dpi=PLOT_DPI)
        plt.close()
        print(f"📊 轨迹图已保存：{img_path}")

        # 保存 CSV
        csv_path = os.path.join(SAVE_DIR, "trajectory_data.csv")
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ref_x', 'ref_y', 'actual_x', 'actual_y'])
                for i in range(len(actual_x)):
                    ref_idx = i % len(ref_path)
                    writer.writerow([
                        ref_path[ref_idx][0],
                        ref_path[ref_idx][1],
                        actual_x[i],
                        actual_y[i]
                    ])
            print(f"📈 数据已保存：{csv_path}")
        except Exception as e:
            print(f"❌ 数据保存失败：{str(e)}")

        print("\n🎉 路径跟踪任务全部完成！")

if __name__ == "__main__":
    main()