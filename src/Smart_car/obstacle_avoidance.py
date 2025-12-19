import time
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

# 无人车状态常量
SAFE_DISTANCE = 50  # 安全距离（厘米）
WARNING_DISTANCE = 30  # 警告距离（厘米）
DANGER_DISTANCE = 15  # 危险距离（厘米）
NORMAL_SPEED = 20  # 正常速度（km/h）
LOW_SPEED = 5  # 低速（km/h）
STOP_SPEED = 0  # 停车速度

# 可视化全局变量
fig, (ax_scene, ax_plot) = plt.subplots(1, 2, figsize=(12, 5))
distance_history = []  # 前方距离历史
speed_history = []  # 车速历史
time_history = []  # 时间轴
car_pos = [5, 2.5]  # 无人车初始位置（x,y）
obstacle_pos = [0, 0]  # 障碍物位置
car_direction = "forward"


class UnmannedCar:
    def __init__(self):
        self.speed = 0
        self.direction = "forward"

    def simulate_sensor(self, direction):
        """模拟传感器测距（加入轻微固定偏移，让障碍物位置可预测）"""
        if direction == "front":
            # 模拟障碍物距离缓慢变化（更贴近实际）
            base_dist = random.randint(10, 60) if len(distance_history) < 5 else distance_history[-1] + random.randint(
                -5, 5)
            distance = max(0, min(100, base_dist))  # 限制0-100cm
        else:
            distance = random.randint(20, 80)  # 左右侧距离

        # 更新障碍物位置（用于可视化）
        global obstacle_pos
        obstacle_pos = [car_pos[0] + distance / 10, car_pos[1]]  # 缩放适配画布
        print(f"[{direction}] 传感器检测距离：{distance} cm")
        return distance

    def adjust_speed(self, new_speed):
        self.speed = new_speed
        print(f"车速调整为：{self.speed} km/h")

    def adjust_direction(self, new_dir):
        global car_direction
        self.direction = new_dir
        car_direction = new_dir
        print(f"行驶方向调整为：{self.direction}")

    def collision_avoidance(self):
        """核心避撞逻辑"""
        front_dist = self.simulate_sensor("front")

        # 记录数据用于绘图
        distance_history.append(front_dist)
        speed_history.append(self.speed)
        time_history.append(len(time_history))

        if front_dist > SAFE_DISTANCE:
            self.adjust_speed(NORMAL_SPEED)
            self.adjust_direction("forward")

        elif WARNING_DISTANCE < front_dist <= SAFE_DISTANCE:
            print("⚠️ 前方接近障碍物，减速！")
            self.adjust_speed(LOW_SPEED)
            self.adjust_direction("forward")

        elif front_dist <= DANGER_DISTANCE:
            print("🚨 前方紧急危险！立即停车！")
            self.adjust_speed(STOP_SPEED)
            self.adjust_direction("stop")

            left_dist = self.simulate_sensor("left")
            right_dist = self.simulate_sensor("right")

            if left_dist > SAFE_DISTANCE:
                print("🔄 左侧有空间，转向左侧避障")
                self.adjust_direction("left")
                self.adjust_speed(LOW_SPEED)
            elif right_dist > SAFE_DISTANCE:
                print("🔄 右侧有空间，转向右侧避障")
                self.adjust_direction("right")
                self.adjust_speed(LOW_SPEED)
            else:
                print("❌ 左右侧均有障碍物，无法避障，保持停车！")


# 初始化可视化场景
def init_visualization():
    # 左侧：场景图（无人车+障碍物）
    ax_scene.set_xlim(0, 15)
    ax_scene.set_ylim(0, 5)
    ax_scene.set_title("无人车避障场景模拟")
    ax_scene.set_xlabel("位置 (cm/10)")
    ax_scene.set_ylabel("位置 (cm/10)")
    ax_scene.grid(True)

    # 右侧：数据曲线图
    ax_plot.set_xlim(0, 20)
    ax_plot.set_ylim(0, max(NORMAL_SPEED + 5, SAFE_DISTANCE + 5))
    ax_plot.set_title("实时数据监控")
    ax_plot.set_xlabel("检测次数")
    ax_plot.set_ylabel("数值")
    ax_plot.grid(True)
    ax_plot.legend(["前方距离 (cm)", "车速 (km/h)"], loc="upper right")
    return ax_scene, ax_plot


# 实时更新可视化
def update_visualization(frame):
    # 清空场景图
    ax_scene.clear()
    ax_scene.set_xlim(0, 15)
    ax_scene.set_ylim(0, 5)
    ax_scene.set_title("无人车避障场景模拟")
    ax_scene.set_xlabel("位置 (cm/10)")
    ax_scene.set_ylabel("位置 (cm/10)")
    ax_scene.grid(True)

    # 绘制无人车（矩形）
    car_color = "green" if car_direction == "forward" else "yellow" if car_direction in ["left", "right"] else "red"
    car = Rectangle((car_pos[0], car_pos[1] - 0.5), 1, 1, color=car_color, label="无人车")
    ax_scene.add_patch(car)

    # 绘制障碍物（圆形）
    obstacle = Circle(obstacle_pos, 0.3, color="black", label="障碍物")
    ax_scene.add_patch(obstacle)

    # 绘制方向标识
    if car_direction == "left":
        ax_scene.arrow(car_pos[0] + 0.5, car_pos[1], -0.3, 0, head_width=0.2, color="blue")
    elif car_direction == "right":
        ax_scene.arrow(car_pos[0] + 0.5, car_pos[1], 0.3, 0, head_width=0.2, color="blue")
    elif car_direction == "forward":
        ax_scene.arrow(car_pos[0] + 0.5, car_pos[1], 0.3, 0, head_width=0.2, color="blue")

    # 更新曲线图
    ax_plot.clear()
    ax_plot.plot(time_history, distance_history, 'b-', label="前方距离 (cm)")
    ax_plot.plot(time_history, speed_history, 'r-', label="车速 (km/h)")
    # 绘制安全阈值线
    ax_plot.axhline(y=SAFE_DISTANCE, color='g', linestyle='--', label="安全距离")
    ax_plot.axhline(y=WARNING_DISTANCE, color='y', linestyle='--', label="警告距离")
    ax_plot.axhline(y=DANGER_DISTANCE, color='r', linestyle='--', label="危险距离")
    ax_plot.set_xlim(max(0, len(time_history) - 20), len(time_history))
    ax_plot.set_ylim(0, max(NORMAL_SPEED + 5, SAFE_DISTANCE + 5))
    ax_plot.set_title("实时数据监控")
    ax_plot.set_xlabel("检测次数")
    ax_plot.set_ylabel("数值")
    ax_plot.grid(True)
    ax_plot.legend(loc="upper right")

    return ax_scene, ax_plot


# 主运行逻辑
if __name__ == "__main__":
    car = UnmannedCar()
    init_visualization()

    # 启动动画更新（每1秒刷新一次，和传感器检测频率同步）
    ani = animation.FuncAnimation(fig, update_visualization, interval=1000, blit=False)


    # 启动无人车避障逻辑（后台运行）
    def run_car():
        print("=== 无人车启动 ===")
        try:
            while True:
                car.collision_avoidance()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n=== 无人车停止 ===")
            car.adjust_speed(STOP_SPEED)
            car.adjust_direction("stop")


    # 多线程运行（避免阻塞可视化）
    import threading

    car_thread = threading.Thread(target=run_car)
    car_thread.daemon = True
    car_thread.start()

    # 显示可视化窗口
    plt.tight_layout()
    plt.show()