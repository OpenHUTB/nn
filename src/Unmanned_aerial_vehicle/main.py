import math
import random
import time


class Drone:
    """无人机核心类，模拟无人机的状态和行为"""

    def __init__(self, start_x=0.0, start_y=0.0, start_z=0.0):
        # 无人机当前位置 (x, y, z) 坐标
        self.x = start_x
        self.y = start_y
        self.z = start_z
        # 无人机飞行速度 (m/s)
        self.speed = 2.0
        # 无人机状态：idle(待机)/flying(飞行)/avoiding(避障)
        self.status = "idle"
        # 目标位置
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        # 避障距离阈值 (米)
        self.obstacle_threshold = 3.0

    def set_target(self, target_x, target_y, target_z):
        """设置无人机的目标位置"""
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.status = "flying"
        print(f"无人机目标已设置：({target_x}, {target_y}, {target_z})，开始飞行")

    def calculate_distance_to_target(self):
        """计算当前位置到目标位置的直线距离"""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dz = self.target_z - self.z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def detect_obstacle(self, obstacles):
        """模拟障碍物检测：检查是否有障碍物在避障阈值范围内"""
        for (ox, oy, oz) in obstacles:
            dx = ox - self.x
            dy = oy - self.y
            dz = oz - self.z
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if distance < self.obstacle_threshold:
                return True, (ox, oy, oz)  # 检测到障碍物，返回障碍物位置
        return False, None

    def avoid_obstacle(self, obstacle_pos):
        """简单避障策略：向障碍物右侧偏移飞行"""
        ox, oy, _ = obstacle_pos
        # 计算偏移方向（右侧1米）
        self.x += (self.x - ox) * 0.5  # 远离障碍物x方向
        self.y += (self.y - oy) * 0.5  # 远离障碍物y方向
        print(f"检测到障碍物({ox}, {oy})，已避障，当前位置：({self.x:.2f}, {self.y:.2f})")
        self.status = "avoiding"

    def update_position(self):
        """更新无人机位置，向目标点移动"""
        if self.status != "flying":
            return

        # 计算方向向量
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dz = self.target_z - self.z
        distance = self.calculate_distance_to_target()

        # 如果距离小于步长，直接到达目标
        if distance < self.speed:
            self.x = self.target_x
            self.y = self.target_y
            self.z = self.target_z
            self.status = "idle"
            print(f"已到达目标位置：({self.x:.2f}, {self.y:.2f}, {self.z:.2f})")
            return

        # 归一化方向向量，按速度移动
        self.x += (dx / distance) * self.speed
        self.y += (dy / distance) * self.speed
        self.z += (dz / distance) * self.speed

    def get_status(self):
        """获取无人机当前状态和位置"""
        return {
            "position": (round(self.x, 2), round(self.y, 2), round(self.z, 2)),
            "status": self.status,
            "distance_to_target": round(self.calculate_distance_to_target(), 2)
        }


# -------------------------- 主程序：无人机自主导航仿真 --------------------------
if __name__ == "__main__":
    # 1. 初始化无人机（起点：0,0,10米高度）
    drone = Drone(start_x=0.0, start_y=0.0, start_z=10.0)

    # 2. 设置目标位置（终点：20, 15, 10米高度）
    drone.set_target(target_x=20.0, target_y=15.0, target_z=10.0)

    # 3. 模拟环境中的障碍物（坐标列表）
    obstacles = [(10, 8, 10), (15, 12, 9)]  # 两个障碍物位置

    # 4. 自主导航主循环
    print("\n=== 无人机自主导航开始 ===")
    while drone.status != "idle":
        # 检测障碍物
        obstacle_detected, obstacle_pos = drone.detect_obstacle(obstacles)
        if obstacle_detected:
            drone.avoid_obstacle(obstacle_pos)
            # 避障后恢复飞行状态
            drone.status = "flying"

        # 更新位置
        drone.update_position()

        # 打印实时状态
        status = drone.get_status()
        print(f"当前位置：{status['position']} | 距离目标：{status['distance_to_target']}米 | 状态：{status['status']}")

        # 模拟实时刷新（每秒更新一次位置）
        time.sleep(1)

    print("\n=== 无人机自主导航完成 ===")