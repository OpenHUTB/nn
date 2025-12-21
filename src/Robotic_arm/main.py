import math
import time


class DroneAutonomousNavigation:
    def __init__(self):
        """初始化无人机导航模拟器（无硬件依赖）"""
        # 模拟无人机当前位置 [纬度, 经度, 高度(m)]
        self.current_position = [39.908823, 116.397470, 10.0]  # 初始位置（天安门附近）
        # 目标位置
        self.target_position = None
        # 导航状态
        self.is_navigating = False

    def set_current_position(self, lat, lon, alt):
        """手动设置当前位置（模拟GPS更新）"""
        self.current_position = [lat, lon, alt]
        print(f"✅ 更新当前位置：纬度{lat:.6f}, 经度{lon:.6f}, 高度{alt:.1f}m")

    def calculate_gps_distance(self, pos1, pos2):
        """
        纯Python实现GPS两点距离计算（半正矢公式）
        :param pos1: [lat, lon, alt] 起点
        :param pos2: [lat, lon, alt] 终点
        :return: 地面距离（米）
        """
        # 地球半径（米）
        EARTH_RADIUS = 6371000.0

        # 转换为弧度
        lat1, lon1 = math.radians(pos1[0]), math.radians(pos1[1])
        lat2, lon2 = math.radians(pos2[0]), math.radians(pos2[1])

        # 计算经纬度差值
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # 半正矢公式核心计算
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = EARTH_RADIUS * c

        return round(distance, 2)

    def generate_straight_path(self, start_pos, target_pos, waypoint_count=5):
        """
        生成直线插值路径（无避障）
        :param waypoint_count: 中间航点数量
        :return: 航点列表 [[lat, lon, alt], ...]
        """
        path = []
        # 计算每个维度的步长
        lat_step = (target_pos[0] - start_pos[0]) / (waypoint_count + 1)
        lon_step = (target_pos[1] - start_pos[1]) / (waypoint_count + 1)
        alt_step = (target_pos[2] - start_pos[2]) / (waypoint_count + 1)

        # 生成中间航点
        for i in range(1, waypoint_count + 1):
            lat = start_pos[0] + lat_step * i
            lon = start_pos[1] + lon_step * i
            alt = start_pos[2] + alt_step * i
            path.append([round(lat, 6), round(lon, 6), round(alt, 1)])

        # 添加最终目标点
        path.append([target_pos[0], target_pos[1], target_pos[2]])
        return path

    def simulate_fly_to_waypoint(self, waypoint):
        """
        模拟飞向单个航点（逐步更新位置）
        :param waypoint: 目标航点 [lat, lon, alt]
        """
        # 每次移动的步长（模拟无人机飞行，每次移动0.00001度经纬度）
        LAT_STEP = 0.00001
        LON_STEP = 0.00001
        ALT_STEP = 0.5  # 高度每次移动0.5米

        # 持续移动直到到达航点（距离<1米）
        while True:
            distance = self.calculate_gps_distance(self.current_position, waypoint)
            if distance < 1.0:
                print(f"✅ 到达航点：{waypoint} (距离{distance}m)")
                break

            # 计算移动方向并更新位置
            current_lat, current_lon, current_alt = self.current_position
            target_lat, target_lon, target_alt = waypoint

            # 纬度调整
            if current_lat < target_lat:
                new_lat = current_lat + LAT_STEP
            elif current_lat > target_lat:
                new_lat = current_lat - LAT_STEP
            else:
                new_lat = current_lat

            # 经度调整
            if current_lon < target_lon:
                new_lon = current_lon + LON_STEP
            elif current_lon > target_lon:
                new_lon = current_lon - LON_STEP
            else:
                new_lon = current_lon

            # 高度调整
            if current_alt < target_alt:
                new_alt = current_alt + ALT_STEP
            elif current_alt > target_alt:
                new_alt = current_alt - ALT_STEP
            else:
                new_alt = current_alt

            # 更新位置
            self.set_current_position(new_lat, new_lon, new_alt)
            # 模拟飞行延迟
            time.sleep(0.1)

    def navigate_to_target(self, target_lat, target_lon, target_alt):
        """
        自主导航主函数（纯算法模拟）
        """
        self.target_position = [target_lat, target_lon, target_alt]
        self.is_navigating = True

        print("\n🚀 开始自主导航任务")
        print(f"📌 起点：{self.current_position}")
        print(f"🎯 终点：{self.target_position}")

        # 1. 生成路径
        path = self.generate_straight_path(self.current_position, self.target_position)
        print(f"\n🗺️  生成路径完成，共{len(path)}个航点：")
        for i, wp in enumerate(path):
            print(f"   航点{i + 1}：{wp}")

        # 2. 依次飞向每个航点
        print("\n✈️  开始飞向目标...")
        for i, waypoint in enumerate(path):
            print(f"\n--- 飞向第{i + 1}个航点 ---")
            self.simulate_fly_to_waypoint(waypoint)

        # 3. 导航完成
        self.is_navigating = False
        print("\n🎉 导航任务完成！已到达目标点")


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 初始化导航模拟器
    drone = DroneAutonomousNavigation()

    # 设置目标点（比如：北京奥林匹克公园，高度50米）
    target_lat = 39.990168
    target_lon = 116.397204
    target_alt = 50.0

    # 执行自主导航
    try:
        drone.navigate_to_target(target_lat, target_lon, target_alt)
    except KeyboardInterrupt:
        print("\n🛑 导航任务被手动终止")
    finally:
        print("\n🛬 无人机已悬停/降落")