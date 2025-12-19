import math

# 定义无人机类，包含核心导航属性和方法
class UAVAutonomousNavigation:
    def __init__(self):
        # 无人机当前位置（经纬度，单位：度）
        self.current_lon = 0.0
        self.current_lat = 0.0
        # 预设航点列表（格式：[纬度, 经度]）
        self.waypoints = []
        # 当前目标航点索引
        self.target_waypoint_idx = 0
        # 到达航点的判定阈值（米）
        self.arrival_threshold = 5.0

    def set_waypoints(self, waypoints_list):
        """设置航点列表"""
        self.waypoints = waypoints_list
        self.target_waypoint_idx = 0  # 重置目标航点索引

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        计算两个经纬度点之间的地面距离（米）
        采用简化的Haversine公式，适用于近距离无人机导航
        """
        # 地球半径（米）
        R = 6371000.0
        # 角度转弧度
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # 经纬度差值
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine公式
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def update_position(self, new_lat, new_lon):
        """更新无人机当前位置"""
        self.current_lat = new_lat
        self.current_lon = new_lon

    def navigate_to_waypoints(self):
        """核心航点导航逻辑：依次飞向每个航点"""
        if not self.waypoints:
            print("未设置航点，导航终止")
            return

        # 获取当前目标航点
        target_lat, target_lon = self.waypoints[self.target_waypoint_idx]
        # 计算当前位置到目标航点的距离
        distance_to_target = self.calculate_distance(
            self.current_lat, self.current_lon, target_lat, target_lon
        )

        print(f"当前位置：({self.current_lat:.6f}, {self.current_lon:.6f})")
        print(f"目标航点{self.target_waypoint_idx+1}：({target_lat:.6f}, {target_lon:.6f})")
        print(f"到目标航点距离：{distance_to_target:.2f} 米")

        # 判断是否到达当前航点
        if distance_to_target < self.arrival_threshold:
            print(f"已到达航点{self.target_waypoint_idx+1}")
            self.target_waypoint_idx += 1
            # 判断是否完成所有航点
            if self.target_waypoint_idx >= len(self.waypoints):
                print("所有航点导航完成！")
                return
            else:
                print(f"切换至下一个航点：{self.target_waypoint_idx+1}")
        else:
            # 生成飞行指令（实际场景中需替换为无人机SDK的飞行控制指令）
            print("飞行指令：向目标航点飞行（航向/速度需根据实际SDK调整）")

# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 初始化无人机导航实例
    uav = UAVAutonomousNavigation()

    # 设置航点列表（示例：3个航点）
    waypoints = [
        (39.908823, 116.397470),  # 航点1：天安门
        (39.997563, 116.337624),  # 航点2：颐和园
        (40.006401, 116.397029)   # 航点3：圆明园
    ]
    uav.set_waypoints(waypoints)

    # 模拟无人机位置更新与导航过程
    # 模拟初始位置（靠近航点1）
    uav.update_position(39.908820, 116.397468)
    uav.navigate_to_waypoints()

    print("-" * 50)

    # 模拟无人机飞到航点1，更新位置
    uav.update_position(39.908823, 116.397470)
    uav.navigate_to_waypoints()