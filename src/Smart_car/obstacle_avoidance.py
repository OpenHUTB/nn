import time
import math
from enum import Enum

# ======================== 常量定义 ========================
# 安全参数（可根据车型调整）
SAFE_DISTANCE = 5.0          # 安全距离（米），低于此值触发预警
EMERGENCY_DISTANCE = 2.0     # 紧急制动距离（米）
MAX_DECELERATION = 8.0       # 最大减速度（m/s²），符合道路安全标准
TTC_THRESHOLD_LOW = 3.0      # 低风险TTC阈值（秒）
TTC_THRESHOLD_HIGH = 1.5     # 高风险TTC阈值（秒）
VEHICLE_MAX_SPEED = 30.0     # 车辆最大速度（m/s）≈108km/h

# ======================== 枚举定义 ========================
class CollisionRiskLevel(Enum):
    """碰撞风险等级"""
    NONE = 0        # 无风险
    LOW = 1         # 低风险（预警）
    MEDIUM = 2      # 中风险（减速）
    HIGH = 3        # 高风险（紧急制动）

class ControlCommand(Enum):
    """车辆控制指令"""
    NORMAL = 0      # 正常行驶
    WARNING = 1     # 预警（声光提示）
    DECELERATE = 2  # 减速
    EMERGENCY_STOP = 3  # 紧急制动

# ======================== 核心类实现 ========================
class Obstacle:
    """障碍物类：模拟感知到的障碍物信息"""
    def __init__(self, distance: float, relative_speed: float, obstacle_type: str):
        """
        :param distance: 障碍物距离（米），正值表示前方
        :param relative_speed: 相对速度（m/s），正值表示靠近
        :param obstacle_type: 障碍物类型（行人/车辆/障碍物）
        """
        self.distance = max(0.0, distance)  # 距离非负
        self.relative_speed = relative_speed
        self.obstacle_type = obstacle_type
        self.update_time = time.time()  # 感知数据更新时间

    def update(self, distance: float, relative_speed: float):
        """更新障碍物感知数据"""
        self.distance = max(0.0, distance)
        self.relative_speed = relative_speed
        self.update_time = time.time()

class CollisionPreventionSystem:
    """碰撞预防系统核心类"""
    def __init__(self):
        self.current_speed = 0.0  # 车辆当前速度（m/s）
        self.obstacle = None      # 感知到的前方障碍物
        self.risk_level = CollisionRiskLevel.NONE
        self.control_command = ControlCommand.NORMAL

    def perception_update(self, obstacle_distance: float, obstacle_relative_speed: float, obstacle_type: str):
        """
        更新感知模块数据
        :param obstacle_distance: 障碍物距离（米）
        :param obstacle_relative_speed: 相对速度（m/s）
        :param obstacle_type: 障碍物类型
        """
        if self.obstacle is None:
            self.obstacle = Obstacle(obstacle_distance, obstacle_relative_speed, obstacle_type)
        else:
            self.obstacle.update(obstacle_distance, obstacle_relative_speed)

    def calculate_ttc(self) -> float:
        """
        计算碰撞时间（Time To Collision, TTC）
        :return: TTC值（秒），无穷大表示无碰撞风险
        """
        if self.obstacle is None or self.obstacle.relative_speed <= 0:
            return float('inf')  # 相对速度≤0，无碰撞风险
        return self.obstacle.distance / self.obstacle.relative_speed

    def evaluate_risk(self):
        """评估碰撞风险等级"""
        ttc = self.calculate_ttc()
        distance = self.obstacle.distance if self.obstacle else float('inf')

        # 风险等级判定逻辑
        if ttc >= TTC_THRESHOLD_LOW or distance >= SAFE_DISTANCE:
            self.risk_level = CollisionRiskLevel.NONE
        elif TTC_THRESHOLD_HIGH <= ttc < TTC_THRESHOLD_LOW or EMERGENCY_DISTANCE <= distance < SAFE_DISTANCE:
            self.risk_level = CollisionRiskLevel.LOW if ttc >= TTC_THRESHOLD_HIGH else CollisionRiskLevel.MEDIUM
        else:
            self.risk_level = CollisionRiskLevel.HIGH

    def generate_control_command(self) -> ControlCommand:
        """根据风险等级生成控制指令"""
        if self.risk_level == CollisionRiskLevel.NONE:
            self.control_command = ControlCommand.NORMAL
        elif self.risk_level == CollisionRiskLevel.LOW:
            self.control_command = ControlCommand.WARNING
        elif self.risk_level == CollisionRiskLevel.MEDIUM:
            self.control_command = ControlCommand.DECELERATE
        else:
            self.control_command = ControlCommand.EMERGENCY_STOP
        return self.control_command

    def execute_control(self, command: ControlCommand) -> float:
        """
        执行车辆控制指令，返回控制后的车速
        :param command: 控制指令
        :return: 调整后的车速（m/s）
        """
        delta_time = 0.1  # 控制周期（秒），模拟实时控制

        if command == ControlCommand.NORMAL:
            # 正常行驶，维持当前速度（可扩展加速逻辑）
            pass
        elif command == ControlCommand.WARNING:
            # 仅预警，不调整车速（声光提示驾驶员）
            print("[预警] 检测到前方障碍物，请注意！")
        elif command == ControlCommand.DECELERATE:
            # 减速：中等减速度（最大减速度的50%）
            deceleration = MAX_DECELERATION * 0.5
            self.current_speed = max(0.0, self.current_speed - deceleration * delta_time)
            print(f"[减速] 当前车速：{self.current_speed:.2f} m/s（原速度：{self.current_speed + deceleration * delta_time:.2f} m/s）")
        elif command == ControlCommand.EMERGENCY_STOP:
            # 紧急制动：最大减速度
            deceleration = MAX_DECELERATION
            self.current_speed = max(0.0, self.current_speed - deceleration * delta_time)
            print(f"[紧急制动] 当前车速：{self.current_speed:.2f} m/s（紧急制动中）")

        # 限制车速不超过最大值
        self.current_speed = min(self.current_speed, VEHICLE_MAX_SPEED)
        return self.current_speed

    def run_cycle(self, obstacle_distance: float, obstacle_relative_speed: float, obstacle_type: str, current_speed: float):
        """
        碰撞预防系统单次运行周期
        :param obstacle_distance: 障碍物距离（米）
        :param obstacle_relative_speed: 相对速度（m/s）
        :param obstacle_type: 障碍物类型
        :param current_speed: 车辆当前速度（m/s）
        """
        # 1. 更新车辆当前速度
        self.current_speed = current_speed

        # 2. 更新感知数据
        self.perception_update(obstacle_distance, obstacle_relative_speed, obstacle_type)

        # 3. 风险评估
        self.evaluate_risk()

        # 4. 生成控制指令
        command = self.generate_control_command()

        # 5. 执行控制
        new_speed = self.execute_control(command)

        # 6. 输出状态信息
        print(f"\n=== 系统状态 ===")
        print(f"障碍物类型：{obstacle_type}")
        print(f"障碍物距离：{self.obstacle.distance:.2f} 米")
        print(f"相对速度：{self.obstacle.relative_speed:.2f} m/s")
        print(f"碰撞时间（TTC）：{self.calculate_ttc():.2f} 秒")
        print(f"风险等级：{self.risk_level.name}")
        print(f"控制指令：{command.name}")
        print(f"当前车速：{new_speed:.2f} m/s")

# ======================== 测试用例 ========================
if __name__ == "__main__":
    # 初始化碰撞预防系统
    cps = CollisionPreventionSystem()

    # 模拟不同场景的运行周期
    test_scenarios = [
        # 场景1：无风险（远距离、低相对速度）
        {"distance": 10.0, "relative_speed": 1.0, "type": "车辆", "speed": 15.0},
        # 场景2：低风险（预警）
        {"distance": 6.0, "relative_speed": 3.0, "type": "行人", "speed": 10.0},
        # 场景3：中风险（减速）
        {"distance": 3.0, "relative_speed": 4.0, "type": "障碍物", "speed": 8.0},
        # 场景4：高风险（紧急制动）
        {"distance": 1.5, "relative_speed": 5.0, "type": "车辆", "speed": 6.0},
    ]

    # 运行测试场景
    for i, scenario in enumerate(test_scenarios):
        print(f"\n==================== 测试场景 {i+1} ====================")
        cps.run_cycle(
            obstacle_distance=scenario["distance"],
            obstacle_relative_speed=scenario["relative_speed"],
            obstacle_type=scenario["type"],
            current_speed=scenario["speed"]
        )
        time.sleep(0.5)  # 模拟时间间隔