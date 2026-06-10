# test_obstacle.py - 独立测试脚本，不需要CARLA

class MockControl:
    def __init__(self):
        self.throttle = 0
        self.brake = 0
        self.steer = 0

class MockVehicle:
    def apply_control(self, control):
        print(f"   → 执行: throttle={control.throttle:.1f}, brake={control.brake:.1f}")

# 从你的 EgoVehicleController 复制过来（只保留核心方法）
import math

class EgoVehicleController:
    def __init__(self) -> None:
        self.controller = None
        self.cruise_speed = 30.0
        self.cruise_enabled = False
    
    def update_ego_vehicle(self, ego_vehicle, control, obstacle_distance=None):
        # 获取当前速度和位置（模拟）
        speed = 30.0  # 模拟速度 km/h

        # ========== 自动避障逻辑 ==========
        if obstacle_distance is not None:
            if obstacle_distance < 3.0:
                control.throttle = 0.0
                control.brake = 1.0
                print(f"⚠️ 紧急刹车！距离障碍物 {obstacle_distance:.1f} 米")
                return
            elif obstacle_distance < 6.0:
                control.throttle = 0.2
                control.brake = 0.3
                print(f"⚠️ 减速慢行，距离障碍物 {obstacle_distance:.1f} 米")
                return
            elif obstacle_distance < 10.0:
                control.throttle = 0.35
                control.brake = 0.1
                print(f"⚠️ 注意前方，距离障碍物 {obstacle_distance:.1f} 米")
                return

        # 正常行驶
        if speed < 30.0:
            control.throttle = 0.5
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = 0.1

# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("障碍物检测功能测试")
    print("=" * 50)
    
    controller = EgoVehicleController()
    mock_car = MockVehicle()
    mock_control = MockControl()
    
    # 测试不同距离
    test_cases = [15.0, 8.0, 5.0, 2.5]
    
    for dist in test_cases:
        print(f"\n>>> 测试距离: {dist} 米")
        controller.update_ego_vehicle(mock_car, mock_control, dist)
    
    print("\n" + "=" * 50)
    print("✅ 测试完成！请截图当前窗口作为运行结果。")
    print("=" * 50)