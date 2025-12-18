"""
AirSimNH 感知驱动自主探索无人机
核心：视觉感知 → 语义理解 → 智能决策 → 安全执行
"""

import airsim
import time
import numpy as np
import cv2
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Tuple, List, Optional


class FlightState(Enum):
    """无人机状态枚举"""
    TAKEOFF = "起飞"
    HOVERING = "悬停观测"
    EXPLORING = "主动探索"
    AVOIDING = "避障机动"
    LANDING = "降落"
    EMERGENCY = "紧急状态"


@dataclass
class PerceptionResult:
    """感知结果数据结构"""
    has_obstacle: bool = False
    obstacle_distance: float = 100.0
    obstacle_direction: float = 0.0  # 障碍物相对方向（弧度）
    terrain_slope: float = 0.0  # 地形坡度
    open_space_score: float = 0.0  # 开阔度评分 (0-1)
    recommended_height: float = -15.0  # 推荐飞行高度
    safe_directions: List[float] = None  # 安全方向列表

    def __post_init__(self):
        if self.safe_directions is None:
            self.safe_directions = []


class PerceptiveExplorer:
    """基于感知的自主探索无人机"""

    def __init__(self, drone_name=""):
        print("=" * 60)
        print("AirSimNH 感知驱动自主探索系统")
        print("=" * 60)

        # 初始化AirSim连接
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.drone_name = drone_name

        # 启用API控制
        self.client.enableApiControl(True, vehicle_name=drone_name)
        self.client.armDisarm(True, vehicle_name=drone_name)

        # 状态管理
        self.state = FlightState.TAKEOFF
        self.state_history = deque(maxlen=20)
        self.emergency_flag = False

        # 感知参数
        self.depth_threshold_near = 5.0  # 近距离警报阈值(米)
        self.depth_threshold_safe = 10.0  # 安全距离阈值(米)
        self.min_ground_clearance = 2.0  # 最小离地间隙(米)
        self.max_pitch_angle = math.radians(15)  # 最大允许俯仰角

        # 探索参数
        self.exploration_time = 180  # 总探索时间(秒)
        self.preferred_speed = 3.0  # 优选速度(m/s)
        self.max_altitude = -30  # 最大海拔(米)
        self.min_altitude = -8  # 最小海拔(米)

        # 记忆系统
        self.visited_positions = deque(maxlen=100)
        self.obstacle_map = {}  # 障碍物位置记忆
        self.traversability_map = {}  # 地形可通行性记忆

        # 性能监控
        self.perception_fps = 0
        self.decision_fps = 0
        self.start_time = time.time()

        print("✅ 系统初始化完成")
        print(f"   开始时间: {time.strftime('%H:%M:%S')}")
        print(f"   预计探索时长: {self.exploration_time}秒")

    def get_depth_perception(self) -> PerceptionResult:
        """获取并分析深度图像，理解环境"""
        result = PerceptionResult()

        try:
            # 请求深度图像（使用正确的DepthPlanar类型）
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    "0",
                    airsim.ImageType.DepthPlanar,
                    pixels_as_float=True,
                    compress=False
                )
            ])

            if not responses or not responses[0]:
                print("⚠ 深度图像获取失败")
                return result

            # 转换深度数据为numpy数组
            depth_img = responses[0]
            depth_array = np.array(depth_img.image_data_float, dtype=np.float32)
            depth_array = depth_array.reshape(depth_img.height, depth_img.width)

            # 分析深度图像的不同区域
            h, w = depth_array.shape

            # 1. 前方近距离区域（紧急避障）
            front_near = depth_array[h // 2:, w // 3:2 * w // 3]
            min_front_distance = np.min(front_near) if front_near.size > 0 else 100

            # 2. 多方向扇形扫描
            directions = []
            scan_angles = [-45, -30, -15, 0, 15, 30, 45]  # 度

            for angle_deg in scan_angles:
                angle_rad = math.radians(angle_deg)
                # 计算对应图像列
                col = int(w / 2 + (w / 2) * math.tan(angle_rad) * 0.5)
                col = max(0, min(w - 1, col))

                # 分析该列的深度
                col_data = depth_array[h // 2:, col]
                if col_data.size > 0:
                    dir_distance = np.percentile(col_data, 25)  # 使用25%分位数（较保守）
                    directions.append((angle_rad, dir_distance))

                    if dir_distance > self.depth_threshold_safe:
                        result.safe_directions.append(angle_rad)

            # 3. 地形分析（通过深度梯度估计坡度）
            ground_region = depth_array[3 * h // 4:, :]
            if ground_region.size > 10:
                row_variances = np.var(ground_region, axis=1)
                result.terrain_slope = np.mean(row_variances) * 100

            # 4. 开阔度评分（基于有效距离像素比例）
            open_pixels = np.sum(depth_array[h // 2:, :] > self.depth_threshold_safe)
            total_pixels = depth_array[h // 2:, :].size
            result.open_space_score = open_pixels / total_pixels if total_pixels > 0 else 0

            # 整合感知结果
            result.has_obstacle = min_front_distance < self.depth_threshold_near
            result.obstacle_distance = min_front_distance

            if directions:
                # 找出最近障碍物的方向
                closest_dir = min(directions, key=lambda x: x[1])
                result.obstacle_direction = closest_dir[0]

            # 根据感知动态调整推荐高度
            if result.terrain_slope > 5:
                result.recommended_height = -20  # 陡峭地形飞高些
            elif result.open_space_score > 0.7:
                result.recommended_height = -12  # 开阔地带可以飞低些
            else:
                result.recommended_height = -15  # 默认高度

            # 更新感知FPS
            self.perception_fps = 1 / (time.time() - self.perception_start)

        except Exception as e:
            print(f"❌ 深度感知异常: {e}")

        return result

    def get_visual_perception(self):
        """获取视觉图像用于高级感知（可选）"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            if responses and responses[0]:
                # 转换为OpenCV格式
                img_data = responses[0].image_data_uint8
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = img_array.reshape(responses[0].height, responses[0].width, 3)

                # 简单颜色分析（示例：寻找绿色植被区域）
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                green_mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
                green_ratio = np.sum(green_mask > 0) / green_mask.size

                return img, green_ratio
        except:
            pass

        return None, 0

    def make_intelligent_decision(self, perception: PerceptionResult) -> Tuple[float, float, float, float]:
        """基于感知结果做出智能决策"""

        # 获取当前位置和状态
        state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity

        # 基础决策：速度、偏航、高度
        target_vx, target_vy, target_z, target_yaw = 0.0, 0.0, perception.recommended_height, 0.0

        # 状态机决策逻辑
        if self.state == FlightState.TAKEOFF:
            target_z = -10  # 起飞到10米
            if pos.z_val < -9.5:
                self.change_state(FlightState.HOVERING)

        elif self.state == FlightState.HOVERING:
            # 悬停观察，缓慢旋转扫描
            target_yaw = (time.time() % 10) * 0.2  # 缓慢旋转

            if len(perception.safe_directions) > 0:
                self.change_state(FlightState.EXPLORING)

        elif self.state == FlightState.EXPLORING:
            # 主动探索模式
            if perception.has_obstacle:
                self.change_state(FlightState.AVOIDING)
                # 紧急制动
                target_vx, target_vy = -vel.x_val, -vel.y_val
            else:
                # 选择最佳探索方向
                if perception.safe_directions:
                    # 优先选择与当前航向相差45-90度的新方向（避免来回摆动）
                    current_yaw = airsim.to_eularian_angles(
                        state.kinematics_estimated.orientation
                    )[2]

                    # 过滤出与当前方向不同的安全方向
                    diverse_dirs = [
                        d for d in perception.safe_directions
                        if abs(d - current_yaw) > math.radians(45)
                    ]

                    if diverse_dirs:
                        best_dir = diverse_dirs[0]
                    else:
                        best_dir = perception.safe_directions[0]

                    # 设置前进速度
                    speed_factor = min(1.0, perception.open_space_score * 1.5)
                    target_vx = self.preferred_speed * speed_factor * math.cos(best_dir)
                    target_vy = self.preferred_speed * speed_factor * math.sin(best_dir)
                else:
                    # 没有安全方向，爬升
                    target_z = pos.z_val - 5
                    self.change_state(FlightState.AVOIDING)

        elif self.state == FlightState.AVOIDING:
            # 避障机动
            if perception.has_obstacle:
                # 根据障碍物方向决定避障策略
                if abs(perception.obstacle_direction) < math.radians(30):
                    # 前方障碍物：爬升
                    target_z = pos.z_val - 3
                    target_vx, target_vy = 0, 0
                else:
                    # 侧方障碍物：向反方向平移
                    avoid_dir = perception.obstacle_direction + math.pi
                    target_vx = 1.5 * math.cos(avoid_dir)
                    target_vy = 1.5 * math.sin(avoid_dir)
            else:
                # 障碍物清除，返回探索
                self.change_state(FlightState.HOVERING)
                time.sleep(1)  # 避障后暂停观察

        elif self.state == FlightState.EMERGENCY:
            # 紧急状态：悬停并准备降落
            target_vx, target_vy, target_yaw = 0, 0, 0
            target_z = max(pos.z_val, -20)  # 限制爬升

        # 确保高度在安全范围内
        target_z = max(self.max_altitude, min(self.min_altitude, target_z))

        return target_vx, target_vy, target_z, target_yaw

    def change_state(self, new_state: FlightState):
        """状态转换"""
        if self.state != new_state:
            print(f"🔄 状态转换: {self.state.value} → {new_state.value}")
            self.state = new_state
            self.state_history.append((time.time(), new_state))

    def run_perception_loop(self):
        """主感知-决策-控制循环"""
        print("\n" + "=" * 60)
        print("启动感知-决策-控制循环")
        print("=" * 60)

        # 起飞
        print("🚀 起飞中...")
        self.client.takeoffAsync(vehicle_name=self.drone_name).join()
        time.sleep(3)
        self.change_state(FlightState.HOVERING)

        # 主循环
        loop_count = 0
        exploration_start = time.time()

        while time.time() - exploration_start < self.exploration_time and not self.emergency_flag:
            loop_start = time.time()
            loop_count += 1

            # 1. 感知阶段
            self.perception_start = time.time()
            perception = self.get_depth_perception()
            # visual_img, green_ratio = self.get_visual_perception()  # 可选

            # 2. 决策阶段
            decision = self.make_intelligent_decision(perception)

            # 3. 控制执行阶段
            target_vx, target_vy, target_z, target_yaw = decision

            # 使用速度控制（更灵活）或位置控制（更精确）
            use_velocity_control = self.state in [FlightState.EXPLORING, FlightState.AVOIDING]

            if use_velocity_control:
                self.client.moveByVelocityZAsync(
                    target_vx, target_vy, target_z, 0.5,  # 持续时间0.5秒
                    vehicle_name=self.drone_name
                )
            else:
                self.client.moveToPositionAsync(
                    0, 0, target_z, 2,  # 相对当前位置移动
                    vehicle_name=self.drone_name
                )

            # 记录当前位置
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            self.visited_positions.append((pos.x_val, pos.y_val, pos.z_val))

            # 性能监控输出
            if loop_count % 20 == 0:
                elapsed = time.time() - exploration_start
                print(f"\n📊 循环{loop_count} | 已运行{elapsed:.1f}s | 状态:{self.state.value}")
                print(f"   感知FPS:{self.perception_fps:.1f} | 障碍:{perception.has_obstacle}")
                print(f"   最近障碍:{perception.obstacle_distance:.1f}m | 开阔度:{perception.open_space_score:.2f}")
                print(f"   位置:({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}m)")
                print(f"   安全方向数:{len(perception.safe_directions)}")

            # 循环频率控制（10Hz）
            loop_time = time.time() - loop_start
            if loop_time < 0.1:
                time.sleep(0.1 - loop_time)

        # 探索结束，准备降落
        print("\n" + "=" * 60)
        print("探索完成，开始返航降落")
        print("=" * 60)

        self.change_state(FlightState.LANDING)
        self.return_to_start()

    def return_to_start(self):
        """返回起始点附近并降落"""
        try:
            # 回到起点附近（简单实现：回到原点）
            print("↩️ 返回起始区域...")
            self.client.moveToPositionAsync(0, 0, -10, 5, vehicle_name=self.drone_name).join()
            time.sleep(2)

            # 降落
            print("🛬 降落中...")
            self.client.landAsync(vehicle_name=self.drone_name).join()
            time.sleep(3)

            # 断开控制
            self.client.armDisarm(False, vehicle_name=self.drone_name)
            self.client.enableApiControl(False, vehicle_name=self.drone_name)

            print("✅ 任务完成，系统安全关闭")

        except Exception as e:
            print(f"❌ 降落异常: {e}")
            print("⚠ 尝试紧急降落...")
            try:
                self.client.landAsync(vehicle_name=self.drone_name).join()
            except:
                pass

    def emergency_stop(self):
        """紧急停止"""
        print("🆘 紧急停止触发！")
        self.emergency_flag = True
        self.change_state(FlightState.EMERGENCY)
        self.client.hoverAsync(vehicle_name=self.drone_name).join()


def main():
    """主程序入口"""
    try:
        # 创建感知探索器
        explorer = PerceptiveExplorer(drone_name="")

        # 设置键盘中断处理
        import signal
        def signal_handler(sig, frame):
            print("\n⚠ 用户中断，正在安全停止...")
            explorer.emergency_stop()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # 运行主循环
        explorer.run_perception_loop()

    except Exception as e:
        print(f"\n❌ 系统异常: {e}")
        import traceback
        traceback.print_exc()

        # 尝试安全降落
        try:
            client = airsim.MultirotorClient()
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except:
            pass


if __name__ == "__main__":
    main()