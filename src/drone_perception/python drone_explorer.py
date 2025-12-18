"""
AirSimNH 感知驱动自主探索无人机 - 集成前视窗口版
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
import queue
import signal
import sys
from typing import Tuple, List, Optional, Dict

# ==================== 前视显示窗口类 ====================

class FrontViewDisplay:
    """
    前视画面显示管理器
    修复了窗口模糊和显示问题
    """

    def __init__(self, window_name="无人机前视画面", width=640, height=480):
        """
        初始化显示窗口

        注意：要获得清晰画面，有两种方法：
        1. 修改AirSim配置文件settings.json，提高相机分辨率（推荐）
        2. 在代码中启用图像锐化处理（下方已实现）
        """
        self.window_name = window_name
        self.window_width = width
        self.window_height = height

        # 图像队列（线程安全）
        self.image_queue = queue.Queue(maxsize=2)
        self.display_active = True
        self.display_thread = None

        # 显示控制
        self.paused = False
        self.show_info = True

        # 图像增强选项
        self.enable_sharpening = True  # 启用图像锐化（改善模糊）
        self.enable_upscaling = False  # 启用图像放大（谨慎使用，可能更模糊）

        # 启动显示线程
        self.start()

    def start(self):
        """启动显示线程"""
        if self.display_thread and self.display_thread.is_alive():
            return

        self.display_active = True
        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="FrontViewDisplay"
        )
        self.display_thread.start()
        print(f"✅ 前视画面窗口已启动: {self.window_name}")
        print("💡 显示窗口控制:")
        print("   - 按 'Q': 关闭显示窗口")
        print("   - 按 'S': 保存当前画面")
        print("   - 按 'P': 暂停/继续视频流")
        print("   - 按 'I': 切换信息叠加层")
        print("💡 画面清晰度提示:")
        print("   当前分辨率: 256x144 (默认较低)")
        print("   如需更高清，请修改settings.json相机配置")

    def stop(self):
        """停止显示线程"""
        self.display_active = False
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
            print("🛑 前视画面显示窗口已停止")

    def update_image(self, image_data: np.ndarray, info: Optional[Dict] = None):
        """
        更新要显示的图像

        Args:
            image_data: OpenCV格式的图像 (BGR)
            info: 叠加信息，如状态、位置等
        """
        if not self.display_active or self.paused or image_data is None:
            return

        try:
            # 图像增强处理
            processed_image = self._enhance_image(image_data)

            # 如果队列已满，丢弃最旧的一帧
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass

            display_packet = {
                'image': processed_image,
                'info': info.copy() if info else {},
                'timestamp': time.time()
            }

            self.image_queue.put_nowait(display_packet)

        except Exception as e:
            print(f"⚠️ 更新图像时出错: {e}")

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """图像增强处理"""
        if image is None or image.size == 0:
            return image

        enhanced = image.copy()

        try:
            # 1. 锐化处理（改善模糊）
            if self.enable_sharpening:
                # 使用拉普拉斯算子进行锐化
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)

            # 2. 对比度增强
            # enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)

            # 3. 可选：放大图像（谨慎使用）
            if self.enable_upscaling and enhanced.shape[0] < 400:
                # 使用双三次插值放大
                new_width = self.window_width
                new_height = self.window_height
                enhanced = cv2.resize(enhanced, (new_width, new_height),
                                     interpolation=cv2.INTER_CUBIC)

            return enhanced

        except Exception as e:
            print(f"⚠️ 图像增强出错: {e}")
            return image

    def _display_loop(self):
        """显示线程主循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        # 初始显示等待画面
        wait_img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(wait_img, "等待无人机图像...", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(self.window_name, wait_img)
        cv2.waitKey(100)  # 关键：给窗口时间初始化

        print("🎬 显示线程开始运行...")

        while self.display_active:
            display_image = None
            info = {}

            try:
                # 获取最新图像
                if not self.image_queue.empty():
                    packet = self.image_queue.get_nowait()
                    display_image = packet['image']
                    info = packet['info']

                    # 清空队列中的旧帧
                    while not self.image_queue.empty():
                        self.image_queue.get_nowait()
            except queue.Empty:
                pass

            # 显示图像
            if display_image is not None:
                # 添加信息叠加
                if self.show_info:
                    display_image = self._add_info_overlay(display_image, info)

                cv2.imshow(self.window_name, display_image)
            elif self.paused:
                # 暂停时显示提示
                blank = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(blank, "PAUSED", (120, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow(self.window_name, blank)
            else:
                # 无图像时保持窗口活动
                pass

            # 关键：必须定期调用waitKey保持窗口响应
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("🔄 用户关闭显示窗口")
                self.display_active = False
                break
            elif key == ord('s') or key == ord('S'):
                self._save_screenshot(display_image)
            elif key == ord('p') or key == ord('P'):
                self.paused = not self.paused
                status = "已暂停" if self.paused else "已恢复"
                print(f"⏸️ 视频流{status}")
            elif key == ord('i') or key == ord('I'):
                self.show_info = not self.show_info
                status = "开启" if self.show_info else "关闭"
                print(f"📊 信息叠加层{status}")
            elif key == ord('h') or key == ord('H'):
                # 切换锐化效果
                self.enable_sharpening = not self.enable_sharpening
                status = "开启" if self.enable_sharpening else "关闭"
                print(f"🔍 图像锐化{status}")

        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)

    def _add_info_overlay(self, image: np.ndarray, info: Dict) -> np.ndarray:
        """在图像上叠加状态信息"""
        try:
            overlay = image.copy()
            height, width = image.shape[:2]

            # 创建半透明信息栏
            info_height = 80
            cv2.rectangle(overlay, (0, 0), (width, info_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # 飞行状态
            state = info.get('state', 'UNKNOWN')
            state_color = (0, 255, 0) if 'EXPLORING' in state else (0, 255, 255) if 'HOVERING' in state else (0, 0, 255)
            cv2.putText(image, f"状态: {state}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            # 位置信息
            pos = info.get('position', (0, 0, 0))
            cv2.putText(image, f"位置: ({pos[0]:.1f}, {pos[1]:.1f}, {-pos[2]:.1f}m)", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 障碍物信息
            obs_dist = info.get('obstacle_distance', 0.0)
            if obs_dist < 100:
                obs_color = (0, 0, 255) if obs_dist < 5.0 else (0, 165, 255) if obs_dist < 10.0 else (0, 255, 0)
                cv2.putText(image, f"障碍: {obs_dist:.1f}m", (width - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, obs_color, 1)

            # 分辨率提示（只在低分辨率时显示）
            if height < 200:
                cv2.putText(image, f"分辨率: {width}x{height}", (width - 120, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            return image
        except Exception as e:
            print(f"⚠️ 添加信息叠加层出错: {e}")
            return image

    def _save_screenshot(self, image: Optional[np.ndarray]):
        """保存当前画面为截图"""
        if image is not None and image.size > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"drone_snapshot_{timestamp}.png"
            cv2.imwrite(filename, image)
            print(f"📸 截图已保存: {filename}")
        else:
            print("⚠️ 无法保存截图：无有效图像数据")


# ==================== 原有类定义 ====================

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
    front_image: Optional[np.ndarray] = None  # 新增：前视图像

    def __post_init__(self):
        if self.safe_directions is None:
            self.safe_directions = []


class PerceptiveExplorer:
    """基于感知的自主探索无人机 - 集成前视窗口版"""

    def __init__(self, drone_name=""):
        print("=" * 60)
        print("AirSimNH 感知驱动自主探索系统 - 集成前视窗口版")
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

        # 性能监控
        self.perception_fps = 0
        self.decision_fps = 0
        self.start_time = time.time()

        # ========== 新增：前视显示窗口 ==========
        self.front_display = FrontViewDisplay(
            window_name=f"无人机前视 - {drone_name or 'AirSimNH'}",
            width=640,
            height=480
        )

        print("✅ 系统初始化完成")
        print(f"   开始时间: {time.strftime('%H:%M:%S')}")
        print(f"   预计探索时长: {self.exploration_time}秒")
        print("🎥 前视窗口已集成，支持交互控制")

    def get_depth_perception(self) -> PerceptionResult:
        """获取并分析深度图像，理解环境 - 集成前视图像"""
        result = PerceptionResult()

        try:
            # ========== 新增：同时获取深度图像和前视图像 ==========
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    "0",  # 相机名称，如"front_center"
                    airsim.ImageType.DepthPlanar,
                    pixels_as_float=True,
                    compress=False
                ),
                airsim.ImageRequest(
                    "0",  # 同一相机获取前视图像
                    airsim.ImageType.Scene,
                    False,
                    False
                )
            ])

            if not responses or len(responses) < 2:
                print("⚠️ 图像获取失败")
                return result

            # 1. 处理深度图像（原有逻辑）
            depth_img = responses[0]
            if depth_img:
                depth_array = np.array(depth_img.image_data_float, dtype=np.float32)
                depth_array = depth_array.reshape(depth_img.height, depth_img.width)

                # 分析深度图像的不同区域
                h, w = depth_array.shape

                # 前方近距离区域（紧急避障）
                front_near = depth_array[h // 2:, w // 3:2 * w // 3]
                min_front_distance = np.min(front_near) if front_near.size > 0 else 100

                # 多方向扇形扫描
                directions = []
                scan_angles = [-45, -30, -15, 0, 15, 30, 45]  # 度

                for angle_deg in scan_angles:
                    angle_rad = math.radians(angle_deg)
                    col = int(w / 2 + (w / 2) * math.tan(angle_rad) * 0.5)
                    col = max(0, min(w - 1, col))

                    col_data = depth_array[h // 2:, col]
                    if col_data.size > 0:
                        dir_distance = np.percentile(col_data, 25)
                        directions.append((angle_rad, dir_distance))

                        if dir_distance > self.depth_threshold_safe:
                            result.safe_directions.append(angle_rad)

                # 地形分析
                ground_region = depth_array[3 * h // 4:, :]
                if ground_region.size > 10:
                    row_variances = np.var(ground_region, axis=1)
                    result.terrain_slope = np.mean(row_variances) * 100

                # 开阔度评分
                open_pixels = np.sum(depth_array[h // 2:, :] > self.depth_threshold_safe)
                total_pixels = depth_array[h // 2:, :].size
                result.open_space_score = open_pixels / total_pixels if total_pixels > 0 else 0

                # 整合感知结果
                result.has_obstacle = min_front_distance < self.depth_threshold_near
                result.obstacle_distance = min_front_distance

                if directions:
                    closest_dir = min(directions, key=lambda x: x[1])
                    result.obstacle_direction = closest_dir[0]

                # 根据感知动态调整推荐高度
                if result.terrain_slope > 5:
                    result.recommended_height = -20
                elif result.open_space_score > 0.7:
                    result.recommended_height = -12
                else:
                    result.recommended_height = -15

            # ========== 新增：处理前视图像 ==========
            front_img = responses[1]
            if front_img and hasattr(front_img, 'image_data_uint8'):
                try:
                    # 转换为OpenCV格式
                    img_array = np.frombuffer(front_img.image_data_uint8, dtype=np.uint8)

                    if len(img_array) > 0:
                        img_rgb = img_array.reshape(front_img.height, front_img.width, 3)
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        result.front_image = img_bgr

                        # 获取当前位置用于显示
                        state = self.client.getMultirotorState(vehicle_name=self.drone_name)
                        pos = state.kinematics_estimated.position

                        # 准备显示信息
                        display_info = {
                            'state': self.state.name,
                            'obstacle_distance': result.obstacle_distance,
                            'position': (pos.x_val, pos.y_val, pos.z_val),
                        }

                        # 更新前视窗口
                        self.front_display.update_image(img_bgr, display_info)

                        # 调试：打印图像信息
                        if hasattr(self, 'loop_count') and self.loop_count % 50 == 0:
                            print(f"📷 前视图像: {front_img.width}x{front_img.height}")
                    else:
                        print("⚠️ 前视图像数据为空")

                except Exception as e:
                    print(f"⚠️ 前视图像处理出错: {e}")

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
        self.loop_count = 0
        exploration_start = time.time()

        while time.time() - exploration_start < self.exploration_time and not self.emergency_flag:
            loop_start = time.time()
            self.loop_count += 1

            # 1. 感知阶段
            self.perception_start = time.time()
            perception = self.get_depth_perception()

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
            if self.loop_count % 20 == 0:
                elapsed = time.time() - exploration_start
                print(f"\n📊 循环{self.loop_count} | 已运行{elapsed:.1f}s | 状态:{self.state.value}")
                print(f"   感知FPS:{self.perception_fps:.1f} | 障碍:{perception.has_obstacle}")
                print(f"   最近障碍:{perception.obstacle_distance:.1f}m | 开阔度:{perception.open_space_score:.2f}")
                print(f"   位置:({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}m)")
                print(f"   安全方向数:{len(perception.safe_directions)}")

                # 显示前视窗口状态
                if hasattr(self, 'front_display'):
                    print(f"   前视窗口: {'活跃' if self.front_display.display_active else '关闭'}")

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

            # 关闭前视窗口
            if hasattr(self, 'front_display'):
                self.front_display.stop()

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

        # 关闭前视窗口
        if hasattr(self, 'front_display'):
            self.front_display.stop()


def main():
    """主程序入口"""
    try:
        # 创建感知探索器
        explorer = PerceptiveExplorer(drone_name="")

        # 设置键盘中断处理
        def signal_handler(sig, frame):
            print("\n⚠ 用户中断，正在安全停止...")
            explorer.emergency_stop()
            sys.exit(0)

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
    print("=" * 70)
    print("重要提示：")
    print("1. 要获得高清画面，请修改AirSim的settings.json配置文件")
    print("   提高相机分辨率（如1280x720）")
    print("2. 当前代码已集成图像锐化处理改善模糊")
    print("3. 前视窗口支持交互控制（按H键切换锐化）")
    print("=" * 70)
    main()