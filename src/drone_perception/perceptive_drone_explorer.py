"""
AirSimNH 感知驱动自主探索无人机 - 工程优化版
核心：视觉感知 → 语义理解 → 智能决策 → 安全执行
集成：配置管理、日志系统、异常恢复、前视窗口显示
版本: 2.0 (工程优化版)
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
import traceback
import logging
from datetime import datetime

# ============ 导入配置文件 ============
try:
    import config
    CONFIG_LOADED = True
except ImportError as e:
    print(f"❌ 无法加载配置文件 config.py: {e}")
    print("正在使用默认配置...")
    CONFIG_LOADED = False
    # 临时默认配置
    class DefaultConfig:
        EXPLORATION = {'TOTAL_TIME': 180, 'PREFERRED_SPEED': 3.0, 'BASE_HEIGHT': -15.0,
                      'MAX_ALTITUDE': -30.0, 'MIN_ALTITUDE': -8.0, 'TAKEOFF_HEIGHT': -10.0}
        PERCEPTION = {'DEPTH_NEAR_THRESHOLD': 5.0, 'DEPTH_SAFE_THRESHOLD': 10.0,
                     'MIN_GROUND_CLEARANCE': 2.0, 'MAX_PITCH_ANGLE_DEG': 15,
                     'SCAN_ANGLES': [-45, -30, -15, 0, 15, 30, 45],
                     'HEIGHT_STRATEGY': {'STEEP_SLOPE': -20.0, 'OPEN_SPACE': -12.0,
                                         'DEFAULT': -15.0, 'SLOPE_THRESHOLD': 5.0,
                                         'OPENNESS_THRESHOLD': 0.7}}
        DISPLAY = {'WINDOW_WIDTH': 640, 'WINDOW_HEIGHT': 480, 'ENABLE_SHARPENING': True,
                  'SHOW_INFO_OVERLAY': True, 'REFRESH_RATE_MS': 30}
        SYSTEM = {'LOG_LEVEL': 'INFO', 'LOG_TO_FILE': True, 'LOG_FILENAME': 'drone_log.txt',
                 'MAX_RECONNECT_ATTEMPTS': 3, 'RECONNECT_DELAY': 2.0,
                 'ENABLE_HEALTH_CHECK': True, 'HEALTH_CHECK_INTERVAL': 10}
        CAMERA = {'DEFAULT_NAME': "0"}
    config = DefaultConfig()


class FlightState(Enum):
    """无人机飞行状态枚举"""
    TAKEOFF = "起飞"
    HOVERING = "悬停观测"
    EXPLORING = "主动探索"
    AVOIDING = "避障机动"
    RETURNING = "返航中"
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
    recommended_height: float = config.PERCEPTION['HEIGHT_STRATEGY']['DEFAULT']
    safe_directions: List[float] = None  # 安全方向列表
    front_image: Optional[np.ndarray] = None  # 前视图像

    def __post_init__(self):
        if self.safe_directions is None:
            self.safe_directions = []


# ==================== 前视显示窗口类 ====================

class FrontViewDisplay:
    """前视画面显示管理器"""

    def __init__(self, window_name="无人机前视画面", width=None, height=None,
                 enable_sharpening=None, show_info=None):
        # 使用配置参数或传入参数
        self.window_name = window_name
        self.window_width = width if width is not None else config.DISPLAY['WINDOW_WIDTH']
        self.window_height = height if height is not None else config.DISPLAY['WINDOW_HEIGHT']
        self.enable_sharpening = (enable_sharpening if enable_sharpening is not None
                                 else config.DISPLAY['ENABLE_SHARPENING'])
        self.show_info = (show_info if show_info is not None
                         else config.DISPLAY['SHOW_INFO_OVERLAY'])

        # 图像队列（线程安全）
        self.image_queue = queue.Queue(maxsize=3)
        self.display_active = True
        self.display_thread = None

        # 显示状态
        self.paused = False

        # 显示统计
        self.display_stats = {
            'fps': 0.0,
            'last_update': time.time(),
            'frame_count': 0
        }

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

    def stop(self):
        """停止显示线程"""
        self.display_active = False
        if self.display_thread:
            self.display_thread.join(timeout=2.0)

    def update_image(self, image_data: np.ndarray, info: Optional[Dict] = None):
        """更新要显示的图像"""
        if not self.display_active or self.paused or image_data is None:
            return

        try:
            # 图像增强处理
            if self.enable_sharpening and image_data is not None and image_data.size > 0:
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
                image_data = cv2.filter2D(image_data, -1, kernel)

            # 如果队列已满，丢弃最旧的一帧
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass

            display_packet = {
                'image': image_data.copy(),
                'info': info.copy() if info else {},
                'timestamp': time.time()
            }

            self.image_queue.put_nowait(display_packet)

        except Exception as e:
            print(f"⚠️ 更新图像时出错: {e}")

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

        print("💡 前视窗口控制:")
        print("   - 按 'Q': 关闭窗口 | 'S': 保存截图")
        print("   - 按 'P': 暂停/继续 | 'I': 切换信息显示")
        print("   - 按 'H': 切换锐化效果")

        while self.display_active:
            display_image = None
            info = {}

            try:
                # 获取最新图像
                if not self.image_queue.empty():
                    packet = self.image_queue.get_nowait()
                    display_image = packet['image']
                    info = packet['info']

                    # 更新统计
                    self._update_stats()

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

            # 键盘事件处理
            key = cv2.waitKey(config.DISPLAY.get('REFRESH_RATE_MS', 30)) & 0xFF

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
                self.enable_sharpening = not self.enable_sharpening
                status = "开启" if self.enable_sharpening else "关闭"
                print(f"🔍 图像锐化{status}")

        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)

    def _update_stats(self):
        """更新显示统计信息"""
        now = time.time()
        self.display_stats['frame_count'] += 1

        if now - self.display_stats['last_update'] >= 1.0:
            self.display_stats['fps'] = self.display_stats['frame_count'] / (now - self.display_stats['last_update'])
            self.display_stats['frame_count'] = 0
            self.display_stats['last_update'] = now

    def _add_info_overlay(self, image: np.ndarray, info: Dict) -> np.ndarray:
        """在图像上叠加状态信息"""
        if image is None or image.size == 0:
            return image

        try:
            overlay = image.copy()
            height, width = image.shape[:2]

            # 创建半透明信息栏
            info_height = 100
            cv2.rectangle(overlay, (0, 0), (width, info_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # 飞行状态
            state = info.get('state', 'UNKNOWN')
            state_color = (0, 255, 0) if '探索' in state else (0, 255, 255) if '悬停' in state else (0, 0, 255)
            cv2.putText(image, f"状态: {state}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

            # 位置信息
            pos = info.get('position', (0, 0, 0))
            cv2.putText(image, f"位置: ({pos[0]:.1f}, {pos[1]:.1f}, {-pos[2]:.1f}m)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 障碍物信息
            obs_dist = info.get('obstacle_distance', 0.0)
            obs_color = (0, 0, 255) if obs_dist < 5.0 else (0, 165, 255) if obs_dist < 10.0 else (0, 255, 0)
            cv2.putText(image, f"障碍: {obs_dist:.1f}m", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, obs_color, 2)

            # 显示统计信息
            fps_text = f"FPS: {self.display_stats['fps']:.1f}"
            cv2.putText(image, fps_text, (width - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            # 清晰度提示（低分辨率时显示）
            if height < 200:
                cv2.putText(image, "提示: 修改config.py可提高分辨率", (10, height-10),
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


# ==================== 主探索器类 ====================

class PerceptiveExplorer:
    """基于感知的自主探索无人机 - 工程优化版"""

    def __init__(self, drone_name=""):
        # 初始化日志系统
        self._setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("AirSimNH 感知驱动自主探索系统 - 工程优化版 初始化")
        self.logger.info("=" * 60)

        # 初始化AirSim连接
        self.client = None
        self.drone_name = drone_name
        self._connect_to_airsim()

        # 启用API控制
        try:
            self.client.enableApiControl(True, vehicle_name=drone_name)
            self.client.armDisarm(True, vehicle_name=drone_name)
            self.logger.info("✅ API控制已启用")
        except Exception as e:
            self.logger.error(f"❌ 启用API控制失败: {e}")
            raise

        # 状态管理
        self.state = FlightState.TAKEOFF
        self.state_history = deque(maxlen=20)
        self.emergency_flag = False

        # 从配置文件读取参数
        self.depth_threshold_near = config.PERCEPTION['DEPTH_NEAR_THRESHOLD']
        self.depth_threshold_safe = config.PERCEPTION['DEPTH_SAFE_THRESHOLD']
        self.min_ground_clearance = config.PERCEPTION['MIN_GROUND_CLEARANCE']
        self.max_pitch_angle = math.radians(config.PERCEPTION['MAX_PITCH_ANGLE_DEG'])
        self.scan_angles = config.PERCEPTION['SCAN_ANGLES']

        # 探索参数
        self.exploration_time = config.EXPLORATION['TOTAL_TIME']
        self.preferred_speed = config.EXPLORATION['PREFERRED_SPEED']
        self.max_altitude = config.EXPLORATION['MAX_ALTITUDE']
        self.min_altitude = config.EXPLORATION['MIN_ALTITUDE']
        self.base_height = config.EXPLORATION['BASE_HEIGHT']
        self.takeoff_height = config.EXPLORATION['TAKEOFF_HEIGHT']

        # 记忆系统
        self.visited_positions = deque(maxlen=100)

        # 性能监控与健康检查
        self.loop_count = 0
        self.start_time = time.time()
        self.last_health_check = 0
        self.reconnect_attempts = 0
        self.last_successful_loop = time.time()

        # 运行统计
        self.stats = {
            'perception_cycles': 0,
            'decision_cycles': 0,
            'exceptions_caught': 0,
            'obstacles_detected': 0,
            'state_changes': 0,
            'front_image_updates': 0,
        }

        # 前视窗口
        self.front_display = None
        self._setup_front_display()

        self.logger.info("✅ 系统初始化完成")
        self.logger.info(f"   开始时间: {datetime.now().strftime('%H:%M:%S')}")
        self.logger.info(f"   预计探索时长: {self.exploration_time}秒")
        self.logger.info(f"   日志级别: {config.SYSTEM['LOG_LEVEL']}")

    def _setup_logging(self):
        """配置日志系统"""
        self.logger = logging.getLogger('DroneExplorer')
        self.logger.setLevel(getattr(logging, config.SYSTEM['LOG_LEVEL']))

        # 清除已有的处理器，避免重复
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # 文件处理器（如果需要）
        if config.SYSTEM['LOG_TO_FILE']:
            try:
                file_handler = logging.FileHandler(config.SYSTEM['LOG_FILENAME'], encoding='utf-8')
                file_format = logging.Formatter('%(asctime)s | %(name)s | %(levelname)-8s | %(message)s')
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)
                self.logger.info(f"📝 日志将保存至: {config.SYSTEM['LOG_FILENAME']}")
            except Exception as e:
                print(f"⚠️ 无法创建日志文件: {e}")

    def _connect_to_airsim(self):
        """连接到AirSim，支持重试机制"""
        max_attempts = config.SYSTEM['MAX_RECONNECT_ATTEMPTS']
        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"🔄 尝试连接到AirSim (第{attempt}次)...")
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                self.logger.info("✅ 成功连接到AirSim")
                self.reconnect_attempts = 0
                return
            except ConnectionRefusedError:
                self.logger.warning(f"❌ 连接被拒绝，请确保AirSim正在运行")
            except Exception as e:
                self.logger.warning(f"❌ 连接失败: {e}")

            if attempt < max_attempts:
                self.logger.info(f"⏳ {config.SYSTEM['RECONNECT_DELAY']}秒后重试...")
                time.sleep(config.SYSTEM['RECONNECT_DELAY'])

        self.logger.error(f"❌ 经过{max_attempts}次尝试后仍无法连接到AirSim")
        self.logger.error("请检查：1. AirSim是否启动 2. 网络设置 3. 防火墙")
        sys.exit(1)

    def _setup_front_display(self):
        """初始化前视显示窗口"""
        try:
            self.front_display = FrontViewDisplay(
                window_name=f"无人机前视 - {self.drone_name or 'AirSimNH'}",
                width=config.DISPLAY['WINDOW_WIDTH'],
                height=config.DISPLAY['WINDOW_HEIGHT'],
                enable_sharpening=config.DISPLAY['ENABLE_SHARPENING'],
                show_info=config.DISPLAY['SHOW_INFO_OVERLAY']
            )
            self.logger.info("🎥 前视窗口已初始化")
        except Exception as e:
            self.logger.error(f"❌ 前视窗口初始化失败: {e}")
            self.front_display = None

    def get_depth_perception(self) -> PerceptionResult:
        """获取并分析深度图像，理解环境"""
        result = PerceptionResult()
        self.stats['perception_cycles'] += 1

        try:
            # 健康检查
            if config.SYSTEM['ENABLE_HEALTH_CHECK']:
                current_time = time.time()
                if current_time - self.last_successful_loop > 10.0:
                    self.logger.warning("⚠️ 感知循环长时间无响应，尝试恢复...")
                    self._check_connection_health()

            # 请求深度图像和前视图像
            camera_name = config.CAMERA['DEFAULT_NAME']
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    camera_name,
                    airsim.ImageType.DepthPlanar,
                    pixels_as_float=True,
                    compress=False
                ),
                airsim.ImageRequest(
                    camera_name,
                    airsim.ImageType.Scene,
                    False,
                    False
                )
            ])

            if not responses or len(responses) < 2:
                self.logger.warning("⚠️ 图像获取失败：响应为空或数量不足")
                return result

            # 处理深度图像
            depth_img = responses[0]
            if depth_img and hasattr(depth_img, 'image_data_float'):
                try:
                    depth_array = np.array(depth_img.image_data_float, dtype=np.float32)
                    depth_array = depth_array.reshape(depth_img.height, depth_img.width)

                    # 分析深度图像的不同区域
                    h, w = depth_array.shape

                    # 前方近距离区域（紧急避障）
                    front_near = depth_array[h // 2:, w // 3:2 * w // 3]
                    min_front_distance = np.min(front_near) if front_near.size > 0 else 100

                    # 多方向扇形扫描
                    directions = []
                    for angle_deg in self.scan_angles:
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
                    if result.has_obstacle:
                        self.stats['obstacles_detected'] += 1

                    if directions:
                        closest_dir = min(directions, key=lambda x: x[1])
                        result.obstacle_direction = closest_dir[0]

                    # 根据感知动态调整推荐高度
                    if result.terrain_slope > config.PERCEPTION['HEIGHT_STRATEGY']['SLOPE_THRESHOLD']:
                        result.recommended_height = config.PERCEPTION['HEIGHT_STRATEGY']['STEEP_SLOPE']
                    elif result.open_space_score > config.PERCEPTION['HEIGHT_STRATEGY']['OPENNESS_THRESHOLD']:
                        result.recommended_height = config.PERCEPTION['HEIGHT_STRATEGY']['OPEN_SPACE']

                except ValueError as e:
                    self.logger.error(f"❌ 深度图像数据转换错误: {e}")
                    return result
                except Exception as e:
                    self.logger.error(f"❌ 深度图像处理异常: {e}")
                    return result

            # 处理前视图像
            front_response = responses[1]
            if front_response and hasattr(front_response, 'image_data_uint8'):
                try:
                    # 转换为OpenCV格式
                    img_array = np.frombuffer(front_response.image_data_uint8, dtype=np.uint8)

                    if len(img_array) > 0:
                        img_rgb = img_array.reshape(front_response.height, front_response.width, 3)
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        result.front_image = img_bgr

                        # 准备显示信息
                        display_info = self._prepare_display_info(result)

                        # 更新前视窗口
                        if self.front_display:
                            self.front_display.update_image(img_bgr, display_info)
                            self.stats['front_image_updates'] += 1

                except Exception as e:
                    self.logger.warning(f"⚠️ 前视图像处理异常: {e}")

            self.last_successful_loop = time.time()

            # 详细日志
            if self.loop_count % 50 == 0:
                self.logger.debug(f"感知结果: 障碍={result.has_obstacle}, 距离={result.obstacle_distance:.1f}m, "
                                f"开阔度={result.open_space_score:.2f}")

        except airsim.ClientException as e:
            self.logger.error(f"❌ AirSim客户端异常: {e}")
            self.stats['exceptions_caught'] += 1
            self._handle_airsim_exception(e)
        except ConnectionError as e:
            self.logger.error(f"❌ 网络连接异常: {e}")
            self.stats['exceptions_caught'] += 1
            self._handle_connection_error()
        except Exception as e:
            self.logger.error(f"❌ 感知过程中发生未知异常: {e}")
            self.logger.debug(f"异常详情: {traceback.format_exc()}")
            self.stats['exceptions_caught'] += 1

        return result

    def _prepare_display_info(self, perception: PerceptionResult) -> Dict:
        """准备显示信息"""
        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            return {
                'state': self.state.value,
                'obstacle_distance': perception.obstacle_distance,
                'position': (pos.x_val, pos.y_val, pos.z_val),
                'loop_count': self.loop_count,
            }
        except:
            return {}

    def _check_connection_health(self):
        """检查连接健康状态"""
        try:
            # 简单的ping测试
            self.client.ping()
            self.logger.info("✅ 连接健康检查通过")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ 连接健康检查失败: {e}")
            if self.reconnect_attempts < config.SYSTEM['MAX_RECONNECT_ATTEMPTS']:
                self.logger.info("尝试重新连接...")
                self._connect_to_airsim()
                self.reconnect_attempts += 1
            return False

    def _handle_airsim_exception(self, exception):
        """处理AirSim特定异常"""
        error_msg = str(exception)
        if "timeout" in error_msg.lower():
            self.logger.warning("⏰ AirSim响应超时，等待后继续...")
            time.sleep(1.0)
        elif "not connected" in error_msg.lower():
            self.logger.error("🔌 与AirSim连接断开")
            self._handle_connection_error()

    def _handle_connection_error(self):
        """处理连接错误"""
        if self.emergency_flag:
            return

        self.logger.warning("🔄 尝试恢复连接...")
        success = self._check_connection_health()
        if not success and not self.emergency_flag:
            self.logger.error("❌ 无法恢复连接，启动紧急降落程序")
            self.emergency_stop()

    def get_visual_perception(self):
        """获取视觉图像用于高级感知（可选）"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            if responses and responses[0]:
                img_data = responses[0].image_data_uint8
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = img_array.reshape(responses[0].height, responses[0].width, 3)

                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                green_mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
                green_ratio = np.sum(green_mask > 0) / green_mask.size

                return img, green_ratio
        except:
            pass

        return None, 0

    def make_intelligent_decision(self, perception: PerceptionResult) -> Tuple[float, float, float, float]:
        """基于感知结果做出智能决策"""
        self.stats['decision_cycles'] += 1

        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity

            target_vx, target_vy, target_z, target_yaw = 0.0, 0.0, perception.recommended_height, 0.0

            if self.state == FlightState.TAKEOFF:
                target_z = self.takeoff_height
                if pos.z_val < self.takeoff_height + 0.5:
                    self.change_state(FlightState.HOVERING)

            elif self.state == FlightState.HOVERING:
                target_yaw = (time.time() % 10) * 0.2

                if len(perception.safe_directions) > 0:
                    self.change_state(FlightState.EXPLORING)

            elif self.state == FlightState.EXPLORING:
                if perception.has_obstacle:
                    self.change_state(FlightState.AVOIDING)
                    target_vx, target_vy = -vel.x_val, -vel.y_val
                else:
                    if perception.safe_directions:
                        current_yaw = airsim.to_eularian_angles(
                            state.kinematics_estimated.orientation
                        )[2]

                        diverse_dirs = [
                            d for d in perception.safe_directions
                            if abs(d - current_yaw) > math.radians(45)
                        ]

                        if diverse_dirs:
                            best_dir = diverse_dirs[0]
                        else:
                            best_dir = perception.safe_directions[0]

                        speed_factor = min(1.0, perception.open_space_score * 1.5)
                        target_vx = self.preferred_speed * speed_factor * math.cos(best_dir)
                        target_vy = self.preferred_speed * speed_factor * math.sin(best_dir)
                    else:
                        target_z = pos.z_val - 5
                        self.change_state(FlightState.AVOIDING)

            elif self.state == FlightState.AVOIDING:
                if perception.has_obstacle:
                    if abs(perception.obstacle_direction) < math.radians(30):
                        target_z = pos.z_val - 3
                        target_vx, target_vy = 0, 0
                    else:
                        avoid_dir = perception.obstacle_direction + math.pi
                        target_vx = 1.5 * math.cos(avoid_dir)
                        target_vy = 1.5 * math.sin(avoid_dir)
                else:
                    self.change_state(FlightState.HOVERING)
                    time.sleep(1)

            elif self.state == FlightState.EMERGENCY:
                target_vx, target_vy, target_yaw = 0, 0, 0
                target_z = max(pos.z_val, -20)

            # 高度安全限制
            target_z = max(self.max_altitude, min(self.min_altitude, target_z))

            return target_vx, target_vy, target_z, target_yaw

        except Exception as e:
            self.logger.error(f"❌ 决策过程异常: {e}")
            return 0.0, 0.0, self.base_height, 0.0

    def change_state(self, new_state: FlightState):
        """状态转换"""
        if self.state != new_state:
            self.logger.info(f"🔄 状态转换: {self.state.value} → {new_state.value}")
            self.state = new_state
            self.state_history.append((time.time(), new_state))
            self.stats['state_changes'] += 1

    def _execute_control_decision(self, decision):
        """执行控制决策，增强异常处理"""
        try:
            target_vx, target_vy, target_z, target_yaw = decision

            if self.state in [FlightState.EXPLORING, FlightState.AVOIDING]:
                self.client.moveByVelocityZAsync(
                    target_vx, target_vy, target_z, 0.5,
                    vehicle_name=self.drone_name
                )
            else:
                self.client.moveToPositionAsync(
                    0, 0, target_z, 2,
                    vehicle_name=self.drone_name
                )

            # 记录位置
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            self.visited_positions.append((pos.x_val, pos.y_val, pos.z_val))

        except Exception as e:
            self.logger.warning(f"⚠️ 控制指令执行失败: {e}")
            try:
                self.client.hoverAsync(vehicle_name=self.drone_name).join()
            except:
                pass

    def _report_status(self, exploration_start, perception):
        """报告系统状态"""
        elapsed = time.time() - exploration_start
        try:
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position

            self.logger.info(f"\n📊 系统状态报告 [循环#{self.loop_count}]")
            self.logger.info(f"   运行时间: {elapsed:.1f}s / {self.exploration_time}s")
            self.logger.info(f"   飞行状态: {self.state.value}")
            self.logger.info(f"   当前位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}m)")
            self.logger.info(f"   环境感知: 障碍{'有' if perception.has_obstacle else '无'} "
                            f"| 距离={perception.obstacle_distance:.1f}m "
                            f"| 开阔度={perception.open_space_score:.2f}")
            self.logger.info(f"   系统统计: 异常{self.stats['exceptions_caught']}次 "
                            f"| 状态切换{self.stats['state_changes']}次")
        except:
            self.logger.info("状态报告: 无法获取无人机状态")

    def run_perception_loop(self):
        """主感知-决策-控制循环"""
        self.logger.info("=" * 60)
        self.logger.info("启动感知-决策-控制主循环")
        self.logger.info("=" * 60)

        try:
            # 起飞
            self.logger.info("🚀 起飞中...")
            self.client.takeoffAsync(vehicle_name=self.drone_name).join()
            time.sleep(2)

            # 上升到目标高度
            self.client.moveToZAsync(self.takeoff_height, 3, vehicle_name=self.drone_name).join()
            self.change_state(FlightState.HOVERING)
            time.sleep(2)

            # 主循环
            exploration_start = time.time()

            while (time.time() - exploration_start < self.exploration_time and
                   not self.emergency_flag):

                self.loop_count += 1
                loop_start = time.time()

                # 1. 感知阶段
                perception = self.get_depth_perception()

                # 2. 决策阶段
                decision = self.make_intelligent_decision(perception)

                # 3. 控制执行阶段
                self._execute_control_decision(decision)

                # 定期状态报告
                if self.loop_count % config.SYSTEM.get('HEALTH_CHECK_INTERVAL', 20) == 0:
                    self._report_status(exploration_start, perception)

                # 循环频率控制
                loop_time = time.time() - loop_start
                if loop_time < 0.1:
                    time.sleep(0.1 - loop_time)

            # 正常结束
            self.logger.info("⏰ 探索时间到，开始返航")
            self._finish_mission()

        except KeyboardInterrupt:
            self.logger.warning("⏹️ 用户中断探索")
            self.emergency_stop()
        except Exception as e:
            self.logger.error(f"❌ 主循环发生异常: {e}")
            self.logger.debug(f"异常堆栈: {traceback.format_exc()}")
            self.emergency_stop()

    def _finish_mission(self):
        """完成任务并生成总结报告"""
        self.logger.info("=" * 60)
        self.logger.info("探索任务完成，开始返航程序")
        self.logger.info("=" * 60)

        self.change_state(FlightState.RETURNING)

        try:
            # 返航
            self.logger.info("↩️ 返回起始区域...")
            self.client.moveToPositionAsync(0, 0, -10, 4, vehicle_name=self.drone_name).join()
            time.sleep(2)

            # 降落
            self.logger.info("🛬 降落中...")
            self.change_state(FlightState.LANDING)
            self.client.landAsync(vehicle_name=self.drone_name).join()
            time.sleep(3)

        except Exception as e:
            self.logger.error(f"❌ 降落过程中出现异常: {e}")

        finally:
            # 无论成功与否，都执行清理
            self._cleanup_system()

            # 生成最终报告
            self._generate_summary_report()

    def _cleanup_system(self):
        """清理系统资源"""
        self.logger.info("🧹 清理系统资源...")

        try:
            self.client.armDisarm(False, vehicle_name=self.drone_name)
            self.client.enableApiControl(False, vehicle_name=self.drone_name)
            self.logger.info("✅ 无人机控制已安全释放")
        except:
            self.logger.warning("⚠️ 释放控制时出现异常")

        # 关闭前视窗口
        if self.front_display:
            self.front_display.stop()
            self.logger.info("✅ 前视窗口已关闭")

    def _generate_summary_report(self):
        """生成运行总结报告"""
        total_time = time.time() - self.start_time

        self.logger.info("\n" + "=" * 60)
        self.logger.info("🏁 任务总结报告")
        self.logger.info("=" * 60)
        self.logger.info(f"   总运行时间: {total_time:.1f}秒")
        self.logger.info(f"   总循环次数: {self.loop_count}")
        if total_time > 0:
            self.logger.info(f"   平均循环频率: {self.loop_count/total_time:.1f} Hz")
        self.logger.info(f"   探索航点数量: {len(self.visited_positions)}")
        self.logger.info(f"   状态切换次数: {self.stats['state_changes']}")
        self.logger.info(f"   检测到障碍次数: {self.stats['obstacles_detected']}")
        self.logger.info(f"   前视图像更新次数: {self.stats['front_image_updates']}")
        self.logger.info(f"   捕获的异常数: {self.stats['exceptions_caught']}")
        self.logger.info(f"   重连尝试次数: {self.reconnect_attempts}")

        # 保存报告到文件
        try:
            report_filename = f"mission_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("AirSimNH 无人机任务报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总运行时间: {total_time:.1f}秒\n")
                f.write(f"总循环次数: {self.loop_count}\n")
                f.write(f"探索航点数量: {len(self.visited_positions)}\n")
                f.write(f"状态切换次数: {self.stats['state_changes']}\n")
                f.write(f"异常捕获次数: {self.stats['exceptions_caught']}\n")
                f.write(f"前视图像更新次数: {self.stats['front_image_updates']}\n")
                f.write("=" * 50 + "\n")
                f.write("飞行航点记录:\n")
                for i, pos in enumerate(self.visited_positions[:20]):  # 只记录前20个航点
                    f.write(f"  航点{i+1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n")
                if len(self.visited_positions) > 20:
                    f.write(f"  ... 还有{len(self.visited_positions)-20}个航点\n")
            self.logger.info(f"📄 详细报告已保存至: {report_filename}")
        except Exception as e:
            self.logger.warning(f"⚠️ 无法保存报告文件: {e}")

    def emergency_stop(self):
        """紧急停止"""
        if self.emergency_flag:
            return

        self.logger.error("\n🆘 紧急停止程序启动!")
        self.emergency_flag = True
        self.change_state(FlightState.EMERGENCY)

        try:
            self.client.hoverAsync(vehicle_name=self.drone_name).join()
            time.sleep(1)
            self.client.landAsync(vehicle_name=self.drone_name).join()
            time.sleep(2)
            self.logger.info("✅ 紧急降落指令已发送")
        except Exception as e:
            self.logger.error(f"⚠️ 紧急降落异常: {e}")

        # 关闭前视窗口
        if self.front_display:
            self.front_display.stop()

        self._cleanup_system()


# ==================== 主程序入口 ====================

def main():
    """主程序入口"""
    # 显示启动信息
    print("=" * 70)
    print("AirSimNH 无人机感知探索系统 - 工程优化版")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置状态: {'已加载' if CONFIG_LOADED else '使用默认配置'}")
    print(f"日志级别: {config.SYSTEM['LOG_LEVEL']}")
    print(f"探索时间: {config.EXPLORATION['TOTAL_TIME']}秒")
    print("=" * 70)

    # 用户确认
    response = input("按 Enter 键开始运行，或输入 'q' 退出: ")
    if response.lower() == 'q':
        print("程序退出")
        return

    explorer = None
    try:
        # 创建感知探索器
        explorer = PerceptiveExplorer(drone_name="")

        # 设置键盘中断处理
        def signal_handler(sig, frame):
            print("\n⚠️ 用户中断，正在安全停止...")
            if explorer:
                explorer.emergency_stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # 运行主循环
        explorer.run_perception_loop()

    except Exception as e:
        print(f"\n❌ 程序启动异常: {e}")
        traceback.print_exc()

        # 尝试安全降落
        try:
            if explorer and explorer.client:
                explorer.client.landAsync().join()
                explorer.client.armDisarm(False)
                explorer.client.enableApiControl(False)
        except:
            pass


if __name__ == "__main__":
    main()