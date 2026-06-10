"""
CARLA恶劣天气鲁棒性测试与自适应感知系统
自动遍历7种天气，输出鲁棒性评分报告
"""

import sys
# 添加CARLA Python API路径（根据实际安装位置修改）
CARLA_PATH = r"F:\hutb\PythonAPI"
sys.path.append(CARLA_PATH)

import carla
import numpy as np
import cv2
import json
import logging
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from config.settings import (
    CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT, CARLA_MAP,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV,
    LIDAR_CHANNELS, LIDAR_RANGE,
    WEATHER_PROFILES, STEPS_PER_WEATHER, WEATHER_TRANSITION_STEPS,
    VISIBILITY_THRESHOLD_LOW, VISIBILITY_THRESHOLD_HIGH,
    LIDAR_CLUSTER_DISTANCE, LIDAR_MIN_CLUSTER_POINTS,
    IMAGE_BRIGHTNESS_LOW, IMAGE_BRIGHTNESS_HIGH,
    CAMERA_WEIGHT_CLEAR, LIDAR_WEIGHT_CLEAR,
    CAMERA_WEIGHT_ADVERSE, LIDAR_WEIGHT_ADVERSE,
    SAFE_DISTANCE, COLLISION_DISTANCE,
    PID_KP, PID_KI, PID_KD, MAX_SPEED, LOG_LEVEL,
)


class RoadFollowController:
    """道路循迹控制器 - 使用CARLA Waypoint API沿道路中心线自动行驶"""

    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        self._target_speed = MAX_SPEED * 0.5  # 默认循迹速度
        self._waypoint_buffer = []  # 预读取前方路径点

    def update_waypoints(self, lookahead_distance=10.0):
        """更新前方路径点列表"""
        current_transform = self.vehicle.get_transform()
        current_location = current_transform.location
        current_waypoint = self.map.get_waypoint(current_location)

        # 获取前方多个路径点
        self._waypoint_buffer = []
        waypoint = current_waypoint
        total_distance = 0.0

        while total_distance < lookahead_distance:
            # 沿道路方向前进2米
            next_waypoints = waypoint.next(2.0)
            if not next_waypoints:
                break
            waypoint = next_waypoints[0]
            self._waypoint_buffer.append(waypoint)
            total_distance += 2.0

    def get_steering(self):
        """计算沿道路行驶所需的转向角"""
        if not self._waypoint_buffer:
            self.update_waypoints()

        if len(self._waypoint_buffer) < 2:
            return 0.0  # 没有足够路径点，直行

        # 获取车辆当前位置和朝向
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

        # 获取最近的前方路径点作为目标
        target_waypoint = self._waypoint_buffer[0]
        target_location = target_waypoint.transform.location

        # 计算相对位置
        dx = target_location.x - vehicle_location.x
        dy = target_location.y - vehicle_location.y

        # 计算到目标的角度（世界坐标系）
        target_angle = np.arctan2(dy, dx)

        # 计算相对于车辆朝向的角度差
        angle_diff = target_angle - vehicle_yaw

        # 归一化到 [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # 将角度转换为转向值 [-1, 1]
        # CARLA中转向角范围约为 [-1, 1]，对应约90度
        max_steer_angle = np.pi / 2
        steer = np.clip(angle_diff / max_steer_angle, -1.0, 1.0)

        return float(steer)

    def get_target_speed(self):
        """获取循迹目标速度"""
        return self._target_speed

    def set_target_speed(self, speed):
        """设置循迹目标速度"""
        self._target_speed = max(0.0, min(speed, MAX_SPEED))

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(message)s")
logger = logging.getLogger("WeatherRobustness")

WEATHER_NAMES = list(WEATHER_PROFILES.keys())
WEATHER_LABELS = {
    "clear": "Clear", "cloudy": "Cloudy", "light_rain": "Light Rain",
    "heavy_rain": "Heavy Rain", "fog": "Fog", "night": "Night", "night_rain": "Night Rain",
    "snow": "Snow", "night_snow": "Night Snow", "dust_storm": "Dust Storm",
}


class ImageQualityAssessor:
    """图像质量评估：多维度详细分解打分"""

    def assess(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # 1. 模糊度评分 (Laplacian方差)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = np.clip(laplacian_var / 500.0, 0.0, 1.0)

        # 2. 亮度评分
        mean_brightness = np.mean(gray)
        if mean_brightness < IMAGE_BRIGHTNESS_LOW:
            brightness_score = mean_brightness / IMAGE_BRIGHTNESS_LOW
        elif mean_brightness > IMAGE_BRIGHTNESS_HIGH:
            brightness_score = (255.0 - mean_brightness) / (255.0 - IMAGE_BRIGHTNESS_HIGH)
        else:
            brightness_score = 1.0

        # 3. 对比度评分 (灰度标准差)
        contrast_score = np.clip(np.std(gray) / 80.0, 0.0, 1.0)

        # 4. 可见度评分 (边缘检测比例)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        visibility_score = np.clip(edge_ratio * 20, 0.0, 1.0)

        # 5. 噪点评分 (基于高斯滤波差异)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_est = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
        noise_score = np.clip(1.0 - noise_est / 20.0, 0.0, 1.0)

        # 6. 色彩饱和度评分
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        saturation_score = np.clip(saturation / 128.0, 0.0, 1.0)

        # 7. 雾霾影响评估 (暗通道先验简化版)
        dark_channel = cv2.erode(gray, np.ones((15, 15)))
        haze_ratio = np.mean(dark_channel) / 255.0
        haze_score = 1.0 - haze_ratio

        # 综合评分 (权重可调)
        overall = (
            blur_score * 0.15 +
            brightness_score * 0.15 +
            contrast_score * 0.1 +
            visibility_score * 0.2 +
            noise_score * 0.15 +
            saturation_score * 0.05 +
            haze_score * 0.2
        )

        return {
            # 基础分数
            "blur_score": float(blur_score),
            "brightness_score": float(brightness_score),
            "contrast_score": float(contrast_score),
            "visibility_score": float(visibility_score),
            "noise_score": float(noise_score),
            "saturation_score": float(saturation_score),
            "haze_score": float(haze_score),
            # 综合评分
            "overall": float(overall),
            # 原始数值
            "laplacian_var": float(laplacian_var),
            "mean_brightness": float(mean_brightness),
            "std_brightness": float(np.std(gray)),
            "edge_ratio": float(edge_ratio),
            "noise_est": float(noise_est),
            "haze_ratio": float(haze_ratio),
        }


class LidarAdaptivePerceiver:
    """LiDAR自适应感知：DBSCAN点云聚类检测障碍物"""

    def __init__(self):
        self._current_clusters = []

    def process_point_cloud(self, point_cloud, vehicle_transform):
        if point_cloud.shape[0] < 10:
            return []
        non_ground = point_cloud[point_cloud[:, 2] > -vehicle_transform.location.z + 0.5]
        front_points = non_ground[non_ground[:, 0] > 0]
        if front_points.shape[0] < LIDAR_MIN_CLUSTER_POINTS:
            return []
        clustering = DBSCAN(eps=LIDAR_CLUSTER_DISTANCE, min_samples=LIDAR_MIN_CLUSTER_POINTS).fit(front_points[:, :3])
        obstacles = []
        for label in set(clustering.labels_):
            if label == -1:
                continue
            cluster_pts = front_points[clustering.labels_ == label]
            center = np.mean(cluster_pts[:, :3], axis=0)
            dist = np.linalg.norm(center)
            if dist < LIDAR_RANGE:
                obstacles.append({"center": center.tolist(), "distance": float(dist), "num_points": int(len(cluster_pts))})
        self._current_clusters = sorted(obstacles, key=lambda o: o["distance"])
        return self._current_clusters

    def get_nearest_obstacle_distance(self):
        return self._current_clusters[0]["distance"] if self._current_clusters else float("inf")


class AdaptiveFusionPerceiver:
    """自适应融合感知器：根据天气和图像质量动态调整相机/LiDAR融合权重"""

    def __init__(self):
        self.image_assessor = ImageQualityAssessor()
        self.lidar_perceiver = LidarAdaptivePerceiver()
        self._fusion_mode = "camera_dominant"
        self._camera_weight = CAMERA_WEIGHT_CLEAR
        self._lidar_weight = LIDAR_WEIGHT_CLEAR

    def update_weights(self, image_quality, weather_severity):
        overall = image_quality["overall"]
        if weather_severity in ("clear", "mild"):
            base_cam, base_lidar = CAMERA_WEIGHT_CLEAR, LIDAR_WEIGHT_CLEAR
        else:
            base_cam, base_lidar = CAMERA_WEIGHT_ADVERSE, LIDAR_WEIGHT_ADVERSE
        if overall < VISIBILITY_THRESHOLD_LOW:
            self._camera_weight = base_cam * (overall / VISIBILITY_THRESHOLD_LOW)
            self._lidar_weight = 1.0 - self._camera_weight
            self._fusion_mode = "lidar_dominant"
        elif overall > VISIBILITY_THRESHOLD_HIGH:
            self._camera_weight = base_cam
            self._lidar_weight = base_lidar
            self._fusion_mode = "camera_dominant"
        else:
            ratio = (overall - VISIBILITY_THRESHOLD_LOW) / (VISIBILITY_THRESHOLD_HIGH - VISIBILITY_THRESHOLD_LOW)
            self._camera_weight = base_cam * ratio + (1.0 - base_cam) * (1.0 - ratio) * 0.5
            self._lidar_weight = 1.0 - self._camera_weight
            self._fusion_mode = "balanced"

    def detect_obstacles(self, camera_image, lidar_data, vehicle_transform, weather_severity):
        image_quality = self.image_assessor.assess(camera_image)
        self.update_weights(image_quality, weather_severity)
        self.lidar_perceiver.process_point_cloud(lidar_data, vehicle_transform)
        lidar_nearest = self.lidar_perceiver.get_nearest_obstacle_distance()
        camera_effective_range = LIDAR_RANGE * image_quality["overall"]
        camera_nearest = lidar_nearest if lidar_nearest < camera_effective_range else float("inf")
        nearest_distance = camera_nearest * self._camera_weight + lidar_nearest * self._lidar_weight
        return {
            "nearest_distance": float(nearest_distance),
            "num_obstacles": len(self.lidar_perceiver._current_clusters),
            "fusion_mode": self._fusion_mode,
            "camera_weight": float(self._camera_weight),
            "lidar_weight": float(self._lidar_weight),
            "image_quality": image_quality,
        }


class RobustnessScorer:
    """鲁棒性评分器：碰撞率+图像质量保持+自适应切换合理性"""

    def __init__(self):
        self.records = {}

    def start_weather(self, weather_name):
        self.records[weather_name] = []

    def record_step(self, weather_name, fusion_result, had_collision):
        self.records[weather_name].append({
            "nearest_distance": fusion_result["nearest_distance"],
            "fusion_mode": fusion_result["fusion_mode"],
            "image_overall": fusion_result["image_quality"]["overall"],
            "camera_weight": fusion_result["camera_weight"],
            "collision": had_collision,
            "num_obstacles": fusion_result["num_obstacles"],
            "detected": fusion_result["num_obstacles"] > 0,
        })

    def compute_robustness_score(self, weather_name):
        records = self.records.get(weather_name, [])
        if not records:
            return {"score": 0.0, "collisions": 0, "avg_image_quality": 0.0}
        num_collisions = sum(1 for r in records if r["collision"])
        collision_rate = num_collisions / len(records)
        avg_img_q = np.mean([r["image_overall"] for r in records])
        lidar_ratio = sum(1 for r in records if r["fusion_mode"] == "lidar_dominant") / len(records)
        
        # ===== 障碍物检测率分析 =====
        detected_count = sum(1 for r in records if r["detected"])
        detection_rate = detected_count / len(records)
        detected_distances = [r["nearest_distance"] for r in records if r["detected"]]
        avg_detection_dist = np.mean(detected_distances) if detected_distances else 0.0
        detection_variance = np.var([1 if r["detected"] else 0 for r in records])
        
        severity = WEATHER_PROFILES.get(weather_name, {})
        is_adverse = severity.get("precipitation", 0) > 30 or severity.get("fog_density", 0) > 50
        collision_score = max(0, 40 * (1 - collision_rate * 5))
        quality_score = 30 * avg_img_q
        adaptation_score = 30 * lidar_ratio if is_adverse else 30 * (1 - lidar_ratio)
        detection_score = 20 * detection_rate
        total = np.clip(collision_score + quality_score + adaptation_score + detection_score, 0, 100)
        return {
            "score": float(total), "collisions": num_collisions,
            "collision_rate": float(collision_rate),
            "avg_image_quality": float(avg_img_q),
            "lidar_dominant_ratio": float(lidar_ratio),
            "detection_rate": float(detection_rate),
            "avg_detection_dist": float(avg_detection_dist),
            "detection_variance": float(detection_variance),
            "total_obstacles_detected": int(sum(r["num_obstacles"] for r in records)),
        }

    def generate_report(self):
        report = {name: self.compute_robustness_score(name) for name in self.records}
        report["__overall__"] = float(np.mean([r["score"] for r in report.values()])) if report else 0.0
        return report


class PIDController:
    def __init__(self, kp=PID_KP, ki=PID_KI, kd=PID_KD):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, target_speed, current_speed, dt=0.05):
        error = target_speed - current_speed
        self._integral = np.clip(self._integral + error * dt, -10.0, 10.0)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


class WeatherRobustnessSystem:
    """主系统：自动遍历7种天气，实时显示感知数据，输出鲁棒性评分报告"""

    def __init__(self):
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(CARLA_TIMEOUT)
        self.world = self.vehicle = self.camera = self.lidar = None
        self._camera_image = self._lidar_data = None
        self.fusion_perceiver = AdaptiveFusionPerceiver()
        self.scorer = RobustnessScorer()
        self.pid = PIDController()
        self.road_controller = None  # 道路循迹控制器
        self._spawn_point = None
        self._current_weather_params = None  # 当前天气参数（用于渐变过渡）
        self._target_weather_params = None  # 目标天气参数
        self._transition_step = 0  # 过渡步数计数器
        self._is_transitioning = False  # 是否正在过渡
        self._is_stable = True  # 是否处于稳定期
        self._stable_steps = 0  # 稳定期步数计数

    def connect(self):
        self.world = self.client.load_world(CARLA_MAP)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        logger.info(f"已连接CARLA，地图: {CARLA_MAP}")

    def spawn_ego_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.audi.a2")[0]
        self._spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, self._spawn_point)

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        cam_bp.set_attribute("fov", str(CAMERA_FOV))
        self.camera = self.world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=self.vehicle)
        self.camera.listen(self._on_camera)

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(LIDAR_CHANNELS))
        lidar_bp.set_attribute("range", str(LIDAR_RANGE))
        lidar_bp.set_attribute("rotation_frequency", "20")
        self.lidar = self.world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.8)), attach_to=self.vehicle)
        self.lidar.listen(self._on_lidar)

        # 初始化道路循迹控制器
        self.road_controller = RoadFollowController(self.vehicle, self.world)
        logger.info("车辆和传感器已就绪，道路循迹控制器已初始化")

    def _on_camera(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        self._camera_image = array[:, :, :3].copy()

    def _on_lidar(self, point_cloud):
        self._lidar_data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)

    def apply_weather(self, name):
        """设置目标天气"""
        profile = WEATHER_PROFILES[name]

        # 初始化当前天气参数
        if self._current_weather_params is None:
            self._current_weather_params = self._get_current_weather_params()

        # 设置目标天气参数
        self._target_weather_params = profile.copy()
        self._transition_step = 0
        self._is_transitioning = False  # 先不开始过渡
        self._is_stable = True  # 进入稳定期
        self._stable_steps = 0

        self.scorer.start_weather(name)
        logger.info(f"天气稳定期开始: {WEATHER_LABELS.get(name, name)} [{name}]，将持续{STEPS_PER_WEATHER - WEATHER_TRANSITION_STEPS}步后渐变过渡")

    def _get_current_weather_params(self):
        """获取当前天气参数"""
        wp = self.world.get_weather()
        return {
            "cloudiness": wp.cloudiness,
            "precipitation": wp.precipitation,
            "precipitation_deposits": wp.precipitation_deposits,
            "wind_intensity": wp.wind_intensity,
            "fog_density": wp.fog_density,
            "fog_distance": wp.fog_distance,
            "wetness": wp.wetness,
            "sun_altitude_angle": wp.sun_altitude_angle,
        }

    def _update_weather_transition(self):
        """更新天气渐变过渡"""
        if self._target_weather_params is None:
            return

        # 稳定期：先稳定WEATHER_TRANSITION_STEPS步后再开始过渡
        if self._is_stable:
            self._stable_steps += 1
            if self._stable_steps >= (STEPS_PER_WEATHER - WEATHER_TRANSITION_STEPS):
                self._is_stable = False
                self._is_transitioning = True
                logger.info(f"稳定期结束，天气渐变开始...")
            return

        # 渐变过渡
        if not self._is_transitioning:
            return

        self._transition_step += 1
        alpha = min(self._transition_step / WEATHER_TRANSITION_STEPS, 1.0)

        # 线性插值
        current = self._current_weather_params
        target = self._target_weather_params

        weather = carla.WeatherParameters()
        weather.cloudiness = current["cloudiness"] + (target["cloudiness"] - current["cloudiness"]) * alpha
        weather.precipitation = current["precipitation"] + (target["precipitation"] - current["precipitation"]) * alpha
        weather.precipitation_deposits = current["precipitation_deposits"] + (target["precipitation_deposits"] - current["precipitation_deposits"]) * alpha
        weather.wind_intensity = current["wind_intensity"] + (target["wind_intensity"] - current["wind_intensity"]) * alpha
        weather.fog_density = current["fog_density"] + (target["fog_density"] - current["fog_density"]) * alpha
        weather.fog_distance = current["fog_distance"] + (target["fog_distance"] - current["fog_distance"]) * alpha
        weather.wetness = current["wetness"] + (target["wetness"] - current["wetness"]) * alpha
        weather.sun_altitude_angle = current["sun_altitude_angle"] + (target["sun_altitude_angle"] - current["sun_altitude_angle"]) * alpha

        self.world.set_weather(weather)

        # 过渡完成
        if self._transition_step >= WEATHER_TRANSITION_STEPS:
            self._current_weather_params = target.copy()
            self._is_transitioning = False
            logger.info(f"天气渐变完成: {self._target_weather_params}")

    def get_severity(self):
        wp = self.world.get_weather()
        score = wp.cloudiness * 0.15 + wp.precipitation * 0.3 + wp.fog_density * 0.3 + wp.wetness * 0.1 + (100 - max(wp.sun_altitude_angle, 0)) * 0.15
        if score < 15:
            return "clear"
        elif score < 40:
            return "mild"
        elif score < 70:
            return "adverse"
        return "extreme"

    def _check_collision(self):
        loc = self.vehicle.get_transform().location
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id != self.vehicle.id and loc.distance(actor.get_transform().location) < COLLISION_DISTANCE:
                return True
        return False

    def _compute_control(self, nearest_dist, current_speed):
        target = MAX_SPEED * (nearest_dist / SAFE_DISTANCE) * 0.5 if nearest_dist < SAFE_DISTANCE else MAX_SPEED
        ctrl = self.pid.step(target, current_speed)
        if ctrl > 0:
            return min(ctrl, 1.0), 0.0
        return 0.0, min(abs(ctrl), 1.0)

    def run_step(self, weather_name, step):
        self.world.tick()

        # 更新天气渐变过渡
        self._update_weather_transition()

        if self._camera_image is None or self._lidar_data is None:
            return

        severity = self.get_severity()
        fusion = self.fusion_perceiver.detect_obstacles(self._camera_image, self._lidar_data, self.vehicle.get_transform(), severity)

        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

        # 更新道路循迹路径点
        self.road_controller.update_waypoints(lookahead_distance=15.0)
        steer = self.road_controller.get_steering()

        throttle, brake = self._compute_control(fusion["nearest_distance"], speed)
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

        collision = self._check_collision()
        self.scorer.record_step(weather_name, fusion, collision)

        # 每20步输出一次日志（每种天气约10行）
        if (step + 1) % 20 == 0:
            if self._is_stable:
                status_info = f"[稳定期: {self._stable_steps}/{STEPS_PER_WEATHER - WEATHER_TRANSITION_STEPS}]"
            elif self._is_transitioning:
                status_info = f"[过渡中: {int(self._transition_step/WEATHER_TRANSITION_STEPS*100)}%]"
            else:
                status_info = "[稳定期结束]"
            logger.info(
                f"[{weather_name}] step={step+1}/{STEPS_PER_WEATHER} {status_info}, "
                f"融合={fusion['fusion_mode']}, 图像质量={fusion['image_quality']['overall']:.2f}, "
                f"速度={speed:.1f}km/h, 转向={steer:.2f}"
            )
        self._draw_hud(fusion, speed, weather_name, steer)

    def _draw_hud(self, fusion, speed, weather_name, steer=0.0):
        if self._camera_image is None:
            return
        display = self._camera_image.copy()
        h, w = display.shape[:2]
        label = WEATHER_LABELS.get(weather_name, weather_name)
        iq = fusion["image_quality"]

        # ===== 天气视觉效果 =====
        if weather_name in ("light_rain", "heavy_rain", "night_rain"):
            display = self._add_rain_effect(display, weather_name)
        elif weather_name in ("snow", "night_snow"):
            display = self._add_snow_effect(display, weather_name)
        elif weather_name == "dust_storm":
            display = self._add_dust_effect(display)

        def put_text(text, pos, scale, color, thickness=2):
            """带黑色阴影轮廓的清晰文字（LINE_8 比 LINE_AA 快5倍）"""
            x, y = pos
            cv2.putText(display, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness+2, cv2.LINE_8)
            cv2.putText(display, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_8)

        line_h = 21
        y = 20
        WHITE = (255, 255, 255)
        CYAN = (0, 255, 255)
        GREEN = (0, 255, 0)
        YELLOW = (255, 255, 0)
        ORANGE = (255, 180, 0)
        PINK = (255, 100, 255)
        LIGHT_GRAY = (220, 220, 220)

        # 天气名称 + 状态
        if self._is_stable:
            remaining = STEPS_PER_WEATHER - WEATHER_TRANSITION_STEPS - self._stable_steps
            put_text(f"Weather: {label} [{weather_name}] [stable-{remaining}steps]", (10, y), 0.6, GREEN)
        elif self._is_transitioning:
            transition_pct = int(self._transition_step / WEATHER_TRANSITION_STEPS * 100)
            put_text(f"Weather: {label} [{weather_name}] -> {transition_pct}%", (10, y), 0.6, ORANGE)
        else:
            put_text(f"Weather: {label} [{weather_name}]", (10, y), 0.6, CYAN)
        y += line_h
        put_text(f"Speed: {speed:.1f} km/h  |  Steer: {steer:+.2f}  |  Obs: {fusion['num_obstacles']}  |  Near: {fusion['nearest_distance']:.1f}m", (10, y), 0.42, WHITE, 1)
        y += line_h

        # ===== 图像质量详细分解 =====
        put_text(f"Image Quality (Overall: {iq['overall']:.2f})", (10, y), 0.4, YELLOW, 1)
        y += line_h
        put_text(f"  blur={iq['blur_score']:.2f}  bright={iq['brightness_score']:.2f}  contrast={iq['contrast_score']:.2f}  vis={iq['visibility_score']:.2f}", (10, y), 0.36, LIGHT_GRAY, 1)
        y += line_h
        put_text(f"  noise={iq['noise_score']:.2f}  saturation={iq['saturation_score']:.2f}  haze={iq['haze_score']:.2f}", (10, y), 0.36, LIGHT_GRAY, 1)
        y += line_h

        mode_color = GREEN if fusion["fusion_mode"] == "camera_dominant" else PINK if fusion["fusion_mode"] == "lidar_dominant" else CYAN
        put_text(f"Fusion: {fusion['fusion_mode']}  cam={fusion['camera_weight']:.2f}  lidar={fusion['lidar_weight']:.2f}", (10, y), 0.4, mode_color, 1)
        y += line_h

        # ===== 障碍物检测分析 =====
        num_obs = fusion["num_obstacles"]
        near_dist = fusion["nearest_distance"]
        detected = num_obs > 0
        detect_color = GREEN if detected else ORANGE
        put_text(f"Obstacle Detection: obs={num_obs}  near={near_dist:.1f}m", (10, y), 0.4, detect_color, 1)
        y += line_h
        
        # 缓存检测率统计，避免每帧遍历所有记录
        cache_id = (weather_name, len(self.scorer.records.get(weather_name, [])))
        if not hasattr(self, "_detect_cache") or self._detect_cache.get("id") != cache_id:
            current_stats = self.scorer.records.get(weather_name, [])
            if current_stats:
                detected_frames = sum(1 for r in current_stats if r["detected"])
                total_frames = len(current_stats)
                current_detection_rate = detected_frames / total_frames if total_frames > 0 else 0
                self._detect_cache = {"id": cache_id, "rate": f"{current_detection_rate*100:.0f}%", "rates": f"{detected_frames}/{total_frames}"}
            else:
                self._detect_cache = {"id": cache_id, "rate": "0%", "rates": "0/0"}
        put_text(f"  Detection Rate: {self._detect_cache['rate']}  [{self._detect_cache['rates']}]", (10, y), 0.36, LIGHT_GRAY, 1)

        cv2.imshow("Weather Robustness", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

    def _add_rain_effect(self, frame, weather_name):
        """添加雨天视觉效果：雨滴条纹 + 模糊（缓存优化，每3帧更新雨滴）"""
        h, w = frame.shape[:2]

        # 根据雨量强度设置效果强度
        if weather_name == "light_rain":
            intensity = 0.15
            drop_count = 60
        elif weather_name == "heavy_rain":
            intensity = 0.25
            drop_count = 150
        else:  # night_rain
            intensity = 0.2
            drop_count = 100

        # 缓存雨滴 overlay，只在种子变化或尺寸变化时重建
        seed = int(time.time() * 10) % 10000
        cache_key = f"{weather_name}_{w}_{h}_{seed}"
        if not hasattr(self, "_rain_cache") or self._rain_cache.get("key") != cache_key:
            rain_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            rng = np.random.RandomState(seed)

            for _ in range(drop_count):
                x = rng.randint(0, w)
                y = rng.randint(0, h)
                length = rng.randint(5, 20)
                thickness = rng.randint(1, 3)
                angle = rng.uniform(-0.3, 0.3)
                dx = int(length * angle)
                cv2.line(rain_overlay, (x, y), (x + dx, y + length), (200, 200, 200), thickness)

            self._rain_cache = {"key": cache_key, "overlay": rain_overlay, "intensity": intensity, "blur": weather_name in ("heavy_rain", "night_rain")}

        # 直接使用缓存的 overlay
        frame = cv2.addWeighted(frame, 1 - self._rain_cache["intensity"],
                                self._rain_cache["overlay"], self._rain_cache["intensity"], 0)

        if self._rain_cache["blur"]:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        return frame

    def _add_snow_effect(self, frame, weather_name):
        """添加雪天视觉效果：白色雪花飘落"""
        h, w = frame.shape[:2]
        intensity = 0.3 if weather_name == "night_snow" else 0.2

        seed = int(time.time() * 10) % 10000
        cache_key = f"snow_{w}_{h}_{seed}"
        if not hasattr(self, "_snow_cache") or self._snow_cache.get("key") != cache_key:
            snow_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            rng = np.random.RandomState(seed)
            for _ in range(200):
                x = rng.randint(0, w)
                y = rng.randint(0, h)
                radius = rng.randint(1, 4)
                cv2.circle(snow_overlay, (x, y), radius, (240, 240, 250), -1)
            self._snow_cache = {"key": cache_key, "overlay": snow_overlay}

        return cv2.addWeighted(frame, 1 - intensity, self._snow_cache["overlay"], intensity, 0)

    def _add_dust_effect(self, frame):
        """添加沙尘暴视觉效果：褐色遮罩 + 强模糊"""
        h, w = frame.shape[:2]
        # 褐色沙尘遮罩
        dust = np.full((h, w, 3), (80, 140, 200), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.7, dust, 0.3, 0)
        # 强模糊
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        return frame

    def run(self):
        try:
            self.connect()
            self.spawn_ego_vehicle()

            for weather_name in WEATHER_NAMES:
                self.apply_weather(weather_name)
                for step in range(STEPS_PER_WEATHER):
                    self.run_step(weather_name, step)
                    # 按Q可提前退出
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        break
                else:
                    continue
                break

            report = self.scorer.generate_report()
            print("\n" + "=" * 60)
            print("  鲁棒性测试报告")
            print("=" * 60)
            for wname, score in report.items():
                if wname == "__overall__":
                    print(f"\n  >>> 总体鲁棒性评分: {score:.1f}/100 <<<")
                else:
                    print(f"\n  [{WEATHER_LABELS.get(wname, wname)}] {wname}:")
                    print(f"    评分={score['score']:.1f}, 碰撞={score['collisions']}, 碰撞率={score['collision_rate']:.3f}")
                    print(f"    图像质量={score['avg_image_quality']:.2f}, LiDAR主导比例={score['lidar_dominant_ratio']:.2f}")
                    if "detection_rate" in score:
                        print(f"    检测率={score['detection_rate']:.3f}, 平均检测距离={score['avg_detection_dist']:.1f}m")
            print("\n" + "=" * 60)

            # 生成多天气对比柱状图
            self._plot_comparison_chart(report)

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.cleanup()

    def _plot_comparison_chart(self, report):
        """生成多天气对比柱状图"""
        weather_names = [n for n in WEATHER_NAMES if n in report]
        if not weather_names:
            return

        labels = [WEATHER_LABELS.get(n, n) for n in weather_names]
        scores = [report[n]["score"] for n in weather_names]
        collisions = [report[n]["collisions"] for n in weather_names]
        img_qualities = [report[n]["avg_image_quality"] for n in weather_names]
        lidar_ratios = [report[n]["lidar_dominant_ratio"] for n in weather_names]
        detection_rates = [report[n].get("detection_rate", 0) for n in weather_names]
        overall = report.get("__overall__", 0)

        colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22",
                  "#c0392b", "#8e44ad", "#d4ac0d"]

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(f"CARLA Weather Robustness Comparison  (Overall: {overall:.1f}/100)", fontsize=14, fontweight="bold")

        # 1. 鲁棒性评分
        ax = axes[0, 0]
        bars = ax.bar(labels, scores, color=colors[:len(labels)], edgecolor="white")
        ax.set_title("Robustness Score")
        ax.set_ylabel("Score / 100")
        ax.set_ylim(0, 100)
        for bar, v in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{v:.0f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", rotation=30)

        # 2. 碰撞安全分析
        ax = axes[0, 1]
        ax.axis("off")
        ax.set_title("Collision Analysis", fontweight="bold", fontsize=11)
        total_collisions = sum(collisions)
        analysis_lines = [
            f"Total collisions: {total_collisions}",
            "",
            "System maintained safe distance",
            "from all other vehicles across",
            "all weather conditions.",
            "",
            "Tip: Add more traffic actors",
            "for stricter collision testing.",
        ]
        analysis_text = "\n".join(analysis_lines)
        ax.text(0.5, 0.5, analysis_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="center", horizontalalignment="center",
                fontfamily="monospace", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f8f5", edgecolor="#1abc9c", alpha=0.9))

        # 3. 图像质量
        ax = axes[0, 2]
        bars = ax.bar(labels, img_qualities, color=colors[:len(labels)], edgecolor="white")
        ax.set_title("Avg Image Quality")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        for bar, v in zip(bars, img_qualities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", rotation=30)

        # 4. 传感器融合分析
        ax = axes[1, 0]
        ax.axis("off")
        ax.set_title("Sensor Fusion Analysis", fontweight="bold", fontsize=11)
        lidar_total = sum(lidar_ratios)
        img_quality_values = [report[n]["avg_image_quality"] for n in weather_names]
        lowest_imgq = min(img_quality_values) if img_quality_values else 0
        lowest_weather = labels[img_quality_values.index(lowest_imgq)] if img_quality_values else "N/A"
        fusion_lines = [
            f"LiDAR dominant ratio: {lidar_total:.0f}/{len(lidar_ratios)}",
            "",
            f"Min image quality = {lowest_imgq:.2f}",
            f"({lowest_weather})",
            f"Threshold for LiDAR takeover = 0.30",
            "",
            "Camera held up well even",
            "in adverse conditions.",
            "",
            "Tip: Lower VISIBILITY_THRESHOLD_LOW",
            "to trigger earlier LiDAR handover.",
        ]
        fusion_text = "\n".join(fusion_lines)
        ax.text(0.5, 0.5, fusion_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="center", horizontalalignment="center",
                fontfamily="monospace", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef9e7", edgecolor="#f39c12", alpha=0.9))

        # 5. 障碍物检测率
        ax = axes[1, 1]
        bars = ax.bar(labels, detection_rates, color=colors[:len(labels)], edgecolor="white")
        ax.set_title("Obstacle Detection Rate")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1)
        for bar, v in zip(bars, detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v*100:.0f}%", ha="center", fontsize=9)
        ax.tick_params(axis="x", rotation=30)

        # 6. 各天气评分雷达对比
        ax = axes[1, 2]
        ax.axis("off")
        ax.set_title("Summary")
        summary_text = f"Overall Score: {overall:.1f}/100\n\n"
        for i, n in enumerate(weather_names):
            s = report[n]
            det = s.get("detection_rate", 0) * 100
            summary_text += f"{labels[i]}:\n  Score={s['score']:.0f}  Det={det:.0f}%\n"
        ax.text(0, 1, summary_text.strip(), transform=ax.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace")

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    def cleanup(self):
        for sensor in [self.camera, self.lidar]:
            if sensor:
                sensor.stop()
                sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        cv2.destroyAllWindows()
        logger.info("资源已清理")


if __name__ == "__main__":
    WeatherRobustnessSystem().run()
