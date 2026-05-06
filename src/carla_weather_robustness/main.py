"""
CARLA恶劣天气鲁棒性测试与自适应感知系统
自动遍历7种天气，输出鲁棒性评分报告
"""

import carla
import numpy as np
import cv2
import json
import logging
from sklearn.cluster import DBSCAN
from config.settings import (
    CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT, CARLA_MAP,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV,
    LIDAR_CHANNELS, LIDAR_RANGE,
    WEATHER_PROFILES, STEPS_PER_WEATHER,
    VISIBILITY_THRESHOLD_LOW, VISIBILITY_THRESHOLD_HIGH,
    LIDAR_CLUSTER_DISTANCE, LIDAR_MIN_CLUSTER_POINTS,
    IMAGE_BRIGHTNESS_LOW, IMAGE_BRIGHTNESS_HIGH,
    CAMERA_WEIGHT_CLEAR, LIDAR_WEIGHT_CLEAR,
    CAMERA_WEIGHT_ADVERSE, LIDAR_WEIGHT_ADVERSE,
    SAFE_DISTANCE, COLLISION_DISTANCE,
    PID_KP, PID_KI, PID_KD, MAX_SPEED, LOG_LEVEL,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(message)s")
logger = logging.getLogger("WeatherRobustness")

WEATHER_NAMES = list(WEATHER_PROFILES.keys())
WEATHER_LABELS = {
    "clear": "晴朗", "cloudy": "多云", "light_rain": "小雨",
    "heavy_rain": "暴雨", "fog": "浓雾", "night": "夜间", "night_rain": "夜间暴雨",
}


class ImageQualityAssessor:
    """图像质量评估：模糊度/亮度/可见度三维打分"""

    def assess(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = np.clip(laplacian_var / 500.0, 0.0, 1.0)
        mean_brightness = np.mean(gray)
        if mean_brightness < IMAGE_BRIGHTNESS_LOW:
            brightness_score = mean_brightness / IMAGE_BRIGHTNESS_LOW
        elif mean_brightness > IMAGE_BRIGHTNESS_HIGH:
            brightness_score = (255.0 - mean_brightness) / (255.0 - IMAGE_BRIGHTNESS_HIGH)
        else:
            brightness_score = 1.0
        visibility_score = np.clip(np.std(gray) / 80.0, 0.0, 1.0)
        overall = blur_score * 0.3 + brightness_score * 0.3 + visibility_score * 0.4
        return {
            "blur_score": float(blur_score), "brightness_score": float(brightness_score),
            "visibility_score": float(visibility_score), "overall": float(overall),
            "laplacian_var": float(laplacian_var), "mean_brightness": float(mean_brightness),
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
        })

    def compute_robustness_score(self, weather_name):
        records = self.records.get(weather_name, [])
        if not records:
            return {"score": 0.0, "collisions": 0, "avg_image_quality": 0.0}
        num_collisions = sum(1 for r in records if r["collision"])
        collision_rate = num_collisions / len(records)
        avg_img_q = np.mean([r["image_overall"] for r in records])
        lidar_ratio = sum(1 for r in records if r["fusion_mode"] == "lidar_dominant") / len(records)
        severity = WEATHER_PROFILES.get(weather_name, {})
        is_adverse = severity.get("precipitation", 0) > 30 or severity.get("fog_density", 0) > 50
        collision_score = max(0, 40 * (1 - collision_rate * 5))
        quality_score = 30 * avg_img_q
        adaptation_score = 30 * lidar_ratio if is_adverse else 30 * (1 - lidar_ratio)
        total = np.clip(collision_score + quality_score + adaptation_score, 0, 100)
        return {
            "score": float(total), "collisions": num_collisions,
            "collision_rate": float(collision_rate),
            "avg_image_quality": float(avg_img_q),
            "lidar_dominant_ratio": float(lidar_ratio),
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
        self._spawn_point = None

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
        logger.info("车辆和传感器已就绪")

    def _on_camera(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        self._camera_image = array[:, :, :3].copy()

    def _on_lidar(self, point_cloud):
        self._lidar_data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)

    def apply_weather(self, name):
        profile = WEATHER_PROFILES[name]
        weather = carla.WeatherParameters()
        weather.cloudiness = profile["cloudiness"]
        weather.precipitation = profile["precipitation"]
        weather.precipitation_deposits = profile["precipitation_deposits"]
        weather.wind_intensity = profile["wind_intensity"]
        weather.fog_density = profile["fog_density"]
        weather.fog_distance = profile["fog_distance"]
        weather.wetness = profile["wetness"]
        weather.sun_altitude_angle = profile["sun_altitude_angle"]
        self.world.set_weather(weather)
        self.scorer.start_weather(name)
        # 重置车辆位置到出生点
        self.vehicle.set_transform(self._spawn_point)
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        logger.info(f"天气已切换: {WEATHER_LABELS.get(name, name)} [{name}]，车辆已重置")

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
        if self._camera_image is None or self._lidar_data is None:
            return

        severity = self.get_severity()
        fusion = self.fusion_perceiver.detect_obstacles(self._camera_image, self._lidar_data, self.vehicle.get_transform(), severity)

        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

        throttle, brake = self._compute_control(fusion["nearest_distance"], speed)
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0))

        collision = self._check_collision()
        self.scorer.record_step(weather_name, fusion, collision)

        # 每20步输出一次日志（每种天气约10行）
        if (step + 1) % 20 == 0:
            logger.info(
                f"[{weather_name}] step={step+1}/{STEPS_PER_WEATHER}, "
                f"融合={fusion['fusion_mode']}, 图像质量={fusion['image_quality']['overall']:.2f}, "
                f"速度={speed:.1f}km/h"
            )
        self._draw_hud(fusion, speed, weather_name)

    def _draw_hud(self, fusion, speed, weather_name):
        if self._camera_image is None:
            return
        display = self._camera_image.copy()
        h, w = display.shape[:2]
        label = WEATHER_LABELS.get(weather_name, weather_name)
        iq = fusion["image_quality"]

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        y = 28
        cv2.putText(display, f"Weather: {label} [{weather_name}]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 28
        cv2.putText(display, f"Speed: {speed:.1f} km/h  |  Obstacles: {fusion['num_obstacles']}  |  Nearest: {fusion['nearest_distance']:.1f}m", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 25
        cv2.putText(display, f"Image: blur={iq['blur_score']:.2f}  bright={iq['brightness_score']:.2f}  vis={iq['visibility_score']:.2f}  overall={iq['overall']:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        mode_color = (0, 255, 0) if fusion["fusion_mode"] == "camera_dominant" else (0, 0, 255) if fusion["fusion_mode"] == "lidar_dominant" else (0, 255, 255)
        cv2.putText(display, f"Fusion: {fusion['fusion_mode']}  cam={fusion['camera_weight']:.2f}  lidar={fusion['lidar_weight']:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
        y += 25
        cv2.putText(display, "CARLA Weather Robustness Test", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Weather Robustness", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

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
            print("\n" + "=" * 60)

            with open("robustness_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print("  报告已保存至 robustness_report.json\n")

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.cleanup()

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
