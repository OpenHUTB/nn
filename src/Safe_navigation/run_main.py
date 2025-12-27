#!/usr/bin/env python3
"""
AirSimNH 无人车完整仿真控制脚本
功能：连接仿真器、手动控制车辆、采集传感器数据、监控车辆状态
"""

import airsim
import time
import numpy as np
import cv2
import json
import os
from datetime import datetime
import threading
from collections import deque



class AirSimNHCarSimulator:
    """AirSim无人车仿真主类"""

    def __init__(self, ip="127.0.0.1", port=41451, vehicle_name="PhysXCar"):
        """
        初始化仿真器连接

        参数:
            ip: AirSim服务器IP地址
            port: AirSim服务器端口
            vehicle_name: 车辆名称，需与settings.json中一致
        """
        self.ip = ip
        self.port = port
        self.vehicle_name = vehicle_name
        self.client = None
        self.is_connected = False
        self.is_api_control_enabled = False
        self.running = False
        self.data_log = []
        self.data_file = None


        # 传感器数据缓存
        self.sensor_data = {
            "camera": deque(maxlen=100),
            "imu": deque(maxlen=1000),
            "gps": deque(maxlen=1000),
            "lidar": deque(maxlen=100)
        }

        # 创建数据保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = f"simulation_data_{timestamp}"
        os.makedirs(self.data_dir, exist_ok=True)

        print(f"数据保存目录: {self.data_dir}")

    def connect(self):
        """连接到AirSim仿真器"""
        try:
            print(f"正在连接到AirSim仿真器 {self.ip}:{self.port}...")

            # 创建客户端连接
            self.client = airsim.CarClient(ip=self.ip, port=self.port)
            self.client.confirmConnection()

            # 检查车辆是否存在
            vehicles = self.client.listVehicles()
            if self.vehicle_name not in vehicles:
                print(f"警告: 车辆 '{self.vehicle_name}' 未找到，可用车辆: {vehicles}")
                # 尝试使用找到的第一个车辆
                if vehicles:
                    self.vehicle_name = vehicles[0]
                    print(f"使用车辆: {self.vehicle_name}")

            self.is_connected = True
            print("✓ 成功连接到AirSim仿真器！")

            return True

        except Exception as e:
            print(f"✗ 连接失败: {e}")
            print("请确保:")
            print("1. AirSimNH环境正在运行 (在虚幻引擎中启动)")
            print("2. settings.json配置正确")
            print("3. 网络连接正常")
            return False


    def enable_api_control(self, enable=True):
        """启用/禁用API控制"""
        try:
            self.client.enableApiControl(enable, vehicle_name=self.vehicle_name)
            self.is_api_control_enabled = enable

            if enable:
                print("✓ API控制已启用")
                # 重置控制到初始状态
                controls = airsim.CarControls()
                controls.throttle = 0
                controls.steering = 0
                controls.brake = 0
                controls.handbrake = False
                self.client.setCarControls(controls, vehicle_name=self.vehicle_name)
            else:
                print("✓ API控制已禁用")

            return True

        except Exception as e:
            print(f"✗ API控制设置失败: {e}")
            return False

    def get_vehicle_state(self):
        """获取完整的车辆状态信息"""
        try:
            state = self.client.getCarState(vehicle_name=self.vehicle_name)

            # 获取车辆物理信息
            kinematics = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)


            state_info = {
                "timestamp": time.time(),
                "speed_kmh": state.speed,
                "speed_ms": state.speed / 3.6,

                "orientation": {
                    "w": kinematics.orientation.w_val,
                    "x": kinematics.orientation.x_val,
                    "y": kinematics.orientation.y_val,
                    "z": kinematics.orientation.z_val
                },
                "gear": state.gear,
                "rpm": state.rpm,
                "max_rpm": state.maxrpm,
                "handbrake": state.handbrake,
                "collision": state.collision.has_collided,
                "collision_count": state.collision.collision_count
            }

            return state_info

        except Exception as e:
            print(f"获取车辆状态失败: {e}")
            return None

    def capture_camera_images(self, camera_names=["front", "back", "left", "right"]):
        """从多个摄像头捕获图像"""
        images = {}

        for cam_name in camera_names:
            try:
                # 请求RGB图像
                responses = self.client.simGetImages([
                    airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False)
                ], vehicle_name=self.vehicle_name)

                if responses and responses[0]:
                    img_response = responses[0]

                    # 将图像数据转换为numpy数组
                    img1d = np.frombuffer(img_response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(img_response.height, img_response.width, 3)

                    # 保存图像到文件
                    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                    filename = f"{self.data_dir}/{cam_name}_{timestamp}.png"
                    cv2.imwrite(filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

                    images[cam_name] = {
                        "filename": filename,
                        "shape": img_rgb.shape,
                        "timestamp": time.time()
                    }

                    # 缓存数据
                    self.sensor_data["camera"].append({
                        "camera": cam_name,
                        "timestamp": time.time(),
                        "filename": filename
                    })

            except Exception as e:
                print(f"摄像头 '{cam_name}' 捕获失败: {e}")

        return images

    def get_imu_data(self):
        """获取IMU传感器数据"""
        try:
            imu_data = self.client.getImuData(imu_name="Imu", vehicle_name=self.vehicle_name)

            data = {
                "timestamp": time.time(),
                "linear_acceleration": {
                    "x": imu_data.linear_acceleration.x_val,
                    "y": imu_data.linear_acceleration.y_val,
                    "z": imu_data.linear_acceleration.z_val
                },
                "angular_velocity": {
                    "x": imu_data.angular_velocity.x_val,
                    "y": imu_data.angular_velocity.y_val,
                    "z": imu_data.angular_velocity.z_val
                },
                "orientation": {
                    "w": imu_data.orientation.w_val,
                    "x": imu_data.orientation.x_val,
                    "y": imu_data.orientation.y_val,
                    "z": imu_data.orientation.z_val
                }
            }

            self.sensor_data["imu"].append(data)
            return data

        except Exception as e:
            print(f"获取IMU数据失败: {e}")
            return None

    def get_gps_data(self):
        """获取GPS数据"""
        try:
            gps_data = self.client.getGpsData(gps_name="Gps", vehicle_name=self.vehicle_name)

            data = {
                "timestamp": time.time(),
                "latitude": gps_data.gnss.geopoint.latitude,
                "longitude": gps_data.gnss.geopoint.longitude,
                "altitude": gps_data.gnss.geopoint.altitude,
                "velocity": {
                    "x": gps_data.gnss.velocity.x_val,
                    "y": gps_data.gnss.velocity.y_val,
                    "z": gps_data.gnss.velocity.z_val
                }
            }

            self.sensor_data["gps"].append(data)
            return data

        except Exception as e:
            print(f"获取GPS数据失败: {e}")
            return None



        参数:
            duration: 演示总时长（秒）
        """
        if not self.is_connected or not self.is_api_control_enabled:
            print("错误: 请先连接并启用API控制")
            return False

        print(f"\n开始手动控制演示 ({duration}秒)...")

        start_time = time.time()
        sequence = 0


        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time


            return True

        except Exception as e:
            print(f"\n✗ 控制演示出错: {e}")
            return False

    def data_collection_demo(self, duration=5):
        """数据采集演示：采集所有传感器数据"""
        print(f"\n开始数据采集演示 ({duration}秒)...")

        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                frame_count += 1

                # 采集所有摄像头图像
                images = self.capture_camera_images()

                # 采集IMU数据
                imu_data = self.get_imu_data()

                # 采集GPS数据
                gps_data = self.get_gps_data()

                # 获取车辆状态
                vehicle_state = self.get_vehicle_state()

                # 记录到日志
                log_entry = {
                    "frame": frame_count,
                    "timestamp": time.time(),
                    "images": len(images),
                    "imu_data": imu_data is not None,
                    "gps_data": gps_data is not None,
                    "vehicle_state": vehicle_state is not None
                }
                self.data_log.append(log_entry)

                print(f"\r采集帧: {frame_count} | "
                      f"图像: {len(images)} | "
                      f"速度: {vehicle_state['speed_kmh'] if vehicle_state else 'N/A':.1f} km/h", end="")

                time.sleep(0.2)  # 5Hz采集频率

            print(f"\n✓ 数据采集完成，共采集 {frame_count} 帧")
            return True

        except Exception as e:
            print(f"\n✗ 数据采集出错: {e}")
            return False

    def save_simulation_data(self):
        """保存所有仿真数据到文件"""
        try:
            # 保存车辆状态日志
            log_file = f"{self.data_dir}/simulation_log.json"
            with open(log_file, 'w') as f:
                json.dump(self.data_log, f, indent=2)

            # 保存传感器数据统计
            stats = {
                "timestamp": datetime.now().isoformat(),
                "vehicle_name": self.vehicle_name,
                "camera_frames": len(self.sensor_data["camera"]),
                "imu_samples": len(self.sensor_data["imu"]),
                "gps_samples": len(self.sensor_data["gps"]),

            }

            stats_file = f"{self.data_dir}/simulation_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)


            # 生成数据报告
            report_file = f"{self.data_dir}/report.txt"
            with open(report_file, 'w') as f:
                f.write("=" * 50 + "\n")
                f.write("AirSim无人车仿真数据报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"仿真时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"车辆名称: {self.vehicle_name}\n")
                f.write(f"摄像头帧数: {stats['camera_frames']}\n")
                f.write(f"IMU采样数: {stats['imu_samples']}\n")
                f.write(f"GPS采样数: {stats['gps_samples']}\n")

                f.write("数据文件:\n")
                for file in os.listdir(self.data_dir):
                    f.write(f"  - {file}\n")

            print(f"\n✓ 仿真数据已保存到: {self.data_dir}")
            print(f"  日志文件: {log_file}")
            print(f"  统计数据: {stats_file}")
            print(f"  报告文件: {report_file}")


            return True

        except Exception as e:
            print(f"✗ 保存数据失败: {e}")
            return False


        """运行完整演示"""
        print("=" * 60)
        print("AirSimNH 无人车完整仿真演示")
        print("=" * 60)

        # 步骤1: 连接仿真器
        if not self.connect():
            return False

        try:
            # 步骤2: 启用API控制
            if not self.enable_api_control(True):
                return False


            # 步骤3: 手动控制演示
            if not self.manual_control_demo(control_duration):
                print("手动控制演示失败，继续其他演示...")

            # 短暂暂停，让车辆完全停止
            time.sleep(2)

            # 步骤4: 数据采集演示
            if not self.data_collection_demo(data_duration):
                print("数据采集演示失败，继续保存数据...")

            # 步骤5: 保存数据
            self.save_simulation_data()

            return True

        finally:
            # 步骤6: 清理和退出
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")

        # 停止车辆
        if self.is_api_control_enabled:
            controls = airsim.CarControls()

            except:
                pass

            # 禁用API控制
            try:
                self.enable_api_control(False)
            except:
                pass

        print("✓ 清理完成")


def main():
    """主函数"""
    # 创建仿真器实例
    simulator = AirSimNHCarSimulator(
        ip="127.0.0.1",
        port=41451,
        vehicle_name="PhysXCar"
    )

    # 运行完整演示
    try:
        simulator.run_full_demo(

            data_duration=10  # 数据采集时长（秒）
        )

        print("\n" + "=" * 60)
        print("仿真演示完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n仿真被用户中断")
        simulator.cleanup()
    except Exception as e:
        print(f"\n仿真出错: {e}")
        simulator.cleanup()


if __name__ == "__main__":
    main()