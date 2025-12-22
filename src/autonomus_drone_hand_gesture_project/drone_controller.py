"""
无人机控制器模块
负责控制无人机的连接、起飞、降落和移动
作者: xiaoshiyuan888
"""

import time
import threading
import math


class SimpleDroneController:
    """简单的无人机控制器"""

    def __init__(self, airsim_module, speech_manager=None, config=None):
        self.airsim = airsim_module
        self.client = None
        self.connected = False
        self.flying = False
        self.speech_manager = speech_manager
        self.config = config

        # 控制参数
        self.velocity = config.get('drone', 'velocity')
        self.duration = config.get('drone', 'duration')
        self.altitude = config.get('drone', 'altitude')
        self.control_interval = config.get('drone', 'control_interval')

        # 控制状态
        self.last_control_time = 0
        self.last_gesture = None
        self.flight_mode = "manual"  # manual, auto, circle, eight, square, return_home

        # 飞行任务状态
        self.auto_flying = False
        self.auto_flight_thread = None
        self.stop_auto_flight = False

        # 飞行高度控制
        self.target_altitude = self.altitude
        self.altitude_change_step = 2.0  # 米

        # 上次语音提示状态
        self.last_connection_announced = False
        self.last_takeoff_announced = False
        self.last_land_announced = False

        print("✓ 增强的无人机控制器已初始化")

    def connect(self):
        """连接AirSim无人机"""
        if self.connected:
            return True

        # 语音提示：正在连接
        if (self.speech_manager and
                self.speech_manager.enabled):
            self.speech_manager.speak('connecting')

        if self.airsim is None:
            print("⚠ AirSim不可用，使用模拟模式")

            # 语音提示：模拟模式
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('simulation_mode')

            self.connected = True
            return True

        print("连接AirSim...")

        try:
            self.client = self.airsim.MultirotorClient()
            self.client.confirmConnection()
            print("✅ 已连接AirSim!")

            # 语音提示：连接成功
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('connected')

            self.client.enableApiControl(True)
            print("✅ API控制已启用")

            self.client.armDisarm(True)
            print("✅ 无人机已武装")

            self.connected = True
            return True

        except Exception as e:
            print(f"❌ 连接失败: {e}")

            # 语音提示：连接失败
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('connection_failed')

            print("\n使用模拟模式继续? (y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                self.connected = True
                print("✅ 使用模拟模式")

                # 语音提示：模拟模式
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('simulation_mode')

                return True

            return False

    def takeoff(self):
        """起飞"""
        if not self.connected:
            return False

        # 语音提示：正在起飞
        if (self.speech_manager and
                self.speech_manager.enabled and
                not self.last_takeoff_announced):
            self.speech_manager.speak('taking_off')
            self.last_takeoff_announced = True
            self.last_land_announced = False

        try:
            if self.airsim is None or self.client is None:
                print("✅ 模拟起飞")
                self.flying = True

                # 语音提示：起飞成功
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('takeoff_success')

                return True

            print("起飞中...")
            self.client.takeoffAsync().join()
            time.sleep(1)

            # 上升到指定高度
            self.target_altitude = self.altitude
            self.client.moveToZAsync(self.altitude, 3).join()

            self.flying = True
            self.flight_mode = "manual"
            print("✅ 无人机成功起飞")

            # 语音提示：起飞成功
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('takeoff_success')

            return True
        except Exception as e:
            print(f"❌ 起飞失败: {e}")

            # 语音提示：起飞失败
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('takeoff_failed')

            return False

    def land(self):
        """降落"""
        if not self.connected:
            return False

        # 停止自动飞行任务
        self.stop_auto_flight_task()

        # 语音提示：正在降落
        if (self.speech_manager and
                self.speech_manager.enabled and
                not self.last_land_announced):
            self.speech_manager.speak('landing')
            self.last_land_announced = True
            self.last_takeoff_announced = False

        try:
            if self.airsim is None or self.client is None:
                print("✅ 模拟降落")
                self.flying = False
                self.flight_mode = "manual"

                # 语音提示：降落成功
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('land_success')

                return True

            print("降落中...")
            self.client.landAsync().join()
            self.flying = False
            self.flight_mode = "manual"
            print("✅ 无人机已降落")

            # 语音提示：降落成功
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('land_success')

            return True
        except Exception as e:
            print(f"降落失败: {e}")
            return False

    def hover(self):
        """悬停"""
        if not self.connected or not self.flying:
            return False

        try:
            if self.airsim is None or self.client is None:
                print("模拟悬停")
                self.flight_mode = "manual"
                return True

            self.client.hoverAsync().join()
            self.flight_mode = "manual"
            print("✅ 无人机悬停")

            # 语音提示：悬停
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('hovering')

            return True
        except Exception as e:
            print(f"悬停失败: {e}")
            return False

    def return_home(self):
        """返航到起飞点"""
        if not self.connected or not self.flying:
            return False

        # 停止当前自动飞行任务
        self.stop_auto_flight_task()

        # 语音提示：返航
        if (self.speech_manager and
                self.speech_manager.enabled):
            self.speech_manager.speak('returning_home')

        try:
            if self.airsim is None or self.client is None:
                print("✅ 模拟返航")
                self.flight_mode = "return_home"
                return True

            print("返航中...")
            self.flight_mode = "return_home"

            # 返回到起点 (0, 0, altitude)
            self.client.moveToPositionAsync(0, 0, self.target_altitude, 5).join()

            # 悬停
            self.client.hoverAsync().join()
            self.flight_mode = "manual"
            print("✅ 返航完成")

            # 语音提示：返航成功
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('return_home_success')

            return True
        except Exception as e:
            print(f"❌ 返航失败: {e}")
            return False

    def start_auto_flight(self):
        """启动自动飞行模式（正方形轨迹）"""
        if not self.connected or not self.flying:
            return False

        # 停止之前的自动飞行任务
        self.stop_auto_flight_task()

        # 语音提示：自动飞行
        if (self.speech_manager and
                self.speech_manager.enabled):
            self.speech_manager.speak('auto_flight_start')

        try:
            if self.airsim is None or self.client is None:
                print("✅ 模拟自动飞行")
                self.flight_mode = "auto"
                return True

            # 启动自动飞行线程
            self.stop_auto_flight = False
            self.auto_flight_thread = threading.Thread(target=self._square_flight_task)
            self.auto_flight_thread.daemon = True
            self.auto_flight_thread.start()

            self.flight_mode = "auto"
            print("✅ 自动飞行模式启动")

            return True
        except Exception as e:
            print(f"❌ 启动自动飞行失败: {e}")
            return False

    def _square_flight_task(self):
        """正方形飞行任务"""
        if self.airsim is None or self.client is None:
            return

        try:
            side_length = 10.0  # 正方形边长（米）
            speed = 3.0  # 飞行速度

            # 四个顶点
            points = [
                (side_length/2, side_length/2, self.target_altitude),
                (side_length/2, -side_length/2, self.target_altitude),
                (-side_length/2, -side_length/2, self.target_altitude),
                (-side_length/2, side_length/2, self.target_altitude),
                (0, 0, self.target_altitude)  # 返回起点
            ]

            for i, point in enumerate(points):
                if self.stop_auto_flight:
                    break

                print(f"自动飞行: 前往点 {i+1}/{len(points)}: {point}")
                self.client.moveToPositionAsync(point[0], point[1], point[2], speed).join()

                # 在每个顶点悬停1秒
                if not self.stop_auto_flight:
                    time.sleep(1)

            if not self.stop_auto_flight:
                # 悬停
                self.client.hoverAsync().join()
                self.flight_mode = "manual"
                print("✅ 自动飞行完成")

                # 语音提示
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('auto_flight_complete')
        except Exception as e:
            print(f"自动飞行任务错误: {e}")

    def start_circle_flight(self):
        """启动圆形盘旋飞行"""
        if not self.connected or not self.flying:
            return False

        # 停止之前的自动飞行任务
        self.stop_auto_flight_task()

        # 语音提示：圆形盘旋
        if (self.speech_manager and
                self.speech_manager.enabled):
            self.speech_manager.speak('circle_flight_start')

        try:
            if self.airsim is None or self.client is None:
                print("✅ 模拟圆形盘旋")
                self.flight_mode = "circle"
                return True

            # 启动圆形飞行线程
            self.stop_auto_flight = False
            self.auto_flight_thread = threading.Thread(target=self._circle_flight_task)
            self.auto_flight_thread.daemon = True
            self.auto_flight_thread.start()

            self.flight_mode = "circle"
            print("✅ 圆形盘旋模式启动")

            return True
        except Exception as e:
            print(f"❌ 启动圆形盘旋失败: {e}")
            return False

    def _circle_flight_task(self):
        """圆形飞行任务"""
        if self.airsim is None or self.client is None:
            return

        try:
            radius = 5.0  # 圆半径（米）
            duration = 30.0  # 飞行时间（秒）
            start_time = time.time()

            while time.time() - start_time < duration and not self.stop_auto_flight:
                # 计算圆上的点
                t = time.time() - start_time
                angle = t * 2 * math.pi / 5  # 每5秒一圈

                x = radius * math.cos(angle)
                y = radius * math.sin(angle)

                # 飞向圆上的点
                self.client.moveToPositionAsync(x, y, self.target_altitude, 2).join()

                # 短暂延迟
                time.sleep(0.1)

            if not self.stop_auto_flight:
                # 返回中心并悬停
                self.client.moveToPositionAsync(0, 0, self.target_altitude, 3).join()
                self.client.hoverAsync().join()
                self.flight_mode = "manual"
                print("✅ 圆形盘旋完成")

                # 语音提示
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('circle_flight_complete')
        except Exception as e:
            print(f"圆形飞行任务错误: {e}")

    def start_eight_flight(self):
        """启动8字形飞行"""
        if not self.connected or not self.flying:
            return False

        # 停止之前的自动飞行任务
        self.stop_auto_flight_task()

        # 语音提示：8字形飞行
        if (self.speech_manager and
                self.speech_manager.enabled):
            self.speech_manager.speak('eight_flight_start')

        try:
            if self.airsim is None or self.client is None:
                print("✅ 模拟8字形飞行")
                self.flight_mode = "eight"
                return True

            # 启动8字形飞行线程
            self.stop_auto_flight = False
            self.auto_flight_thread = threading.Thread(target=self._eight_flight_task)
            self.auto_flight_thread.daemon = True
            self.auto_flight_thread.start()

            self.flight_mode = "eight"
            print("✅ 8字形飞行模式启动")

            return True
        except Exception as e:
            print(f"❌ 启动8字形飞行失败: {e}")
            return False

    def _eight_flight_task(self):
        """8字形飞行任务"""
        if self.airsim is None or self.client is None:
            return

        try:
            radius = 5.0  # 半径（米）
            duration = 40.0  # 飞行时间（秒）
            start_time = time.time()

            while time.time() - start_time < duration and not self.stop_auto_flight:
                # 计算8字形参数方程
                t = time.time() - start_time
                scale = t * 2 * math.pi / 10  # 每10秒一个完整8字

                # 8字形参数方程
                x = radius * math.sin(scale)
                y = radius * math.sin(scale) * math.cos(scale)

                # 飞向8字形上的点
                self.client.moveToPositionAsync(x, y, self.target_altitude, 2).join()

                # 短暂延迟
                time.sleep(0.1)

            if not self.stop_auto_flight:
                # 返回中心并悬停
                self.client.moveToPositionAsync(0, 0, self.target_altitude, 3).join()
                self.client.hoverAsync().join()
                self.flight_mode = "manual"
                print("✅ 8字形飞行完成")

                # 语音提示
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('eight_flight_complete')
        except Exception as e:
            print(f"8字形飞行任务错误: {e}")

    def increase_altitude(self):
        """增加飞行高度"""
        if not self.connected or not self.flying:
            return False

        try:
            self.target_altitude -= self.altitude_change_step  # AirSim中Z轴向下为负

            # 语音提示：增加高度
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('altitude_increasing')

            if self.airsim is None or self.client is None:
                print(f"✅ 模拟增加高度到: {-self.target_altitude:.1f}米")
                return True

            print(f"增加高度到: {-self.target_altitude:.1f}米")
            self.client.moveToZAsync(self.target_altitude, 3).join()
            print("✅ 高度调整完成")

            return True
        except Exception as e:
            print(f"❌ 调整高度失败: {e}")
            return False

    def decrease_altitude(self):
        """降低飞行高度"""
        if not self.connected or not self.flying:
            return False

        try:
            self.target_altitude += self.altitude_change_step  # AirSim中Z轴向下为负

            # 确保高度不低于最小值
            min_altitude = -20.0
            if self.target_altitude > min_altitude:
                self.target_altitude = min_altitude

            # 语音提示：降低高度
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('altitude_decreasing')

            if self.airsim is None or self.client is None:
                print(f"✅ 模拟降低高度到: {-self.target_altitude:.1f}米")
                return True

            print(f"降低高度到: {-self.target_altitude:.1f}米")
            self.client.moveToZAsync(self.target_altitude, 3).join()
            print("✅ 高度调整完成")

            return True
        except Exception as e:
            print(f"❌ 调整高度失败: {e}")
            return False

    def set_altitude(self, altitude=None):
        """设置特定飞行高度"""
        if not self.connected or not self.flying:
            return False

        try:
            if altitude is None:
                # 使用默认高度
                altitude = self.altitude

            self.target_altitude = altitude

            # 语音提示：设置高度
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak_direct(f"设置高度到{-altitude:.1f}米")

            if self.airsim is None or self.client is None:
                print(f"✅ 模拟设置高度到: {-altitude:.1f}米")
                return True

            print(f"设置高度到: {-altitude:.1f}米")
            self.client.moveToZAsync(altitude, 3).join()
            print("✅ 高度设置完成")

            return True
        except Exception as e:
            print(f"❌ 设置高度失败: {e}")
            return False

    def stop_auto_flight_task(self):
        """停止自动飞行任务"""
        if self.auto_flight_thread and self.auto_flight_thread.is_alive():
            self.stop_auto_flight = True
            self.auto_flight_thread.join(timeout=2.0)
            self.auto_flight_thread = None
            self.flight_mode = "manual"
            print("✅ 自动飞行任务已停止")

    def move_by_gesture(self, gesture, confidence):
        """根据手势移动"""
        if not self.connected or not self.flying:
            return False

        # 检查控制间隔
        current_time = time.time()
        if current_time - self.last_control_time < self.control_interval:
            return False

        # 检查置信度阈值
        min_confidence = self.config.get('gesture', 'min_confidence')
        if confidence < min_confidence:
            # 低置信度语音提示
            if (self.speech_manager and
                    self.speech_manager.enabled and
                    confidence < min_confidence * 0.8):
                self.speech_manager.speak('gesture_low_confidence')
            return False

        try:
            if self.airsim is None or self.client is None:
                print(f"模拟移动: {gesture}")
                self.last_control_time = current_time
                self.last_gesture = gesture
                return True

            success = False

            if gesture == "Up":
                self.client.moveByVelocityZAsync(0, 0, -self.velocity, self.duration)
                success = True
            elif gesture == "Down":
                self.client.moveByVelocityZAsync(0, 0, self.velocity, self.duration)
                success = True
            elif gesture == "Left":
                self.client.moveByVelocityAsync(-self.velocity, 0, 0, self.duration)
                success = True
            elif gesture == "Right":
                self.client.moveByVelocityAsync(self.velocity, 0, 0, self.duration)
                success = True
            elif gesture == "Forward":
                self.client.moveByVelocityAsync(0, -self.velocity, 0, self.duration)
                success = True
            elif gesture == "Backward":
                self.client.moveByVelocityAsync(0, self.velocity, 0, self.duration)
                success = True
            elif gesture == "Stop":
                self.client.hoverAsync()
                success = True
                self.flight_mode = "manual"
                # 悬停语音提示
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('hovering')
            elif gesture == "Hover":
                self.client.hoverAsync()
                success = True
                self.flight_mode = "manual"
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('hovering')
            elif gesture == "ReturnHome":
                success = self.return_home()
            elif gesture == "AutoFlight":
                success = self.start_auto_flight()
            elif gesture == "CircleFlight":
                success = self.start_circle_flight()
            elif gesture == "EightFlight":
                success = self.start_eight_flight()
            elif gesture == "SquareFlight":
                success = self.start_auto_flight()  # 复用自动飞行
            elif gesture == "IncreaseAltitude":
                success = self.increase_altitude()
            elif gesture == "DecreaseAltitude":
                success = self.decrease_altitude()
            elif gesture == "SetAltitude":
                success = self.set_altitude()
            elif gesture == "Grab":
                # 抓取动作（模拟）
                print("执行抓取动作")
                success = True
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('gesture_grab')
            elif gesture == "Release":
                # 释放动作（模拟）
                print("执行释放动作")
                success = True
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('gesture_release')
            elif gesture == "RotateCW":
                # 顺时针旋转
                print("顺时针旋转")
                success = True
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('gesture_rotate_cw')
            elif gesture == "RotateCCW":
                # 逆时针旋转
                print("逆时针旋转")
                success = True
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('gesture_rotate_ccw')
            elif gesture == "TakePhoto":
                # 拍照/截图
                print("执行拍照")
                success = True
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('gesture_photo')

            if success:
                self.last_control_time = current_time
                self.last_gesture = gesture

            return success
        except Exception as e:
            print(f"控制命令失败: {e}")
            return False

    def emergency_stop(self):
        """紧急停止"""
        if self.connected:
            try:
                # 停止自动飞行任务
                self.stop_auto_flight_task()

                if self.flying and self.client is not None:
                    print("紧急降落...")

                    # 语音提示：紧急停止
                    if (self.speech_manager and
                            self.speech_manager.enabled):
                        self.speech_manager.speak('emergency_stop')

                    self.land()
                if self.client is not None:
                    self.client.armDisarm(False)
                    self.client.enableApiControl(False)
                    print("✅ 紧急停止完成")
            except:
                pass

        self.connected = False
        self.flying = False
        self.flight_mode = "manual"

    def get_flight_mode(self):
        """获取当前飞行模式"""
        return self.flight_mode