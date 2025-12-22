"""
无人机控制器模块
负责控制无人机的连接、起飞、降落和移动
作者: xiaoshiyuan888
"""

import time


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

        # 上次语音提示状态
        self.last_connection_announced = False
        self.last_takeoff_announced = False
        self.last_land_announced = False

        print("✓ 简单的无人机控制器已初始化")

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
            self.client.moveToZAsync(self.altitude, 3).join()

            self.flying = True
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

                # 语音提示：降落成功
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('land_success')

                return True

            print("降落中...")
            self.client.landAsync().join()
            self.flying = False
            print("✅ 无人机已降落")

            # 语音提示：降落成功
            if (self.speech_manager and
                    self.speech_manager.enabled):
                self.speech_manager.speak('land_success')

            return True
        except Exception as e:
            print(f"降落失败: {e}")
            return False

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
            elif gesture == "Stop":
                self.client.hoverAsync()
                success = True
                # 悬停语音提示
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('hovering')
            elif gesture == "Hover":
                self.client.hoverAsync()
                success = True
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('hovering')
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
            elif gesture == "ReturnHome":
                # 返航
                print("执行返航")
                self.client.moveToPositionAsync(0, 0, self.altitude, 5)
                success = True
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('gesture_return_home')
            elif gesture == "AutoFlight":
                # 自动飞行模式
                print("启动自动飞行模式")
                success = True
                if (self.speech_manager and
                        self.speech_manager.enabled):
                    self.speech_manager.speak('gesture_auto_flight')

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