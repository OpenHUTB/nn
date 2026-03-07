import airsim
import time
import numpy as np
import cv2
import os


def detect_obstacles(image):
    """
    检测图像中的障碍物（红色和蓝色物体）
    返回：是否有障碍物，障碍物位置信息
    """
    if image is None or image.size == 0:
        return False, None

    try:
        # 转换为HSV颜色空间，更容易进行颜色检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义红色范围（红色在HSV中分布在两个区间）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # 定义蓝色范围
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])

        # 创建掩码
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 合并掩码
        combined_mask = cv2.bitwise_or(red_mask, blue_mask)

        # 形态学操作，去除噪声
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 找出所有轮廓
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 障碍物信息
        obstacles_detected = []
        height, width = image.shape[:2]
        center_x = width // 2

        for contour in contours:
            # 忽略太小的区域（可能是噪声）
            if cv2.contourArea(contour) < 500:
                continue

            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 计算障碍物中心位置
            obstacle_center = x + w // 2

            # 判断障碍物在左侧还是右侧
            if obstacle_center < center_x - 50:
                position = "左侧"
            elif obstacle_center > center_x + 50:
                position = "右侧"
            else:
                position = "正前方"

            # 计算距离（基于障碍物大小，越大表示越近）
            area_ratio = (w * h) / (width * height)
            if area_ratio > 0.15:
                distance = "非常近"
            elif area_ratio > 0.08:
                distance = "较近"
            else:
                distance = "较远"

            # 在图像上绘制边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{position} {distance}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            obstacles_detected.append({
                'position': position,
                'distance': distance,
                'area': w * h,
                'bbox': (x, y, w, h)
            })

        # 添加障碍物计数信息
        if obstacles_detected:
            cv2.putText(image, f"障碍物数量: {len(obstacles_detected)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return len(obstacles_detected) > 0, obstacles_detected
    except Exception as e:
        print(f"障碍物检测出错: {e}")
        return False, None


def get_camera_image(client):
    """获取摄像头图像的辅助函数"""
    try:
        # 尝试多种方法获取图像
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        if responses and len(responses) > 0:
            response = responses[0]

            # 检查是否有图像数据
            if response.image_data_uint8:
                # 获取图像数据
                img_data = response.image_data_uint8

                # 确保有高度和宽度信息
                if response.height > 0 and response.width > 0:
                    img_height = response.height
                    img_width = response.width

                    # 转换为numpy数组
                    img = np.frombuffer(img_data, dtype=np.uint8).reshape(img_height, img_width, 3)

                    # AirSim返回的是RGBA格式，需要转换为BGR用于OpenCV显示
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                    return img, img_width, img_height
                else:
                    # 如果没有高度宽度信息，尝试其他方式解析
                    print("警告: 图像尺寸信息不完整，尝试自动解析")
                    # 假设图像是3通道的
                    img = np.frombuffer(img_data, dtype=np.uint8)
                    # 尝试计算可能的尺寸
                    possible_height = int(np.sqrt(len(img) / 3))
                    possible_width = len(img) // (3 * possible_height)
                    if possible_width * possible_height * 3 == len(img):
                        img = img.reshape(possible_height, possible_width, 3)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        return img, possible_width, possible_height
        return None, 0, 0
    except Exception as e:
        print(f"获取图像时出错: {e}")
        return None, 0, 0


def main():
    print("=== 尝试连接 AirSimNH (使用 airsim 1.8.1) ===")
    print("重要：请先确保虚幻引擎中的 AirSimNH 已点击播放(Play)！\n")

    try:
        # 1. 创建客户端
        client = airsim.CarClient()
        print("✓ 客户端对象创建成功")

        # 2. 确认连接
        client.confirmConnection()
        print("✓ 已连接到AirSim仿真服务器")

        # 3. 启用控制
        client.enableApiControl(True)
        print("✓ API控制已启用")

        # 4. 获取车辆状态
        car_state = client.getCarState()
        print(f"✓ 车辆状态获取成功 - 速度: {car_state.speed} km/h")

        # 5. 测试摄像头可用性
        print("\n>>> 测试摄像头可用性...")
        camera_info = client.simGetCameraInfo("0")
        print(f"✓ 摄像头信息: {camera_info}")

        # 尝试获取图像
        print("正在获取第一帧图像...")
        img, width, height = get_camera_image(client)

        if img is not None:
            print(f"✓ 成功获取图像! 分辨率: {width}x{height}")

            # 检测障碍物
            has_obstacles, obstacles_info = detect_obstacles(img)
            if has_obstacles:
                print(f"⚠ 初始检测到 {len(obstacles_info)} 个障碍物")

            # 显示图像
            cv2.imshow('AirSim Camera - 障碍物检测', img)
            cv2.waitKey(1)
            print("✓ 摄像头图像已显示")
        else:
            print("⚠ 无法获取摄像头图像，将使用模拟数据进行演示")
            # 创建一个空白图像用于显示
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Camera feed unavailable", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('AirSim Camera - 障碍物检测', img)
            cv2.waitKey(1)

        # 6. 精确90度转弯演示（带障碍物检测）
        print("\n>>> 连接成功！开始精确90度转弯演示（带障碍物检测）...")
        controls = airsim.CarControls()

        # 直行到路口
        controls.throttle = 0.5
        controls.steering = 0.0
        client.setCarControls(controls)
        print("直行前往路口...")

        # 行驶过程中实时检测障碍物
        last_obstacle_check = time.time()

        for i in range(26):
            time.sleep(1)

            # 每次循环都尝试获取图像（不只是每3秒）
            img, width, height = get_camera_image(client)

            if img is not None:
                # 检测障碍物
                has_obstacles, obstacles_info = detect_obstacles(img)

                # 添加时间戳和行驶信息
                cv2.putText(img, f"Time: {i + 1}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, f"Speed: {car_state.speed:.1f} km/h", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if has_obstacles:
                    print(f"  ⚠ 行驶中 ({i + 1}秒) - 检测到 {len(obstacles_info)} 个障碍物!")

                    # 根据障碍物信息调整驾驶行为
                    for obs in obstacles_info:
                        if obs['position'] == "正前方" and obs['distance'] in ["非常近", "较近"]:
                            print(f"    前方有障碍物({obs['distance']})，减速避让!")
                            controls.throttle = 0.2  # 减速
                            client.setCarControls(controls)
                            time.sleep(0.5)

                            # 如果障碍物很近，尝试绕行
                            if obs['distance'] == "非常近":
                                print("    障碍物非常近，尝试绕行...")
                                controls.steering = -0.3
                                controls.throttle = 0.3
                                client.setCarControls(controls)
                                time.sleep(1)

                                # 回正
                                controls.steering = 0.0
                                controls.throttle = 0.5
                                client.setCarControls(controls)

                        elif obs['position'] == "左侧" and obs['distance'] == "非常近":
                            print("    左侧有很近的障碍物，向右微调")
                            controls.steering = 0.2
                            client.setCarControls(controls)
                            time.sleep(0.5)
                            controls.steering = 0.0
                            client.setCarControls(controls)

                        elif obs['position'] == "右侧" and obs['distance'] == "非常近":
                            print("    右侧有很近的障碍物，向左微调")
                            controls.steering = -0.2
                            client.setCarControls(controls)
                            time.sleep(0.5)
                            controls.steering = 0.0
                            client.setCarControls(controls)

                # 显示图像
                cv2.imshow('AirSim Camera - 障碍物检测', img)
                cv2.waitKey(1)

                if i % 3 == 0:  # 每3秒打印一次状态
                    print(f"  行驶中 ({i + 1}秒) - 摄像头已更新")
            else:
                print(f"  行驶中 ({i + 1}秒) - 等待摄像头图像...")

            # 更新车辆状态
            car_state = client.getCarState()

        # 到达路口，完全停车
        controls.throttle = 0.0
        controls.brake = 1.0
        client.setCarControls(controls)
        print("到达路口，停车...")
        time.sleep(1)

        # 缓慢起步并适度转向
        controls.brake = 0.0
        controls.throttle = 0.25
        controls.steering = 0.7
        client.setCarControls(controls)
        print("缓慢起步转弯...")
        time.sleep(4)

        # 稍微回正一点方向盘
        controls.steering = 0.5
        client.setCarControls(controls)
        print("调整转向角度...")
        time.sleep(2)

        # 完全回正方向盘
        controls.steering = 0.0
        client.setCarControls(controls)
        print("转弯完成，直行...")
        time.sleep(5)

        # 缓慢减速停止
        controls.throttle = 0.2
        client.setCarControls(controls)
        print("减速中...")
        time.sleep(2)

        controls.brake = 1.0
        controls.throttle = 0.0
        client.setCarControls(controls)
        print("停车...")
        time.sleep(1)

        # 关闭摄像头窗口
        cv2.destroyAllWindows()
        print("演示结束。")

        # 7. 释放控制
        client.enableApiControl(False)
        print("控制权已释放。")

    except ConnectionRefusedError:
        print("\n✗ 连接被拒绝。")
        print("  最可能的原因：虚幻引擎中的 AirSimNH 仿真没有启动。")
        print("  请打开虚幻引擎，加载AirSimNH项目，并点击顶部工具栏的蓝色【播放】(▶)按钮。")
    except Exception as e:
        print(f"\n✗ 连接过程中出错: {e}")
        import traceback
        traceback.print_exc()
        print("  其他可能原因：防火墙阻止、端口占用或配置文件错误。")


if __name__ == "__main__":
    main()