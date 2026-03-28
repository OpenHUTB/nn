import airsim
import time
import numpy as np
import cv2

def detect_obstacles(image):
    """检测红蓝障碍物，返回是否有障碍+障碍信息+标注后图像"""
    if image is None or image.size == 0:
        return False, [], image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 红色阈值
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 蓝色阈值
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    combine_mask = cv2.bitwise_or(red_mask, blue_mask)

    kernel = np.ones((5, 5), np.uint8)
    combine_mask = cv2.morphologyEx(combine_mask, cv2.MORPH_OPEN, kernel)
    combine_mask = cv2.morphologyEx(combine_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combine_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obs_list = []
    h_img, w_img = image.shape[:2]
    cx_img = w_img // 2

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        obs_cx = x + w // 2

        if obs_cx < cx_img - 50:
            pos = "左侧"
        elif obs_cx > cx_img + 50:
            pos = "右侧"
        else:
            pos = "正前方"

        area_ratio = (w*h)/(w_img*h_img)
        if area_ratio > 0.15:
            dist = "非常近"
        elif area_ratio > 0.08:
            dist = "较近"
        else:
            dist = "较远"

        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(image, f"{pos} {dist}", (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        obs_list.append({"pos":pos,"dist":dist})

    cv2.putText(image, f"Obstacle Num:{len(obs_list)}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    return len(obs_list)>0, obs_list, image


def get_camera_image(client):
    """从AirSim获取车载相机画面"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    if not responses:
        return None
    res = responses[0]
    if not res.image_data_uint8:
        return None

    img_np = np.frombuffer(res.image_data_uint8, dtype=np.uint8)
    try:
        img = img_np.reshape(res.height, res.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    except:
        return None
    return img


def main():
    print("=== AirSim 无人车+视觉避障+90°右转控制 ===")
    print("请先打开UE5并点击Play运行仿真环境！\n")

    try:
        client = airsim.CarClient()
        client.confirmConnection()
        print("✓ 成功连接AirSim仿真")

        client.enableApiControl(True)
        print("✓ 车辆API控制已开启")

        # 初始化控制
        controls = airsim.CarControls()

        # -------- 1 直行开往路口 --------
        print("\n1. 直行前往路口...")
        controls.throttle = 0.5
        controls.steering = 0.0
        client.setCarControls(controls)

        start_t = time.time()
        while time.time()-start_t < 26:
            frame = get_camera_image(client)
            if frame is not None:
                has_obs, _, frame = detect_obstacles(frame)
                cv2.imshow("Camera View - Obstacle Detect", frame)
                if has_obs:
                    print("!!!检测到前方障碍物，请注意")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        # -------- 2 路口停车 --------
        print("2. 到达路口，停车等待...")
        controls.throttle = 0.0
        controls.brake = 1.0
        client.setCarControls(controls)
        time.sleep(2)

        # -------- 3 90度右转 --------
        print("3. 开始90°右转...")
        controls.brake = 0.0
        controls.throttle = 0.3
        controls.steering = 1.0
        client.setCarControls(controls)

        start_t = time.time()
        while time.time()-start_t < 6:
            frame = get_camera_image(client)
            if frame is not None:
                _, _, frame = detect_obstacles(frame)
                cv2.imshow("Camera View - Obstacle Detect", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        # -------- 4 回正直行 --------
        print("4. 转弯完成，回正直行...")
        controls.steering = 0.0
        client.setCarControls(controls)

        start_t = time.time()
        while time.time()-start_t < 8:
            frame = get_camera_image(client)
            if frame is not None:
                _, _, frame = detect_obstacles(frame)
                cv2.imshow("Camera View - Obstacle Detect", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        # -------- 5 减速停车 --------
        print("5. 减速准备停车...")
        controls.throttle = 0.2
        client.setCarControls(controls)
        time.sleep(2)

        controls.brake = 1.0
        controls.throttle = 0.0
        controls.steering = 0.0
        client.setCarControls(controls)
        time.sleep(1)

        print("✅ 全程行驶演示结束")
        client.enableApiControl(False)

    except ConnectionRefusedError:
        print("\n❌ 连接失败：未启动UE5+AirSim仿真/未点Play")
    except KeyboardInterrupt:
        print("\n程序手动退出")
    except Exception as e:
        print(f"\n❌运行异常：{e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()