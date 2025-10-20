import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import tempfile
from pathlib import Path


def setup_client():
    """初始化并配置 AirSim 客户端"""
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    print(f"API Control enabled: {client.isApiControlEnabled()}")
    return client


def setup_image_directory():
    """创建图像保存目录"""
    tmp_dir = Path(tempfile.gettempdir()) / "airsim_car"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to {tmp_dir}")
    return tmp_dir


def execute_maneuver(client, throttle, steering, duration, gear_mode=None, description=""):
    """执行单次驾驶操作"""
    car_controls = airsim.CarControls()
    car_controls.throttle = throttle
    car_controls.steering = steering

    if gear_mode == "reverse":
        car_controls.is_manual_gear = True
        car_controls.manual_gear = -1

    client.setCarControls(car_controls)
    print(description)
    time.sleep(duration)

    # 还原手动档
    if gear_mode == "reverse":
        car_controls.is_manual_gear = False
        car_controls.manual_gear = 0
        client.setCarControls(car_controls)


def apply_brake(client, duration=3):
    """应用刹车"""
    car_controls = airsim.CarControls()
    car_controls.brake = 1
    client.setCarControls(car_controls)
    print("Apply brakes")
    time.sleep(duration)


def save_image(response, filename):
    """根据图像类型保存图像"""
    if response.pixels_as_float:
        print(f"Type {response.image_type}, size {len(response.image_data_float)}")
        airsim.write_pfm(f'{filename}.pfm', airsim.get_pfm_array(response))
    elif response.compress:
        print(f"Type {response.image_type}, size {len(response.image_data_uint8)}")
        airsim.write_file(f'{filename}.png', response.image_data_uint8)
    else:
        print(f"Type {response.image_type}, size {len(response.image_data_uint8)}")
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        cv2.imwrite(f'{filename}.png', img_rgb)


def capture_images(client, tmp_dir, idx):
    """捕获并保存多种类型的图像"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthVis),
        airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True),
        airsim.ImageRequest("1", airsim.ImageType.Scene),
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)
    ])
    print(f'Retrieved images: {len(responses)}')

    for response_idx, response in enumerate(responses):
        filename = tmp_dir / f"{idx}_{response.image_type}_{response_idx}"
        save_image(response, str(filename))


def main():
    """主函数"""
    client = setup_client()
    tmp_dir = setup_image_directory()

    try:
        for idx in range(3):
            # 获取车辆状态
            car_state = client.getCarState()
            print(f"Speed {car_state.speed}, Gear {car_state.gear}")

            # 执行各种驾驶操作
            execute_maneuver(client, 0.5, 0, 3, description="Go Forward")
            execute_maneuver(client, 0.5, 1, 3, description="Go Forward, steer right")
            execute_maneuver(client, -0.5, 0, 3, gear_mode="reverse", description="Go reverse")
            apply_brake(client)

            # 捕获图像
            capture_images(client, tmp_dir, idx)

    finally:
        # 确保资源正确释放
        client.reset()
        client.enableApiControl(False)
        print("Control released and simulator reset")


if __name__ == "__main__":
    main()