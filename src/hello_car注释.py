import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import tempfile

# 连接到 AirSim 模拟器
client = airsim.CarClient()
client.confirmConnection()  # 确认连接成功
client.enableApiControl(True)  # 启用 API 控制权（由代码控制车辆）
print("API Control enabled: %s" % client.isApiControlEnabled())
car_controls = airsim.CarControls()  # 创建车辆控制对象

# 在临时目录中保存图片
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)  # 尝试创建目录
except OSError:
    if not os.path.isdir(tmp_dir):
        raise  # 如果不是目录且创建失败，抛出异常

for idx in range(3):
    # 获取车辆状态
    car_state = client.getCarState()
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # 前进（直行）
    car_controls.throttle = 0.5  # 油门，范围通常 [-1,1]
    car_controls.steering = 0    # 方向盘归中
    client.setCarControls(car_controls)  # 发送控制指令
    print("Go Forward")
    time.sleep(3)   # 等待 3 秒，让车辆行驶一段时间

    # 前进并向右转
    car_controls.throttle = 0.5
    car_controls.steering = 1    # 向右打满方向（具体取值范围视环境而定）
    client.setCarControls(car_controls)
    print("Go Forward, steer right")
    time.sleep(3)   # 等待 3 秒

    # 倒车
    car_controls.throttle = -0.5      # 反向油门用于倒车
    car_controls.is_manual_gear = True
    car_controls.manual_gear = -1     # 手动档设置到倒档
    car_controls.steering = 0
    client.setCarControls(car_controls)
    print("Go reverse, steer right")
    time.sleep(3)   # 等待 3 秒
    # 还原变速器到自动模式
    car_controls.is_manual_gear = False
    car_controls.manual_gear = 0

    # 刹车
    car_controls.brake = 1
    client.setCarControls(car_controls)
    print("Apply brakes")
    time.sleep(3)   # 等待 3 秒（完全刹停）
    car_controls.brake = 0  # 释放刹车

    # 从车辆摄像头获取多种类型的图像
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthVis),  # 深度可视化图像（深度伪彩）
        airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), # 透视投影的深度图（以浮点数形式返回）
        airsim.ImageRequest("1", airsim.ImageType.Scene), # 场景图像（压缩的 png 格式）
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  # 场景图像（未压缩的 RGB 数组）
    print('Retrieved images: %d' % len(responses))

    for response_idx, response in enumerate(responses):
        filename = os.path.join(tmp_dir, f"{idx}_{response.image_type}_{response_idx}")

        if response.pixels_as_float:
            # 当图像以浮点数像素返回（例如透视深度）时，写入 PFM 文件
            print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
        elif response.compress: # png 压缩格式
            # 当图像数据为压缩的二进制（如 PNG）时，直接写入文件
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        else: # 未压缩的原始数组（RGB）
            # 将字节数组转换为 numpy 数组并 reshape 成 H x W x 3，然后用 OpenCV 保存
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # 将字节流转成 numpy 一维数组
            img_rgb = img1d.reshape(response.height, response.width, 3) # 重构为 HxWx3 的 RGB 图像
            cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # 保存为 PNG 文件

# 恢复到初始状态（重置模拟器）
client.reset()

# 释放 API 控制权，由用户/手动控制接管
client.enableApiControl(False)