# scripts/subscriber.py
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float32MultiArray, Bool
from cv_bridge import CvBridge
from utils import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CHART_WIDTH, CHART_HEIGHT, MAX_HISTORY_FRAMES,
    TRACK_WINDOW_WIDTH, TRACK_WINDOW_HEIGHT, FONT_SCALE_SMALL, FONT_SCALE_MEDIUM,
    LINE_THICKNESS, SUPPORTED_WEATHERS, COCO_CLASS_NAMES,
    custom_draw_bounding_boxes, draw_dynamic_chart, convert_image_format,
    draw_3d_objects, SHOW_VEHICLES_3D, SHOW_TRAFFIC_LIGHTS, SHOW_TRAFFIC_LIGHTS_STATE
)

# 全局变量
bridge = CvBridge()
history_frames = []
history_vehicles = []
history_max_dist = []
current_weather = "ClearNoon"
tracked_vehicles = {}
frame_counter = 0

def image_callback(msg):
    """原始图像回调函数"""
    global frame_counter
    frame_counter += 1
    try:
        # 将ROS Image消息转换为OpenCV图像
        cv_image = bridge.imgmsg_to_cv2(msg, "bgra8")
        process_image(cv_image)
    except Exception as e:
        rospy.logerr(f"图像处理出错：{e}")

def compressed_image_callback(msg):
    """压缩图像回调函数（可选）"""
    global frame_counter
    frame_counter += 1
    try:
        cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        # 补全4通道（如果需要）
        cv_image = np.dstack((cv_image, np.ones((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint8)*255))
        process_image(cv_image)
    except Exception as e:
        rospy.logerr(f"压缩图像处理出错：{e}")

def weather_callback(msg):
    """天气信息回调函数"""
    global current_weather
    current_weather = msg.data

def vehicle_data_callback(msg):
    """车辆数据回调函数"""
    global history_frames, history_vehicles, history_max_dist
    # 解析数据：[current_vehicles, max_distance, x1, y1, x2, y2]
    current_vehicles = int(msg.data[0])
    max_distance = msg.data[1]
    # 更新历史数据
    history_frames.append(frame_counter)
    history_vehicles.append(current_vehicles)
    history_max_dist.append(max_distance)
    # 截断数据
    if len(history_frames) > MAX_HISTORY_FRAMES:
        history_frames.pop(0)
        history_vehicles.pop(0)
        history_max_dist.pop(0)

def process_image(cv_image):
    """处理图像并可视化"""
    global tracked_vehicles

    # 模拟车辆跟踪数据（实际可从话题中解析）
    boxes = []
    ids = []
    track_data = {}
    # 这里简化处理，实际可根据vehicle_data话题中的数据填充
    if len(history_vehicles) > 0 and history_vehicles[-1] > 0:
        boxes = np.array([[100, 100, 200, 200]])  # 示例框
        ids = [1]  # 示例ID
        tracked_vehicles[1] = {'color': (0, 255, 0), 'distance': 20.0}  # 示例数据
        track_data = tracked_vehicles

    # 绘制2D框
    labels = np.array([2] * len(boxes))  # car类别
    output = custom_draw_bounding_boxes(
        cv_image, boxes, labels, COCO_CLASS_NAMES, ids, track_data
    ) if len(boxes) > 0 else cv_image

    # 注意：3D绘制需要Carla的world/camera/vehicle对象，这里如果需要可通过ROS参数或服务传递，
    # 简化版可注释3D绘制，仅保留2D可视化
    # output_3d, _, _ = draw_3d_objects(output, world, camera, vehicle, K, K_b)

    # 转换图像格式
    output_rgb = convert_image_format(output)
    # 绘制动态图表
    chart_image = draw_dynamic_chart(history_frames, history_vehicles, history_max_dist)
    # 拼接图像
    combined_image = np.hstack((output_rgb, chart_image))

    # 绘制天气信息
    cv2.putText(
        combined_image, f"Weather: {current_weather}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MEDIUM, (255, 255, 255), 2
    )
    # 绘制控制提示
    cv2.putText(
        combined_image, "Press Q:Quit",
        (10, CAMERA_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (255, 255, 255), 1
    )

    # 绘制跟踪窗口
    track_window = np.zeros((TRACK_WINDOW_HEIGHT, TRACK_WINDOW_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        track_window, "Vehicle Tracking Monitor", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MEDIUM, (255, 255, 255), LINE_THICKNESS
    )
    cv2.putText(
        track_window, f"Current Weather: {current_weather}",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (255, 255, 255), 1
    )
    y_offset = 90
    for vid, data in list(tracked_vehicles.items())[:10]:
        if y_offset > TRACK_WINDOW_HEIGHT - 20:
            break
        color = data['color']
        dist = data.get('distance', 0.0)
        cv2.putText(
            track_window, f"ID: {vid} | Dist: {dist:.1f}m", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, color, 1
        )
        y_offset += 30

    # 显示窗口
    cv2.imshow('2D Ground Truth + Visualization', combined_image)
    cv2.imshow('Vehicle Tracking Monitor', track_window)
    # 按键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("用户按下Q键退出")
        cv2.destroyAllWindows()

def carla_subscriber_node():
    # 初始化ROS节点
    rospy.init_node('carla_data_subscriber', anonymous=True)

    # 订阅话题
    rospy.Subscriber('/carla/camera/image_raw', Image, image_callback)
    # rospy.Subscriber('/carla/camera/image_compressed', CompressedImage, compressed_image_callback)  # 可选
    rospy.Subscriber('/carla/weather/current', String, weather_callback)
    rospy.Subscriber('/carla/vehicle/track_data', Float32MultiArray, vehicle_data_callback)

    # 保持节点运行
    rospy.spin()

    # 清理窗口
    cv2.destroyAllWindows()

if __name__ == '__main__':
    carla_subscriber_node()
