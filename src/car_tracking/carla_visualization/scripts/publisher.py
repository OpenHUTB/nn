# scripts/publisher.py
#!/usr/bin/env python3
import carla
import queue
import rospy
import time
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float32MultiArray, Bool
from cv_bridge import CvBridge
from utils import (
    CAMERA_WIDTH, CAMERA_HEIGHT, WEATHER_SWITCH_INTERVAL, SUPPORTED_WEATHERS,
    EDGES, build_projection_matrix, get_image_point, point_in_canvas,
    get_2d_box_from_3d_edges, get_vehicle_color, clear_npc, clear_static_vehicle,
    set_weather, get_random_weather, SHOW_VEHICLES_3D, SHOW_TRAFFIC_LIGHTS, SHOW_TRAFFIC_LIGHTS_STATE
)

# 全局变量
bridge = CvBridge()
image_queue = queue.Queue()

def camera_callback(image, queue):
    """相机回调函数，将图像存入队列"""
    queue.put(np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))

def carla_publisher_node():
    # 初始化ROS节点
    rospy.init_node('carla_data_publisher', anonymous=True)

    # 创建发布者
    pub_image = rospy.Publisher('/carla/camera/image_raw', Image, queue_size=10)  # 原始图像
    pub_compressed_image = rospy.Publisher('/carla/camera/image_compressed', CompressedImage, queue_size=10)  # 压缩图像
    pub_vehicle_data = rospy.Publisher('/carla/vehicle/track_data', Float32MultiArray, queue_size=10)  # 车辆跟踪数据
    pub_weather = rospy.Publisher('/carla/weather/current', String, queue_size=10)  # 天气信息
    pub_3d_switch = rospy.Publisher('/carla/3d/switch', Bool, queue_size=10)  # 3D显示开关（示例）

    # 初始化Carla客户端
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 生成主车辆
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.try_spawn_actor(vehicle_bp, carla.Transform(spawn_points[0]))
    if not vehicle:
        rospy.logerr("主车辆生成失败！")
        return
    vehicle.set_autopilot(True)

    # 生成相机
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    camera_init_trans = carla.Transform(carla.Location(x=1, z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
    camera.listen(lambda image: camera_callback(image, image_queue))

    # 清理NPC
    clear_npc(world)
    clear_static_vehicle(world)

    # 生成NPC车辆
    for _ in range(50):
        vehicle_bp_list = bp_lib.filter('vehicle')
        car_bp = [bp for bp in vehicle_bp_list if int(bp.get_attribute('number_of_wheels')) == 4]
        if car_bp:
            npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))
            if npc:
                npc.set_autopilot(True)

    # 初始化参数
    fov = camera_bp.get_attribute("fov").as_float()
    K = build_projection_matrix(CAMERA_WIDTH, CAMERA_HEIGHT, fov)
    K_b = build_projection_matrix(CAMERA_WIDTH, CAMERA_HEIGHT, fov, is_behind_camera=True)
    tracked_vehicles = {}
    frame_counter = 0
    current_weather = "ClearNoon"
    set_weather(world, current_weather)
    last_weather_switch_time = time.time()

    # 频率设置（与Carla同步，20Hz）
    rate = rospy.Rate(20)

    try:
        while not rospy.is_shutdown():
            world.tick()
            frame_counter += 1

            # 自动切换天气
            auto_switch = False
            if WEATHER_SWITCH_INTERVAL > 0:
                current_time = time.time()
                if current_time - last_weather_switch_time >= WEATHER_SWITCH_INTERVAL:
                    current_weather = get_random_weather()
                    set_weather(world, current_weather)
                    last_weather_switch_time = current_time
                    auto_switch = True

            # 发布天气信息
            pub_weather.publish(current_weather)

            # 处理键盘控制（ROS参数服务器或话题控制，这里简化为示例）
            # 可通过订阅话题来修改3D显示开关
            pub_3d_switch.publish(SHOW_VEHICLES_3D)

            # 获取相机图像
            if not image_queue.empty():
                image = image_queue.get()
                # 转换为ROS Image消息并发布
                ros_image = bridge.cv2_to_imgmsg(image, encoding="bgra8")
                ros_image.header.stamp = rospy.Time.now()
                pub_image.publish(ros_image)
                # 发布压缩图像（可选，减少带宽）
                compressed_image = bridge.cv2_to_compressed_imgmsg(image[..., :3], "jpg")
                pub_compressed_image.publish(compressed_image)

            # 采集车辆跟踪数据
            vehicle_data = Float32MultiArray()
            boxes = []
            ids = []
            track_data = {}
            current_vehicles = 0
            max_distance = 0.0

            for npc in world.get_actors().filter('*vehicle*'):
                if npc.id != vehicle.id:
                    dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                    max_distance = max(max_distance, dist)
                    if dist < 50:
                        forward_vec = vehicle.get_transform().get_forward_vector()
                        ray = npc.get_transform().location - vehicle.get_transform().location
                        if forward_vec.dot(ray) > 0:
                            verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
                            points_2d = []
                            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
                            for vert in verts:
                                ray0 = vert - camera.get_transform().location
                                cam_forward_vec = camera.get_transform().get_forward_vector()
                                if cam_forward_vec.dot(ray0) > 0:
                                    p = get_image_point(vert, K, world_2_camera)
                                else:
                                    p = get_image_point(vert, K_b, world_2_camera)
                                points_2d.append(p)
                            x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(
                                points_2d, EDGES, CAMERA_HEIGHT, CAMERA_WIDTH
                            )
                            if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                                if point_in_canvas((x_min, y_min), CAMERA_HEIGHT, CAMERA_WIDTH) and \
                                        point_in_canvas((x_max, y_max), CAMERA_HEIGHT, CAMERA_WIDTH):
                                    ids.append(npc.id)
                                    boxes.append([x_min, y_min, x_max, y_max])
                                    if npc.id not in tracked_vehicles:
                                        tracked_vehicles[npc.id] = {'color': get_vehicle_color(npc.id)}
                                    tracked_vehicles[npc.id]['distance'] = dist
                                    track_data[npc.id] = tracked_vehicles[npc.id]
                                    current_vehicles += 1

            # 封装车辆数据并发布（示例：车辆数量、最大距离、第一个框的坐标）
            vehicle_data.data = [current_vehicles, max_distance] + (boxes[0] if boxes else [0,0,0,0])
            pub_vehicle_data.publish(vehicle_data)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"发布节点出错：{e}")
    finally:
        # 清理资源
        if camera:
            camera.stop()
            camera.destroy()
        if vehicle:
            vehicle.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        rospy.loginfo("Carla资源已清理")

if __name__ == '__main__':
    carla_publisher_node()
