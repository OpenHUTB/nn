import carla
import random
import queue
import numpy as np
import cv2
import cvips_utils as utils

# ================= 配置区域 =================
IMAGE_W, IMAGE_H = 640, 360 
FOV = 90.0
TARGET_FPS = 30
# ===========================================

def get_w2c_matrix(cam_transform):
    """构建绝对对齐的 W2C 矩阵"""
    world_2_cam_ue = np.linalg.inv(utils.get_matrix(cam_transform))
    calibration = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    return np.dot(calibration, world_2_cam_ue)

def draw_3d_box_generic(img, transform, bb, is_walker, K, w2c):
    """
    通用 3D 画框函数
    transform: 物体的世界变换 (carla.Transform)
    bb: 物体的碰撞盒 (carla.BoundingBox)
    is_walker: 是否为行人 (决定颜色)
    """
    # 1. 获取物体的世界变换矩阵
    obj_to_world = utils.get_matrix(transform)
    
    # 2. 计算 8 个顶点 (考虑中心偏移 loc 和 范围 ext)
    ext = bb.extent
    loc = bb.location
    corners = np.array([
        [loc.x+ext.x, loc.y+ext.y, loc.z+ext.z, 1], [loc.x+ext.x, loc.y-ext.y, loc.z+ext.z, 1],
        [loc.x+ext.x, loc.y-ext.y, loc.z-ext.z, 1], [loc.x+ext.x, loc.y+ext.y, loc.z-ext.z, 1],
        [loc.x-ext.x, loc.y+ext.y, loc.z+ext.z, 1], [loc.x-ext.x, loc.y-ext.y, loc.z+ext.z, 1],
        [loc.x-ext.x, loc.y-ext.y, loc.z-ext.z, 1], [loc.x-ext.x, loc.y+ext.y, loc.z-ext.z, 1]
    ])
    
    pixels = []
    for corner in corners:
        world_pos = np.dot(obj_to_world, corner)
        p = utils.get_image_point(carla.Location(x=world_pos[0], y=world_pos[1], z=world_pos[2]), K, w2c)
        if p is None: return img
        pixels.append(tuple(p))

    color = (0,0,255) if is_walker else (0,255,0)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for s, e in edges:
        cv2.line(img, pixels[s], pixels[e], color, 1)
    return img

def main():
    # 1. 环境初始化
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0/TARGET_FPS
    world.apply_settings(settings)
    
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    # 2. 健壮生成主车
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    ego_vehicle = None
    print("正在寻找空位生成主车...")
    while ego_vehicle is None:
        spawn_point = random.choice(spawn_points)
        ego_vehicle = world.try_spawn_actor(bp_lib.find('vehicle.tesla.model3'), spawn_point)
    
    ego_vehicle.set_autopilot(True, tm.get_port())
    print(f"主车已就绪，位置: {spawn_point.location}")

    # 3. 配置 6 摄像头
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMAGE_W))
    cam_bp.set_attribute('image_size_y', str(IMAGE_H))
    cam_bp.set_attribute('fov', str(FOV))
    
    mounts = {
        'Front':      carla.Transform(carla.Location(x=1.5, z=2.0), carla.Rotation(yaw=0)),
        'FrontLeft':  carla.Transform(carla.Location(x=1.5, z=2.0), carla.Rotation(yaw=-60)),
        'FrontRight': carla.Transform(carla.Location(x=1.5, z=2.0), carla.Rotation(yaw=60)),
        'Back':       carla.Transform(carla.Location(x=-1.5, z=2.0), carla.Rotation(yaw=180)),
        'BackLeft':   carla.Transform(carla.Location(x=-1.5, z=2.0), carla.Rotation(yaw=-120)),
        'BackRight':  carla.Transform(carla.Location(x=-1.5, z=2.0), carla.Rotation(yaw=120))
    }
    
    cams = {}; queues = {}
    for name, trans in mounts.items():
        c = world.spawn_actor(cam_bp, trans, attach_to=ego_vehicle)
        q = queue.Queue(); c.listen(q.put)
        cams[name] = c; queues[name] = q

    K = utils.build_projection_matrix(IMAGE_W, IMAGE_H, FOV)

    # 4. 扫描地图静态车 (ID 10 代表所有车辆类型)
    print("正在扫描地图静态车辆...")
    static_vehicles = world.get_environment_objects(10)
    print(f"扫描完成：找到 {len(static_vehicles)} 辆静态车")

    # 5. 主循环
    print("\n🚀 全场景检测启动！按 'q' 退出")
    try:
        while True:
            current_frame = world.tick()
            snapshot = world.get_snapshot()
            
            # 严格帧对齐取图
            imgs = {}
            for name, q in queues.items():
                while True:
                    data = q.get()
                    if data.frame == current_frame:
                        imgs[name] = data
                        break
                    if data.frame > current_frame: break # 防止死循环
            if len(imgs) < 6: continue

            # 获取动态 NPC
            all_actors = list(world.get_actors().filter('vehicle.*')) + \
                         list(world.get_actors().filter('walker.pedestrian.*'))
            
            frame_list = []
            display_order = ['FrontLeft', 'Front', 'FrontRight', 'BackLeft', 'Back', 'BackRight']
            
            for name in display_order:
                raw = imgs[name]
                img = np.reshape(np.frombuffer(raw.raw_data, dtype="uint8"), (IMAGE_H, IMAGE_W, 4))[:,:,:3].copy()
                cam_sn = snapshot.find(cams[name].id)
                
                if cam_sn:
                    w2c = get_w2c_matrix(cam_sn.get_transform())
                    cam_loc = cam_sn.get_transform().location

                    # --- 绘制动态物体 ---
                    for actor in all_actors:
                        if actor.id == ego_vehicle.id: continue
                        actor_sn = snapshot.find(actor.id)
                        if actor_sn:
                            if actor_sn.get_transform().location.distance(cam_loc) < 45:
                                is_walker = 'walker' in actor.type_id
                                img = draw_3d_box_generic(img, actor_sn.get_transform(), actor.bounding_box, is_walker, K, w2c)

                    # --- 绘制静态物体 (路边车) ---
                    for obj in static_vehicles:
                        if obj.transform.location.distance(cam_loc) < 45:
                            img = draw_3d_box_generic(img, obj.transform, obj.bounding_box, False, K, w2c)
                
                cv2.putText(img, name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                frame_list.append(img)

            # 拼接显示
            combined = np.vstack([np.hstack(frame_list[:3]), np.hstack(frame_list[3:])])
            cv2.imshow("CVIPS Full Scene Detection", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        print("正在清理环境...")
        settings = world.get_settings(); settings.synchronous_mode = False; world.apply_settings(settings)
        ego_vehicle.destroy()
        for c in cams.values(): c.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()