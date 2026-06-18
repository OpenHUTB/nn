import carla
import math
import random
import time
import numpy as np

def get_trajectory(transform, speed, steer, steps=15, dt=0.2):
    """
    使用简化的运动学自行车模型预测车辆未来的轨迹点
    """
    x = transform.location.x
    y = transform.location.y
    z = transform.location.z
    yaw = math.radians(transform.rotation.yaw)
    
    # 假设最小速度，以便在静止时也能画出预测线
    v = max(speed, 5.0) 
    wheelbase = 3.0 # 假设轴距为 3 米
    max_steer_angle = math.radians(45.0) # 最大转向角 45 度
    delta = steer * max_steer_angle
    
    points = []
    for _ in range(steps):
        points.append(carla.Location(x=x, y=y, z=z))
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += (v / wheelbase) * math.tan(delta) * dt
        
    return points

def check_collision(trajectory, obstacles, safe_distance=3.0):
    """
    检查轨迹是否会撞上障碍物
    """
    for point in trajectory:
        for obs in obstacles:
            obs_loc = obs.get_location()
            # 计算轨迹点到障碍物的距离
            dist = math.sqrt((point.x - obs_loc.x)**2 + (point.y - obs_loc.y)**2)
            if dist < safe_distance:
                return True # 发生碰撞
    return False

def main():
    # 1. 连接到 Carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    actor_list = []
    
    try:
        # 2. 生成自车 (Ego Vehicle)
        ego_bp = blueprint_library.filter('model3')[0]
        ego_bp.set_attribute('role_name', 'ego')
        spawn_points = world.get_map().get_spawn_points()
        ego_spawn_point = spawn_points[0] # 选定一个初始点
        ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
        actor_list.append(ego_vehicle)
        print("已生成自车 (Ego Vehicle)")

        # 3. 在自车前方生成一个障碍车 (静止或缓慢行驶)
        obs_bp = blueprint_library.filter('vehicle.carlamotors.carlacola')[0]
        # 计算正前方 30 米的位置
        forward_vec = ego_spawn_point.get_forward_vector()
        obs_spawn_point = carla.Transform(
            ego_spawn_point.location + forward_vec * 30.0,
            ego_spawn_point.rotation
        )
        obstacle_vehicle = world.spawn_actor(obs_bp, obs_spawn_point)
        actor_list.append(obstacle_vehicle)
        print("已生成障碍车 (Obstacle Vehicle)")
        
        # 将视角移动到自车后方 (Spectator)
        spectator = world.get_spectator()

        # 4. 主控循环：轨迹采样与避障
        # 候选转向角度：从左打死到右打死
        candidate_steers = [-0.8, -0.4, 0.0, 0.4, 0.8]
        
        while True:
            # 更新观察者视角，跟随自车
            transform = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=50.0, x=-10.0),
                carla.Rotation(pitch=-90.0) # 俯视角，方便观察轨迹
            ))
            
            # 如果想看和截图一样的第三人称视角，请注释上面两行，解开下面这三行的注释：
            # back_cam_loc = transform.location - transform.get_forward_vector() * 8 + carla.Location(z=3)
            # back_cam_rot = transform.rotation
            # spectator.set_transform(carla.Transform(back_cam_loc, back_cam_rot))

            # 获取自车速度
            velocity = ego_vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # 获取周围的其他车辆作为障碍物 (排除自车)
            all_vehicles = world.get_actors().filter('vehicle.*')
            obstacles = [v for v in all_vehicles if v.id != ego_vehicle.id]
            
            best_steer = 0.0
            min_cost = float('inf')
            action_found = False
            
            # 5. 遍历评估每条候选轨迹
            for steer in candidate_steers:
                trajectory = get_trajectory(transform, speed, steer)
                is_collision = check_collision(trajectory, obstacles)
                
                # 绘制轨迹线 (Debug 渲染)
                color = carla.Color(255, 0, 0) if is_collision else carla.Color(0, 255, 0)
                for i in range(len(trajectory) - 1):
                    # 画出带有厚度的线，模拟图片中的效果
                    world.debug.draw_line(
                        trajectory[i] + carla.Location(z=0.5), 
                        trajectory[i+1] + carla.Location(z=0.5), 
                        thickness=0.2, 
                        color=color, 
                        life_time=0.1
                    )
                
                # 评估代价 (Cost Function)
                if not is_collision:
                    # 倾向于选择转向角小的轨迹（走直线）
                    cost = abs(steer)
                    if cost < min_cost:
                        min_cost = cost
                        best_steer = steer
                        action_found = True
            
            # 6. 执行控制命令
            if action_found:
                # 找到安全路径，执行转向并保持油门
                control = carla.VehicleControl(steer=best_steer, throttle=0.5, brake=0.0)
            else:
                # 前方全被堵死，紧急刹车
                control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
                
            ego_vehicle.apply_control(control)
            
            time.sleep(0.1) # 控制频率 10Hz

    except KeyboardInterrupt:
        print("\n用户中断，退出程序。")
    finally:
        print("正在清理生成的车辆...")
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        print("清理完成。")

if __name__ == '__main__':
    main()