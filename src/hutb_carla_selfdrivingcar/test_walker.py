import carla
import random

def test_walker_spawn():
    # 连接到模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    
    # 检查行人蓝图
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    print(f"可用行人蓝图数量: {len(walker_bps)}")
    
    # 测试生成单个行人
    if len(walker_bps) > 0:
        try:
            # 获取随机位置
            random_location = world.get_random_location_from_navigation()
            print(f"\n随机位置: {random_location}")
            
            if random_location:
                # 选择第一个行人蓝图
                bp = walker_bps[0]
                print(f"使用蓝图: {bp.id}")
                
                # 设置生成变换
                spawn_transform = carla.Transform(
                    carla.Location(x=random_location.x, y=random_location.y, z=random_location.z + 0.5),
                    carla.Rotation(yaw=0)
                )
                
                # 生成行人
                walker = world.spawn_actor(bp, spawn_transform)
                print(f"行人生成成功，ID: {walker.id}")
                
                # 创建控制器
                controller_bp = bp_lib.find('controller.ai.walker')
                controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
                print(f"控制器生成成功，ID: {controller.id}")
                
                # 启动控制器
                controller.start()
                target_location = world.get_random_location_from_navigation()
                if target_location:
                    controller.go_to_location(target_location)
                    controller.set_max_speed(1.5)
                    print("行人开始行走")
                
                # 等待3秒
                import time
                time.sleep(3)
                
                # 清理
                controller.stop()
                controller.destroy()
                walker.destroy()
                print("测试完成，资源已清理")
            else:
                print("无法获取随机位置")
        except Exception as e:
            print(f"生成行人失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("没有找到行人蓝图")

if __name__ == "__main__":
    test_walker_spawn()