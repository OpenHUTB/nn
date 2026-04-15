import carla
import time
import random


def get_spawn_points(world):
    """获取地图的spawn points

    Args:
        world: CARLA世界对象

    Returns:
        list: spawn points列表
    """
    return world.get_map().get_spawn_points()


def get_small_car_blueprints(blueprint_library):
    """只获取小汽车蓝图，避免大车碰撞

    Args:
        blueprint_library: 蓝图库

    Returns:
        list: 小汽车蓝图列表
    """
    # 获取所有车辆蓝图
    all_vehicles = blueprint_library.filter('vehicle')

    # 过滤条件：只保留小汽车
    small_cars = []
    for bp in all_vehicles:
        bp_id = bp.id.lower()
        # 排除所有非小汽车类型
        if ('bike' in bp_id or 'bicycle' in bp_id or 'pedestrian' in bp_id or
            'walker' in bp_id or 'animal' in bp_id or 'cross' in bp_id or
            'vespa' in bp_id or 'harley' in bp_id or 'low_rider' in bp_id or
            'omafiets' in bp_id or 'zx125' in bp_id or
            'truck' in bp_id or 'bus' in bp_id or 'van' in bp_id or
            'hgv' in bp_id or 'sprinter' in bp_id or 'fusorosa' in bp_id or
            'ambulance' in bp_id or 'police' in bp_id or 'crown' in bp_id or
            'patrol' in bp_id or 'gtv' in bp_id or 'wrangler' in bp_id or
            'cybertruck' in bp_id or 't2' in bp_id or 'c3' in bp_id):
            continue
        # 只保留小汽车
        small_cars.append(bp)

    # 如果过滤后为空，回退到使用所有车辆（但排除明显的非汽车）
    if not small_cars:
        for bp in all_vehicles:
            bp_id = bp.id.lower()
            if not ('bike' in bp_id or 'bicycle' in bp_id or 'pedestrian' in bp_id or
                    'walker' in bp_id or 'animal' in bp_id or 'cross' in bp_id):
                small_cars.append(bp)

    return small_cars if small_cars else all_vehicles


def is_intersection_area(spawn_point):
    """判断生成点是否位于路口区域

    通过检查生成点周围是否有多个方向的生成点来判断是否为路口

    Args:
        spawn_point: 生成点

    Returns:
        bool: 如果是路口区域返回True
    """
    # 这个函数需要访问spawn_points，所以需要在外部传入或全局访问
    # 简化实现：根据spawn_point的位置特征判断
    # 路口通常有更复杂的道路连接，可以通过rotation的变化来判断

    # 简单的启发式判断：
    # 1. 如果rotation的角度接近0, 90, 180, 270度，可能是在主路上
    # 2. 如果rotation角度较为随机，可能是在路口附近

    rotation_yaw = spawn_point.rotation.yaw % 360
    # 如果rotation在45-135或225-315度之间，可能是在路口区域
    if (45 < rotation_yaw < 135) or (225 < rotation_yaw < 315):
        return True

    return False


def group_spawn_points_by_road(spawn_points, group_size=10):
    """将spawn points按道路分组

    简化版实现：根据位置将spawn points分组，每组代表一条道路

    Args:
        spawn_points: spawn points列表
        group_size: 每组的最大大小

    Returns:
        list: 分组后的spawn points列表
    """
    if not spawn_points:
        return []

    # 使用简单的聚类算法将相近的spawn points分组
    groups = []
    used_points = set()

    for i, point in enumerate(spawn_points):
        if i in used_points:
            continue

        # 创建新组
        current_group = [point]
        used_points.add(i)

        # 查找附近的其他spawn points
        for j, other_point in enumerate(spawn_points):
            if j in used_points:
                continue

            # 计算两点之间的距离
            distance = ((point.location.x - other_point.location.x) ** 2 +
                       (point.location.y - other_point.location.y) ** 2) ** 0.5

            # 如果距离较近（在同一道路上），加入同一组
            if distance < 50.0:  # 50米范围内认为是同一条路
                current_group.append(other_point)
                used_points.add(j)

                # 如果组大小达到限制，停止添加
                if len(current_group) >= group_size:
                    break

        groups.append(current_group)

    # 如果分组后仍然太多，进一步合并
    if len(groups) > 20:
        # 合并一些小分组
        merged_groups = []
        temp_group = []

        for group in groups:
            temp_group.extend(group)
            if len(temp_group) >= group_size * 2:
                merged_groups.append(temp_group)
                temp_group = []

        if temp_group:
            merged_groups.append(temp_group)

        groups = merged_groups if merged_groups else groups

    return groups


def spawn_vehicles(world, client, num_vehicles=100):
    """在地图上生成车辆，避免车辆碰撞

    Args:
        world: CARLA世界对象
        client: CARLA客户端对象
        num_vehicles: 要生成的车辆数量

    Returns:
        list: 生成的车辆actor列表
    """
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = get_small_car_blueprints(blueprint_library)

    if not vehicle_blueprints:
        print("❌ 没有找到合适的车辆蓝图！")
        return []

    vehicles = []
    spawn_points = get_spawn_points(world)

    if not spawn_points:
        print("❌ 没有找到合适的生成点！")
        return []

    # 获取地图信息，了解道路布局
    map_name = world.get_map().name
    print(f"🗺️  地图信息: {map_name}, spawn points数量: {len(spawn_points)}")

    # 按道路分组spawn points
    road_groups = group_spawn_points_by_road(spawn_points, group_size=3)  # 每组最多3辆车

    print(f"🛣️  将生成点分为 {len(road_groups)} 个道路组")

    # 严格限制总车辆数，避免转弯和直行碰撞
    max_total_vehicles = min(num_vehicles, len(road_groups), 40)  # 最大40辆，每路最多1辆
    if max_total_vehicles < num_vehicles:
        print(f"⚠️  为避免转弯直行碰撞，限制生成 {max_total_vehicles} 辆车（原计划 {num_vehicles} 辆）")
        num_vehicles = max_total_vehicles

    # 获取交通管理器
    traffic_manager = client.get_trafficmanager()

    # 设置全局交通参数，减少碰撞
    try:
        traffic_manager.global_percentage_speed_difference(-20)  # 整体减速20%，更安全
        traffic_manager.ignore_lights_percentage(100)  # 全局忽略红绿灯
    except:
        pass

    # 为每辆车选择不同的道路，避免同一路口车辆过多
    used_roads = set()
    for i in range(num_vehicles):
        # 找到未使用的道路
        available_roads = [idx for idx in range(len(road_groups)) if idx not in used_roads]

        if not available_roads:
            # 如果所有道路都用了，重新开始
            used_roads.clear()
            available_roads = list(range(len(road_groups)))

        # 随机选择一个未使用的道路
        road_idx = random.choice(available_roads)
        used_roads.add(road_idx)
        current_group = road_groups[road_idx]

        # 在组内选择不同的spawn points
        success = False
        for offset in range(min(20, len(current_group))):  # 尝试位置数
            actual_index = offset % len(current_group)
            spawn_point = current_group[actual_index]

            # 随机选择正常的车辆型号
            blueprint = random.choice(vehicle_blueprints)

            # 使用spawn point的位置和方向
            transform = carla.Transform(spawn_point.location, spawn_point.rotation)

            # 尝试多次生成，避免碰撞
            for attempt in range(5):
                try:
                    vehicle = world.spawn_actor(blueprint, transform)

                    # 设置车辆属性 - 更保守的参数
                    traffic_manager.ignore_lights_percentage(vehicle, 100)  # 忽略红绿灯
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, -25)  # 比限速慢25%，非常平稳

                    # 启用自动驾驶
                    vehicle.set_autopilot(True)

                    # 长时间等待，确保车辆稳定后再生成下一辆
                    time.sleep(1.0)  # 每辆车间隔1秒

                    vehicles.append(vehicle)
                    success = True
                    print(f"🚗 成功生成车辆 {len(vehicles)}/{num_vehicles}: {blueprint.id} (道路 {road_idx})")
                    break

                except Exception as e:
                    error_msg = str(e).lower()
                    if "collision" in error_msg or "spawn" in error_msg:
                        # 如果是碰撞或生成错误，尝试调整位置
                        time.sleep(0.5)
                        continue
                    else:
                        print(f"❌ 生成车辆 {len(vehicles)+1} 失败: {e}")
                        break

            if success:
                break

        if not success:
            print(f"❌ 生成车辆 {len(vehicles)+1} 失败: 无法找到合适的位置")
            # 如果连续失败，提前退出
            if len(vehicles) > 0 and len(vehicles) % 3 == 0:
                print(f"⚠️  已连续失败，可能是地图容量已满，当前成功生成 {len(vehicles)} 辆车")
                break

    return vehicles


def main():
    """主函数"""
    print("=" * 60)
    print("CARLA 车辆生成器 - 启动")
    print("=" * 60)

    vehicles = []

    try:
        # 连接到CARLA服务器
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10.0)  # 设置超时时间
        print(f"✅ 成功连接到CARLA服务器")

        # 获取世界
        world = client.get_world()
        print(f"🌍 当前地图: {world.get_map().name}")

        # 生成车辆
        num_vehicles = 100
        print(f"🚗 开始生成 {num_vehicles} 辆正常行驶车辆...")
        vehicles = spawn_vehicles(world, client, num_vehicles)

        if not vehicles:
            print("❌ 没有成功生成任何车辆！")
            return

        print(f"✅ 成功生成 {len(vehicles)} 辆车辆")
        print("🎮 所有车辆已启用自动驾驶，沿道路行驶")
        print("🚦 红绿灯已禁用，车辆持续行驶")
        print("\n💡 提示：使用录屏软件录制视频，按 Ctrl+C 停止程序")

        # 主循环 - 保持程序运行
        while True:
            try:
                time.sleep(1)
                # 可以在这里添加监控代码

            except KeyboardInterrupt:
                print("\n🛑 用户中断，正在清理...")
                break
            except Exception as e:
                print(f"❌ 程序错误: {e}")
                break

    except KeyboardInterrupt:
        print("\n🛑 用户中断，正在清理...")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        # 清理车辆
        print("🧹 清理车辆资源...")
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        print("✅ 完成！")


if __name__ == "__main__":
    main()