#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 车辆批量生成器

模块名称: carla_launcher
功能描述: 在CARLA仿真器中批量生成自动驾驶车辆，避免碰撞
核心功能:
    - CARLA服务器连接诊断
    - 智能车辆蓝图筛选（只保留小汽车）
    - 按道路分组生成车辆，避免路口拥堵
    - 交通管理器配置，优化行驶行为


依赖库:
    - carla: CARLA仿真器API
    - time: 时间控制
    - random: 随机选择
    - socket: 网络连接检测

运行环境:
    - Carla 0.9.13+ 运行在 localhost:2000
    - Python 3.7+
"""

import carla
import time
import random
import socket


# ==================== 连接管理 ====================

def test_connection(host="localhost", port=2000):
    """
    测试CARLA服务器连接状态
    
    功能说明:
        先通过socket检测端口是否开放，再尝试建立CARLA客户端连接。
        双重检测机制，提供更清晰的错误诊断。
    
    参数:
        host (str): CARLA服务器地址，默认 localhost
        port (int): CARLA服务器端口，默认 2000
    
    返回:
        carla.Client: 成功连接返回客户端对象，失败返回 None
    
    使用场景:
        在主程序启动前验证仿真器是否已运行
    
    示例:
        >>> client = test_connection()
        >>> if client:
        ...     world = client.get_world()
    """
    # ----- 第一步：端口检测 -----
    # 创建TCP socket用于检测端口是否开放
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)  # 2秒超时，避免长时间等待
    
    # connect_ex 返回0表示连接成功，非0表示失败
    if sock.connect_ex((host, port)) != 0:
        print(f"[X] 端口 {port} 未开放，请确认Carla服务器已启动")
        sock.close()
        return None
    sock.close()  # 关闭socket，准备建立CARLA连接
    
    # ----- 第二步：建立CARLA连接 -----
    try:
        # 创建CARLA客户端
        client = carla.Client(host, port)
        # 设置超时时间10秒（比socket稍长，适应CARLA初始化）
        client.set_timeout(10.0)
        # 尝试获取世界对象，验证连接是否正常
        world = client.get_world()
        print(f"[✓] 成功连接到Carla服务器: {host}:{port}")
        return client
    except Exception as e:
        print(f"[X] 连接失败: {e}")
        return None


# ==================== 生成点管理 ====================

def get_spawn_points(world):
    """
    获取地图的所有车辆生成点
    
    功能说明:
        从CARLA世界对象中提取所有预定义的车辆生成位置。
        这些生成点是地图中合法、安全的车辆初始位置。
    
    参数:
        world (carla.World): CARLA世界对象
    
    返回:
        list: carla.Transform 对象列表，每个包含位置和朝向
    
    使用场景:
        - 生成车辆前获取可用的生成位置
        - 分析地图的生成点分布
    
    示例:
        >>> spawn_points = get_spawn_points(world)
        >>> print(f"共有 {len(spawn_points)} 个生成点")
    """
    return world.get_map().get_spawn_points()


# ==================== 车辆蓝图筛选 ====================

def get_small_car_blueprints(blueprint_library):
    """
    筛选小汽车蓝图，排除大型车辆和两轮车
    
    功能说明:
        从CARLA蓝图库中过滤出适合城市道路的小汽车，
        排除卡车、公交车、自行车、摩托车等，避免碰撞和堵塞。
    
    参数:
        blueprint_library (carla.BlueprintLibrary): CARLA蓝图库
    
    返回:
        list: 小汽车蓝图列表
    
    排除规则:
        1. 两轮车: bike, bicycle, vespa, harley, zx125
        2. 行人/动物: pedestrian, walker, animal
        3. 大型车辆: truck, bus, van, hgv, sprinter
        4. 警车/救护车: police, ambulance, crown, patrol
        5. 特殊车辆: cybertruck, wrangler, gtv
    
    使用场景:
        生成大量自动驾驶车辆时，避免大车造成拥堵
    
    示例:
        >>> blueprints = get_small_car_blueprints(world.get_blueprint_library())
        >>> print(f"找到 {len(blueprints)} 种小汽车模型")
    """
    # 获取所有车辆蓝图
    all_vehicles = blueprint_library.filter('vehicle')
    
    # ----- 第一轮过滤：严格排除 -----
    small_cars = []
    for bp in all_vehicles:
        bp_id = bp.id.lower()
        
        # 排除条件：包含以下关键词的都不是小汽车
        exclude_keywords = [
            'bike', 'bicycle', 'pedestrian', 'walker', 'animal',
            'cross', 'vespa', 'harley', 'low_rider', 'omafiets', 'zx125',
            'truck', 'bus', 'van', 'hgv', 'sprinter', 'fusorosa',
            'ambulance', 'police', 'crown', 'patrol', 'gtv', 'wrangler',
            'cybertruck', 't2', 'c3'
        ]
        
        # 检查是否包含排除关键词
        is_excluded = any(keyword in bp_id for keyword in exclude_keywords)
        if not is_excluded:
            small_cars.append(bp)
    
    # ----- 回退方案：如果过滤后为空，使用宽松过滤 -----
    if not small_cars:
        print("[WARN] 严格过滤后无车辆，使用宽松过滤")
        for bp in all_vehicles:
            bp_id = bp.id.lower()
            # 只排除明显的非汽车
            if not any(k in bp_id for k in ['bike', 'bicycle', 'pedestrian', 
                                            'walker', 'animal', 'cross']):
                small_cars.append(bp)
    
    # 最终回退：返回所有车辆
    return small_cars if small_cars else all_vehicles


# ==================== 生成点分组 ====================

def is_intersection_area(spawn_point):
    """
    判断生成点是否位于路口区域（简化版）
    
    功能说明:
        通过检查生成点的朝向角度来判断是否可能位于路口。
        路口的车辆通常朝向角度变化较大。
    
    参数:
        spawn_point (carla.Transform): 生成点变换信息
    
    返回:
        bool: True表示可能是路口区域，False表示非路口
    
    判断逻辑:
        - 如果朝向角度在 45-135° 或 225-315° 之间，判定为路口
        - 这些角度对应转弯或斜向行驶
    
    注意:
        这是一个启发式判断，精确判断需要分析道路连接图
    
    示例:
        >>> if is_intersection_area(spawn_point):
        ...     print("路口区域，减少车辆生成")
    """
    # 获取yaw角度（偏航角）并归一化到0-360度
    rotation_yaw = spawn_point.rotation.yaw % 360
    
    # 路口区域的车辆通常需要转弯，朝向不是正方向
    # 45-135度：斜向或右转方向
    # 225-315度：左转或斜向方向
    if (45 < rotation_yaw < 135) or (225 < rotation_yaw < 315):
        return True
    
    return False


def group_spawn_points_by_road(spawn_points, group_size=10):
    """
    将生成点按道路分组
    
    功能说明:
        使用距离聚类将相近的生成点归为同一组，每组代表一条道路。
        这样可以避免同一道路生成过多车辆。
    
    参数:
        spawn_points (list): 生成点列表
        group_size (int): 每组最大生成点数量
    
    返回:
        list: 分组后的生成点列表，每个元素是一个道路组
    
    聚类算法:
        1. 遍历所有未使用的生成点
        2. 以当前点为起点创建新组
        3. 查找50米范围内的其他生成点加入同一组
        4. 重复直到所有点都被分组
    
    使用场景:
        - 生成车辆时为每个道路分配不同车辆
        - 避免同一路段车辆过多造成拥堵
    
    示例:
        >>> groups = group_spawn_points_by_road(spawn_points, group_size=3)
        >>> print(f"将 {len(spawn_points)} 个点分为 {len(groups)} 组")
    """
    if not spawn_points:
        return []
    
    # ----- 使用简单聚类算法分组 -----
    groups = []
    used_points = set()  # 记录已使用的点索引
    
    for i, point in enumerate(spawn_points):
        # 跳过已使用的点
        if i in used_points:
            continue
        
        # 创建新组，以当前点为起点
        current_group = [point]
        used_points.add(i)
        
        # 查找附近的其他生成点
        for j, other_point in enumerate(spawn_points):
            if j in used_points:
                continue
            
            # 计算欧氏距离（忽略Z轴）
            distance = ((point.location.x - other_point.location.x) ** 2 +
                       (point.location.y - other_point.location.y) ** 2) ** 0.5
            
            # 50米范围内认为是同一条道路（CARLA中道路宽度约10-20米）
            if distance < 50.0:
                current_group.append(other_point)
                used_points.add(j)
                
                # 达到组大小上限，停止添加
                if len(current_group) >= group_size:
                    break
        
        groups.append(current_group)
    
    # ----- 后处理：合并过多的小分组 -----
    if len(groups) > 20:
        merged_groups = []
        temp_group = []
        
        for group in groups:
            temp_group.extend(group)
            if len(temp_group) >= group_size * 2:
                merged_groups.append(temp_group)
                temp_group = []
        
        # 处理剩余的点
        if temp_group:
            merged_groups.append(temp_group)
        
        groups = merged_groups if merged_groups else groups
    
    return groups


# ==================== 车辆生成主函数 ====================

def spawn_vehicles(world, client, num_vehicles=100):
    """
    在地图上批量生成自动驾驶车辆
    
    功能说明:
        智能地在CARLA地图上生成指定数量的自动驾驶车辆，
        采用多种策略避免车辆碰撞：
        1. 按道路分组，每条道路最多1辆车
        2. 车辆间生成间隔1秒
        3. 交通管理器配置减速
        4. 生成点碰撞重试机制
    
    参数:
        world (carla.World): CARLA世界对象
        client (carla.Client): CARLA客户端对象
        num_vehicles (int): 要生成的车辆数量，默认100
    
    返回:
        list: 成功生成的车辆Actor列表
    
    生成策略:
        - 每条道路最多1辆车（避免路口拥堵）
        - 总车辆数上限40辆（保证流畅）
        - 每辆车间隔1秒生成
        - 车辆速度比限速慢25%
    
    使用场景:
        快速填充CARLA城市交通，用于测试自动驾驶算法
    
    示例:
        >>> vehicles = spawn_vehicles(world, client, num_vehicles=50)
        >>> print(f"成功生成 {len(vehicles)} 辆车")
    """
    blueprint_library = world.get_blueprint_library()
    
    # ----- 第一步：获取小汽车蓝图 -----
    vehicle_blueprints = get_small_car_blueprints(blueprint_library)
    if not vehicle_blueprints:
        print("❌ 没有找到合适的车辆蓝图！")
        return []
    
    # ----- 第二步：获取生成点 -----
    spawn_points = get_spawn_points(world)
    if not spawn_points:
        print("❌ 没有找到合适的生成点！")
        return []
    
    # 打印地图信息
    map_name = world.get_map().name
    print(f"🗺️  地图信息: {map_name}, spawn points数量: {len(spawn_points)}")
    
    # ----- 第三步：按道路分组生成点 -----
    # 每组最多3个生成点，实际每条道路只生成1辆车
    road_groups = group_spawn_points_by_road(spawn_points, group_size=3)
    print(f"🛣️  将生成点分为 {len(road_groups)} 个道路组")
    
    # ----- 第四步：限制总车辆数 -----
    # 为减少碰撞，每条道路最多1辆车，总车辆数不超过道路数
    max_total_vehicles = min(num_vehicles, len(road_groups), 40)
    if max_total_vehicles < num_vehicles:
        print(f"⚠️  为避免转弯直行碰撞，限制生成 {max_total_vehicles} 辆车（原计划 {num_vehicles} 辆）")
        num_vehicles = max_total_vehicles
    
    # ----- 第五步：配置交通管理器 -----
    traffic_manager = client.get_trafficmanager()
    try:
        # 全局减速20%，使行驶更安全
        traffic_manager.global_percentage_speed_difference(-20)
        # 忽略所有红绿灯，保证车辆持续行驶
        traffic_manager.ignore_lights_percentage(100)
    except Exception as e:
        print(f"[WARN] 交通管理器配置失败: {e}")
    
    # ----- 第六步：逐辆生成车辆 -----
    vehicles = []
    used_roads = set()  # 记录已使用的道路
    
    for i in range(num_vehicles):
        # 6.1 选择未使用的道路
        available_roads = [idx for idx in range(len(road_groups)) if idx not in used_roads]
        
        if not available_roads:
            # 所有道路都用完了，重新开始（但通常不会发生）
            used_roads.clear()
            available_roads = list(range(len(road_groups)))
        
        # 随机选择一个道路
        road_idx = random.choice(available_roads)
        used_roads.add(road_idx)
        current_group = road_groups[road_idx]
        
        # 6.2 在道路组中尝试生成车辆
        success = False
        for offset in range(min(20, len(current_group))):
            # 轮询组内的生成点
            actual_index = offset % len(current_group)
            spawn_point = current_group[actual_index]
            
            # 随机选择车辆型号
            blueprint = random.choice(vehicle_blueprints)
            
            # 使用生成点的位置和朝向（完全覆盖，不使用偏移）
            transform = carla.Transform(spawn_point.location, spawn_point.rotation)
            
            # 6.3 重试机制：最多尝试5次
            for attempt in range(5):
                try:
                    vehicle = world.spawn_actor(blueprint, transform)
                    
                    # 6.4 配置车辆参数
                    # 忽略红绿灯（确保持续行驶）
                    traffic_manager.ignore_lights_percentage(vehicle, 100)
                    # 比限速慢25%，行驶更平稳
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, -25)
                    
                    # 启用自动驾驶模式
                    vehicle.set_autopilot(True)
                    
                    # 车辆间延迟1秒，避免瞬时大量生成造成碰撞
                    time.sleep(1.0)
                    
                    vehicles.append(vehicle)
                    success = True
                    print(f"🚗 成功生成车辆 {len(vehicles)}/{num_vehicles}: {blueprint.id} (道路 {road_idx})")
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    # 碰撞错误：稍等重试
                    if "collision" in error_msg or "spawn" in error_msg:
                        time.sleep(0.5)
                        continue
                    else:
                        # 其他错误：打印后放弃
                        print(f"❌ 生成车辆 {len(vehicles)+1} 失败: {e}")
                        break
            
            if success:
                break
        
        # 6.5 处理连续失败
        if not success:
            print(f"❌ 生成车辆 {len(vehicles)+1} 失败: 无法找到合适的位置")
            # 连续失败3次，可能是地图容量已满，提前退出
            if len(vehicles) > 0 and len(vehicles) % 3 == 0:
                print(f"⚠️  已连续失败，可能是地图容量已满，当前成功生成 {len(vehicles)} 辆车")
                break
    
    return vehicles


# ==================== 主函数 ====================

def main():
    """
    主函数：程序入口
    
    执行流程:
        1. 连接CARLA服务器
        2. 获取世界对象
        3. 批量生成自动驾驶车辆
        4. 保持程序运行，等待用户中断
        5. 清理资源
    
    使用说明:
        1. 确保CARLA仿真器已启动
        2. 运行脚本: python carla_launcher.py
        3. 按 Ctrl+C 停止程序并清理车辆
    
    注意事项:
        - 车辆数量自动限制，避免过度占用资源
        - 所有车辆会自动沿道路行驶
        - 红绿灯被禁用，车辆持续行驶
    """
    print("=" * 60)
    print("CARLA 车辆生成器 - 启动")
    print("=" * 60)
    
    vehicles = []
    
    try:
        # ----- 1. 连接CARLA -----
        print("=== 连接诊断 ===")
        client = test_connection()
        if not client:
            return
        
        # ----- 2. 获取世界 -----
        world = client.get_world()
        print(f"🌍 当前地图: {world.get_map().name}")
        
        # ----- 3. 生成车辆 -----
        num_vehicles = 100  # 目标车辆数（实际会自适应调整）
        print(f"🚗 开始生成 {num_vehicles} 辆正常行驶车辆...")
        vehicles = spawn_vehicles(world, client, num_vehicles)
        
        if not vehicles:
            print("❌ 没有成功生成任何车辆！")
            return
        
        # ----- 4. 打印统计 -----
        print(f"✅ 成功生成 {len(vehicles)} 辆车辆")
        print("🎮 所有车辆已启用自动驾驶，沿道路行驶")
        print("🚦 红绿灯已禁用，车辆持续行驶")
        print("\n💡 提示：使用录屏软件录制视频，按 Ctrl+C 停止程序")
        
        # ----- 5. 保持主循环 -----
        while True:
            try:
                time.sleep(1)
                # 可以在此添加监控代码，如统计车辆数量
                # active_count = sum(1 for v in vehicles if v.is_alive)
                # print(f"活跃车辆: {active_count}/{len(vehicles)}")
                
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
        # ----- 6. 清理资源 -----
        print("🧹 清理车辆资源...")
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        print("✅ 完成！")


# ==================== 程序入口 ====================

if __name__ == "__main__":
    """
    当脚本被直接运行时（而不是作为模块导入），执行 main() 函数
    """
    main()