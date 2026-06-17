import carla
import random
from queue import Queue

SpawnActor = carla.command.SpawnActor

weather_presets = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.MidRainyNoon,
    # carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.SoftRainNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.CloudySunset,
    carla.WeatherParameters.WetSunset,
    carla.WeatherParameters.WetCloudySunset,
    carla.WeatherParameters.MidRainSunset,
    # carla.WeatherParameters.HardRainSunset,
    carla.WeatherParameters.SoftRainSunset
]

motorcycle_list = ["vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja", "vehicle.vespa.zx125", "vehicle.yamaha.yzf"]
bike_list = ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]
emergency_list = ["vehicle.dodge.charger_police", "vehicle.dodge.charger_police_2020", "vehicle.carlamotors.firetruck", "vehicle.ford.ambulance"]

def spawn_actors(client, world, num_vehicles, num_walkers):
    bp_lib = world.get_blueprint_library()

    actor_list = []
    walkers_list = []
    all_id = []

    # spawn vehicle
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    # Get the map spawn points
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    # 修复：在调用方法前检查 vehicle 是否生成成功
    if vehicle is None:
        # 抛出异常，让外层的 try-except 捕获
        raise RuntimeError("主车辆生成失败，生成点可能被占用") 
    
    vehicle.set_autopilot(True)
    # collect all actors to destroy when we quit the script
    actor_list.append(vehicle)

    # generate npc vehicle
    vehicle_bps = [x for x in bp_lib.filter('*vehicle*') if int(x.get_attribute('number_of_wheels')) == 4]
    for i in range(num_vehicles):
        vehicle_npc = random.choice(vehicle_bps)
        # vehicle_npc = bp_lib.find('vehicle.carlamotors.firetruck')
        npc = world.try_spawn_actor(vehicle_npc, random.choice(spawn_points))

        # set light state
        # light_state = carla.VehicleLightState(carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2)
        if npc:
            npc.set_autopilot(True)
            # apply light state
            # npc.set_light_state(light_state)
            actor_list.append(npc)
    
    # spawn pedestrians
    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    percentagePedestriansRunning = 0.0      # how many pedestrians will run
    percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(num_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            pass
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
            # print("Spawned pedestrian")
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            pass
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put together the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
    
    return actor_list, walkers_list, all_id


# 新增 camera_mode='front' 参数
def spawn_sensors(world, vehicle, inference=False, camera_mode='front'):
    bp_lib = world.get_blueprint_library()
    sensor_list = []
    q_list = []
    sensor_idxs = {}

    idx = 0
    tick_queue = Queue()
    world.on_tick(tick_queue.put)
    q_list.append(tick_queue)
    sensor_idxs['tick'] = idx
    idx = idx+1

    # 新增：根据模式动态计算传感器的生成坐标和俯仰角
    if camera_mode == 'bev':
        # BEV俯视模式：位于车顶正上方 15 米，镜头垂直向下转动 -90 度
        sensor_init_trans = carla.Transform(
            carla.Location(x=0.0, z=15.0), 
            carla.Rotation(pitch=-90.0)
        )
    else:
        # 默认前视模式 (Dashcam)
        sensor_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
    # Spawn RGB camera
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('bloom_intensity','1')
    camera_bp.set_attribute('fov','100')
    camera = world.spawn_actor(camera_bp, sensor_init_trans, attach_to=vehicle)
    
    cam_queue = Queue()
    camera.listen(cam_queue.put)
    sensor_list.append(camera)
    q_list.append(cam_queue)
    sensor_idxs['rgb'] = idx
    idx = idx+1
    print('RGB camera ready')

    if not inference:
        # Spawn depth camera
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute('fov','100')
        depth = world.spawn_actor(depth_bp, sensor_init_trans, attach_to=vehicle)

        depth_queue = Queue()
        depth.listen(depth_queue.put)
        sensor_list.append(depth)
        q_list.append(depth_queue)
        sensor_idxs['depth'] = idx
        idx = idx+1
        print('Depth camera ready')

        # Spawn LIDAR sensor
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '1120000')
        lidar_bp.set_attribute('upper_fov', '40')
        lidar_bp.set_attribute('lower_fov', '-40')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('rotation_frequency', '40')
        lidar = world.spawn_actor(lidar_bp, sensor_init_trans, attach_to=vehicle)

        lidar_queue = Queue()
        lidar.listen(lidar_queue.put)
        sensor_list.append(lidar)
        q_list.append(lidar_queue)
        sensor_idxs['lidar'] = idx
        idx = idx+1
        print('LIDAR ready')

        # Spawn semantic segmentation camera
        semantic_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('fov','100')
        semantic = world.spawn_actor(semantic_bp, sensor_init_trans, attach_to=vehicle)

        semantic_queue = Queue()
        semantic.listen(semantic_queue.put)
        sensor_list.append(semantic)
        q_list.append(semantic_queue)
        sensor_idxs['semantic'] = idx
        idx = idx+1
        print('Semantic camera ready')

    return sensor_list, q_list, sensor_idxs
