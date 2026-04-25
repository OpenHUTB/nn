import carla
import time

print("。。。 正在持续等待CARLA仿真器连接...")

# 官方原生标准连接，彻底解决之前IP连接报错
while True:
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        print("。。。 CARLA 服务端连接成功！！！")
        break
    except Exception as e:
        print(f"。。。 连接重试中... 报错：{e}")
        time.sleep(1)

# 固定生成辨识度最高的特斯拉Model3车辆
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

# 出生点精准固定在【你当前画面正前方这条主干道】，车辆直接刷在你视野里
spawn_points = world.get_map().get_spawn_points()
spawn_point = spawn_points[86]

# 生成车辆 + 开启CARLA官方原生自动驾驶
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)
print("。。。车辆生成完成！自动驾驶已启动！")

# 程序保持后台运行，只维持车辆自动驾驶，不再改动任何游戏视角
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # 终端按 Ctrl+C 安全停止，自动销毁车辆
    vehicle.destroy()
    print("\n 。。。程序已安全停止，车辆已销毁")
