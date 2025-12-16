# cvips_smart_final.py
"""
CVIPS 智能版本 - 不再关闭运行的CARLA，并修复所有问题
"""

import sys
import os
import time
import random  # 添加这行
import traceback
from datetime import datetime

print("=" * 70)
print("CVIPS 数据生成器 - 智能版本")
print("=" * 70)

# ============================================================
# 1. 设置CARLA路径
# ============================================================
print("\n[1/5] 设置CARLA路径...")
CARLA_EGG = r"D:\carla\carla0914\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.14-py3.7-win-amd64.egg"

if os.path.exists(CARLA_EGG):
    sys.path.append(CARLA_EGG)
    print(f"✓ CARLA路径: {os.path.basename(CARLA_EGG)}")
else:
    print(f"✗ 找不到egg文件: {CARLA_EGG}")
    sys.exit(1)

# ============================================================
# 2. 导入CARLA
# ============================================================
print("\n[2/5] 导入CARLA模块...")
try:
    import carla

    print("✓ CARLA模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)


# ============================================================
# 3. 智能连接CARLA服务器
# ============================================================
def smart_connect_to_carla(max_retries=10, retry_delay=3):
    """智能连接CARLA服务器，不关闭已有服务器"""
    print(f"\n[3/5] 连接到CARLA服务器 (最多尝试{max_retries}次)...")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  尝试 {attempt}/{max_retries}...")

            # 创建客户端
            client = carla.Client('localhost', 2000)
            client.set_timeout(15.0)

            # 获取服务器版本
            server_version = client.get_server_version()
            print(f"  ✓ 连接成功! 服务器版本: {server_version}")

            # 获取世界
            world = client.get_world()
            print(f"  ✓ 地图: {world.get_map().name}")

            return client, world

        except Exception as e:
            error_msg = str(e)
            print(f"  尝试 {attempt} 失败: {error_msg[:80]}...")

            # 给出具体建议
            if "time-out" in error_msg:
                if attempt == 1:
                    print(f"  ℹ 请确保CARLA服务器正在运行")
                    print(f"  ℹ 如果CARLA正在启动中，请等待几秒钟")
                elif attempt == 3:
                    print(f"  ℹ 如果CARLA窗口无响应，请尝试在窗口中点击一下")

            if attempt < max_retries:
                print(f"  等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"\n✗ 所有连接尝试失败")
                print(f"\n请检查:")
                print(f"1. CARLA服务器是否正在运行 (应该能看到3D窗口)")
                print(f"2. CARLA窗口是否在前台 (尝试点击一下CARLA窗口)")
                print(f"3. 如果CARLA刚启动，可能需要更多时间加载")
                return None, None

    return None, None


# 智能连接
client, world = smart_connect_to_carla()

if not client or not world:
    print("\n" + "=" * 70)
    print("连接失败！")
    print("=" * 70)
    sys.exit(1)

# ============================================================
# 4. 创建简单场景
# ============================================================
print("\n[4/5] 创建数据收集场景...")

try:
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"cvips_data/success_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"输出目录: {output_dir}")

    # 保存配置
    with open(f"{output_dir}/config.txt", "w") as f:
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"地图: {world.get_map().name}\n")

    # 设置异步模式（更稳定）
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    # 设置简单天气
    weather = carla.WeatherParameters(
        sun_altitude_angle=90,
        cloudiness=0,
        precipitation=0,
        fog_density=0
    )
    world.set_weather(weather)
    print("✓ 天气设置完成")

    # 生成车辆
    blueprint_lib = world.get_blueprint_library()

    # 选择简单车辆
    vehicle_bp = None
    vehicle_types = [
        'vehicle.tesla.model3',
        'vehicle.audi.tt',
        'vehicle.nissan.micra',
        'vehicle.mini.cooperst'
    ]

    for vtype in vehicle_types:
        if blueprint_lib.filter(vtype):
            vehicle_bp = random.choice(blueprint_lib.filter(vtype))
            break

    if not vehicle_bp:
        vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))

    # 获取生成点
    spawn_points = world.get_map().get_spawn_points()
    if spawn_points:
        spawn_point = random.choice(spawn_points)
        print(f"使用生成点: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

        # 生成车辆
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"✓ 生成车辆: {vehicle.type_id}")

        # 设置自动驾驶
        vehicle.set_autopilot(True)

        # 添加简单摄像头
        camera_bp = blueprint_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')

        # 前摄像头
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=1.4),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # 图像保存回调
        frame_count = [0]


        def save_image(image):
            frame_count[0] += 1
            if frame_count[0] <= 30:  # 只保存30张
                image.save_to_disk(f"{output_dir}/frame_{frame_count[0]:03d}.png")
                if frame_count[0] % 10 == 0:
                    print(f"    已保存 {frame_count[0]}/30 帧")


        camera.listen(save_image)
        print("✓ 摄像头已安装")

        # 收集数据
        print("\n[5/5] 收集数据 (15秒)...")
        print("按 Ctrl+C 可提前结束")

        start_time = time.time()
        try:
            for i in range(15):
                print(f"  进度: {i + 1}/15 秒")
                time.sleep(1.0)

            print(f"\n✓ 数据收集完成!")
            print(f"  总帧数: {frame_count[0]}")
            print(f"  数据保存到: {output_dir}")

        except KeyboardInterrupt:
            print(f"\n数据收集中断，已保存 {frame_count[0]} 帧")

        # 清理
        print("\n清理场景...")
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        print("✓ 场景已清理")

    else:
        print("⚠ 没有找到生成点，跳过车辆生成")

except Exception as e:
    print(f"✗ 创建场景时出错: {e}")
    traceback.print_exc()

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 70)
print("🎉 CVIPS 数据生成完成！")
print("=" * 70)
print(f"CARLA服务器仍在运行，可以继续使用")
print("=" * 70)

input("\n按Enter键退出...")