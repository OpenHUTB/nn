import carla
import pygame
import numpy as np
import random
import sys
import queue

# 尝试导入 CARLA 的官方导航模块
try:
    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    print("错误: 找不到 'agents' 模块。请务必将此脚本放在 CARLA 的 PythonAPI/examples/ 目录下运行！")
    sys.exit(1)

class CarlaDisplay:
    """处理 Pygame 窗口渲染和摄像头数据接收"""
    def __init__(self, width=800, height=600):
        pygame.init()
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Real-time Navigation")
        self.font = pygame.font.SysFont('mono', 16, bold=True)
        self.image_queue = queue.Queue()

    def process_camera_data(self, image):
        """将 CARLA 传感器图像转换为 Pygame Surface"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1] # BGR 转 RGB
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.image_queue.put(surface)

    def render(self, vehicle):
        """在屏幕上渲染画面和 HUD 数据"""
        try:
            # 获取最新图像
            surface = self.image_queue.get(block=False)
            self.display.blit(surface, (0, 0))
        except queue.Empty:
            pass

        # 获取车辆遥测数据
        v = vehicle.get_velocity()
        speed = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)) # km/h
        control = vehicle.get_control()
        transform = vehicle.get_transform()
        loc = transform.location

        # 准备 HUD 文本
        info_text = [
            f"Vehicle: Ford Ambulance",
            f"Speed:   {speed} km/h",
            f"Loc:     ({loc.x:.1f}, {loc.y:.1f})",
            f"Throttle:{control.throttle:.2f}",
            f"Steer:   {control.steer:.2f}",
            f"Brake:   {control.brake:.2f}"
        ]

        # 绘制半透明黑色背景框
        overlay = pygame.Surface((250, 180))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        self.display.blit(overlay, (10, 10))

        # 渲染文本
        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.display.blit(text_surface, (20, 20 + i * 25))

        pygame.display.flip()

def main():
    actor_list = []
    pygame_display = CarlaDisplay(width=800, height=600)

    # 1. 连接 CARLA
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    try:
        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 2. 生成福特救护车
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.ford.ambulance')
        spawn_points = carla_map.get_spawn_points()
        start_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, start_point)
        actor_list.append(vehicle)

        # 3. 安装 RGB 摄像头 (用于 Pygame 显示)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        # 将摄像头放置在车辆后方，模拟第三人称视角
        camera_transform = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-15.0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(pygame_display.process_camera_data)

        # 4. 路线规划与画线 (青色导航线)
        destination_point = random.choice(spawn_points).location
        grp = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
        route = grp.trace_route(start_point.location, destination_point)

        for i in range(len(route) - 1):
            w1 = route[i][0].transform.location + carla.Location(z=0.5)
            w2 = route[i+1][0].transform.location + carla.Location(z=0.5)
            world.debug.draw_line(w1, w2, thickness=0.2, color=carla.Color(0, 255, 255), life_time=1000.0)

        # 5. 初始化自动驾驶 Agent
        agent = BasicAgent(vehicle, target_speed=30)
        agent.set_destination(destination_point)

        print("开始导航！请在弹出的 Pygame 窗口中观看。按 ESC 退出。")

        # 6. 主循环
        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            world.tick()

            # 处理 Pygame 事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            # Agent 控制逻辑
            if agent.done():
                print("到达目的地！")
                break
            control = agent.run_step()
            vehicle.apply_control(control)

            # 更新 UI 渲染
            pygame_display.render(vehicle)

    finally:
        print("清理世界...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()