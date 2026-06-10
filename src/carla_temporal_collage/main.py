import carla
import time

def main():
    HOST = "127.0.0.1"
    PORT = 2000
    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)
        world = client.get_world()
        print("CARLA客户端连接成功！")
        print("当前地图：", world.get_map().name)

        # 同步仿真配置
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 常驻等待退出
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n程序正常退出")
    except Exception as err:
        print("连接失败：", err)
        print("请先打开CarlaUE4.exe")

if __name__ == "__main__":
    main()