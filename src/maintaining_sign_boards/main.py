import carla
import random
import time

# 主函数
def main():
    actor_list = []
    try:
        # 连接CARLA模拟器
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        print("连接CARLA模拟器成功")

    finally:
        # 清理Actor
        print("清理actors")
        for actor in actor_list:
            actor.destroy()
        print("完成.")

if __name__ == "__main__":
    main()