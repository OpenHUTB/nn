import carla
import time

def debug_traffic_light():
    # 连接到模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # 获取所有交通灯
    actors = world.get_actors()
    traffic_lights = actors.filter('traffic.traffic_light')
    
    print(f"找到 {len(traffic_lights)} 个交通灯")
    
    # 打印每个交通灯的状态
    for tl in traffic_lights:
        state = tl.get_state()
        state_name = str(state).split('.')[-1]
        print(f"交通灯 {tl.id}: {state_name}")
    
    # 实时监控交通灯状态变化
    print("\n开始监控交通灯状态（按Ctrl+C退出）")
    try:
        while True:
            world.tick()
            for tl in traffic_lights:
                state = tl.get_state()
                state_name = str(state).split('.')[-1]
                print(f"\r交通灯 {tl.id}: {state_name}", end='')
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n监控结束")

if __name__ == "__main__":
    debug_traffic_light()