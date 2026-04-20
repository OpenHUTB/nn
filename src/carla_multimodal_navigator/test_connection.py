# --------------------------
# 测试连接到CARLA服务器
# --------------------------
import carla


def test_connection():
    """测试连接到CARLA服务器"""
    print("测试连接到CARLA服务器...")

    try:
        # 尝试连接
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        print("连接成功！")
        
        # 获取服务器版本
        server_version = client.get_server_version()
        print(f"服务器版本: {server_version}")
        
        # 获取客户端版本
        client_version = client.get_client_version()
        print(f"客户端版本: {client_version}")
        
        # 检查版本是否匹配
        if server_version != client_version:
            print("警告：版本不匹配！")
            print("这可能会导致兼容性问题")
        else:
            print("版本匹配，连接正常")

        # 尝试获取地图信息
        try:
            world = client.get_world()
            map_name = world.get_map().name
            print(f"当前地图: {map_name}")
        except Exception as e:
            print(f"获取地图信息时出错: {e}")

    except Exception as e:
        print(f"连接失败: {e}")
        print("请确保:")
        print("1. CARLA服务器正在运行")
        print("2. 服务器端口为2000")


if __name__ == "__main__":
    test_connection()
