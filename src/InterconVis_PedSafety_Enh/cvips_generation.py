import carla
import argparse
import time

# 完整代码：参数解析+CARLA连接+地图加载
if __name__ == "__main__":
    # 1. 解析命令行参数（带输入校验）
    parser = argparse.ArgumentParser(description='CVIPS场景数据生成工具')
    parser.add_argument('--town', type=str, required=True, choices=['Town04', 'Town10HD', 'Town07'], 
                        help='CARLA城镇地图（支持：Town04/Town10HD/Town07）')
    parser.add_argument('--intersection', type=str, required=True, choices=['3way', '4way'], 
                        help='路口类型（3way=三叉路口，4way=四岔路口）')
    parser.add_argument('--weather', type=str, required=True, choices=['clear', 'rainy', 'cloudy'], 
                        help='天气条件（clear=晴天，rainy=雨天，cloudy=阴天）')
    parser.add_argument('--time_of_day', type=str, required=True, choices=['noon', 'sunset', 'night'], 
                        help='时段（noon=中午，sunset=日落，night=夜晚）')
    args = parser.parse_args()

    try:
        # 2. 连接CARLA服务器（设置超时重试机制）
        client = carla.Client('localhost', 2000)
        client.set_timeout(15.0)
        client.load_world(args.town)  # 加载指定城镇
        world = client.get_world()
        print(f"✅ 成功连接CARLA并加载地图：{args.town}")
    except Exception as e:
        print(f"❌ 连接失败：{str(e)}，请检查CARLA服务器是否启动")