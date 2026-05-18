#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import sys

try:
    import carla
except ImportError:
    print("正在加载 CARLA 库...")
    try:
        sys.path.append("D:/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla/dist/")
        import carla
    except:
        print("错误：无法找到 CARLA 库，请检查你的CARLA安装路径！")
        exit()

import argparse


def main():
    argparser = argparse.ArgumentParser(description="CARLA 录制文件阻塞车辆检测工具")
    argparser.add_argument('--host', default='127.0.0.1', help='CARLA 服务器地址')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='CARLA 端口号')
    argparser.add_argument('-f', '--recorder_filename', default="test1.rec", help='录制文件名')
    argparser.add_argument('-t', '--time', default=10.0, type=float, help='判定阻塞的最小时间（秒）')
    argparser.add_argument('-d', '--distance', default=50.0, type=float, help='判定静止的最小距离（厘米）')
    
    args = argparser.parse_args()

    try:
        # 连接 CARLA 服务器
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        print(f"正在读取录制文件：{args.recorder_filename}")
        print("=" * 50)
        
        # 执行阻塞检测
        result = client.show_recorder_actors_blocked(
            args.recorder_filename, 
            args.time, 
            args.distance
        )
        
        print(result)

    except Exception as e:
        print(f"运行出错：{str(e)}")
        print("\n请确保：")
        print("1. CARLA 已启动")
        print("2. 已生成 test1.rec 文件")
    finally:
        print('\n✅ 运行完成！')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n❌ 用户停止运行。')
    finally:
        pass