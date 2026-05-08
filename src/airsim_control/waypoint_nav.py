"""
航点导航主入口（PID控制 + 3D实时可视化）
用法:
  python waypoint_nav.py                      # 交互式添加航点
  python waypoint_nav.py --waypoints 10,0,-5 10,10,-5 0,10,-10  # 命令行指定
  python waypoint_nav.py --loop              # 循环模式
  python waypoint_nav.py --no-visual        # 关闭可视化
"""
import argparse
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from client.drone_client import DroneClient
from agents.waypoint_agent import WaypointAgent


def parse_waypoints(wp_strings):
    """解析航点参数 'x,y,z' 格式，z为高度（向上为正）"""
    waypoints = []
    for s in wp_strings:
        try:
            x, y, z = map(float, s.split(','))
            waypoints.append((x, y, z))  # 直接存储，z表示高度
        except ValueError:
            print(f"警告: 忽略无效航点 '{s}'，格式应为 x,y,z")
    return waypoints


def main():
    parser = argparse.ArgumentParser(description='航点导航 - PID控制 + 3D可视化')
    parser.add_argument('--waypoints', nargs='*', default=[],
                        help='航点列表，格式: x,y,z x,y,z ...')
    parser.add_argument('--loop', action='store_true', help='循环模式')
    parser.add_argument('--reach-threshold', type=float, default=0.5,
                        help='到达阈值（米），默认0.5')
    parser.add_argument('--kp', type=float, default=2.0, help='P参数')
    parser.add_argument('--ki', type=float, default=0.02, help='I参数')
    parser.add_argument('--kd', type=float, default=1.0, help='D参数')
    parser.add_argument('--max-vel', type=float, default=1.5, help='最大速度')
    parser.add_argument('--no-visual', action='store_true', help='关闭3D可视化')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='可视化更新间隔（秒）')
    args = parser.parse_args()

    # 解析航点
    waypoints = parse_waypoints(args.waypoints)

    print("=== 航点导航系统 ===\n")
    print(f"PID参数: kp={args.kp}, ki={args.ki}, kd={args.kd}")
    print(f"最大速度: {args.max_vel}m/s")
    print(f"到达阈值: {args.reach_threshold}m")
    print(f"循环模式: {'是' if args.loop else '否'}")
    print(f"3D可视化: {'否' if args.no_visual else '是'}\n")

    # 连接无人机
    print("连接无人机...")
    client = DroneClient(interval=0.1, root_path='./')

    # 起飞
    print("无人机起飞中...")
    client.start()
    import time
    time.sleep(3)  # 等待起飞稳定

    # 检查起飞是否成功
    pos = client.get_state().kinematics_estimated.position
    print(f"起飞后位置: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f} (NED)")
    print(f"当前高度约: {-pos.z_val:.1f}m")
    if -pos.z_val < 1.0:
        print("警告: 起飞可能失败，无人机高度过低！")
    else:
        print("起飞成功!\n")

    # 创建Agent
    agent = WaypointAgent(
        client=client,
        waypoints=waypoints if waypoints else None,
        reach_threshold=args.reach_threshold,
        kp=args.kp, ki=args.ki, kd=args.kd,
        max_vel=args.max_vel,
        update_interval=args.interval
    )

    # 如果没有航点，交互式添加
    if not waypoints:
        agent.add_waypoint_interactive()

    if not agent.planner.waypoints:
        print("未添加航点，退出")
        return

    print(f"\n航点列表:")
    for i, wp in enumerate(agent.planner.waypoints):
        print(f"  {i+1}. ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")

    # 等待用户确认
    input("\n按Enter开始导航，Ctrl+C取消...")

    # 运行导航
    agent.run(loop=args.loop)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
