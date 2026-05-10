#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""航点导航入口（含降落后返航功能）"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from client.drone_client import DroneClient
from agents.waypoint_agent import WaypointAgent


def _print_wp_list(agent):
    """打印航点列表及当前模式"""
    print("\n  序号 |      坐标      | 降落 | 返航 | 飞行速度 | 下降速度 | 悬停时间")
    print("  " + "-" * 72)
    for i, wp in enumerate(agent.planner.waypoints):
        land = "●" if wp.is_landing else "○"
        home = "●" if wp.is_return_home else "○"
        print(f"  {i+1:>4} | ({wp.x:>5.1f},{wp.y:>5.1f},{wp.z:>5.1f}) |   {land}  |  {home}  | {wp.fly_speed:>7.1f}m/s | {wp.descend_speed:>7.1f}m/s | {wp.hover_time:>5.1f}s")


def waypoint_edit_menu(agent):
    """航点编辑菜单：配置模式、执行选择、开始飞行"""
    while True:
        _print_wp_list(agent)
        print("""
  操作选项:
    l <序号>   - 开启/关闭该航点的降落模式
    r <序号>   - 开启/关闭该航点的返航模式
    d <序号>   - 删除该航点
    a          - 继续添加航点
    s <序号> <速度> - 修改该航点的飞行速度 (m/s)
    v <序号> <速度> - 修改该航点的下降速度 (m/s)
    t <序号> <秒数> - 修改该航点的悬停时间 (s)
    exec        - 开始执行当前航点列表
    q           - 退出""")

        inp = input("\n操作> ").strip().lower()
        if not inp:
            continue

        parts = inp.split()
        cmd = parts[0]
        idx = None
        if len(parts) > 1:
            try:
                idx = int(parts[1]) - 1
            except ValueError:
                print("  无效的序号")
                continue

        if cmd == 'l':
            if idx is None or not (0 <= idx < len(agent.planner.waypoints)):
                print("  序号无效")
                continue
            wp = agent.planner.waypoints[idx]
            wp.is_landing = not wp.is_landing
            print(f"  WP{idx+1} 降落: {'开启' if wp.is_landing else '关闭'}")

        elif cmd == 'r':
            if idx is None or not (0 <= idx < len(agent.planner.waypoints)):
                print("  序号无效")
                continue
            wp = agent.planner.waypoints[idx]
            wp.is_return_home = not wp.is_return_home
            print(f"  WP{idx+1} 返航: {'开启' if wp.is_return_home else '关闭'}")

        elif cmd == 'd':
            if idx is None or not (0 <= idx < len(agent.planner.waypoints)):
                print("  序号无效")
                continue
            removed = agent.planner.waypoints.pop(idx)
            agent.navigator.trajectory = []
            print(f"  已删除 WP{idx+1}: ({removed.x}, {removed.y}, {removed.z})")

        elif cmd == 'a':
            print("  继续添加 (输入 'done' 返回编辑菜单)")
            agent.add_waypoint_interactive(show_help=False)

        elif cmd in ('s', 'v', 't'):
            if idx is None or len(parts) < 3:
                print("  格式: s <序号> <速度值>")
                continue
            try:
                val = float(parts[2])
            except ValueError:
                print("  无效的数值")
                continue
            if not (0 <= idx < len(agent.planner.waypoints)):
                print("  序号无效")
                continue
            wp = agent.planner.waypoints[idx]
            if cmd == 's':
                wp.fly_speed = max(0.1, val)
                print(f"  WP{idx+1} 飞行速度: {wp.fly_speed:.1f} m/s")
            elif cmd == 'v':
                wp.descend_speed = max(0.1, val)
                print(f"  WP{idx+1} 下降速度: {wp.descend_speed:.1f} m/s")
            elif cmd == 't':
                wp.hover_time = max(0.0, val)
                print(f"  WP{idx+1} 悬停时间: {wp.hover_time:.1f} s")

        elif cmd == 'exec':
            if not agent.planner.waypoints:
                print("  航点列表为空，请先添加航点")
                continue
            print("\n" + "=" * 50)
            return True  # 开始飞行

        elif cmd == 'q':
            return False  # 退出


def main():
    client = DroneClient(interval=0.05)

    print("\n=== 初始化 ===")
    print("正在连接 AirSim...")
    client.start()

    state = client.get_state()
    pos = state.kinematics_estimated.position
    print(f"起飞位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})")

    waypoints = []
    agent = WaypointAgent(
        client,
        waypoints=waypoints,
        reach_threshold=2.0,
        kp=1.0, ki=0.01, kd=0.5,
        max_vel=1.0
    )

    print("\n=== 自行设置航点 ===")
    print("输入格式: x y z")
    print("示例: 10 10 10")
    print("输入 'done' 完成添加")
    agent.add_waypoint_interactive()

    if not agent.planner.waypoints:
        print("未添加任何航点，退出。")
        return

    # 进入编辑/执行菜单
    if waypoint_edit_menu(agent):
        agent.run(loop=False)

    print("\n飞行结束。")


if __name__ == "__main__":
    main()
