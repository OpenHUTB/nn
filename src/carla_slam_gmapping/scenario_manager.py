#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA场景管理器
功能：切换CARLA地图，支持不同场景的重复实验
"""

import rospy
import carla
import sys
import time


class ScenarioManager:
    def __init__(self, host='localhost', port=2000, timeout=10.0):
        """
        初始化场景管理器
        
        Args:
            host: CARLA服务器地址
            port: CARLA服务器端口
            timeout: 连接超时时间
        """
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            self.world = self.client.get_world()
            print(f"Connected to CARLA server at {host}:{port}")
            print(f"Current map: {self.world.get_map().name}")
        except Exception as e:
            print(f"Failed to connect to CARLA: {e}")
            sys.exit(1)
    
    def list_available_maps(self):
        """列出所有可用的地图"""
        maps = self.client.get_available_maps()
        print("\n=== Available Maps ===")
        for i, map_name in enumerate(maps, 1):
            print(f"{i}. {map_name}")
        return maps
    
    def switch_map(self, map_name):
        """
        切换地图
        
        Args:
            map_name: 地图名称（如 'Town01', 'Town02' 等）
        """
        print(f"\nSwitching to map: {map_name}")
        try:
            # 如果不包含完整路径，自动补全
            if not map_name.startswith('/Game/Carla/Maps'):
                map_name = f'/Game/Carla/Maps/{map_name}'
            
            self.world = self.client.load_world(map_name)
            print(f"Map switched successfully!")
            print(f"Current map: {self.world.get_map().name}")
            time.sleep(2)  # 等待地图加载完成
            return True
        except Exception as e:
            print(f"Failed to switch map: {e}")
            return False
    
    def get_spawn_points(self):
        """获取当前地图的所有出生点"""
        spawn_points = self.world.get_map().get_spawn_points()
        print(f"\n=== Spawn Points ({len(spawn_points)} total) ===")
        for i, point in enumerate(spawn_points[:10], 1):  # 只显示前10个
            loc = point.location
            print(f"{i}. x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}, yaw={point.rotation.yaw:.2f}")
        if len(spawn_points) > 10:
            print(f"... and {len(spawn_points) - 10} more")
        return spawn_points
    
    def set_weather(self, weather_preset='ClearNoon'):
        """
        设置天气
        
        Args:
            weather_preset: 天气预设名称
        """
        weather_presets = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
            'CloudySunset': carla.WeatherParameters.CloudySunset,
            'WetSunset': carla.WeatherParameters.WetSunset,
        }
        
        if weather_preset in weather_presets:
            self.world.set_weather(weather_presets[weather_preset])
            print(f"Weather set to: {weather_preset}")
        else:
            print(f"Unknown weather preset: {weather_preset}")
            print(f"Available presets: {', '.join(weather_presets.keys())}")


def main():
    if len(sys.argv) < 2:
        print("\n=== Scenario Manager Usage ===")
        print("1. List available maps:")
        print("   python3 scenario_manager.py list")
        print("\n2. Switch to a specific map:")
        print("   python3 scenario_manager.py switch <map_name>")
        print("   Example: python3 scenario_manager.py switch Town02")
        print("\n3. Show spawn points:")
        print("   python3 scenario_manager.py spawns")
        print("\n4. Set weather:")
        print("   python3 scenario_manager.py weather <preset>")
        print("   Example: python3 scenario_manager.py weather CloudyNoon")
        print("\n5. Interactive mode:")
        print("   python3 scenario_manager.py interactive")
        return
    
    try:
        manager = ScenarioManager()
        
        command = sys.argv[1].lower()
        
        if command == "list":
            manager.list_available_maps()
        
        elif command == "switch":
            if len(sys.argv) < 3:
                print("Error: Please specify map name")
                print("Example: python3 scenario_manager.py switch Town02")
            else:
                manager.switch_map(sys.argv[2])
        
        elif command == "spawns":
            manager.get_spawn_points()
        
        elif command == "weather":
            if len(sys.argv) < 3:
                print("Error: Please specify weather preset")
            else:
                manager.set_weather(sys.argv[2])
        
        elif command == "interactive":
            print("\n=== Interactive Scenario Manager ===")
            manager.list_available_maps()
            
            while True:
                print("\nCommands:")
                print("  switch <map>  - Switch to map")
                print("  spawns        - Show spawn points")
                print("  weather <preset> - Set weather")
                print("  list          - List maps")
                print("  q             - Quit")
                
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue
                
                if cmd[0] == 'q':
                    break
                elif cmd[0] == 'switch' and len(cmd) > 1:
                    manager.switch_map(cmd[1])
                elif cmd[0] == 'spawns':
                    manager.get_spawn_points()
                elif cmd[0] == 'weather' and len(cmd) > 1:
                    manager.set_weather(cmd[1])
                elif cmd[0] == 'list':
                    manager.list_available_maps()
                else:
                    print("Unknown command or missing argument")
        
        else:
            print(f"Unknown command: {command}")
            print("Run without arguments to see usage")
    
    except KeyboardInterrupt:
        print("\nScenario Manager interrupted")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
