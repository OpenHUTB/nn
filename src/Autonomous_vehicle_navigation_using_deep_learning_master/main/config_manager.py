"""
配置管理器 - 封装config.py功能，管理CARLA模拟器配置
"""

import sys
import os
import glob
import re
import socket
import textwrap
import datetime
import time
# 添加CARLA路径
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class ConfigManager:
    """配置管理器 - 负责CARLA模拟器的配置"""
    
    def __init__(self, client=None, host='localhost', port=2000):
        """
        初始化配置管理器
        
        Args:
            client: 可选的CARLA客户端对象
            host: CARLA服务器主机
            port: CARLA服务器端口
        """
        if client:
            self.client = client
            self.world = client.get_world()
        else:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
        
        print("⚙️ 配置管理器初始化完成")
    
    def get_available_maps(self):
        """获取可用地图列表"""
        maps = [m.replace('/Game/Carla/Maps/', '') for m in self.client.get_available_maps()]
        return sorted(maps)
    
    def get_weather_presets(self):
        """获取天气预设列表"""
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), x) for x in presets]
    
    def get_available_blueprints(self, filter_pattern='*'):
        """获取可用蓝图列表"""
        blueprint_library = self.world.get_blueprint_library()
        blueprints = [bp.id for bp in blueprint_library.filter(filter_pattern)]
        return sorted(blueprints)
    
    def load_map(self, map_name):
        """
        加载地图
        
        Args:
            map_name: 地图名称
            
        Returns:
            bool: 是否成功
        """
        try:
            available_maps = self.get_available_maps()
            
            if map_name not in available_maps:
                print(f"❌ 地图 '{map_name}' 不存在")
                print(f"可用地图: {', '.join(available_maps)}")
                return False
            
            print(f"🗺️  加载地图: {map_name}")
            self.world = self.client.load_world(map_name)
            
            # 等待地图加载完成
            time.sleep(2.0)
            
            print(f"✅ 地图加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 加载地图失败: {e}")
            return False
    
    def set_weather(self, weather_preset):
        """
        设置天气
        
        Args:
            weather_preset: 天气预设名称
            
        Returns:
            bool: 是否成功
        """
        try:
            if not hasattr(carla.WeatherParameters, weather_preset):
                print(f"❌ 天气预设 '{weather_preset}' 不存在")
                return False
            
            weather = getattr(carla.WeatherParameters, weather_preset)
            self.world.set_weather(weather)
            
            print(f"☀️ 设置天气: {weather_preset}")
            return True
            
        except Exception as e:
            print(f"❌ 设置天气失败: {e}")
            return False
    
    def set_weather_custom(self, 
                          cloudiness=0.0,
                          precipitation=0.0,
                          precipitation_deposits=0.0,
                          wind_intensity=0.0,
                          sun_azimuth_angle=0.0,
                          sun_altitude_angle=75.0,
                          fog_density=0.0,
                          fog_distance=0.0,
                          wetness=0.0):
        """
        设置自定义天气
        
        Args:
            各种天气参数
            
        Returns:
            bool: 是否成功
        """
        try:
            weather = carla.WeatherParameters(
                cloudiness=cloudiness,
                precipitation=precipitation,
                precipitation_deposits=precipitation_deposits,
                wind_intensity=wind_intensity,
                sun_azimuth_angle=sun_azimuth_angle,
                sun_altitude_angle=sun_altitude_angle,
                fog_density=fog_density,
                fog_distance=fog_distance,
                wetness=wetness
            )
            
            self.world.set_weather(weather)
            
            print(f"🌤️  设置自定义天气")
            print(f"  云量: {cloudiness}%")
            print(f"  降水量: {precipitation}%")
            print(f"  雾密度: {fog_density}")
            
            return True
            
        except Exception as e:
            print(f"❌ 设置自定义天气失败: {e}")
            return False
    
    def set_fixed_fps(self, fps=20.0):
        """
        设置固定帧率
        
        Args:
            fps: 帧率 (0表示可变帧率)
            
        Returns:
            bool: 是否成功
        """
        try:
            settings = self.world.get_settings()
            
            if fps > 0:
                settings.fixed_delta_seconds = 1.0 / fps
                print(f"📊 设置固定帧率: {fps} FPS")
            else:
                settings.fixed_delta_seconds = None
                print("📊 设置可变帧率")
            
            self.world.apply_settings(settings)
            return True
            
        except Exception as e:
            print(f"❌ 设置帧率失败: {e}")
            return False
    
    def set_synchronous_mode(self, enabled=True, fixed_delta_seconds=0.05):
        """
        设置同步模式
        
        Args:
            enabled: 是否启用同步模式
            fixed_delta_seconds: 固定时间步长
            
        Returns:
            bool: 是否成功
        """
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = enabled
            
            if enabled:
                settings.fixed_delta_seconds = fixed_delta_seconds
                print(f"⏱️  启用同步模式，时间步长: {fixed_delta_seconds}s")
            else:
                print("⏱️  禁用同步模式")
            
            self.world.apply_settings(settings)
            return True
            
        except Exception as e:
            print(f"❌ 设置同步模式失败: {e}")
            return False
    
    def set_rendering_mode(self, enabled=True):
        """
        设置渲染模式
        
        Args:
            enabled: 是否启用渲染
            
        Returns:
            bool: 是否成功
        """
        try:
            settings = self.world.get_settings()
            settings.no_rendering_mode = not enabled
            self.world.apply_settings(settings)
            
            print(f"🎨 渲染模式: {'启用' if enabled else '禁用'}")
            return True
            
        except Exception as e:
            print(f"❌ 设置渲染模式失败: {e}")
            return False
    
    def set_streaming_distance(self, tile_distance=300.0, actor_distance=100.0):
        """
        设置流式加载距离
        
        Args:
            tile_distance: 贴图流式距离
            actor_distance: 演员活跃距离
            
        Returns:
            bool: 是否成功
        """
        try:
            settings = self.world.get_settings()
            settings.tile_stream_distance = tile_distance
            settings.actor_active_distance = actor_distance
            self.world.apply_settings(settings)
            
            print(f"📡 设置流式距离: 贴图={tile_distance}m, 演员={actor_distance}m")
            return True
            
        except Exception as e:
            print(f"❌ 设置流式距离失败: {e}")
            return False
    
    def inspect_simulation(self):
        """检查模拟器状态"""
        try:
            address = f'{self.client.host}:{self.client.port}'
            elapsed_time = self.world.get_snapshot().timestamp.elapsed_seconds
            elapsed_time = datetime.timedelta(seconds=int(elapsed_time))
            
            actors = self.world.get_actors()
            settings = self.world.get_settings()
            
            # 获取当前天气
            weather = 'Custom'
            current_weather = self.world.get_weather()
            for preset, name in self.get_weather_presets():
                if current_weather == preset:
                    weather = name
            
            # 获取帧率
            if settings.fixed_delta_seconds is None:
                frame_rate = 'variable'
            else:
                fps = 1.0 / settings.fixed_delta_seconds
                frame_rate = f'{settings.fixed_delta_seconds*1000:.2f} ms ({fps:.0f} FPS)'
            
            # 打印信息
            print("\n" + "="*60)
            print("CARLA模拟器状态检查")
            print("="*60)
            print(f"地址:     {address:>30}")
            print(f"版本:     {self.client.get_server_version():>30}")
            print(f"地图:     {self.world.get_map().name:>30}")
            print(f"天气:     {weather:>30}")
            print(f"运行时间: {elapsed_time:>30}")
            print(f"帧率:     {frame_rate:>30}")
            print(f"渲染:     {'禁用' if settings.no_rendering_mode else '启用':>30}")
            print(f"同步模式: {'禁用' if not settings.synchronous_mode else '启用':>30}")
            print(f"\n演员统计:")
            print(f"  总演员数: {len(actors):>25}")
            print(f"  观察者:   {len(actors.filter('spectator')):>25}")
            print(f"  静态物体: {len(actors.filter('static.*')):>25}")
            print(f"  交通标志: {len(actors.filter('traffic.*')):>25}")
            print(f"  车辆:     {len(actors.filter('vehicle.*')):>25}")
            print(f"  行人:     {len(actors.filter('walker.*')):>25}")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"❌ 检查模拟器状态失败: {e}")
            return False
    
    def apply_default_settings(self):
        """应用默认设置"""
        print("\n⚙️ 应用默认设置...")
        
        settings_applied = []
        
        # 启用渲染
        if self.set_rendering_mode(enabled=True):
            settings_applied.append("渲染")
        
        # 禁用同步模式（提高性能）
        if self.set_synchronous_mode(enabled=False):
            settings_applied.append("同步模式")
        
        # 设置默认天气
        if self.set_weather("ClearNoon"):
            settings_applied.append("天气")
        
        # 设置固定时间步长
        try:
            import config as cfg
            if self.set_fixed_fps(fps=1/cfg.FIXED_DELTA_SECONDS):
                settings_applied.append(f"帧率({1/cfg.FIXED_DELTA_SECONDS:.1f}FPS)")
        except:
            if self.set_fixed_fps(fps=0):
                settings_applied.append("帧率(可变)")
        
        # 设置流式距离
        if self.set_streaming_distance(tile_distance=300.0, actor_distance=100.0):
            settings_applied.append("流式距离")
        
        if settings_applied:
            print(f"✅ 已应用设置: {', '.join(settings_applied)}")
        else:
            print("⚠️ 未应用任何设置")
        
        return len(settings_applied) > 0
    
    def get_current_settings(self):
        """获取当前设置"""
        settings = self.world.get_settings()
        
        current_weather = self.world.get_weather()
        weather_name = 'Custom'
        
        for preset, name in self.get_weather_presets():
            if current_weather == preset:
                weather_name = name
                break
        
        return {
            'map': self.world.get_map().name,
            'weather': weather_name,
            'synchronous_mode': settings.synchronous_mode,
            'no_rendering': settings.no_rendering_mode,
            'fixed_delta_seconds': settings.fixed_delta_seconds,
            'fps': 1.0 / settings.fixed_delta_seconds if settings.fixed_delta_seconds else 0,
            'tile_stream_distance': settings.tile_stream_distance,
            'actor_active_distance': settings.actor_active_distance
        }