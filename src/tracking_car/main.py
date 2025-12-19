# main.py
import argparse
import carla
import queue
import random
import cv2
import numpy as np
import time
import os
import sys
import yaml
import torch
import open3d as o3d
import psutil
from dataclasses import dataclass
from datetime import datetime

# ======================== 导入简化后的核心模块 ========================
from core import SORT, DetThread, load_yolo_model
from sensors import CameraSensor, LiDARSensor
from utils import (
    setup_logger, clear_actors, spawn_npcs, safe_spawn_ego,
    WeatherEnhancer, draw, FrameBuf, FPS, Recorder, valid_img
)

# ======================== 配置类（从YAML加载） ========================
@dataclass
class Config:
    # 基础网络配置
    host: str = "localhost"
    port: int = 2000
    num_npcs: int = 20
    
    # 图像配置
    img_width: int = 640
    img_height: int = 480
    
    # 检测配置
    conf_thres: float = 0.5
    iou_thres: float = 0.3
    yolo_model: str = "yolov8n.pt"
    yolo_imgsz_max: int = 320
    yolo_iou: float = 0.45
    yolo_quantize: bool = False
    
    # 跟踪配置
    max_age: int = 5
    min_hits: int = 3
    kf_dt: float = 0.05
    max_speed: float = 50.0
    
    # 可视化配置
    window_width: int = 1280
    window_height: int = 720
    smooth_alpha: float = 0.2
    fps_window_size: int = 15
    display_fps: int = 30
    track_history_len: int = 20
    track_line_width: int = 2
    track_alpha: float = 0.6
    
    # 行为分析配置
    stop_speed_thresh: float = 1.0
    stop_frames_thresh: int = 5
    overtake_speed_ratio: float = 1.5
    overtake_dist_thresh: float = 50.0
    lane_change_thresh: float = 0.5
    brake_accel_thresh: float = 2.0
    turn_angle_thresh: float = 15.0
    danger_dist_thresh: float = 10.0
    predict_frames: int = 10
    
    # 环境配置
    default_weather: str = "clear"
    auto_adjust_detection: bool = True
    
    # LiDAR配置
    use_lidar: bool = True
    lidar_channels: int = 32
    lidar_range: float = 100.0
    lidar_points_per_second: int = 500000
    fuse_lidar_vision: bool = True
    
    # 数据记录配置
    record_data: bool = True
    record_dir: str = "track_records"
    record_format: str = "csv"
    record_fps: int = 10
    save_screenshots: bool = False
    
    # 3D可视化配置
    use_3d_visualization: bool = True
    pcd_view_size: int = 800

    @classmethod
    def from_yaml(cls, yaml_path):
        """从YAML文件加载配置"""
        if not os.path.exists(yaml_path):
            print(f"⚠️ 配置文件 {yaml_path} 不存在，使用默认配置")
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # 过滤有效字段，避免配置文件中有多余字段导致报错
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        # 类型转换（确保配置值类型正确）
        for k, v in filtered_data.items():
            try:
                filtered_data[k] = cls.__dataclass_fields__[k].type(v)
            except:
                del filtered_data[k]
        return cls(**filtered_data)

# ======================== 加载天气参数 ========================
def load_weather_params(cfg):
    """从配置文件加载天气参数并转换为CARLA WeatherParameters"""
    # 兼容旧配置：如果配置文件中有weather_params字段，优先使用
    if hasattr(cfg, 'weather_params') and isinstance(cfg.weather_params, dict):
        weather_dict = {}
        for weather_name, params in cfg.weather_params.items():
            weather_params = carla.WeatherParameters(
                sun_altitude_angle=params.get('sun_altitude_angle', 75.0),
                sun_azimuth_angle=params.get('sun_azimuth_angle', 180.0),
                cloudiness=params.get('cloudiness', 0.0),
                precipitation=params.get('precipitation', 0.0),
                precipitation_deposits=params.get('precipitation_deposits', 0.0),
                wind_intensity=params.get('wind_intensity', 0.0),
                fog_density=params.get('fog_density', 0.0),
                fog_distance=params.get('fog_distance', 0.0),
                fog_falloff=params.get('fog_falloff', 1.0),
                wetness=params.get('wetness', 0.0),
                scattering_intensity=params.get('scattering_intensity', 0.0)
            )
            weather_dict[weather_name] = weather_params
        return weather_dict
    else:
        # 配置文件中无天气参数时，使用硬编码默认值
        print("⚠️ 配置文件中未找到天气参数，使用默认值")
        return {
            'clear':carla.WeatherParameters(0.0,0.0,0.0,0.0,180.0,75.0,0.0,0.0,1.0,0.0,0.0),
            'rain':carla.WeatherParameters(80.0,80.0,50.0,30.0,180.0,45.0,20.0,50.0,0.8,80.0,0.5),
            'fog':carla.WeatherParameters(90.0,0.0,0.0,10.0,180.0,30.0,70.0,20.0,0.5,10.0,0.8),
            'night':carla.WeatherParameters(20.0,0.0,0.0,0.0,0.0,-90.0,10.0,100.0,0.7,0.0,1.0),
            'cloudy':carla.WeatherParameters(90.0,0.0,0.0,20.0,180.0,60.0,10.0,100.0,0.9,0.0,0.3),
            'snow':carla.WeatherParameters(90.0,90.0,80.0,40.0,180.0,20.0,30.0,30.0,0.6,50.0,0.7)
        }

# ======================== 主函数 ========================
def main():
    # 1. 初始化日志
    logger = setup_logger()
    
    # 2. 解析命令行参数
    parser = argparse.ArgumentParser(description="CARLA多目标跟踪系统")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（默认：config.yaml）")
    parser.add_argument("--host", help="CARLA主机（覆盖配置文件）")
    parser.add_argument("--port", type=int, help="CARLA端口（覆盖配置文件）")
    parser.add_argument("--conf-thres", type=float, help="检测置信度阈值（覆盖配置文件）")
    parser.add_argument("--weather", help="初始天气（可选：clear/rain/fog/night/cloudy/snow）")
    args = parser.parse_args()
    
    # 3. 加载配置
    cfg = Config.from_yaml(args.config)
    # 命令行参数覆盖配置文件（优先级更高）
    if args.host: cfg.host = args.host
    if args.port: cfg.port = args.port
    if args.conf_thres: cfg.conf_thres = args.conf_thres
    # 加载天气参数
    WEATHER = load_weather_params(cfg)
    if args.weather and args.weather in WEATHER: 
        cfg.default_weather = args.weather
    current_weather = cfg.default_weather

    # 4. 初始化资源变量（避免未定义报错）
    ego = None
    camera = None
    lidar = None
    det_thread = None
    vis = None
    recorder = Recorder(cfg)

    try:
        # 5. 连接CARLA服务器
        print(f"🔌 正在连接CARLA服务器 {cfg.host}:{cfg.port}...")
        client = carla.Client(cfg.host, cfg.port)
        client.set_timeout(20.0)  # 超时时间20秒
        world = client.get_world()
        
        # 设置同步模式（保证帧率稳定）
        print("⚙️ 配置CARLA同步模式...")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20Hz
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        
        # 初始化交通管理器
        try:
            tm = client.get_trafficmanager(8000)
            tm.set_global_distance_to_leading_vehicle(2.0)
            tm.set_respawn_dormant_vehicles(True)
            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(50.0)
            tm.global_percentage_speed_difference(0)
            print("✅ 交通管理器初始化成功")
        except Exception as e:
            print(f"⚠️ 交通管理器初始化失败: {e}，使用默认设置")

        # 6. 设置初始天气
        print(f"🌤️ 设置初始天气：{current_weather}")
        world.set_weather(WEATHER[current_weather])

        # 7. 生成自车和NPC车辆
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("❌ 无可用生成点，退出程序")
            return
        
        # 生成自车
        print("🚗 正在生成自车...")
        ego = safe_spawn_ego(world, spawn_points)
        if ego is None:
            return
        ego.set_autopilot(True, tm_port=8000)
        
        # 生成NPC车辆
        print(f"🚙 正在生成{cfg.num_npcs}辆NPC车辆...")
        npc_count = spawn_npcs(world, cfg.num_npcs, spawn_points)
        print(f"✅ 成功生成NPC车辆：{npc_count} 辆")

        # 8. 初始化传感器
        # 相机传感器
        print("📷 初始化相机传感器...")
        camera = CameraSensor(world, ego, cfg).start()
        
        # LiDAR传感器
        lidar = None
        lidar_proc = None
        if cfg.use_lidar:
            print("📡 初始化LiDAR传感器...")
            lidar = LiDARSensor(world, ego, cfg).start()
            lidar_proc = lidar.get_detector()

        # 9. 初始化YOLO检测模型
        print("🤖 加载YOLO检测模型...")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ 使用计算设备: {dev}")
        model = load_yolo_model(cfg, dev)

        # 10. 初始化天气增强器
        weather_enhancer = WeatherEnhancer(cfg)
        weather_enhancer.set_weather(current_weather)

        # 11. 启动检测线程（异步推理，避免阻塞主循环）
        print("⚡ 启动检测线程...")
        in_q = queue.Queue(maxsize=2)
        out_q = queue.Queue(maxsize=2)
        det_thread = DetThread(model, cfg, weather_enhancer, in_q, out_q, dev)
        det_thread.start()
        print("✅ 推理线程已启动")

        # 12. 初始化跟踪器和可视化
        tracker = SORT(cfg)
        frame_buf = FrameBuf((cfg.img_height, cfg.img_width, 3))
        fps_counter = FPS(cfg.fps_window_size)
        # 创建可视化窗口
        cv2.namedWindow("CARLA Object Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CARLA Object Tracking", cfg.window_width, cfg.window_height)

        # 13. 初始化3D点云可视化
        vis = None
        if cfg.use_3d_visualization and cfg.use_lidar and lidar_proc is not None:
            print("🖥️ 初始化3D点云可视化...")
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="LiDAR Point Cloud", width=cfg.pcd_view_size, height=cfg.pcd_view_size)
            print("✅ 3D点云可视化窗口已启动")

        # 14. 主循环
        print("\n=====================================")
        print("🚀 CARLA多目标跟踪系统已启动！")
        print("💡 操作说明：")
        print("   - 按ESC键退出程序")
        print("   - 按W键切换天气")
        print("=====================================\n")
        
        frame_count = 0
        last_display_time = time.time()
        weather_list = list(WEATHER.keys())
        
        while True:
            # 同步世界步长
            world.tick()
            
            # 帧率控制（保证显示帧率稳定）
            current_time = time.time()
            elapsed = current_time - last_display_time
            target_interval = 1.0 / cfg.display_fps
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            last_display_time = current_time
            
            # 获取相机图像
            img = camera.get_data(timeout=0.1)
            if img is None:
                img = frame_buf.get()  # 使用缓冲帧避免黑屏
            else:
                frame_buf.update(img)  # 更新缓冲帧
            
            # 提交推理任务到检测线程
            if not in_q.full():
                in_q.put(img.copy())
            
            # 获取检测结果
            dets = np.array([])
            try:
                _, dets = out_q.get_nowait()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"⚠️ 获取检测结果异常: {e}")
            
            # LiDAR目标检测
            lidar_dets = lidar_proc.detect() if (cfg.use_lidar and lidar_proc) else []
            
            # 更新跟踪器（融合视觉+LiDAR检测结果）
            boxes, ids, cls_ids = tracker.update(dets, (cfg.img_width//2, cfg.img_height//2), lidar_dets)
            
            # 计算实时FPS
            fps = fps_counter.update()
            
            # 绘制可视化界面
            display_img = draw(
                img, boxes, ids, cls_ids, tracker.tracks,
                fps=fps, det_cnt=len(dets), cfg=cfg, w=current_weather
            )
            cv2.imshow("CARLA Object Tracking", display_img)
            
            # 更新3D点云可视化
            if cfg.use_3d_visualization and vis and lidar_proc:
                pcd = lidar_proc.get_3d()
                if pcd is not None:
                    vis.clear_geometries()
                    vis.add_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
            
            # 数据记录（跟踪结果、性能指标）
            recorder.record(tracker.tracks, dets, fps)
            # 定期保存截图
            if cfg.save_screenshots and frame_count%30==0:
                recorder.save_ss(display_img, current_weather)
            
            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                print("\n🛑 用户按下ESC键，正在退出程序...")
                break
            elif key == ord('w') or key == ord('W'):  # W键切换天气
                current_idx = weather_list.index(current_weather)
                current_weather = weather_list[(current_idx+1)%len(weather_list)]
                world.set_weather(WEATHER[current_weather])
                weather_enhancer.set_weather(current_weather)
                print(f"🌤️ 已切换天气到: {current_weather} (可选：{', '.join(weather_list)})")
            
            frame_count +=1

    except KeyboardInterrupt: 
        print("\n🛑 用户中断程序（Ctrl+C），正在退出...")
    except Exception as e: 
        print(f"\n❌ 运行错误: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误堆栈，方便调试
    finally:
        # 资源清理（关键：避免CARLA残留Actor）
        print("\n🧹 开始清理资源...")
        
        # 停止检测线程
        if det_thread is not None and det_thread.is_alive():
            det_thread.stop()
            det_thread.join(timeout=2.0)
            print("✅ 检测线程已停止")
        
        # 关闭3D可视化窗口
        if vis is not None:
            try:
                vis.destroy_window()
                print("✅ 3D可视化窗口已关闭")
            except:
                pass
        
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        print("✅ OpenCV窗口已关闭")
        
        # 关闭数据记录器
        recorder.close()
        print("✅ 数据记录器已关闭")
        
        # 销毁传感器
        if lidar is not None:
            lidar.stop()
        if camera is not None:
            camera.stop()
        
        # 销毁自车
        if ego is not None and ego.is_alive:
            try: 
                ego.destroy()
                print("✅ 自车已销毁")
            except Exception as e:
                print(f"⚠️ 销毁自车失败: {e}")
        
        # 清理所有NPC和残留传感器
        clear_actors(world)
        print("✅ 所有NPC和传感器已清理")
        
        # 恢复CARLA世界设置（关闭同步模式）
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("✅ CARLA同步模式已关闭")
        
        print("\n🎉 所有资源清理完成，程序正常退出！")

if __name__ == "__main__":
    main()