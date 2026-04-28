import cv2
import time
import queue
import numpy as np
import carla
import argparse  # [新增] 引入命令行参数解析库

from config import config
from utils.carla_client import CarlaClient
from models.yolo_detector import YOLODetector
from utils.visualization import (
    draw_results, draw_safe_zone, show_fps, show_confidence_value,
    create_confidence_trackbar, get_confidence_threshold,
    Tracker, draw_trajectories, draw_tracking_ids,
    draw_risk_heatmap, draw_object_count, draw_detection_cost_bar,
    trigger_aeb_animation, draw_aeb_warning,
    draw_depth_overlay
)
from utils.planner import SimplePlanner
from utils.logger import PerformanceLogger


# [新增] 参数解析函数
def parse_arguments():
    parser = argparse.ArgumentParser(description="Autonomous Vehicle Object Detection System")

    parser.add_argument("--host", default=config.carla_host, help="CARLA Host IP")
    parser.add_argument("--port", type=int, default=config.carla_port, help="CARLA Port")
    parser.add_argument("--no-render", action="store_true", help="Disable OpenCV rendering window (Headless mode)")
    parser.add_argument("--demo", action="store_true", help="使用演示模式（模拟图像，无 CARLA）")
    parser.add_argument("--in-carla", action="store_true", help="在 CARLA 模拟器窗口中显示检测结果（推荐）")

    return parser.parse_args()


def generate_demo_frame():
    """生成模拟道路图像用于测试"""
    # 创建道路背景（灰色）
    frame = np.ones((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8) * 100
    
    # 绘制模拟道路
    cv2.rectangle(frame, (0, config.CAMERA_HEIGHT//2), (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), (80, 80, 80), -1)
    
    # 绘制车道线
    for i in range(0, config.CAMERA_WIDTH, 50):
        cv2.line(frame, (i, config.CAMERA_HEIGHT//2), (i, config.CAMERA_HEIGHT//2 + 30), (255, 255, 255), 2)
    
    # 添加一些模拟车辆（红色矩形）
    cv2.rectangle(frame, (300, 200), (500, 350), (0, 0, 200), -1)
    cv2.rectangle(frame, (100, 400), (250, 500), (0, 0, 200), -1)
    
    return frame


def main():
    args = parse_arguments()
    print("[Main] 初始化模块...")
    
    # 初始化目标跟踪器
    tracker = Tracker()
    
    # 演示模式
    if args.demo:
        print("[INFO] 运行模式: 演示模式（模拟图像）")
        detector = YOLODetector(
            cfg_path=config.yolo_cfg_path,
            weights_path=config.yolo_weights_path,
            names_path=config.yolo_names_path,
            conf_thres=config.conf_thres,
            nms_thres=config.nms_thres
        )
        detector.load_model()
        planner = SimplePlanner()
        logger = PerformanceLogger(log_dir=config.LOG_DIR)
        
        # 创建窗口和滑动条
        cv2.namedWindow("CARLA Object Detection - DEMO")
        create_confidence_trackbar("CARLA Object Detection - DEMO")
        
        print("[Main] 演示模式开始 (按 'q' 退出, 按 'r' 重置阈值)...")
        try:
            frame_count = 0
            while True:
                start_time = time.time()
                
                # 获取当前置信度阈值
                current_conf_thres = get_confidence_threshold("CARLA Object Detection - DEMO")
                config.conf_thres = current_conf_thres
                
                # 生成模拟帧
                frame = generate_demo_frame()
                
                # --- 感知 ---
                t0 = time.time()
                results = detector.detect(frame)
                t1 = time.time()

                # 更新跟踪器
                tracker.update(results)

                # --- 规划 ---
                is_brake, warning_msg = planner.plan(results)

                # --- 控制 ---
                if is_brake:
                    print(f"[控制] 刹车: {warning_msg}")
                    trigger_aeb_animation(frame_count)

                # --- 记录数据 ---
                total_time = time.time() - start_time
                fps = 1 / total_time
                logger.log_step(fps, len(results))

                # --- 可视化 ---
                if not args.no_render:
                    display_frame = draw_results(
                        draw_safe_zone(frame.copy()), 
                        results, 
                        detector.classes, 
                        fps=fps,
                        tracker=tracker,
                        detect_time=t1 - t0,
                        total_time=total_time
                    )
                    
                    # AEB 警告动画
                    if is_brake:
                        display_frame = draw_aeb_warning(display_frame, frame_count)
                    
                    cv2.imshow("CARLA Object Detection - DEMO", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r') or key == ord('R'):
                        cv2.setTrackbarPos("Confidence", "CARLA Object Detection - DEMO", 50)
                        config.conf_thres = 0.5
                
                frame_count += 1
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n[Main] 用户中断程序")
        finally:
            logger.close()
            cv2.destroyAllWindows()
            print("[Main] 程序已退出")
        return

    # 正常 CARLA 模式
    if args.no_render:
        print("[INFO] 运行模式: Headless (无窗口渲染)")

    detector = YOLODetector(
        cfg_path=config.yolo_cfg_path,
        weights_path=config.yolo_weights_path,
        names_path=config.yolo_names_path,
        conf_thres=config.conf_thres,
        nms_thres=config.nms_thres
    )
    detector.load_model()

    planner = SimplePlanner()
    logger = PerformanceLogger(log_dir=config.LOG_DIR)

    client = CarlaClient(host=args.host, port=args.port)

    if not client.connect():
        return

    client.spawn_vehicle()
    client.setup_camera()

    # 创建窗口和滑动条
    cv2.namedWindow("CARLA Object Detection")
    create_confidence_trackbar()
    
    print("[Main] 开始主循环 (按 Ctrl+C 或 'q' 退出, 按 'r' 重置阈值)...")
    
    if args.in_carla:
        print("[INFO] 显示模式: 在 CARLA 模拟器窗口中显示检测结果")
        print("[INFO] 检测到的车辆, 白色框 = 车辆边界框")
    elif args.no_render:
        print("[INFO] 显示模式: 无窗口 (Headless)")
    else:
        print("[INFO] 显示模式: OpenCV 窗口")
    
    try:
        frame_count = 0
        while True:
            try:
                # --- 感知 ---
                t0 = time.time()
                
                # 获取当前置信度阈值
                current_conf_thres = get_confidence_threshold()
                config.conf_thres = current_conf_thres
                
                try:
                    frame = client.image_queue.get(timeout=0.1)
                    results = detector.detect(frame)
                except queue.Empty:
                    results = []
                
                t1 = time.time()
                
                # 更新跟踪器
                tracker.update(results)
                
                if args.in_carla:
                    client.follow_vehicle()
                    client.draw_vehicle_boxes()
                    
                    if results:
                        client.draw_detection_in_carla(results)
                        if frame_count % 100 == 0:
                            print(f"[DEBUG] 检测到 {len(results)} 个目标")
                
                # --- 规划 ---
                is_brake, warning_msg = planner.plan(results)

                # --- 控制 ---
                if is_brake:
                    trigger_aeb_animation(frame_count)
                pass

                # --- 记录数据 ---
                total_time = time.time() - t0
                fps = 1 / total_time if total_time > 0 else 0
                logger.log_step(fps, len(results))

                # --- 在 OpenCV 窗口中显示 ---
                if not args.no_render and not args.in_carla:
                    try:
                        frame = client.image_queue.get(timeout=0.01)
                        display_frame = draw_results(
                            draw_safe_zone(frame.copy()), 
                            results, 
                            detector.classes,
                            fps=fps,
                            tracker=tracker,
                            detect_time=t1 - t0,
                            total_time=total_time
                        )
                        
                        # AEB 警告动画
                        if is_brake:
                            display_frame = draw_aeb_warning(display_frame, frame_count)
                        
                        cv2.imshow("CARLA Object Detection", display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('r') or key == ord('R'):
                            cv2.setTrackbarPos("Confidence", "CARLA Object Detection", 50)
                            config.conf_thres = 0.5
                    except queue.Empty:
                        pass

                frame_count += 1
                time.sleep(0.05)

            except KeyboardInterrupt:
                break

    except KeyboardInterrupt:
        print("\n[Main] 用户中断程序")

    finally:
        print("[Main] 正在清理资源...")
        client.destroy_actors()
        logger.close()
        if not args.no_render:
            cv2.destroyAllWindows()
        print("[Main] 程序已退出")


if __name__ == "__main__":
    main()