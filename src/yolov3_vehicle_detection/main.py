import cv2
import time
import queue
import numpy as np
import carla
import argparse  # [新增] 引入命令行参数解析库
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import config
from utils.carla_client import CarlaClient
from models.yolo_detector import YOLODetector
from utils.visualization import draw_results, draw_safe_zone
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
    # 1. 解析命令行参数
    args = parse_arguments()

    print("[Main] 初始化模块...")
    
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
        
        print("[Main] 演示模式开始 (按 'q' 退出)...")
        try:
            frame_count = 0
            while True:
                start_time = time.time()
                
                # 生成模拟帧
                frame = generate_demo_frame()
                
                # --- 感知 ---
                results = detector.detect(frame)

                # --- 规划 ---
                is_brake, warning_msg = planner.plan(results)

                # --- 控制 ---
                if is_brake:
                    print(f"[控制] 刹车: {warning_msg}")

                # --- 记录数据 ---
                fps = 1 / (time.time() - start_time)
                logger.log_step(fps, len(results))

                # --- 可视化 ---
                if not args.no_render:
                    display_frame = draw_results(draw_safe_zone(frame.copy()), results, detector.classes)
                    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(display_frame, "DEMO MODE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if is_brake:
                        cv2.putText(display_frame, "EMERGENCY BRAKING!", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        cv2.putText(display_frame, warning_msg, (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    cv2.imshow("CARLA Object Detection - DEMO", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
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

    # [Mod] 使用命令行参数初始化客户端
    client = CarlaClient(host=args.host, port=args.port)

    if not client.connect():
        return

    client.spawn_vehicle()
    client.setup_camera()

    print("[Main] 开始主循环 (按 Ctrl+C 退出)...")
    
    # 模式选择提示
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
                start_time = time.time()
                
                # 处理摄像头图像（用于检测）
                try:
                    frame = client.image_queue.get(timeout=0.1)
                    results = detector.detect(frame)
                except queue.Empty:
                    results = []
                
                # --- 在 CARLA 中绘制所有车辆边界框 ---
                if args.in_carla:
                    # 第三人称跟随主车辆
                    client.follow_vehicle()
                    client.draw_vehicle_boxes()
                    
                    # 绘制 YOLO 检测结果（如果有）
                    if results:
                        client.draw_detection_in_carla(results)
                        if frame_count % 100 == 0:  # 每100帧打印一次
                            print(f"[DEBUG] 检测到 {len(results)} 个目标")
                
                # --- 智能绕行避让控制（状态机持续运行）---
                client.apply_smart_avoidance()
                
                # --- 规划 ---
                is_brake, warning_msg = planner.plan(results)

                # --- 控制 ---
                # [临时禁用刹车] if client.vehicle:
                # [临时禁用刹车]     if is_brake:
                # [临时禁用刹车]         client.vehicle.set_autopilot(False)
                # [临时禁用刹车]         control = carla.VehicleControl()
                # [临时禁用刹车]         control.throttle = 0.0
                # [临时禁用刹车]         control.brake = 1.0
                # [临时禁用刹车]         client.vehicle.apply_control(control)
                # [临时禁用刹车]     else:
                # [临时禁用刹车]         client.vehicle.set_autopilot(True)
                pass

                # --- 记录数据 ---
                fps = 1 / (time.time() - start_time)
                logger.log_step(fps, len(results))

                # --- 在 OpenCV 窗口中显示 ---
                if not args.no_render and not args.in_carla:
                    try:
                        frame = client.image_queue.get(timeout=0.01)
                        display_frame = draw_results(draw_safe_zone(frame.copy()), results, detector.classes)
                        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if is_brake:
                            cv2.putText(display_frame, "EMERGENCY BRAKING!", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                            cv2.putText(display_frame, warning_msg, (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow("CARLA Object Detection", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except queue.Empty:
                        pass
                
                # --- 在 CARLA 模拟器中绘制调试信息 ---
                if args.in_carla and not args.no_render:
                    client.draw_debug_info_in_carla()

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
        # 只有创建了窗口才需要销毁
        if not args.no_render:
            cv2.destroyAllWindows()
        print("[Main] 程序已退出")


if __name__ == "__main__":
    main()