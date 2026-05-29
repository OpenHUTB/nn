import cv2
import time
import queue
import numpy as np
import carla
import argparse

from config import config
from utils.carla_client import CarlaClient
from models.yolo_detector import YOLODetector
from utils.visualization import draw_results, draw_safe_zone
from utils.planner import SimplePlanner
from utils.logger import PerformanceLogger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Autonomous Vehicle Object Detection System")
    parser.add_argument("--host", default=config.carla_host, help="CARLA Host IP")
    parser.add_argument("--port", type=int, default=config.carla_port, help="CARLA Port")
    parser.add_argument("--no-render", action="store_true", help="Disable OpenCV rendering window (Headless mode)")
    parser.add_argument("--demo", action="store_true", help="使用演示模式（模拟图像，无 CARLA）")
    parser.add_argument("--in-carla", action="store_true", help="在 CARLA 模拟器窗口中显示检测结果（推荐）")
    return parser.parse_args()


def generate_demo_frame():
    frame = np.ones((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8) * 100
    cv2.rectangle(frame, (0, config.CAMERA_HEIGHT//2), (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), (80, 80, 80), -1)
    for i in range(0, config.CAMERA_WIDTH, 50):
        cv2.line(frame, (i, config.CAMERA_HEIGHT//2), (i, config.CAMERA_HEIGHT//2 + 30), (255, 255, 255), 2)
    cv2.rectangle(frame, (300, 200), (500, 350), (0, 0, 200), -1)
    cv2.rectangle(frame, (100, 400), (250, 500), (0, 0, 200), -1)
    return frame


def main():
    args = parse_arguments()

    print("[Main] 初始化模块...")
    
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
                frame = generate_demo_frame()
                results = detector.detect(frame)
                is_brake, warning_msg = planner.plan(results)
                if is_brake:
                    print(f"[控制] 刹车: {warning_msg}")
                fps = 1 / (time.time() - start_time)
                logger.log_step(fps, len(results))

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

    print("[Main] 开始主循环 (按 Ctrl+C 退出)...")
    
    if args.in_carla:
        print("[INFO] 显示模式: 在 CARLA 模拟器窗口中显示检测结果")
        print("[INFO] 检测到的车辆, 白色框 = 车辆边界框")
    elif args.no_render:
        print("[INFO] 显示模式: 无窗口 (Headless)")
    else:
        print("[INFO] 显示模式: OpenCV 窗口")

    frame_count = 0
    try:
        while True:
            start_time = time.time()
            
            results = []
            frame = None
            try:
                frame = client.image_queue.get(timeout=0.02)
                results = detector.detect(frame)
            except queue.Empty:
                if frame_count % 200 == 0:
                    print("[DEBUG] 等待图像数据...")
                pass
            except Exception as e:
                if frame_count % 100 == 0:
                    print(f"[WARNING] 检测异常: {e}")
                pass
            
            is_brake, warning_msg = planner.plan(results)

            fps = 1 / (time.time() - start_time)
            logger.log_step(fps, len(results))
            
            if args.in_carla:
                try:
                    # 推进世界模拟（同步模式下必须调用）
                    client.tick()
                    
                    client.follow_vehicle()
                    client.draw_vehicle_boxes()
                    
                    if results:
                        client.draw_detection_in_carla(results, detector.classes)
                    
                    # 获取车辆速度（m/s 转换为 km/h）
                    speed_kmh = 0
                    if client.vehicle:
                        velocity = client.vehicle.get_velocity()
                        speed_kmh = velocity.length() * 3.6
                    
                    # 获取主视角画面（车内摄像头）并显示
                    if frame is not None:
                        display_frame = draw_results(draw_safe_zone(frame.copy()), results, detector.classes)
                        
                        # 添加 HUD 信息到左上角
                        cv2.putText(display_frame, f"Speed: {speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(display_frame, f"Detections: {len(results)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"View: {client.get_current_camera_name()}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
                        cv2.putText(display_frame, "Press Q to Quit", (display_frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                        
                        if is_brake:
                            cv2.putText(display_frame, "!!! EMERGENCY BRAKING !!!", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        
                        cv2.imshow("CARLA Object Detection", display_frame)
                    
                    # 处理键盘输入
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('1'):
                        client.switch_camera('front')
                    elif key == ord('2'):
                        client.switch_camera('back')
                    elif key == ord('3'):
                        client.switch_camera('left')
                    elif key == ord('4'):
                        client.switch_camera('right')
                    
                    # 检测碰撞并自动重置
                    client.check_collision_and_reset()
                    
                    # 控制台显示FPS和速度信息
                    if frame_count % 30 == 0:
                        print(f"[INFO] FPS: {fps:.1f} | 速度: {speed_kmh:.1f} km/h | 检测目标: {len(results)}")
                except Exception as e:
                    if frame_count % 100 == 0:
                        print(f"[WARNING] CARLA 绘制异常: {e}")
                    pass

            if not args.no_render and not args.in_carla:
                try:
                    if frame is not None:
                        display_frame = draw_results(draw_safe_zone(frame.copy()), results, detector.classes)
                        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cam_name = client.get_current_camera_name()
                        cv2.putText(display_frame, f"视角: {cam_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        cv2.putText(display_frame, "1-前 2-后 3-左 4-右", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1)
                        
                        if is_brake:
                            cv2.putText(display_frame, "EMERGENCY BRAKING!", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                            cv2.putText(display_frame, warning_msg, (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        cv2.imshow("CARLA Object Detection", display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('1'):
                            client.switch_camera('front')
                        elif key == ord('2'):
                            client.switch_camera('back')
                        elif key == ord('3'):
                            client.switch_camera('left')
                        elif key == ord('4'):
                            client.switch_camera('right')
                            
                except Exception as e:
                    if frame_count % 100 == 0:
                        print(f"[WARNING] OpenCV 显示异常: {e}")
                    pass
            
            if args.in_carla:
                try:
                    import sys
                    if sys.platform == 'win32':
                        import msvcrt
                        if msvcrt.kbhit():
                            key = msvcrt.getch()
                            if isinstance(key, bytes):
                                key = key.decode('utf-8')
                            if key == 'q' or key == 'Q':
                                break
                            elif key == '1':
                                client.switch_camera('front')
                                print("[INFO] 已切换到前视视角")
                            elif key == '2':
                                client.switch_camera('back')
                                print("[INFO] 已切换到后视视角")
                            elif key == '3':
                                client.switch_camera('left')
                                print("[INFO] 已切换到左视视角")
                            elif key == '4':
                                client.switch_camera('right')
                                print("[INFO] 已切换到右视视角")
                except Exception as e:
                    if frame_count % 100 == 0:
                        print(f"[WARNING] 键盘处理异常: {e}")
                    pass

            frame_count += 1
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Main] 用户中断程序")

    finally:
        print("[Main] 正在清理资源...")
        client.destroy_actors()
        logger.close()
        if not args.no_render and not args.in_carla:
            cv2.destroyAllWindows()
        print("[Main] 程序已退出")


if __name__ == "__main__":
    main()
