import cv2
import time
import queue
import numpy as np
import carla
import argparse  # [新增] 引入命令行参数解析库

from config import config
from utils.carla_client import CarlaClient
from models.yolo_detector import YOLODetector
from utils.visualization import draw_results, draw_safe_zone
from utils.planner import SimplePlanner
from utils.logger import PerformanceLogger


# [新增] 参数解析函数
def parse_arguments():
    parser = argparse.ArgumentParser(description="Autonomous Vehicle Object Detection System")

    parser.add_argument("--host", default=config.CARLA_HOST, help="CARLA Host IP")
    parser.add_argument("--port", type=int, default=config.CARLA_PORT, help="CARLA Port")
    parser.add_argument("--no-render", action="store_true", help="Disable OpenCV rendering window (Headless mode)")

    return parser.parse_args()


def main():
    # 1. 解析命令行参数
    args = parse_arguments()

    print("[Main] 初始化模块...")
    # 打印运行模式
    if args.no_render:
        print("[INFO] 运行模式: Headless (无窗口渲染)")

    detector = YOLODetector(
        cfg_path=config.YOLO_CONFIG_PATH,
        weights_path=config.YOLO_WEIGHTS_PATH,
        names_path=config.YOLO_NAMES_PATH,
        conf_thres=config.CONFIDENCE_THRESHOLD,
        nms_thres=config.NMS_THRESHOLD
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

    print("[Main] 开始主循环 (按 'q' 退出)...")
    try:
        while True:
            try:
                frame = client.image_queue.get(timeout=2.0)

                # --- 感知 ---
                start_time = time.time()
                results = detector.detect(frame)

                # --- 规划 ---
                is_brake, warning_msg = planner.plan(results)

                # --- 控制 ---
                if client.vehicle:
                    if is_brake:
                        client.vehicle.set_autopilot(False)
                        control = carla.VehicleControl()
                        control.throttle = 0.0
                        control.brake = 1.0
                        client.vehicle.apply_control(control)
                    else:
                        client.vehicle.set_autopilot(True)

                # --- 记录数据 ---
                fps = 1 / (time.time() - start_time)
                logger.log_step(fps, len(results))

                # --- 可视化 (根据 --no-render 参数决定是否显示) ---
                if not args.no_render:
                    frame = draw_safe_zone(frame)
                    frame = draw_results(frame, results, detector.classes)

                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if is_brake:
                        cv2.putText(frame, "EMERGENCY BRAKING!", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),
                                    4)
                        cv2.putText(frame, warning_msg, (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    cv2.imshow("CARLA Object Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            except queue.Empty:
                continue

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