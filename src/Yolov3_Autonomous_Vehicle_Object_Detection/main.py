import cv2
import time
import queue
import numpy as np
import carla

from config import config
from utils.carla_client import CarlaClient
from models.yolo_detector import YOLODetector
from utils.visualization import draw_results
from utils.planner import SimplePlanner
# [新增] 引入日志模块
from utils.logger import PerformanceLogger


def main():
    # 1. 初始化各模块
    print("[Main] 初始化模块...")
    detector = YOLODetector(
        cfg_path=config.YOLO_CONFIG_PATH,
        weights_path=config.YOLO_WEIGHTS_PATH,
        names_path=config.YOLO_NAMES_PATH,
        conf_thres=config.CONFIDENCE_THRESHOLD,
        nms_thres=config.NMS_THRESHOLD
    )
    detector.load_model()

    planner = SimplePlanner()

    # [新增] 初始化日志记录器
    logger = PerformanceLogger()

    client = CarlaClient()
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

                # --- 数据记录 (Logging) ---
                fps = 1 / (time.time() - start_time)

                # 计算平均置信度
                avg_conf = 0
                if len(results) > 0:
                    avg_conf = np.mean([res[5] for res in results])

                # 写入 TensorBoard
                logger.log_step(fps, len(results), avg_conf)

                # --- 可视化 ---
                frame = draw_results(frame, results, detector.classes)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if is_brake:
                    cv2.putText(frame, "EMERGENCY BRAKING!", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

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
        # [新增] 关闭日志
        logger.close()
        cv2.destroyAllWindows()
        print("[Main] 程序已退出")


if __name__ == "__main__":
    main()