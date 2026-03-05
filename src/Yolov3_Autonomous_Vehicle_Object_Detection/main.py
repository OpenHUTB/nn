import cv2
import time
import queue
import numpy as np
import carla  # [新增] 需要用到 carla.VehicleControl

from config import config
from utils.carla_client import CarlaClient
from models.yolo_detector import YOLODetector
from utils.visualization import draw_results
# [新增] 引入规划器
from utils.planner import SimplePlanner


def main():
    # 1. 初始化 YOLOv3 检测器
    print("[Main] 初始化 YOLOv3 检测器...")
    detector = YOLODetector(
        cfg_path=config.YOLO_CONFIG_PATH,
        weights_path=config.YOLO_WEIGHTS_PATH,
        names_path=config.YOLO_NAMES_PATH,
        conf_thres=config.CONFIDENCE_THRESHOLD,
        nms_thres=config.NMS_THRESHOLD
    )
    detector.load_model()

    # 2. 初始化规划器 (决策层)
    print("[Main] 初始化自动驾驶规划器...")
    planner = SimplePlanner()

    # 3. 初始化 CARLA 客户端
    print("[Main] 初始化 CARLA 客户端...")
    client = CarlaClient()
    if not client.connect():
        return

        # 4. 生成车辆和传感器
    client.spawn_vehicle()
    client.setup_camera()

    print("[Main] 开始主循环 (按 'q' 退出)...")
    try:
        while True:
            try:
                # --- 获取数据 ---
                frame = client.image_queue.get(timeout=2.0)

                # --- 感知 (Perception) ---
                start_time = time.time()
                results = detector.detect(frame)

                # --- 规划与决策 (Planning) ---
                # 根据感知结果判断是否需要刹车
                is_brake, warning_msg = planner.plan(results)

                # --- 控制 (Control) ---
                if client.vehicle:
                    if is_brake:
                        # 紧急情况：关闭自动驾驶，强制刹车
                        client.vehicle.set_autopilot(False)
                        control = carla.VehicleControl()
                        control.throttle = 0.0
                        control.brake = 1.0
                        control.hand_brake = False
                        client.vehicle.apply_control(control)
                    else:
                        # 正常情况：恢复自动驾驶
                        client.vehicle.set_autopilot(True)

                # --- 可视化 (Visualization) ---
                fps = 1 / (time.time() - start_time)

                # 绘制检测框
                frame = draw_results(frame, results, detector.classes)

                # 绘制 FPS
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 如果触发刹车，绘制巨大的红色警告
                if is_brake:
                    cv2.putText(frame, "EMERGENCY BRAKING!", (150, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    cv2.putText(frame, warning_msg, (180, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # 显示窗口
                cv2.imshow("CARLA Autonomous Driving - Object Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except queue.Empty:
                print("[Warning] 等待传感器数据超时...")
                continue

    except KeyboardInterrupt:
        print("\n[Main] 用户中断程序")

    finally:
        print("[Main] 正在清理资源...")
        client.destroy_actors()
        cv2.destroyAllWindows()
        print("[Main] 程序已退出")


if __name__ == "__main__":
    main()