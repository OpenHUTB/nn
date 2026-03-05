import cv2
import time
import queue
import numpy as np

from config import config
from utils.carla_client import CarlaClient
from models.yolo_detector import YOLODetector
# [新增] 引入可视化模块
from utils.visualization import draw_results


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

    # 2. 初始化 CARLA 客户端
    print("[Main] 初始化 CARLA 客户端...")
    client = CarlaClient()
    if not client.connect():
        return

        # 3. 生成车辆和传感器
    client.spawn_vehicle()
    client.setup_camera()

    print("[Main] 开始主循环 (按 'q' 退出)...")
    try:
        while True:
            try:
                # 4. 获取图像
                frame = client.image_queue.get(timeout=2.0)

                # 5. 目标检测
                start_time = time.time()
                results = detector.detect(frame)
                fps = 1 / (time.time() - start_time)

                # 6. 绘制结果 (调用新模块)
                frame = draw_results(frame, results, detector.classes)

                # 显示 FPS
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 7. 显示窗口
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