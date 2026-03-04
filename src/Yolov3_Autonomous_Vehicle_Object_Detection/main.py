import cv2
import time
import queue
import numpy as np

from config import config
from utils.carla_client import CarlaClient
from models.yolo_detector import YOLODetector


def draw_results(image, results, classes):
    """
    简单绘制检测结果 (临时函数，后续会移动到 utils/visualization.py)
    """
    for (x, y, w, h, class_id, conf) in results:
        label = str(classes[class_id])
        color = (0, 255, 0)  # 绿色框

        # 画框
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        # 写字
        text = "{}: {:.4f}".format(label, conf)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def main():
    # 1. 初始化 YOLO 检测器
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
        return  # 连接失败直接退出

    # 3. 生成车辆和传感器
    client.spawn_vehicle()
    client.setup_camera()

    print("[Main] 开始主循环 (按 'q' 退出)...")
    try:
        while True:
            # 4. 从队列获取图像 (带超时机制)
            try:
                frame = client.image_queue.get(timeout=2.0)

                # 5. 执行目标检测
                start_time = time.time()
                results = detector.detect(frame)
                end_time = time.time()

                # 计算 FPS
                fps = 1 / (end_time - start_time)

                # 6. 绘制结果
                frame = draw_results(frame, results, detector.classes)

                # 显示 FPS
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 7. 显示窗口
                cv2.imshow("CARLA Autonomous Driving - Object Detection", frame)

                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except queue.Empty:
                print("[Warning] 等待传感器数据超时...")
                continue

    except KeyboardInterrupt:
        print("\n[Main] 用户中断程序")

    finally:
        # 8. 清理资源
        print("[Main] 正在清理资源...")
        client.destroy_actors()
        cv2.destroyAllWindows()
        print("[Main] 程序已退出")


if __name__ == "__main__":
    main()