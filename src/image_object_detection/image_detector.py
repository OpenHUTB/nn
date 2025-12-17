# image_detector.py
# 功能：对单张静态图像执行目标检测，并可视化结果

import cv2
import numpy as np
import traceback
import os


class ImageDetector:
    """
    静态图像检测器类。
    接收一个检测引擎（如 YOLO 模型封装），加载指定路径的图像，
    执行目标检测，并在窗口中显示带标注的结果，同时打印检测信息到控制台。
    """

    def __init__(self, detection_engine):
        """
        初始化图像检测器。

        参数:
            detection_engine: 实现 detect(frame) 方法的对象（如 DetectionEngine 实例）
        """
        self.engine = detection_engine

    def detect_static_image(self, image_path):
        """
        对指定路径的静态图像执行目标检测。

        参数:
            image_path (str): 待检测图像的文件路径
        """
        print(f"Loading image: {image_path}")

        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"Error: Image file not found - {image_path}")
            return

        # 使用 OpenCV 读取图像（BGR 格式）
        frame = cv2.imread(image_path)
        if frame is None or frame.size == 0:
            print("Error: Failed to load image")
            return

        print("Running detection...")

        # 调用检测引擎进行推理，获取可视化图像和原始结果
        annotated_frame, results = self.engine.detect(frame)

        # 安全检查：确保返回的图像有效
        if annotated_frame is None or annotated_frame.size == 0:
            print("Error: Invalid annotated frame")
            return

        # 打印检测到的目标信息（类别、置信度）
        self._display_results(results)

        # 在独立窗口中显示带检测框的图像
        self._show_image(annotated_frame, "YOLO_Static_Detection")

    def _display_results(self, results):
        """
        私有方法：解析并打印检测结果到控制台。

        参数:
            results (List[Results]): YOLO 返回的检测结果列表（通常长度为1）
        """
        # 若结果为空或无检测框，提示用户
        if not results:
            print("No objects detected.")
            return

        result = results[0]  # YOLO 总是返回至少一个 Results 对象
        boxes = result.boxes
        if len(boxes) == 0:
            print("No objects detected.")
            return

        print(f"Detected {len(boxes)} object(s):")
        names_list = self.engine.model.names  # 获取模型的类别名称列表

        # 遍历每个检测框，提取类别索引、置信度和类别名
        for i, box in enumerate(boxes):
            try:
                cls_index = int(box.cls.item())  # 类别索引（tensor → int）
                confidence = box.conf.item()     # 置信度（tensor → float）
                # 安全获取类别名称，防止索引越界
                cls_name = names_list[cls_index] if 0 <= cls_index < len(names_list) else f"unknown_{cls_index}"
                print(f" {i+1}. {cls_name} (confidence: {confidence:.2f})")
            except Exception as e:
                print(f" Warning: Failed to parse box {i+1}: {e}")

    def _show_image(self, annotated_frame, window_name="YOLO_Static_Detection"):
        """
        私有方法：使用 OpenCV 显示带标注的图像。

        参数:
            annotated_frame (np.ndarray): 带检测框的图像（HWC, BGR）
            window_name (str): 显示窗口的标题
        """
        try:
            # 关闭可能已存在的同名窗口，避免残留
            cv2.destroyAllWindows()

            # 创建可调整大小的窗口（便于查看大图）
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)  # 初始窗口大小

            # 显示图像，并等待任意按键关闭
            cv2.imshow(window_name, annotated_frame)
            cv2.waitKey(0)  # 0 表示无限等待，直到按键
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Failed to display image: {e}")
            traceback.print_exc()  # 打印完整错误栈，便于调试
