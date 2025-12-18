# image_detector.py
# 功能：封装 YOLOv8 模型的加载与推理逻辑，提供干净的检测接口

from ultralytics import YOLO
import io
import sys
import numpy as np


class ImageDetector:
    """
    目标检测引擎类。
    负责加载 YOLO 模型并对输入图像帧执行推理，
    同时屏蔽模型内部的冗余打印输出（如进度条、日志等），
    使主程序输出更整洁。
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        """
        初始化检测引擎。

        参数:
            model_path (str): YOLO 模型文件路径或名称（如 'yolov8n.pt'）
            conf_threshold (float): 置信度阈值，低于此值的检测结果将被过滤
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # 加载 YOLO 模型（在初始化时完成，避免每次检测重复加载）
        self.model = self._load_model()

    def _load_model(self):
        """
        私有方法：加载 YOLO 模型，并抑制其标准输出和错误输出。
        原因：YOLO 在加载模型或首次推理时会自动打印信息（如设备、尺寸等），
        这些信息在 GUI 或自动化脚本中属于干扰。通过临时重定向 stdout/stderr 来静默加载。

        返回:
            YOLO: 已加载的模型实例
        """
        # 保存原始的标准输出和错误流
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        # 临时重定向到 StringIO 缓冲区（丢弃所有输出）
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            # 加载模型（若为官方模型且未下载，会自动联网下载）
            model = YOLO(self.model_path)
        finally:
            # 确保无论是否出错，都恢复原始输出流
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return model

    def detect(self, frame: np.ndarray):
        """
        对单帧图像执行目标检测。

        参数:
            frame (np.ndarray): 输入图像，格式为 HWC（高度, 宽度, 通道），BGR 或 RGB 均可（YOLO 内部会处理）

        返回:
            tuple:
                - annotated_frame (np.ndarray): 带有检测框、标签和置信度的可视化图像（HWC, BGR 格式）
                - results (List[ultralytics.engine.results.Results]): 原始检测结果对象列表（通常长度为1）

        注意:
            - 即使没有检测到任何目标，YOLO 仍会返回一个 Results 对象；
            - annotated_frame 始终是有效图像（无检测时即为原图）。
        """
        # 保存原始输出流
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        # 临时静音 YOLO 的 verbose 输出（如 "image 1/1 ..."）
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            # 执行推理：传入图像、置信度阈值，并关闭详细日志（verbose=False）
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # 使用 YOLO 内置的 plot() 方法生成带标注的图像（numpy array, HWC）
        annotated_frame = results[0].plot()  # results 总是包含至少一个元素
        return annotated_frame, results
