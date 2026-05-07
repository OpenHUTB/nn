# detection_engine.py
# 功能：封装 YOLOv8 模型的加载与推理逻辑，提供干净的检测接口
# 修改：增加距离估计和危险等级显示

from ultralytics import YOLO
import io
import sys
import os
import cv2  # 新增导入


class ModelLoadError(Exception):
    """模型加载失败专用异常"""
    pass


class DetectionEngine:
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
            # 判断是本地文件还是官方模型名
            if os.path.isfile(self.model_path):
                model = YOLO(self.model_path)
            else:
                # 尝试作为 Ultralytics 官方模型加载（会自动下载）
                model = YOLO(self.model_path)
            return model
        except FileNotFoundError as e:
            raise ModelLoadError(f"Model file not found: {self.model_path}") from e
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg:
                raise ModelLoadError(
                    "GPU memory insufficient. Try using CPU or a smaller model (e.g., yolov8n.pt)."
                ) from e
            elif "AssertionError" in msg and ("model" in msg or "state_dict" in msg):
                raise ModelLoadError(f"Corrupted or incompatible model weights: {self.model_path}") from e
            else:
                raise ModelLoadError(f"Runtime error during model loading: {msg}") from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading YOLO model: {e}") from e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _estimate_distance(self, box_height, known_height=1.6, focal_length=700):
        """根据检测框高度估算距离（米）"""
        if box_height < 1:
            return 999.9
        return (known_height * focal_length) / box_height

    def _get_danger_level(self, distance):
        """根据距离判定危险等级"""
        if distance < 10:
            return "DANGER"
        elif distance < 20:
            return "WARNING"
        else:
            return "SAFE"

    def detect(self, frame):
        """
        对单帧图像执行目标检测。

        参数:
            frame (np.ndarray): 输入图像，格式为 HWC（高度, 宽度, 通道），BGR 或 RGB 均可（YOLO 内部会处理）

        返回:
            tuple:
                - annotated_frame (np.ndarray): 带有检测框、标签和置信度的可视化图像（HWC, BGR 格式）
                - results (List[ultralytics.engine.results.Results]): 原始检测结果对象列表（通常长度为1）
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
            annotated_frame = results[0].plot()   # 获取 YOLO 默认标注图

            # ---------- 新增：添加距离估计和危险等级 ----------
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    # 获取坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 计算框高度
                    box_height = y2 - y1
                    # 估算距离
                    distance = self._estimate_distance(box_height)
                    # 判定危险等级
                    danger = self._get_danger_level(distance)
                    # 准备显示的文字
                    info_text = f"{danger} {distance:.1f}m"
                    # 在框的上方绘制文字（y1-15 偏移避免与原始标签重叠）
                    cv2.putText(annotated_frame, info_text, (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # -------------------------------------------------

            return annotated_frame, results
        except Exception as e:
            # 不抛出异常，而是返回原图以维持流程
            print(f"⚠️ Warning: Detection failed on current frame: {e}")
            return frame.copy(), []
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr

