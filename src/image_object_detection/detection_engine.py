# detection_engine.py
from ultralytics import YOLO
import io
import sys

class DetectionEngine:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = self._load_model()
    
    def _load_model(self):
        """加载YOLO模型并抑制输出"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            model = YOLO(self.model_path)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return model
    
    def detect(self, frame):
        """
        对输入帧进行检测。
        返回: (annotated_frame: np.ndarray, results: List[Results])
        即使无检测框，annotated_frame 也是原始图像（HWC格式）。
        """
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # YOLOv8 always returns at least one result
        annotated_frame = results[0].plot()  # numpy array (H, W, C)
        return annotated_frame, results
