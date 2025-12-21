# model_manager.py
"""
模型管理器模块。
提供模型热切换能力，封装 DetectionEngine 的重新初始化逻辑。
"""

from detection_engine import DetectionEngine, ModelLoadError

class ModelManager:
    """管理当前使用的 DetectionEngine 实例，支持动态切换模型。"""

    def __init__(self, initial_model_path, conf_threshold):
        """
        初始化模型管理器。
        :param initial_model_path: 初始模型路径
        :param conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.engine = None
        self._load_model(initial_model_path)

    def _load_model(self, model_path):
        """内部方法：加载模型并创建 DetectionEngine 实例。"""
        try:
            self.engine = DetectionEngine(model_path=model_path, conf_threshold=self.conf_threshold)
            print(f"✅ Model successfully loaded: {model_path}")
            return True
        except ModelLoadError as e:
            print(f"❌ Failed to load model '{model_path}': {e}")
            return False

    def switch_model(self, new_model_path):
        """
        尝试切换到新模型。
        :param new_model_path: 新模型路径（如 'yolov8s.pt' 或 './models/custom.pt'）
        :return: bool，是否切换成功
        """
        success = self._load_model(new_model_path)
        if success:
            print(f"🔄 Detection engine now using: {new_model_path}")
        else:
            print("⚠️ Model switch failed. Keeping current model.")
        return success

    def get_current_engine(self):
        """返回当前可用的 DetectionEngine 实例。"""
        return self.engine