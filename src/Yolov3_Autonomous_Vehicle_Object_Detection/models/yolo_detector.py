import cv2
import numpy as np
import os


class YOLODetector:
    """
    YOLOv3 目标检测器封装类
    负责加载模型权重、图像预处理、前向推理及结果后处理（NMS）
    """

    def __init__(self, cfg_path, weights_path, names_path, conf_thres=0.5, nms_thres=0.4):
        """
        初始化检测器参数
        """
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.names_path = names_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

        # 核心变量
        self.net = None
        self.output_layers = None
        self.classes = []

    def load_model(self):
        """
        加载 Darknet 网络模型与类别标签
        """
        # 1. 检查文件是否存在
        if not os.path.exists(self.weights_path) or not os.path.exists(self.cfg_path):
            raise FileNotFoundError(f"[ERROR] 模型文件缺失！请先运行 python download_weights.py")

        print(f"[INFO] 正在加载 YOLO 模型...\n配置: {self.cfg_path}\n权重: {self.weights_path}")

        # 2. 使用 OpenCV DNN 模块加载网络
        try:
            self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        except Exception as e:
            raise RuntimeError(f"[ERROR] 模型加载失败: {e}")

        # 3. 设置后端 (默认为 OpenCV + CPU，若有 GPU 可修改为 CUDA)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 4. 获取输出层名称 (兼容不同版本的 OpenCV)
        layer_names = self.net.getLayerNames()
        try:
            # OpenCV 4.x+ 返回的是 numpy 数组，需要 flatten
            out_layers_indices = self.net.getUnconnectedOutLayers().flatten()
            self.output_layers = [layer_names[i - 1] for i in out_layers_indices]
        except AttributeError:
            # 旧版本 OpenCV 处理方式
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # 5. 加载类别名称
        if os.path.exists(self.names_path):
            with open(self.names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            print(f"[WARNING] 类别文件 {self.names_path} 不存在，类别名称将为空。")

        print(f"[INFO] 模型加载成功！共 {len(self.classes)} 个类别。")

    def detect(self, image):
        """
        执行目标检测的主流水线 (待实现)
        """
        if self.net is None:
            print("[ERROR] 模型未加载，请先调用 load_model()")
            return []

        # 下一步实现这里
        return []

    def _preprocess(self, image):
        pass

    def _post_process(self, outputs, height, width):
        pass