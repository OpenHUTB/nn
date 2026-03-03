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

        self.net = None
        self.output_layers = None
        self.classes = []

    def load_model(self):
        """
        加载 Darknet 网络模型与类别标签
        """
        if not os.path.exists(self.weights_path) or not os.path.exists(self.cfg_path):
            raise FileNotFoundError(f"[ERROR] 模型文件缺失！请先运行 python download_weights.py")

        print(f"[INFO] 正在加载 YOLO 模型...\n配置: {self.cfg_path}\n权重: {self.weights_path}")

        try:
            self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        except Exception as e:
            raise RuntimeError(f"[ERROR] 模型加载失败: {e}")

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        try:
            out_layers_indices = self.net.getUnconnectedOutLayers().flatten()
            self.output_layers = [layer_names[i - 1] for i in out_layers_indices]
        except AttributeError:
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        if os.path.exists(self.names_path):
            with open(self.names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            print(f"[WARNING] 类别文件 {self.names_path} 不存在，类别名称将为空。")

        print(f"[INFO] 模型加载成功！共 {len(self.classes)} 个类别。")

    def detect(self, image):
        """
        执行目标检测的主流水线
        """
        if self.net is None:
            print("[ERROR] 模型未加载，请先调用 load_model()")
            return []

        (H, W) = image.shape[:2]

        # 1. 预处理
        blob = self._preprocess(image)

        # 2. 推理
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # 3. 后处理 (解析框 & NMS)
        return self._post_process(outputs, H, W)

    def _preprocess(self, image):
        """
        内部函数：图像预处理
        """
        return cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    def _post_process(self, outputs, height, width):
        """
        内部函数：解析网络输出，过滤低置信度框，执行 NMS
        """
        boxes = []
        confidences = []
        class_ids = []

        # 遍历三个尺度的输出层
        for output in outputs:
            # 遍历每个检测框
            for detection in output:
                # detection 格式: [x, y, w, h, objectness, class_scores...]
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # 1. 过滤掉低置信度的框
                if confidence > self.conf_thres:
                    # 还原坐标 (center_x, center_y, w, h) -> (x, y, w, h)
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")

                    # 计算左上角坐标
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 2. 执行非极大值抑制 (NMS)
        # 返回的是保留下来的框的索引
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.nms_thres)

        results = []
        if len(indices) > 0:
            # flatten() 用于兼容不同版本的 OpenCV 返回格式
            for i in indices.flatten():
                # 提取最终结果: [x, y, w, h, class_id, confidence]
                (x, y, w, h) = boxes[i]
                results.append([x, y, w, h, class_ids[i], confidences[i]])

        return results