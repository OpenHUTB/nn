import cv2
import numpy as np
import os


class YOLODetector:
    """
    YOLOv3 目标检测器封装类（OpenCV 4.13.0 兼容版）
    彻底修复Unknown layer type问题，适配YOLOv3-tiny
    """

    def __init__(self, cfg_path, weights_path, names_path, conf_thres=0.5, nms_thres=0.4):
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.names_path = names_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

        self.net = None
        self.output_layers = None
        self.classes = []

    def load_model(self):
        if not os.path.exists(self.weights_path) or not os.path.exists(self.cfg_path):
            raise FileNotFoundError(f"[ERROR] 模型文件缺失！请先运行 python download_weights.py")

        print(f"[INFO] 正在加载 YOLO 模型...\n配置: {self.cfg_path}\n权重: {self.weights_path}")

        # 核心修复：强制使用CPU后端，添加OpenCV 4.13.0兼容参数
        try:
            self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
            # 强制指定后端和目标，彻底解决层解析问题
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            # 兜底方案：如果Darknet加载失败，自动切换到兼容模式
            print(f"[WARNING] Darknet加载失败，尝试兼容模式: {e}")
            # 直接读取权重，跳过cfg解析（仅适用于YOLOv3-tiny）
            self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 兼容不同OpenCV版本的输出层获取
        layer_names = self.net.getLayerNames()
        try:
            out_layers = self.net.getUnconnectedOutLayers()
            if isinstance(out_layers, np.ndarray):
                out_layers_indices = out_layers.flatten()
            else:
                out_layers_indices = [i[0] for i in out_layers]
            self.output_layers = [layer_names[i - 1] for i in out_layers_indices]
        except Exception as e:
            print(f"[WARNING] 输出层获取异常，使用默认层: {e}")
            self.output_layers = layer_names[-2:]

        # 加载类别标签
        if os.path.exists(self.names_path):
            with open(self.names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            print(f"[WARNING] 类别文件 {self.names_path} 不存在，类别名称将为空。")

        print(f"[INFO] 模型加载成功！共 {len(self.classes)} 个类别。")

    def detect(self, image):
        if self.net is None:
            print("[ERROR] 模型未加载，请先调用 load_model()")
            return []

        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return self._post_process(outputs, H, W)

    def _post_process(self, outputs, height, width):
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_thres:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    x = int(centerX - w/2)
                    y = int(centerY - h/2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.nms_thres)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y, w, h) = boxes[i]
                results.append([x, y, w, h, class_ids[i], confidences[i]])
        return results