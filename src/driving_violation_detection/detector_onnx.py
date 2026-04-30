import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class YOLOOnnxDetector:
    def __init__(
        self,
        model_path,
        class_names=None,
        conf_threshold=0.35,
        iou_threshold=0.45,
        input_size=640,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.class_names = class_names or ["TrafficSign"]

    def letterbox(self, image):
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)

        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return canvas, scale, pad_x, pad_y

    def preprocess(self, image):
        letterboxed, scale, pad_x, pad_y = self.letterbox(image)
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        return blob, scale, pad_x, pad_y

    def _prepare_predictions(self, outputs):
        preds = np.squeeze(outputs)

        if preds.ndim != 2:
            raise ValueError(f"Unexpected ONNX output shape: {outputs.shape}")

        # Ultralytics YOLO ONNX commonly exports either:
        # 1. [num_preds, 4 + num_classes]
        # 2. [4 + num_classes, num_preds]
        if preds.shape[0] < preds.shape[1] and preds.shape[0] <= 128:
            preds = preds.T

        # Single-class exports can be [x, y, w, h, score], i.e. 5 values.
        if preds.shape[1] < 5:
            raise ValueError(f"Model output does not look like YOLO detection output: {outputs.shape}")

        return preds

    def postprocess(self, image, outputs, scale, pad_x, pad_y):
        preds = self._prepare_predictions(outputs)
        image_h, image_w = image.shape[:2]

        boxes = []
        confidences = []
        class_ids = []

        for pred in preds:
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]

            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < self.conf_threshold:
                continue

            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale

            x1 = int(max(0, min(image_w - 1, round(x1))))
            y1 = int(max(0, min(image_h - 1, round(y1))))
            x2 = int(max(0, min(image_w - 1, round(x2))))
            y2 = int(max(0, min(image_h - 1, round(y2))))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        if len(indices) == 0:
            return []

        detections = []
        for index in np.array(indices).flatten():
            x, y, w, h = boxes[index]
            class_id = class_ids[index]
            label = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            detections.append(
                Detection(
                    label=label,
                    confidence=confidences[index],
                    xmin=x,
                    ymin=y,
                    xmax=x + w,
                    ymax=y + h,
                )
            )

        return detections

    def detect(self, image):
        blob, scale, pad_x, pad_y = self.preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self.postprocess(image, outputs, scale, pad_x, pad_y)


def draw_detections(image, detections):
    result = image.copy()
    for det in detections:
        cv2.rectangle(result, (det.xmin, det.ymin), (det.xmax, det.ymax), (0, 0, 255), 2)
        text = f"{det.label} {det.confidence:.2f}"
        cv2.putText(result, text, (det.xmin, max(20, det.ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return result


def load_class_names(class_names_path):
    if not class_names_path:
        return ["TrafficSign"]

    with open(class_names_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    return names or ["TrafficSign"]


def run_image(detector, image_path, save_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    detections = detector.detect(image)
    vis = draw_detections(image, detections)

    print(f"detections: {len(detections)}")
    for det in detections:
        print(
            f"{det.label} conf={det.confidence:.3f} "
            f"box=({det.xmin},{det.ymin},{det.xmax},{det.ymax})"
        )

    if save_path:
        cv2.imwrite(save_path, vis)

    cv2.imshow("detector", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(detector, source, save_path=None):
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open source: {source}")

    writer = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.detect(frame)
            vis = draw_detections(frame, detections)

            if save_path and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, 20.0, (vis.shape[1], vis.shape[0]))

            if writer is not None:
                writer.write(vis)

            cv2.imshow("detector", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def build_argparser():
    parser = argparse.ArgumentParser(description="Minimal YOLO ONNX detector for this project.")
    parser.add_argument("--model", required=True, help="Path to an exported YOLO ONNX model.")
    parser.add_argument("--image", help="Run detection on a single image.")
    parser.add_argument("--video", help="Run detection on a video file or camera index, e.g. 0.")
    parser.add_argument("--classes", help="Optional text file with one class name per line.")
    parser.add_argument("--save", help="Optional output image/video path.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    return parser


def main():
    args = build_argparser().parse_args()

    if not args.image and not args.video:
        raise ValueError("Please provide --image or --video.")

    class_names = load_class_names(args.classes)
    detector = YOLOOnnxDetector(
        model_path=args.model,
        class_names=class_names,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.imgsz,
    )

    if args.image:
        run_image(detector, args.image, args.save)
    else:
        run_video(detector, args.video, args.save)


if __name__ == "__main__":
    main()
