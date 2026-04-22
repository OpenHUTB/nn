# custom_faster_rcnn.py

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2
import numpy as np
import os

class CustomFasterRCNN:
    def __init__(self, num_classes, backbone_name="resnet50", pretrained_backbone=True, device=None, model_dir="models"):
        """
        Initialize Faster-RCNN for object detection.

        Args:
            num_classes (int): Number of classes including background.
            backbone_name (str): Backbone network.
            pretrained_backbone (bool): Use pretrained backbone.
            device (torch.device): Device to run inference.
            model_dir (str): Folder to save/load model weights.
        """
        # self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._select_device()
        self.num_classes = num_classes
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = self._build_model(backbone_name, pretrained_backbone)
        self.model.to(self.device)
        self.model.eval()
        
    def _select_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Selected Device: {self.device}")


    def _build_model(self, backbone_name, pretrained_backbone):
        if backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=pretrained_backbone)
            modules = list(backbone.children())[:-2]
            backbone = torch.nn.Sequential(*modules)
            backbone.out_channels = 2048
        elif backbone_name == "mobilenet_v3_large":
            backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained_backbone).features
            backbone.out_channels = 960
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )

        model = FasterRCNN(
            backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        return model

    @torch.no_grad()
    def predict(self, frame, threshold=0.5):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torchvision.transforms.functional.to_tensor(img_rgb).to(self.device)
        if (self.device == "cuda"):
            img_tensor = img_tensor.half()  # convert to FP16
            self.model.half()
        outputs = self.model([img_tensor])[0]

        boxes, labels, scores = [], [], []
        for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
            if score >= threshold:
                boxes.append(box.cpu().numpy().astype(int))
                labels.append(int(label.cpu()))
                scores.append(float(score.cpu()))
        return boxes, labels, scores

    @staticmethod
    def draw_detections(frame, boxes, labels=None, scores=None, color=(0, 255, 0)):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if labels is not None and scores is not None:
                text = f"{labels[i]}:{scores[i]:.2f}"
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def save(self, filename="faster_rcnn.pth"):
        """Save model weights to model_dir"""
        path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, filename="faster_rcnn.pth"):
        """Load model weights from model_dir"""
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = CustomFasterRCNN(num_classes=4)

    # Optional: load pretrained weights
    # detector.load("my_faster_rcnn.pth")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, labels, scores = detector.predict(frame)
        frame = detector.draw_detections(frame, boxes, labels, scores)
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) == 27:
            break

    # Optional: save model weights
    # detector.save("my_faster_rcnn.pth")

    cap.release()
    cv2.destroyAllWindows()
