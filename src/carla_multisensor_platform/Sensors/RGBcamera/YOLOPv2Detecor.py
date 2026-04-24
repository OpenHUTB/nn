import os
import cv2
import time
import torch
import carla
import pygame
import traceback
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import Tuple, List, Optional, Dict, Any

import torch.jit
import torch.nn.functional as F

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# OMP_NUM_THREADS = 4
# print(f"OMP_NUM_THREADS = {os.getenv('OMP_NUM_THREADS')}")

class YOLOPv2Config:
    def __init__(self,
                 model_path: str = 'Sensors/RGBcamera/model/pretrained/yolopv2.pt',
                 confidence_threshold: float = 0.8,
                 iou_threshold: float = 0.4,
                 processing_width: int = 640,  # Model's expected input width
                 processing_height: int = 640,  # Model's expected input height
                 processing_interval: float = 0.033,
                 known_car_width: float = 1.8,
                 known_car_height: float = 1.5,
                 fov: float = 90.0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.processing_width = processing_width
        self.processing_height = processing_height
        self.processing_interval = processing_interval
        self.known_car_width = known_car_width
        self.known_car_height = known_car_height
        self.fov = fov

class YOLOPv2Detector:
    def __init__(self):
        self.config = YOLOPv2Config()
        self.model = None
        self.model_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_model()
        
        # Define colors for visualization
        self.colors = {
            'drivable': (0, 255, 0),  # Green for drivable area
            'lane': (0, 0, 255),      # Red for lane markings
            'vehicle': (255, 0, 0)    # Blue for vehicles
        }
           
    def _initialize_model(self) -> None:
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.config.model_path}")
        
        # Load model to the selected device (GPU if available, otherwise CPU)
        self.model = torch.jit.load(self.config.model_path, map_location=self.device)
        self.model.eval()
        
        self.model_loaded = True
        if self.device == 'cuda':
            self._cuda_warmup()
        
    def _cuda_warmup(self):
        dummy_input = torch.zeros(1, 3, self.config.processing_width, self.config.processing_height).to(self.device)
        
        with torch.no_grad():
            for _ in range(7):
                _ = self.model(dummy_input)
        torch.cuda.synchronize()

    def estimate_distance(self, bbox_width: int, bbox_height: int, image_width: int) -> float:
        if isinstance(bbox_width, (np.ndarray, torch.Tensor)):
            bbox_width = bbox_width.item() if bbox_width.size == 1 else bbox_width[0]
        if isinstance(bbox_height, (np.ndarray, torch.Tensor)):
            bbox_height = bbox_height.item() if bbox_height.size == 1 else bbox_height[0]
        if bbox_width <= 0 or bbox_height <= 0:
            return 100.0
            
        focal_length = (image_width / 2) / np.tan(np.radians(self.config.fov / 2))
        
        distance_width = (self.config.known_car_width * focal_length) / bbox_width
        distance_height = (self.config.known_car_height * focal_length) / bbox_height
        
        distance = distance_width
        # print(distance.shape)
        if distance < 1.0:
            distance = distance_height
        if distance > 100.0:
            distance = min(distance_width, distance_height)
            
        return round(distance, 1)

    @lru_cache(maxsize=100)
    def _cached_detect(self, image_hash: int, timestamp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.model_loaded:
            return np.zeros((640, 640, 3), dtype=np.uint8), {
                'vehicles': [],
                'drivable_area': None,
                'lane_markings': None,
                'using_traditional': True
            }
        

        img = self._preprocess_image(self._current_image)
        with torch.no_grad():
            pred = self.model(img)
        
        detections = self._process_predictions(pred)
        output = self._draw_results(self._current_image, detections)
        
        # torch.cuda.synchronize()
               
        return output, detections

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.orig_h, self.orig_w = image.shape[:2]
        self.scale_x = self.orig_w / self.config.processing_width
        self.scale_y = self.orig_h / self.config.processing_height
        self._current_image = image

        image_hash = hash((image.shape, float(image.mean())))
        current_time = time.time()
        
        # Check if we should use cached results based on processing interval
        if hasattr(self, '_last_detect_time') and current_time - self._last_detect_time < self.config.processing_interval:
            return self._cached_detect(image_hash, self._last_detect_time)

        self._last_detect_time = current_time

        return self._cached_detect(image_hash, current_time)
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2
        dh /= 2

        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img_padded, r, (dw, dh)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.resize(image, (self.config.processing_width, self.config.processing_height))
        # image, self.ratio, self.pad = self.letterbox(image, (self.config.processing_width, self.config.processing_height))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device).contiguous()
        image = image
        return image

    def split_for_trace_model(self, predictions, anchor_grid):
        all_boxes = []
        all_scores = []
        strides = [8, 16, 32]
        
        for i in range(3):
            batch_size, _, ny, nx = predictions[i].shape

            # Reshape: [B, 255, ny, nx] --> [B, 3, 85, ny, nx] --> [B, 3, ny, nx, 85]
            predictions[i] = predictions[i].view(batch_size, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous() # [1, 3, 80, 80, 85]
            grid = self. _make_grid(nx, ny).to(predictions[i].device) # [1, 1, 80, 80, 2]
            y = predictions[i].sigmoid()  # [1, 3, 80, 80, 85]
            anchor = anchor_grid[i].view(1, 3, 1, 1, 2).to(self.device).contiguous()     # [1, 3, 1, 1, 2]

            xy = (y[..., 0:2] * 2 - 0.5 + grid) * strides[i] # center x/y
            wh = (y[..., 2:4] * 2) ** 2 * anchor     # width/height
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2

            boxes = torch.cat((x1y1, x2y2), dim=-1)  # [B, 3, nx, ny, 4]
            scores = y[..., 4:5] * y[..., 5:]       # [B, 3, ny, nx, num_classes]

            all_boxes.append(boxes.view(batch_size, -1, 4))
            all_scores.append(scores.view(batch_size, -1, scores.shape[-1]))
        
        boxes = torch.cat(all_boxes, dim=1).squeeze(0)      # [N, 4]
        scores = torch.cat(all_scores, dim=1).squeeze(0)    # [N, num_classes]
        return boxes, scores
        
    def _make_grid(self, nx: int=80, ny: int=80):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), dim=2).view((1, 1, ny, nx, 2)).float()

    def rescale_coordinate(self, x1, y1, x2, y2):
        x_scale = self._current_image.shape[1] / self.config.processing_width
        y_scale = self._current_image.shape[0] / self.config.processing_height

        x1 = int(x1 * x_scale)
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)

        return x1, y1, x2, y2

    def _process_predictions(self, pred) -> Dict[str, Any]:
        detections = {
            'vehicles': [],
            'drivable_area': None,
            'lane_markings': None,
            'using_traditional': False
        }
        
        try:
            # YOLOPv2 returns a tuple of (detections_tuple, drivable_area, lane_markings)
            if not isinstance(pred, tuple) or len(pred) < 3:
                print(f"Unexpected prediction format: {type(pred)}, length: {len(pred) if isinstance(pred, tuple) else 'N/A'}")
                return detections
            
            [det_tensor, anchor_grid], da_seg, ll_seg = pred

            boxes, scores = self.split_for_trace_model(det_tensor, anchor_grid)

            boxes[:, [0, 2]] *= self.scale_x
            boxes[:, [1, 3]] *= self.scale_y
            boxes[:, 0].clamp_(0, self.orig_w)  # x1
            boxes[:, 1].clamp_(0, self.orig_h)  # y1
            boxes[:, 2].clamp_(0, self.orig_w)  # x2
            boxes[:, 3].clamp_(0, self.orig_h)  # y2


            if len(boxes) > 0:
                filtered = self.non_max_suppression(boxes, scores, self.config.iou_threshold)
                for box in filtered:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    distance = self.estimate_distance(bbox_width, bbox_height, self.orig_w)

                    detections['vehicles'].append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': int(cls),
                        'distance': distance
                    })

            # Process drivable area segmentation
            if isinstance(da_seg, torch.Tensor):
                da_mask = da_seg.squeeze().cpu().numpy()
                if da_mask.shape[0] == 2:
                    da_mask = da_mask[1]
                if da_mask.ndim == 2:
                    da_mask = cv2.resize(da_mask, (self.orig_w, self.orig_h), interpolation=cv2.INTER_NEAREST)
                    da_mask = (da_mask > 0.5).astype(np.uint8) * 255
                    detections['drivable_area'] = da_mask
            
            # Process lane line segmentation
            if isinstance(ll_seg, torch.Tensor):
                ll_mask = ll_seg.squeeze().cpu().numpy()
                if ll_mask.ndim == 2:
                    ll_mask = cv2.resize(ll_mask, (self.orig_w, self.orig_h), interpolation=cv2.INTER_NEAREST)
                    ll_mask = (ll_mask > 0.5).astype(np.uint8) * 255
                    detections['lane_markings'] = ll_mask
            
            return detections
            
        except Exception as e:
            # print(f"Error processing predictions: {str(e)}")
            traceback.print_exc()
            return detections
        
    def bbox_ious(self, box1, box2):
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])

        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou
    
    def non_max_suppression(self, boxes, scores, iou_threshold):
        CAR_CLASS_ID = 3
        scores = scores[:, CAR_CLASS_ID]
        conf_mask = scores > self.config.confidence_threshold
            
        class_boxes = boxes[conf_mask]
        class_scores = scores[conf_mask]

        sorted_indices = torch.argsort(class_scores, descending=True)
        class_boxes = class_boxes[sorted_indices]
        class_scores = class_scores[sorted_indices]

        keep = []

        while len(class_boxes) > 0:
            box = class_boxes[0]
            score = class_scores[0]
            keep.append(torch.cat([box, score.unsqueeze(0), torch.tensor([CAR_CLASS_ID], device=box.device)]))

            if len(class_boxes) == 1:
                break

            IoU = self.bbox_ious(box.unsqueeze(0), class_boxes[1:])
            class_boxes = class_boxes[1:][IoU < iou_threshold]
            class_scores = class_scores[1:][IoU < iou_threshold]

        return keep
    
    def _draw_results(self, image: np.ndarray, detections: Dict[str, Any]) -> np.ndarray:
        try:
            output = image.copy()
            
            # Draw drivable area
            if detections['drivable_area'] is not None:
                try:
                    # Create overlay for drivable area
                    overlay = output.copy()
                    mask = detections['drivable_area'] > 0
                    overlay[mask] = self.colors['drivable']
                    # Blend with original image
                    cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
                except Exception as e:
                    print(f"Error drawing drivable area: {str(e)}")
            
            # Draw lane markings
            if detections['lane_markings'] is not None:
                try:
                    # Create overlay for lane markings
                    overlay = output.copy()
                    mask = detections['lane_markings'] > 0
                    overlay[mask] = self.colors['lane']
                    # Blend with original image
                    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
                except Exception as e:
                    print(f"Error drawing lane markings: {str(e)}")
            
            # Draw vehicle detections
            for vehicle in detections['vehicles']:
                try:
                    x1, y1, x2, y2 = vehicle['bbox']
                    label = f"car - {vehicle['distance']}m"
                    
                    # Draw bounding box
                    cv2.rectangle(output, (x1, y1), (x2, y2), self.colors['vehicle'], 2)
                    
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(output, (x1, y1-label_h-4), (x1+label_w, y1), self.colors['vehicle'], -1)
                    
                    # Draw label text
                    cv2.putText(output, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                except Exception as e:
                    print(f"Error drawing vehicle detection: {str(e)}")
            
            # Draw status text
            status = "YOLOPv2 Detection" if not detections['using_traditional'] else "Traditional Detection"
            device_name = str(self.device).split(':')[-1]  # Extract device name (e.g., 'cuda:0' -> '0')
            cv2.putText(output, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(output, f"Device: {device_name}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return output
            
        except Exception as e:
            print(f"Error in _draw_results: {str(e)}")
            return image.copy()