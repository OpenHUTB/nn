import os
import cv2
import time
import carla
import pygame
import logging
import numpy as np
from ultralytics import YOLO
from functools import lru_cache
from typing import Tuple, List, Optional, Dict, Any

# Suppress YOLO logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

class LaneDetectionConfig:
    def __init__(self,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 rho: int = 1,
                 theta: float = np.pi/180,
                 threshold: int = 50,
                 min_line_length: int = 100,
                 max_line_gap: int = 50,
                 roi_top: float = 0.5,
                 roi_bottom: float = 1.0,
                 lane_width_pixels: int = 100,
                 min_lane_length: int = 50):
            
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.lane_width_pixels = lane_width_pixels
        self.min_lane_length = min_lane_length

class LaneDetector:
    def __init__(self):
        self.config = LaneDetectionConfig()
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        edges = self._preprocess_frame(frame)
        masked_edges = self._apply_roi(edges)
        lines = self._detect_lines(masked_edges)
        return self._process_lines(frame, lines)
    
    def edges_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        edges = self._preprocess_frame(frame)
        masked_edges = self._apply_roi(edges)
        edges_frame = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2RGB)
        return edges_frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)
    
    def _apply_roi(self, edges: np.ndarray) -> np.ndarray:
        roi_mask = self._create_roi_mask(edges.shape)
        return cv2.bitwise_and(edges, roi_mask)
    
    def _create_roi_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        height, width = shape
        roi_top = int(height * self.config.roi_top)
        
        vertices = np.array([
            (self.config.lane_width_pixels//4, height),
            (width//2 - self.config.lane_width_pixels, roi_top),
            (width//2 + self.config.lane_width_pixels, roi_top),
            (width-self.config.lane_width_pixels//4, height)
        ], dtype=np.int32)
        
        return cv2.fillConvexPoly(np.zeros(shape, dtype=np.uint8), vertices, 255)
    
    def _detect_lines(self, masked_edges: np.ndarray) -> Optional[np.ndarray]:
        return cv2.HoughLinesP(
            masked_edges,
            self.config.rho,
            self.config.theta,
            self.config.threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap
        )
    
    def _process_lines(self, frame: np.ndarray, lines: Optional[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        output = frame.copy()
        detected_lanes = []
        
        if lines is not None:
            left_lines, right_lines = self._separate_lanes(lines)
            detected_lanes.extend(self._average_lanes(left_lines, right_lines, frame.shape[0]))
            self._draw_lanes(output, detected_lanes)
        
        return output, detected_lanes
    
    def _separate_lanes(self, lines: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if abs(slope) < 0.5 or length < self.config.min_lane_length:
                continue
                
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def _average_lanes(self, left_lines: List[np.ndarray], right_lines: List[np.ndarray], 
                      height: int) -> List[np.ndarray]:
        lanes = []
        
        if left_lines:
            left_lane = self._average_lane_lines(left_lines, height)
            if left_lane is not None:
                lanes.append(left_lane)
        
        if right_lines:
            right_lane = self._average_lane_lines(right_lines, height)
            if right_lane is not None:
                lanes.append(right_lane)
        
        return lanes
    
    def _average_lane_lines(self, lines: List[np.ndarray], height: int) -> Optional[np.ndarray]:
        if not lines:
            return None
            
        slopes = []
        intercepts = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            slopes.append(slope)
            intercepts.append(intercept)
        
        if not slopes:
            return None
            
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        
        y1 = height
        y2 = int(height * self.config.roi_top * 1.2)
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        
        return np.array([x1, y1, x2, y2])
    
    def _draw_lanes(self, frame: np.ndarray, lanes: List[np.ndarray]) -> None:
        for lane in lanes:
            cv2.line(frame, 
                    (int(lane[0]), int(lane[1])),
                    (int(lane[2]), int(lane[3])),
                    (0, 255, 0), 2)
    
    def draw_lane_area(self, frame: np.ndarray, lanes: List[np.ndarray]) -> np.ndarray:
        if len(lanes) != 2:
            return frame
            
        output = frame.copy()
        left_lane, right_lane = lanes
        
        pts = np.array([
            [left_lane[0], left_lane[1]],
            [left_lane[2], left_lane[3]],
            [right_lane[2], right_lane[3]],
            [right_lane[0], right_lane[1]]
        ], np.int32)
        
        overlay = output.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
        
        return output

class CarDetectionConfig:
    def __init__(self,
                 model_path: str = 'Sensors/RGBcamera/model/pretrained/yolov8n.pt',
                 confidence_threshold: float = 0.6,
                 iou_threshold: float = 0.5,
                 max_detections: int = 10,
                 processing_width: int = 640,
                 processing_height: int = 360,
                 processing_interval: float = 0.05,
                 known_car_width: float = 1.8,
                 known_car_height: float = 1.5,
                 fov: float = 90.0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.processing_width = processing_width
        self.processing_height = processing_height
        self.processing_interval = processing_interval
        self.known_car_width = known_car_width
        self.known_car_height = known_car_height
        self.fov = fov

class CarDetector:
    def __init__(self):
        self.config = CarDetectionConfig()
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        self.model = YOLO(self.config.model_path)
        # Set model parameters directly
        self.model.conf = self.config.confidence_threshold
        self.model.iou = self.config.iou_threshold
        self.model.max_det = self.config.max_detections
        self.model.verbose = False
        self.model.show = False
        self.model.save = False
        self.model.save_txt = False
        self.model.save_conf = False
        self.model.save_crop = False
        self.model.show_labels = False
        self.model.show_conf = False
        self.model.hide_labels = True
        self.model.hide_conf = True
        self.model.half = False
        self.model.dnn = False
        self.model.plots = False
    
    @lru_cache(maxsize=2)  # Cache last 2 frames to handle potential frame skipping
    def _cached_detection(self, image_bytes: bytes, width: int, height: int) -> Tuple[np.ndarray, float]:
        """Helper method to enable caching of detection results. Takes image as bytes to make it hashable."""
        image = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 3))
        process_image = self._resize_if_needed(image)
        predictions = self._run_detection(process_image)
        output_image = self._process_detections(predictions, image.copy(), width, height)
        return output_image, time.time()

    def detect(self, image: np.ndarray) -> np.ndarray:
        # Convert image to bytes for caching
        image_bytes = image.tobytes()
        height, width = image.shape[:2]
        
        # Get cached result if available
        output_image, _ = self._cached_detection(image_bytes, width, height)
        return output_image
    
    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        if image.shape[1] != self.config.processing_width:
            return cv2.resize(image, (self.config.processing_width, self.config.processing_height))
        return image
    
    def _run_detection(self, image: np.ndarray) -> Any:
        return self.model.predict(
            source=image,
            stream=True,
            verbose=False,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections
        )
    
    def _process_detections(self, predictions: Any, output_image: np.ndarray, 
                          original_width: int, original_height: int) -> np.ndarray:
        for result in predictions:
            boxes = result.boxes
            for box in boxes:
                if not self._is_valid_detection(box):
                    continue
                    
                x1, y1, x2, y2 = self._get_scaled_coordinates(box, original_width, original_height)
                self._draw_detection(output_image, box, x1, y1, x2, y2, original_width)
        
        return output_image
    
    def _is_valid_detection(self, box: Any) -> bool:
        confidence_score = box.conf[0]
        class_id = int(box.cls[0])
        return (self.model.names[class_id] in ['car', 'truck'] and 
                confidence_score >= self.config.confidence_threshold)
    
    def _get_scaled_coordinates(self, box: Any, original_width: int, 
                              original_height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if original_width != self.config.processing_width:
            scale_x = original_width / self.config.processing_width
            scale_y = original_height / self.config.processing_height
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        return x1, y1, x2, y2
    
    def _draw_detection(self, image: np.ndarray, box: Any, x1: int, y1: int, 
                       x2: int, y2: int, original_width: int) -> None:
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        if bbox_width > 20 and bbox_height > 20:
            distance = self.estimate_distance(bbox_width, bbox_height, original_width)
            if 1.0 <= distance <= 100.0:
                label = f'{self.model.names[int(box.cls[0])]} {distance}m'
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def estimate_distance(self, bbox_width: int, bbox_height: int, image_width: int) -> float:
        focal_length = (image_width / 2) / np.tan(np.radians(self.config.fov / 2))
        
        distance_width = (self.config.known_car_width * focal_length) / bbox_width
        distance_height = (self.config.known_car_height * focal_length) / bbox_height
        
        distance = distance_width
        if distance < 1.0:
            distance = distance_height
        if distance > 100.0:
            distance = min(distance_width, distance_height)
            
        return round(distance, 1)