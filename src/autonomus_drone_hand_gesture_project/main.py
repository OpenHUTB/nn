# -*- coding: utf-8 -*-
"""
Gesture Controlled AirSim Drone - Optimized Version
Pure OpenCV implementation with English interface
Author: xiaoshiyuan888
"""

import sys
import os
import time
import traceback
import json
import math
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from collections import deque, Counter

print("=" * 60)
print("Gesture Controlled Drone - Optimized Version")
print("=" * 60)

# ========== Fix import path ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# ========== Core module imports ==========
def safe_import():
    """Safely import all modules"""
    modules_status = {}

    try:
        from PIL import Image, ImageDraw, ImageFont
        modules_status['PIL'] = True
        print("[PIL] ✓ Image processing library ready")
    except Exception as e:
        modules_status['PIL'] = False
        print(f"[PIL] ✗ Import failed: {e}")
        return None, modules_status

    try:
        import cv2
        import numpy as np
        modules_status['OpenCV'] = True
        print("[OpenCV] ✓ Computer vision library ready")
    except Exception as e:
        modules_status['OpenCV'] = False
        print(f"[OpenCV] ✗ Import failed: {e}")
        return None, modules_status

    airsim_module = None
    try:
        airsim_module = __import__('airsim')
        modules_status['AirSim'] = True
        print(f"[AirSim] ✓ Successfully imported")
    except ImportError:
        print("\n" + "!" * 60)
        print("⚠ AirSim library NOT FOUND!")
        print("!" * 60)
        print("To install AirSim, run:")
        print("1. First install: pip install msgpack-rpc-python")
        print("2. Then install: pip install airsim")
        print("\nOr from source:")
        print("  pip install git+https://github.com/microsoft/AirSim.git")
        print("!" * 60)

        print("\nContinue without AirSim? (y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            sys.exit(1)

    return {
        'cv2': cv2,
        'np': np,
        'PIL': {'Image': Image, 'ImageDraw': ImageDraw, 'ImageFont': ImageFont},
        'airsim': airsim_module
    }, modules_status


# Execute imports
libs, status = safe_import()
if not status.get('OpenCV', False) or not status.get('PIL', False):
    print("\n❌ Core libraries missing, cannot start.")
    input("Press Enter to exit...")
    sys.exit(1)

print("-" * 60)
print("✅ Environment check passed, initializing...")
print("-" * 60)

# Unpack libraries
cv2, np = libs['cv2'], libs['np']
Image, ImageDraw, ImageFont = libs['PIL']['Image'], libs['PIL']['ImageDraw'], libs['PIL']['ImageFont']


# ========== Configuration Manager ==========
class ConfigManager:
    """Configuration Manager"""

    def __init__(self):
        self.config_file = os.path.join(current_dir, 'gesture_config.json')
        self.default_config = {
            'camera': {
                'index': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'gesture': {
                'skin_lower': [0, 30, 60],
                'skin_upper': [25, 255, 255],
                'min_hand_area': 3000,
                'history_size': 10,
                'smooth_frames': 5,
                'min_confidence': 0.6,
                'detection_interval': 1,
                'single_finger_angle_threshold': 60  # 新增：单个手指角度阈值
            },
            'drone': {
                'velocity': 2.5,
                'duration': 0.3,
                'altitude': -10.0,
                'control_interval': 0.3
            },
            'display': {
                'show_fps': True,
                'show_confidence': True,
                'show_help': True,
                'show_contours': True,
                'show_bbox': True,
                'show_fingertips': True  # 新增：显示指尖
            },
            'performance': {
                'target_fps': 30,
                'resize_factor': 1.0,
                'enable_multiprocessing': False
            }
        }
        self.config = self.load_config()

    def load_config(self):
        """Load configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    config = self.default_config.copy()
                    self._merge_config(config, loaded_config)
                    print("✓ Configuration loaded from file")
                    return config
            except Exception as e:
                print(f"⚠ Failed to load config: {e}, using defaults")
                return self.default_config.copy()
        else:
            print("✓ Using default configuration")
            return self.default_config.copy()

    def _merge_config(self, base, update):
        """Recursively merge configurations"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def save_config(self):
        """Save configuration"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print("✓ Configuration saved")
        except Exception as e:
            print(f"⚠ Failed to save config: {e}")

    def get(self, *keys):
        """Get configuration value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def set(self, *keys, value):
        """Set configuration value"""
        if len(keys) == 0:
            return

        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        self.save_config()


config = ConfigManager()


# ========== Robust Gesture Recognizer ==========
class RobustGestureRecognizer:
    """Robust Gesture Recognizer - Pure OpenCV Implementation"""

    def __init__(self):
        self.config = config.get('gesture')

        # Gesture history and smoothing
        self.history_size = self.config['history_size']
        self.gesture_history = deque(maxlen=self.history_size)
        self.confidence_history = deque(maxlen=self.history_size)
        self.current_gesture = "Waiting"
        self.current_confidence = 0.0

        # Hand tracking
        self.last_hand_position = None
        self.hand_tracking = False
        self.track_window = None

        # Performance statistics
        self.process_times = deque(maxlen=30)
        self.frame_counter = 0
        self.detection_interval = self.config['detection_interval']

        # Gesture color mapping
        self.gesture_colors = {
            "Stop": (0, 0, 255),  # Red
            "Forward": (0, 255, 0),  # Green
            "Up": (255, 255, 0),  # Cyan
            "Down": (255, 0, 255),  # Purple
            "Left": (255, 165, 0),  # Orange
            "Right": (0, 165, 255),  # Light Blue
            "Waiting": (200, 200, 200),  # Gray
            "Error": (255, 0, 0)  # Blue
        }

        # Skin detection parameters
        self.skin_lower = np.array(self.config['skin_lower'], dtype=np.uint8)
        self.skin_upper = np.array(self.config['skin_upper'], dtype=np.uint8)

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=True
        )

        # Morphological operation kernels
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Fingertip detection
        self.prev_fingertips = None
        self.fingertip_stability_threshold = 3

        print("✓ Robust gesture recognizer initialized (Pure OpenCV)")

    def preprocess_frame(self, frame):
        """Preprocess frame"""
        # Flip image horizontally for intuitive control
        frame = cv2.flip(frame, 1)

        # Adjust size for performance
        resize_factor = config.get('performance', 'resize_factor')
        if resize_factor != 1.0:
            new_width = int(frame.shape[1] * resize_factor)
            new_height = int(frame.shape[0] * resize_factor)
            frame = cv2.resize(frame, (new_width, new_height))

        return frame

    def detect_skin(self, frame):
        """Detect skin regions"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin mask
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Combine skin mask and foreground mask
        combined_mask = cv2.bitwise_and(skin_mask, fg_mask)

        # Morphological operations
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # Gaussian blur
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

        return combined_mask

    def find_hand_contours(self, mask):
        """Find hand contours"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest contour (assumed to be hand)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter out too small contours
            if area < self.config['min_hand_area']:
                continue

            # Calculate contour perimeter
            perimeter = cv2.arcLength(contour, True)

            # Filter out too simple contours
            if perimeter < 100:
                continue

            return contour

        return None

    def detect_fingertips_advanced(self, contour):
        """Advanced fingertip detection"""
        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Find convex hull
        hull = cv2.convexHull(approx, returnPoints=False)

        if hull is None or len(hull) < 3:
            return [], 0

        # Find convexity defects
        defects = cv2.convexityDefects(approx, hull)

        if defects is None:
            return [], 0

        fingertips = []
        defects_points = []

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])

            defects_points.append((start, end, far))

            # Calculate triangle sides
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            # Calculate angle
            if b * c != 0:
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                angle_degrees = math.degrees(angle)

                # If angle is less than 90 degrees, it might be a finger
                if angle_degrees < 90:
                    # Check if this is a deep defect (far from hull)
                    if d > 1000:  # Adjust this threshold based on your needs
                        # Both start and end points could be fingertips
                        for point in [start, end]:
                            if point not in fingertips:
                                fingertips.append(point)

        # If we found few fingertips, try alternative method
        if len(fingertips) <= 2:
            # Use convex hull points as fingertips
            hull_points = cv2.convexHull(approx, returnPoints=True)

            # Find extreme points in hull
            if len(hull_points) > 0:
                hull_points = hull_points.reshape(-1, 2)

                # Sort by y-coordinate (top to bottom)
                hull_points = hull_points[hull_points[:, 1].argsort()]

                # Take top points as fingertips
                num_fingertips = min(5, len(hull_points))
                for i in range(num_fingertips):
                    point = tuple(hull_points[i])
                    if point not in fingertips:
                        fingertips.append(point)

        return fingertips, len(fingertips)

    def analyze_hand_contour(self, contour, frame_shape):
        """Analyze hand contour"""
        if contour is None:
            return None, 0.0

        # Calculate contour area
        area = cv2.contourArea(contour)

        # Calculate contour center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, 0.0

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Advanced fingertip detection
        fingertips, finger_count = self.detect_fingertips_advanced(contour)

        # Calculate hand position (normalized)
        h_img, w_img = frame_shape[:2]
        norm_x = cx / w_img
        norm_y = cy / h_img

        # Calculate gesture confidence
        confidence = self.calculate_confidence(area, finger_count, len(contour))

        # Return result
        result = {
            'contour': contour,
            'center': (cx, cy),
            'bbox': (x, y, x + w, y + h),
            'finger_count': finger_count,
            'fingertips': fingertips,
            'area': area,
            'position': (norm_x, norm_y),
            'confidence': confidence
        }

        return result, confidence

    def calculate_confidence(self, area, finger_count, contour_length):
        """Calculate gesture confidence"""
        confidence = 0.5  # Base confidence

        # Confidence based on area
        if area > 5000:
            confidence += 0.2
        elif area > 3000:
            confidence += 0.1

        # Confidence based on finger count
        if 0 <= finger_count <= 5:
            confidence += 0.2

        # Confidence based on contour complexity
        if contour_length > 200:
            confidence += 0.1

        return min(confidence, 1.0)

    def recognize_gesture(self, hand_data):
        """Recognize gesture based on hand data"""
        if hand_data is None:
            return "Waiting", 0.3

        finger_count = hand_data['finger_count']
        norm_x, norm_y = hand_data['position']
        confidence = hand_data['confidence']
        fingertips = hand_data.get('fingertips', [])

        # Special case: single finger detection
        if finger_count == 1:
            # Additional check for single finger gesture
            if len(fingertips) == 1:
                # Check if the fingertip is above the hand center (typical for pointing)
                cx, cy = hand_data['center']
                fingertip = fingertips[0]

                if fingertip[1] < cy - 20:  # Fingertip is above hand center
                    return "Forward", confidence * 0.9
                else:
                    return "Stop", confidence * 0.7
            return "Forward", confidence * 0.8

        # No fingers (fist)
        elif finger_count == 0:
            return "Stop", confidence * 0.9

        # Two fingers (V sign)
        elif finger_count == 2:
            return "Forward", confidence * 0.7

        # Three fingers
        elif finger_count == 3:
            # Judge direction based on position
            if norm_y < 0.4:
                return "Up", confidence * 0.7
            elif norm_y > 0.6:
                return "Down", confidence * 0.7
            else:
                return "Stop", confidence * 0.6

        # Four or five fingers (open hand)
        elif finger_count >= 4:
            # Judge direction based on hand position
            if norm_x < 0.3:
                return "Left", confidence * 0.8
            elif norm_x > 0.7:
                return "Right", confidence * 0.8
            elif norm_y < 0.4:
                return "Up", confidence * 0.8
            elif norm_y > 0.6:
                return "Down", confidence * 0.8
            else:
                return "Stop", confidence * 0.7

        return "Waiting", confidence * 0.5

    def smooth_gesture(self, new_gesture, new_confidence):
        """Smooth gesture output"""
        # Add to history
        self.gesture_history.append(new_gesture)
        self.confidence_history.append(new_confidence)

        # If not enough history, return directly
        if len(self.gesture_history) < 3:
            self.current_gesture = new_gesture
            self.current_confidence = new_confidence
            return new_gesture, new_confidence

        # Count most common gesture
        gesture_counter = Counter(self.gesture_history)
        most_common_gesture, most_common_count = gesture_counter.most_common(1)[0]

        # Calculate average confidence
        avg_confidence = np.mean(list(self.confidence_history))

        # If the most common gesture appears enough times, use it
        smooth_frames = self.config['smooth_frames']
        if most_common_count >= smooth_frames:
            self.current_gesture = most_common_gesture
            self.current_confidence = avg_confidence

        return self.current_gesture, self.current_confidence

    def visualize_detection(self, frame, hand_data, gesture, confidence):
        """Visualize detection results"""
        if hand_data is None:
            return frame

        # Get display configuration
        show_contours = config.get('display', 'show_contours')
        show_bbox = config.get('display', 'show_bbox')
        show_fingertips = config.get('display', 'show_fingertips')

        # Draw contour
        if show_contours and 'contour' in hand_data:
            cv2.drawContours(frame, [hand_data['contour']], -1, (0, 255, 0), 2)

        # Draw bounding box
        if show_bbox and 'bbox' in hand_data:
            x1, y1, x2, y2 = hand_data['bbox']
            color = self.gesture_colors.get(gesture, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Show gesture label
            label = f"{gesture}"
            if config.get('display', 'show_confidence'):
                label += f" ({confidence:.0%})"

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw text background
            cv2.rectangle(frame,
                          (x1, y1 - text_height - 10),
                          (x1 + text_width, y1),
                          color, -1)

            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw center point
        if 'center' in hand_data:
            cx, cy = hand_data['center']
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Center", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw fingertips
        if show_fingertips and 'fingertips' in hand_data:
            for i, point in enumerate(hand_data['fingertips']):
                cv2.circle(frame, point, 4, (255, 0, 0), -1)
                cv2.putText(frame, f"F{i + 1}", (point[0] + 5, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Show finger count
        if 'finger_count' in hand_data:
            finger_text = f"Fingers: {hand_data['finger_count']}"
            cv2.putText(frame, finger_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def recognize(self, frame):
        """Recognize gesture"""
        start_time = time.time()

        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)

            # Detect every few frames to improve performance
            if self.frame_counter % self.detection_interval != 0:
                self.frame_counter += 1
                return self.current_gesture, self.current_confidence, processed_frame

            # Detect skin
            skin_mask = self.detect_skin(processed_frame)

            # Find hand contour
            hand_contour = self.find_hand_contours(skin_mask)

            # Analyze hand contour
            hand_data, confidence = self.analyze_hand_contour(hand_contour, processed_frame.shape)

            # Recognize gesture
            if hand_data is not None:
                gesture, raw_confidence = self.recognize_gesture(hand_data)
                confidence = max(confidence, raw_confidence)
            else:
                gesture, confidence = "Waiting", 0.3

            # Smooth gesture
            final_gesture, final_confidence = self.smooth_gesture(gesture, confidence)

            # Visualize results
            if hand_data is not None:
                processed_frame = self.visualize_detection(
                    processed_frame, hand_data, final_gesture, final_confidence
                )

            # Update counter
            self.frame_counter += 1

            # Calculate processing time
            process_time = (time.time() - start_time) * 1000
            self.process_times.append(process_time)

            return final_gesture, final_confidence, processed_frame

        except Exception as e:
            print(f"⚠ Gesture recognition error: {e}")
            return "Error", 0.0, frame

    def get_performance_stats(self):
        """Get performance statistics"""
        if len(self.process_times) == 0:
            return 0.0

        return np.mean(list(self.process_times))

    def set_simulated_gesture(self, gesture):
        """Set simulated gesture"""
        self.current_gesture = gesture
        self.current_confidence = 0.9


# ========== Simple Drone Controller ==========
class SimpleDroneController:
    """Simple Drone Controller"""

    def __init__(self, airsim_module):
        self.airsim = airsim_module
        self.client = None
        self.connected = False
        self.flying = False

        # Control parameters
        self.velocity = config.get('drone', 'velocity')
        self.duration = config.get('drone', 'duration')
        self.altitude = config.get('drone', 'altitude')
        self.control_interval = config.get('drone', 'control_interval')

        # Control state
        self.last_control_time = 0
        self.last_gesture = None

        print("✓ Simple drone controller initialized")

    def connect(self):
        """Connect to AirSim drone"""
        if self.connected:
            return True

        if self.airsim is None:
            print("⚠ AirSim not available, using simulation mode")
            self.connected = True
            return True

        print("Connecting to AirSim...")

        try:
            self.client = self.airsim.MultirotorClient()
            self.client.confirmConnection()
            print("✅ Connected to AirSim!")

            self.client.enableApiControl(True)
            print("✅ API control enabled")

            self.client.armDisarm(True)
            print("✅ Drone armed")

            self.connected = True
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\nContinue with simulation mode? (y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                self.connected = True
                print("✅ Using simulation mode")
                return True

            return False

    def takeoff(self):
        """Take off"""
        if not self.connected:
            return False

        try:
            if self.airsim is None or self.client is None:
                print("✅ Simulated takeoff")
                self.flying = True
                return True

            print("Taking off...")
            self.client.takeoffAsync().join()
            time.sleep(1)

            # Ascend to specified altitude
            self.client.moveToZAsync(self.altitude, 3).join()

            self.flying = True
            print("✅ Drone took off successfully")
            return True
        except Exception as e:
            print(f"❌ Takeoff failed: {e}")
            return False

    def land(self):
        """Land"""
        if not self.connected:
            return False

        try:
            if self.airsim is None or self.client is None:
                print("✅ Simulated landing")
                self.flying = False
                return True

            print("Landing...")
            self.client.landAsync().join()
            self.flying = False
            print("✅ Drone landed")
            return True
        except Exception as e:
            print(f"Landing failed: {e}")
            return False

    def move_by_gesture(self, gesture, confidence):
        """Move based on gesture"""
        if not self.connected or not self.flying:
            return False

        # Check control interval
        current_time = time.time()
        if current_time - self.last_control_time < self.control_interval:
            return False

        # Check confidence threshold
        min_confidence = config.get('gesture', 'min_confidence')
        if confidence < min_confidence:
            return False

        try:
            if self.airsim is None or self.client is None:
                print(f"Simulated move: {gesture}")
                self.last_control_time = current_time
                self.last_gesture = gesture
                return True

            success = False

            if gesture == "Up":
                self.client.moveByVelocityZAsync(0, 0, -self.velocity, self.duration)
                success = True
            elif gesture == "Down":
                self.client.moveByVelocityZAsync(0, 0, self.velocity, self.duration)
                success = True
            elif gesture == "Left":
                self.client.moveByVelocityAsync(-self.velocity, 0, 0, self.duration)
                success = True
            elif gesture == "Right":
                self.client.moveByVelocityAsync(self.velocity, 0, 0, self.duration)
                success = True
            elif gesture == "Forward":
                self.client.moveByVelocityAsync(0, -self.velocity, 0, self.duration)
                success = True
            elif gesture == "Stop":
                self.client.hoverAsync()
                success = True

            if success:
                self.last_control_time = current_time
                self.last_gesture = gesture

            return success
        except Exception as e:
            print(f"Control command failed: {e}")
            return False

    def emergency_stop(self):
        """Emergency stop"""
        if self.connected:
            try:
                if self.flying and self.client is not None:
                    print("Emergency landing...")
                    self.land()
                if self.client is not None:
                    self.client.armDisarm(False)
                    self.client.enableApiControl(False)
                    print("✅ Emergency stop complete")
            except:
                pass

        self.connected = False
        self.flying = False


# ========== Simple UI Renderer ==========
class SimpleUIRenderer:
    """Simple UI Renderer"""

    def __init__(self):
        # Color definitions
        self.colors = {
            'title': (0, 255, 255),  # Cyan
            'connected': (0, 255, 0),  # Green
            'disconnected': (0, 0, 255),  # Red
            'flying': (0, 255, 0),  # Green
            'landed': (255, 165, 0),  # Orange
            'warning': (0, 165, 255),  # Light Blue
            'info': (255, 255, 255),  # White
            'help': (255, 200, 100)  # Light Orange
        }

        print("✓ Simple UI renderer initialized")

    def draw_status_bar(self, frame, drone_controller, gesture, confidence, fps, process_time):
        """Draw status bar"""
        h, w = frame.shape[:2]

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Title
        title = "Gesture Controlled Drone"
        cv2.putText(frame, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['title'], 2)

        # Connection status
        status_color = self.colors['connected'] if drone_controller.connected else self.colors['disconnected']
        status_text = f"Drone: {'Connected' if drone_controller.connected else 'Disconnected'}"
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

        # Flight status
        flight_color = self.colors['flying'] if drone_controller.flying else self.colors['landed']
        flight_text = f"Flight: {'Flying' if drone_controller.flying else 'Landed'}"
        cv2.putText(frame, flight_text, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, flight_color, 1)

        # Gesture information
        if confidence > 0.7:
            gesture_color = (0, 255, 0)  # Green
        elif confidence > 0.5:
            gesture_color = (255, 165, 0)  # Orange
        else:
            gesture_color = (200, 200, 200)  # Gray

        gesture_text = f"Gesture: {gesture}"
        if config.get('display', 'show_confidence'):
            gesture_text += f" ({confidence:.0%})"

        cv2.putText(frame, gesture_text, (w // 2, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 1)

        # Performance information
        if config.get('display', 'show_fps'):
            perf_text = f"FPS: {fps:.1f}"
            if process_time > 0:
                perf_text += f" | Delay: {process_time:.1f}ms"

            cv2.putText(frame, perf_text, (w - 200, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)

        return frame

    def draw_help_bar(self, frame):
        """Draw help bar"""
        if not config.get('display', 'show_help'):
            return frame

        h, w = frame.shape[:2]

        # Draw bottom help bar
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)

        # Help text
        help_text = "C:Connect  Space:Takeoff/Land  ESC:Exit  W/A/S/D/F/X:Keyboard"
        cv2.putText(frame, help_text, (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['help'], 1)

        return frame

    def draw_warning(self, frame, message):
        """Draw warning message"""
        h, w = frame.shape[:2]

        # Draw warning at top
        warning_bg = np.zeros((40, w, 3), dtype=np.uint8)
        warning_bg[:, :] = (0, 69, 255)  # Orange

        frame[100:140, 0:w] = cv2.addWeighted(
            frame[100:140, 0:w], 0.3,
            warning_bg, 0.7, 0
        )

        # Draw warning text
        cv2.putText(frame, message, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['warning'], 1)

        return frame


# ========== Performance Monitor ==========
class PerformanceMonitor:
    """Performance Monitor"""

    def __init__(self):
        self.frame_times = deque(maxlen=60)
        self.last_update = time.time()
        self.fps = 0
        self.frame_count = 0

    def update(self):
        """Update performance data"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_count += 1

        # Calculate FPS once per second
        if current_time - self.last_update >= 1.0:
            if len(self.frame_times) > 1:
                time_diff = self.frame_times[-1] - self.frame_times[0]
                if time_diff > 0:
                    self.fps = len(self.frame_times) / time_diff
                else:
                    self.fps = 0
            self.last_update = current_time

    def get_stats(self):
        """Get performance statistics"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count
        }


# ========== Main Program ==========
def main():
    """Main function"""
    # Initialize components
    print("Initializing components...")

    gesture_recognizer = RobustGestureRecognizer()
    drone_controller = SimpleDroneController(libs['airsim'])
    ui_renderer = SimpleUIRenderer()
    performance_monitor = PerformanceMonitor()

    # Initialize camera
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('camera', 'width'))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('camera', 'height'))
            cap.set(cv2.CAP_PROP_FPS, config.get('camera', 'fps'))

            # Get actual parameters
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

            print(f"✓ Camera initialized")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  FPS: {actual_fps}")
        else:
            print("❌ Camera not available, using simulation mode")
            cap = None
    except Exception as e:
        print(f"⚠ Camera init failed: {e}")
        cap = None

    # Display welcome message
    print("\n" + "=" * 60)
    print("Gesture Controlled Drone - Optimized Version")
    print("=" * 60)
    print("System Status:")
    print(f"  Camera: {'Connected' if cap else 'Simulation Mode'}")
    print(f"  Gesture Recognition: Pure OpenCV")
    print(f"  AirSim: {'Available' if libs['airsim'] else 'Simulation Mode'}")
    print("=" * 60)

    # Display operation instructions
    print("\nOperation Instructions:")
    print("1. Press [C] to connect drone (AirSim simulator)")
    print("2. Press [SPACE] to takeoff/land")
    print("3. Gesture Control:")
    print("   - Fist (0 fingers): Stop")
    print("   - Index finger (1 finger): Forward")
    print("   - Open hand (4-5 fingers): Control direction by hand position")
    print("   * Gesture confidence > 60% to execute")
    print("4. Keyboard Control:")
    print("   [W]Up [S]Down [A]Left [D]Right [F]Forward [X]Stop")
    print("5. Press [H] to toggle help display")
    print("6. Press [R] to reset gesture recognition")
    print("7. Press [ESC] to exit safely")
    print("=" * 60)
    print("Program started successfully!")
    print("-" * 60)

    # Keyboard gesture mapping
    key_to_gesture = {
        ord('w'): "Up", ord('W'): "Up",
        ord('s'): "Down", ord('S'): "Down",
        ord('a'): "Left", ord('A'): "Left",
        ord('d'): "Right", ord('D'): "Right",
        ord('f'): "Forward", ord('F'): "Forward",
        ord('x'): "Stop", ord('X'): "Stop",
    }

    # Main loop
    print("\nEntering main loop, press ESC to exit...")

    try:
        while True:
            # Update performance monitor
            performance_monitor.update()

            # Read camera frame
            if cap:
                ret, frame = cap.read()
                if not ret:
                    # Create blank frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    gesture, confidence = "Camera Error", 0.0
                else:
                    # Gesture recognition
                    gesture, confidence, frame = gesture_recognizer.recognize(frame)
            else:
                # Simulation mode
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                gesture, confidence = gesture_recognizer.current_gesture, gesture_recognizer.current_confidence

            # Get performance statistics
            perf_stats = performance_monitor.get_stats()
            process_time = gesture_recognizer.get_performance_stats()

            # Draw UI
            frame = ui_renderer.draw_status_bar(
                frame, drone_controller, gesture, confidence,
                perf_stats['fps'], process_time
            )

            frame = ui_renderer.draw_help_bar(frame)

            # Show connection warning
            if not drone_controller.connected:
                warning_msg = "⚠ Press C to connect drone, or use simulation mode"
                frame = ui_renderer.draw_warning(frame, warning_msg)

            # Show image
            window_title = "Gesture Controlled Drone - Press ESC to exit"
            cv2.imshow(window_title, frame)

            # ========== Keyboard Control ==========
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key
                print("\nExiting program...")
                break

            elif key == ord('c') or key == ord('C'):
                if not drone_controller.connected:
                    drone_controller.connect()

            elif key == 32:  # Space key
                if drone_controller.connected:
                    if drone_controller.flying:
                        drone_controller.land()
                    else:
                        drone_controller.takeoff()
                    time.sleep(0.5)

            elif key == ord('h') or key == ord('H'):
                # Toggle help display
                current = config.get('display', 'show_help')
                config.set('display', 'show_help', value=not current)
                print(f"Help display: {'ON' if not current else 'OFF'}")

            elif key == ord('f') or key == ord('F'):
                # Toggle FPS display
                current = config.get('display', 'show_fps')
                config.set('display', 'show_fps', value=not current)
                print(f"FPS display: {'ON' if not current else 'OFF'}")

            elif key == ord('r') or key == ord('R'):
                # Reset gesture recognition
                print("Resetting gesture recognition...")
                gesture_recognizer = RobustGestureRecognizer()

            elif key == ord('t') or key == ord('T'):
                # Toggle fingertip display
                current = config.get('display', 'show_fingertips')
                config.set('display', 'show_fingertips', value=not current)
                print(f"Fingertip display: {'ON' if not current else 'OFF'}")

            elif key in key_to_gesture:
                # Keyboard control
                simulated_gesture = key_to_gesture[key]
                gesture_recognizer.set_simulated_gesture(simulated_gesture)
                gesture = simulated_gesture
                confidence = 0.9
                if drone_controller.connected and drone_controller.flying:
                    drone_controller.move_by_gesture(gesture, confidence)

            # Real gesture control
            current_time = time.time()
            if (gesture and gesture != "Waiting" and
                    gesture != "Camera Error" and gesture != "Error" and
                    drone_controller.connected and drone_controller.flying):
                drone_controller.move_by_gesture(gesture, confidence)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nProgram error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup resources
        print("\nCleaning up resources...")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        drone_controller.emergency_stop()
        config.save_config()

        print("Program exited safely")
        print("=" * 60)
        print("\nThank you for using Gesture Controlled Drone System!")
        input("Press Enter to exit...")


# ========== Program Entry Point ==========
if __name__ == "__main__":
    main()