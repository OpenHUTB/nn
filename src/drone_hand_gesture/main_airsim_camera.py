# -*- coding: utf-8 -*-
"""
Drone Hand Gesture Control - AirSim Version WITH CAMERA!
Combined hand gesture control and drone camera display!
"""

import cv2
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from airsim_controller import AirSimController

# Import hand detector
USE_ENHANCED_DETECTOR = False
HAS_ENHANCED = False

try:
    import mediapipe as mp
    from gesture_detector_enhanced import EnhancedGestureDetector
    USE_ENHANCED_DETECTOR = True
    HAS_ENHANCED = True
except ImportError:
    try:
        from gesture_detector import GestureDetector
    except ImportError:
        print("No gesture detector found!")
        sys.exit(1)


def main(show_window=True):
    print("=" * 70)
    print("Drone Hand Gesture Control - AirSim with CAMERA!")
    print("=" * 70)
    print()
    
    # [1/4] Connect to AirSim
    print("[1/4] Connecting to AirSim...")
    controller = AirSimController()
    
    if not controller.connect():
        print("\n[ERROR] AirSim connection failed!")
        print("\nMake sure AirSim is running!")
        print("\nPress any key to exit...")
        input()
        return
    
    print("[SUCCESS] Connected to AirSim!")
    
    # [2/4] Initialize gesture detector
    print("\n[2/4] Initializing gesture detector...")
    
    if HAS_ENHANCED:
        detector = EnhancedGestureDetector(use_ml=False)
        print("[SUCCESS] Using enhanced detector!")
    else:
        from gesture_detector import GestureDetector
        detector = GestureDetector()
        print("[SUCCESS] Using basic detector!")
    
    # [3/4] Initialize camera (user webcam)
    print("\n[3/4] Initializing user webcam...")
    cap = None
    for cam_id in [1, 0]:
        print(f"  Trying webcam {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"[SUCCESS] Webcam {cam_id} ready!")
                break
            else:
                cap.release()
                cap = None
    
    if not cap or not cap.isOpened():
        print("[ERROR] No webcam available!")
        controller.disconnect()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # [4/4] System ready!
    print("\n[4/4] System READY!")
    print()
    print("=" * 70)
    print("Gesture Controls:")
    print("  Open Palm   = Takeoff")
    print("  Closed Fist = Land")
    print("  Point Up    = Move Up")
    print("  Point Down  = Move Down")
    print("  Victory     = Move Forward")
    print("  Thumb Up    = Move Backward")
    print("  OK Sign     = Hover")
    print("  Thumb Down  = Stop")
    print()
    print("Keyboard Controls:")
    print("  SPACE       = Takeoff/Land")
    print("  T           = Manual Takeoff")
    print("  L           = Manual Land")
    print("  H           = Hover")
    print("  Q/ESC       = Exit")
    print("=" * 70)
    print()
    
    # State variables
    is_flying = False
    last_command_time = 0
    last_processed_gesture = ""
    last_processed_time = 0
    command_cooldown = 1.5
    frame_count = 0
    start_time = time.time()
    
    print("[INFO] Press SPACE or T to takeoff!")
    
    try:
        while True:
            # Read user webcam
            ret, gesture_frame = cap.read()
            if not ret:
                print("[WARNING] Can't read user webcam!")
                break
            
            gesture_frame = cv2.flip(gesture_frame, 1)
            frame_count += 1
            
            # Hand gesture detection
            debug_frame, gesture, confidence, _ = detector.detect_gestures(gesture_frame, simulation_mode=False)
            
            # Get drone camera
            drone_camera_frame = controller.get_camera_image()
            
            # Display FPS on gesture frame
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Gesture: {gesture} ({confidence:.2f})", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Process gesture commands
            current_time = time.time()
            in_cooldown = current_time - last_command_time <= command_cooldown
            same_gesture = (gesture == last_processed_gesture and
                           current_time - last_processed_time < 2.0)
            
            command = detector.get_command(gesture)
            
            # Configure thresholds for reliable detection
            gesture_thresholds = {
                "open_palm": 0.5, "closed_fist": 0.55, "victory": 0.55,
                "thumb_up": 0.55, "thumb_down": 0.55, "pointing_up": 0.5,
                "pointing_down": 0.5, "ok_sign": 0.6
            }
            threshold = gesture_thresholds.get(gesture, 0.5)
            
            if (gesture not in ["no_hand", "hand_detected", "none"] and
                    confidence > threshold and not in_cooldown and
                    not same_gesture and command != "none"):
                
                print(f"[CMD] Gesture: {gesture} -> Execute: {command}")
                controller.send_command(command)
                
                if command == "takeoff":
                    is_flying = True
                elif command == "land":
                    is_flying = False
                
                last_command_time = current_time
                last_processed_gesture = gesture
                last_processed_time = current_time
            
            # Show combined view!
            if show_window:
                # Prepare drone camera frame (or placeholder)
                if drone_camera_frame is not None:
                    # Resize drone camera to match gesture frame height
                    h, w = debug_frame.shape[:2]
                    drone_camera_frame = cv2.resize(drone_camera_frame, (w, h))
                else:
                    # Create placeholder for drone camera
                    drone_camera_frame = np.ones((debug_frame.shape[0], debug_frame.shape[1], 3), dtype=np.uint8) * 50
                    cv2.putText(drone_camera_frame, "DRONE CAMERA - NO SIGNAL", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(drone_camera_frame, "Connect AirSim to see drone view!", 
                               (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Stack the two views!
                combined_view = np.hstack([debug_frame, drone_camera_frame])
                
                cv2.imshow("Drone Control - LEFT: User, RIGHT: Drone", combined_view)
                
                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("\n[INFO] Exiting...")
                    break
                elif key == ord(' '):
                    if is_flying:
                        print("[INFO] Landing...")
                        controller.land()
                        is_flying = False
                    else:
                        print("[INFO] Taking off...")
                        controller.takeoff()
                        is_flying = True
                    time.sleep(0.5)
                elif key == ord('t') or key == ord('T'):
                    if not is_flying:
                        print("[INFO] Manual takeoff...")
                        controller.takeoff()
                        is_flying = True
                elif key == ord('l') or key == ord('L'):
                    if is_flying:
                        print("[INFO] Manual landing...")
                        controller.land()
                        is_flying = False
                elif key == ord('h') or key == ord('H'):
                    print("[INFO] Hovering...")
                    controller.hover()
            
            # Print state info periodically
            if is_flying and frame_count % 60 == 0:
                state = controller.get_state()
                print(f"[STATUS] Position: {state['position']}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted!")
    
    finally:
        print("\n[INFO] Cleaning up...")
        
        # Land if still flying
        if is_flying:
            print("[INFO] Landing...")
            controller.land()
        
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        controller.disconnect()
        
        if hasattr(detector, "release"):
            detector.release()
        
        print("[SUCCESS] Safe exit!")


if __name__ == "__main__":
    main()
