"""
Drone Vision Navigation System
Optimized for Abandoned Park Environment
Pure English interface to avoid encoding issues
Fixed KeyboardInterrupt handling
Improved image analysis (no extra dependencies)
"""

import airsim
import cv2
import numpy as np
import time
import json
import os
import sys
from datetime import datetime
import random


class DroneVisionSystem:
    """Drone vision navigation system"""

    def __init__(self):
        """Initialize system"""
        self.clear_screen()
        print("=" * 70)
        print("DRONE VISION NAVIGATION SYSTEM v2.0 (Enhanced)")
        print("=" * 70)
        print("Optimized for Abandoned Park Environment")
        print("-" * 70)

        # Try to connect to simulator
        try:
            print("Connecting to AirSim simulator...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("✓ Connection successful!")
        except Exception as e:
            print(f"✗ Connection failed: {str(e)}")
            print("\nPlease ensure:")
            print("1. AbandonedPark.exe is running")
            print("2. Simulator is fully loaded")
            print("3. Switched to drone mode if needed")
            self.client = None

        # System status
        self.running = False
        self.flying = False
        self.battery = 100.0
        self.current_env = "Unknown"
        self.env_confidence = 0.0
        self.emergency = False

        # Create directories
        self.create_folders()

        # Initialize classifier (enhanced version)
        self.classifier = EnvironmentClassifier()

        print("System initialization complete!")
        print("=" * 70)

    def clear_screen(self):
        """Clear console screen"""
        if sys.platform == 'win32':
            os.system('cls')
        else:
            os.system('clear')

    def create_folders(self):
        """Create project folders"""
        folders = ['data/images', 'data/logs', 'models', 'debug']
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def log(self, message):
        """Log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"

        # Print to console
        print(log_msg)

        # Save to file
        log_file = f"data/logs/drone_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")

    def show_status(self):
        """Display system status"""
        status_text = f"""
Current Status:
  Flight Status: {'Flying' if self.flying else 'Landed'}
  Environment: {self.current_env} ({self.env_confidence:.1%})
  Battery: {self.battery:.1f}%
  Emergency Mode: {'Yes' if self.emergency else 'No'}
  System Running: {'Yes' if self.running else 'No'}
        """
        print(status_text)

    def takeoff(self, height=15):
        """Take off to specified height"""
        if not self.client:
            print("Not connected to simulator!")
            return False

        try:
            self.log(f"Taking off to {height} meters...")

            # Unlock drone
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            time.sleep(1)

            # Take off
            self.client.takeoffAsync().join()
            time.sleep(2)

            # Ascend to specified height
            self.client.moveToZAsync(-height, 3).join()

            self.flying = True
            self.log("Takeoff successful!")
            return True

        except Exception as e:
            self.log(f"Takeoff failed: {str(e)}")
            return False

    def land(self):
        """Land the drone"""
        try:
            self.log("Landing...")
            self.client.landAsync().join()
            time.sleep(2)

            # Lock drone
            self.client.armDisarm(False)
            self.client.enableApiControl(False)

            self.flying = False
            self.log("Landing successful!")
            return True

        except Exception as e:
            self.log(f"Landing failed: {str(e)}")
            return False

    def capture_image(self):
        """Capture image from drone camera"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            if responses:
                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)

                # Convert to BGR format
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                return img_bgr

        except Exception as e:
            self.log(f"Image capture failed: {str(e)}")

        return None

    def analyze_environment(self, image):
        """Analyze environment using enhanced classifier"""
        return self.classifier.classify(image)

    def update_battery(self):
        """Update battery status"""
        if self.flying:
            self.battery -= 0.05  # Flying consumption
        else:
            self.battery -= 0.01  # Standby consumption

        if self.battery < 0:
            self.battery = 0

        # Check low battery
        if self.battery < 20 and not self.emergency:
            self.log(f"Warning: Low battery ({self.battery:.1f}%)")
            if self.battery < 10:
                self.log("Critical: Very low battery!")
                self.emergency = True

    def save_image_data(self, image, environment, confidence):
        """Save image data"""
        if image is None:
            return

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"data/images/img_{timestamp}_{environment}.jpg"

        # Save image
        cv2.imwrite(filename, image)

        # Record to JSON
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "filename": filename,
            "environment": environment,
            "confidence": float(confidence),
            "battery": float(self.battery),
            "flying": self.flying
        }

        json_file = "data/images/images_log.json"

        try:
            # Read existing data
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    all_data = json.load(f)
            else:
                all_data = []

            # Add new data
            all_data.append(data)

            # Keep only recent 1000 records
            if len(all_data) > 1000:
                all_data = all_data[-1000:]

            # Save
            with open(json_file, 'w') as f:
                json.dump(all_data, f, indent=2)

        except Exception as e:
            self.log(f"Data save failed: {str(e)}")

    def navigate(self, environment, confidence):
        """Navigate based on environment"""
        if not self.flying:
            return

        # Choose action based on environment
        actions = {
            "Ruins": self.navigate_ruins,
            "Building": self.navigate_building,
            "Forest": self.navigate_forest,
            "Road": self.navigate_road,
            "Sky": self.navigate_sky,
            "Water": self.navigate_water,
            "Fire": self.navigate_fire,
            "Animal": self.navigate_animal,
            "Vehicle": self.navigate_vehicle
        }

        # Execute corresponding navigation function
        action_func = actions.get(environment, self.navigate_default)
        action_func(confidence)

    def navigate_ruins(self, confidence):
        """Navigate in ruins"""
        self.log("Exploring ruins...")
        # Move slowly to avoid collisions
        self.client.moveByVelocityAsync(2, 0, 0, 2).join()
        time.sleep(1)

    def navigate_building(self, confidence):
        """Navigate around buildings"""
        self.log("Avoiding building...")
        # Rise to avoid collision
        self.client.moveByVelocityAsync(0, 0, -1, 1).join()
        # Move right
        self.client.moveByVelocityAsync(0, 2, 0, 2).join()

    def navigate_forest(self, confidence):
        """Navigate through forest"""
        self.log("Moving through forest...")
        # Rise slightly and move slowly
        self.client.moveByVelocityAsync(1.5, 0, -0.5, 2).join()

    def navigate_road(self, confidence):
        """Navigate along road"""
        self.log("Following road...")
        # Move at normal speed
        self.client.moveByVelocityAsync(3, 0, 0, 3).join()

    def navigate_sky(self, confidence):
        """Navigate in open sky"""
        self.log("Open sky, normal flight...")
        # Move faster
        self.client.moveByVelocityAsync(4, 0, 0, 3).join()

    def navigate_water(self, confidence):
        """Avoid water"""
        self.log("Avoiding water area...")
        # Rise immediately
        self.client.moveByVelocityAsync(0, 0, -2, 1).join()
        # Move backward
        self.client.moveByVelocityAsync(-2, 0, 0, 2).join()

    def navigate_fire(self, confidence):
        """Navigate fire emergency"""
        self.log("Fire detected! Emergency response...")
        self.emergency = True
        # Emergency ascent
        self.client.moveToZAsync(-30, 5).join()
        # Send alert
        self.log("Fire alert!")

    def navigate_animal(self, confidence):
        """Navigate around animals"""
        self.log("Animal detected, keeping distance...")
        # Hover and observe
        self.client.hoverAsync().join()
        time.sleep(3)
        # Move back slowly
        self.client.moveByVelocityAsync(-1, 0, 0, 2).join()

    def navigate_vehicle(self, confidence):
        """Navigate around vehicles"""
        self.log("Vehicle detected, following...")
        # Follow at distance
        self.client.moveByVelocityAsync(2, 0, 0, 2).join()

    def navigate_default(self, confidence):
        """Default navigation"""
        self.log("Unknown environment, conservative exploration...")
        # Move slowly
        self.client.moveByVelocityAsync(1, 0, 0, 2).join()

    def display_image(self, image, environment, confidence):
        """Display image with information"""
        if image is None:
            return

        # Create display image
        display_img = image.copy()
        height, width = display_img.shape[:2]

        # Add environment info
        env_text = f"Env: {environment}"
        conf_text = f"Conf: {confidence:.1%}"
        bat_text = f"Battery: {self.battery:.1f}%"

        # Set text color
        color = (0, 255, 0)  # Green
        if confidence < 0.6:
            color = (0, 255, 255)  # Yellow
        if confidence < 0.4:
            color = (0, 0, 255)  # Red

        # Add text to image
        cv2.putText(display_img, env_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(display_img, conf_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_img, bat_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(display_img, timestamp, (width - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show image
        cv2.imshow('Drone Vision System', display_img)

    def run_mission(self, mission_time=300):
        """Run main mission loop"""
        if not self.client:
            print("Cannot run: Not connected to simulator")
            return

        self.clear_screen()
        print("=" * 70)
        print("STARTING DRONE MISSION")
        print(f"Mission time: {mission_time} seconds")
        print("=" * 70)
        print("CONTROLS:")
        print("  Q - Quit program")
        print("  L - Manual landing")
        print("  R - Return home")
        print("  S - Save current image")
        print("  P - Pause/Resume")
        print("-" * 70)

        # Take off
        if not self.takeoff():
            return

        self.running = True
        start_time = time.time()
        frame_count = 0
        last_env = "Unknown"

        try:
            while self.running and (time.time() - start_time) < mission_time:
                frame_count += 1

                # Update battery
                self.update_battery()

                # Check emergency
                if self.emergency and self.flying:
                    self.log("Emergency! Returning home...")
                    self.client.moveToPositionAsync(0, 0, -20, 5).join()
                    self.land()
                    break

                # Capture image
                image = self.capture_image()

                if image is not None:
                    # Analyze environment
                    try:
                        environment, confidence = self.analyze_environment(image)
                        last_env = environment
                        self.current_env = environment
                        self.env_confidence = confidence
                    except Exception as e:
                        self.log(f"Environment analysis failed: {str(e)}")
                        environment, confidence = "Unknown", 0.0
                        self.current_env = "Unknown"
                        self.env_confidence = 0.0

                    # Save data
                    if confidence > 0.7 or frame_count % 5 == 0:
                        self.save_image_data(image, environment, confidence)

                    # Display image
                    self.display_image(image, environment, confidence)

                    # Navigation decision
                    try:
                        self.navigate(environment, confidence)
                    except Exception as e:
                        self.log(f"Navigation error: {str(e)}")

                # Show status (every 20 frames)
                if frame_count % 20 == 0:
                    elapsed = int(time.time() - start_time)
                    remaining = mission_time - elapsed
                    self.clear_screen()
                    print(f"MISSION IN PROGRESS...")
                    print(f"Elapsed: {elapsed} sec | Remaining: {remaining} sec")
                    print(f"Environment: {last_env} ({self.env_confidence:.1%})")
                    print(f"Battery: {self.battery:.1f}%")
                    print(f"Frames processed: {frame_count}")
                    print("-" * 40)
                    print("CONTROLS: Q-Quit, L-Land, R-Return, S-Save, P-Pause")

                # Check keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    self.log("User quit")
                    break
                elif key == ord('l'):
                    self.land()
                    break
                elif key == ord('r'):
                    self.log("Manual return home")
                    self.client.moveToPositionAsync(0, 0, -20, 5).join()
                    self.land()
                    break
                elif key == ord('s'):
                    if image is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"debug/snapshot_{timestamp}.jpg"
                        cv2.imwrite(filename, image)
                        self.log(f"Saved snapshot: {filename}")
                elif key == ord('p'):
                    self.running = not self.running
                    status = "Paused" if not self.running else "Resumed"
                    self.log(f"Mission {status}")

                # Small delay to control processing speed
                time.sleep(0.1)

            # Mission complete
            self.log("Mission complete!")

        except KeyboardInterrupt:
            self.log("Mission interrupted by user (Ctrl+C)")
            print("\nMission interrupted. Landing drone...")
        except Exception as e:
            self.log(f"Runtime error: {str(e)}")
        finally:
            # Cleanup
            if self.flying:
                try:
                    self.land()
                except:
                    pass
            cv2.destroyAllWindows()
            try:
                self.generate_report()
            except:
                pass

    def generate_report(self):
        """Generate mission report"""
        report = {
            "mission": "Drone Vision Navigation",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "summary": {
                "battery_final": round(self.battery, 1),
                "environment_detected": self.current_env,
                "emergency_activated": self.emergency
            },
            "files": {
                "images": "data/images/",
                "logs": "data/logs/",
                "debug": "debug/"
            }
        }

        report_file = f"data/logs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.log(f"Report saved: {report_file}")
        except Exception as e:
            self.log(f"Report save failed: {str(e)}")


class EnvironmentClassifier:
    """Enhanced environment classifier (no extra dependencies)"""

    def __init__(self):
        self.environments = [
            "Ruins", "Building", "Forest", "Road",
            "Sky", "Water", "Fire", "Animal", "Vehicle"
        ]
        # Base weights for fallback
        self.weights = {
            "Ruins": 0.35, "Building": 0.20, "Forest": 0.15,
            "Road": 0.10, "Sky": 0.08, "Water": 0.05,
            "Fire": 0.02, "Animal": 0.03, "Vehicle": 0.02
        }

    def classify(self, image):
        """Main classification interface"""
        if image is None:
            return "Unknown", 0.0

        # Extract enhanced features
        features = self._extract_enhanced_features(image)

        # Rule-based classification
        env, conf = self._rule_based_v2(features)

        # If rules fail, use weighted random
        if env == "Unknown":
            env, conf = self._weighted_random(features)

        return env, conf

    def _extract_enhanced_features(self, image):
        """Extract rich features using only OpenCV and numpy"""
        h, w = image.shape[:2]
        features = {}

        # ----- LAB color statistics (mean, std) -----
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_mean, l_std = cv2.meanStdDev(lab[:, :, 0])
        a_mean, a_std = cv2.meanStdDev(lab[:, :, 1])
        b_mean, b_std = cv2.meanStdDev(lab[:, :, 2])
        features['l_mean'] = l_mean[0][0]
        features['l_std'] = l_std[0][0]
        features['a_mean'] = a_mean[0][0]
        features['a_std'] = a_std[0][0]
        features['b_mean'] = b_mean[0][0]
        features['b_std'] = b_std[0][0]

        # ----- Texture features (gradient magnitude, Laplacian variance) -----
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        features['grad_mean'] = np.mean(mag)
        features['grad_std'] = np.std(mag)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['lap_var'] = np.var(laplacian)

        # Edge density (Canny)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (h * w)

        # ----- Global color ratios (HSV) -----
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        features['blue_ratio'] = np.sum(blue_mask > 0) / (h * w)

        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        features['green_ratio'] = np.sum(green_mask > 0) / (h * w)

        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        features['red_ratio'] = np.sum(red_mask > 0) / (h * w)

        features['brightness'] = np.mean(gray)
        features['gray_variance'] = np.var(gray)

        # ----- Sky detection (improved) -----
        if h > 10:
            top_blue = np.sum(blue_mask[:h//3, :] > 0) / (w * h//3)
            bottom_blue = np.sum(blue_mask[2*h//3:, :] > 0) / (w * h//3)
            features['is_sky'] = (top_blue > 0.25) and (top_blue > bottom_blue * 1.8) and (features['l_mean'] > 180)
        else:
            features['is_sky'] = False

        # ----- Spatial grid color ratios (3x3) -----
        grid_h, grid_w = h // 3, w // 3
        for i in range(3):
            for j in range(3):
                roi = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                if roi.size == 0:
                    continue
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                blue_roi = cv2.inRange(roi_hsv, (100, 50, 50), (130, 255, 255))
                features[f'blue_grid_{i}_{j}'] = np.sum(blue_roi > 0) / (grid_h * grid_w)
                green_roi = cv2.inRange(roi_hsv, (40, 50, 50), (80, 255, 255))
                features[f'green_grid_{i}_{j}'] = np.sum(green_roi > 0) / (grid_h * grid_w)
                red_roi1 = cv2.inRange(roi_hsv, (0, 50, 50), (10, 255, 255))
                red_roi2 = cv2.inRange(roi_hsv, (170, 50, 50), (180, 255, 255))
                red_roi = cv2.bitwise_or(red_roi1, red_roi2)
                features[f'red_grid_{i}_{j}'] = np.sum(red_roi > 0) / (grid_h * grid_w)

        return features

    def _rule_based_v2(self, f):
        """Improved rule-based classification using enhanced features"""
        # Extract common features
        blue = f.get('blue_ratio', 0)
        green = f.get('green_ratio', 0)
        red = f.get('red_ratio', 0)
        edges = f.get('edge_density', 0)
        grad_mean = f.get('grad_mean', 0)
        lap_var = f.get('lap_var', 0)
        bright = f.get('brightness', 0)
        is_sky = f.get('is_sky', False)
        l_mean = f.get('l_mean', 128)

        # Rule 1: Fire (high red + high brightness + low texture)
        if red > 0.25 and bright > 200 and grad_mean < 15:
            return "Fire", 0.75

        # Rule 2: Sky (high blue + is_sky + high L)
        if blue > 0.3 and is_sky and l_mean > 180:
            return "Sky", 0.85

        # Rule 3: Water (high blue + not sky + low edges)
        if blue > 0.25 and not is_sky and edges < 0.03:
            return "Water", 0.70

        # Rule 4: Forest (high green + high texture)
        if green > 0.3 and grad_mean > 20:
            return "Forest", 0.80

        # Rule 5: Ruins (high edges + high gray variance + low brightness)
        if edges > 0.07 and f.get('gray_variance', 0) > 1200 and bright < 120:
            return "Ruins", 0.85

        # Rule 6: Building (medium edges + low brightness + low green)
        if edges > 0.05 and bright < 100 and green < 0.1:
            return "Building", 0.75

        # Rule 7: Road (low edges + medium brightness + low color saturation)
        if edges < 0.02 and 100 < bright < 200 and blue < 0.1 and green < 0.1 and red < 0.1:
            return "Road", 0.70

        # Rule 8: Animal/Vehicle (simple red + edges)
        if red > 0.15 and edges > 0.04:
            # Could be improved with spatial patterns, but fallback to vehicle
            return "Vehicle", 0.60

        # Unknown
        return "Unknown", 0.0

    def _weighted_random(self, features):
        """Weighted random fallback (similar to original)"""
        adj_weights = self.weights.copy()
        blue = features.get('blue_ratio', 0)
        green = features.get('green_ratio', 0)
        edges = features.get('edge_density', 0)

        if blue > 0.2:
            adj_weights["Sky"] *= 1.5
            if blue > 0.3:
                adj_weights["Water"] *= 0.5
        if green > 0.15:
            adj_weights["Forest"] *= 2.0
        if edges > 0.05:
            adj_weights["Ruins"] *= 1.8

        total = sum(adj_weights.values())
        if total > 0:
            probs = [adj_weights[env] / total for env in self.environments]
        else:
            probs = [1 / len(self.environments)] * len(self.environments)

        env = random.choices(self.environments, weights=probs)[0]

        base_conf = 0.6
        if env == "Ruins" and edges > 0.04:
            base_conf += 0.15
        elif env == "Forest" and green > 0.15:
            base_conf += 0.1
        elif env == "Sky" and blue > 0.2:
            base_conf += 0.1

        return env, min(base_conf, 0.9)


def safe_input(prompt):
    """Safe input function with KeyboardInterrupt handling"""
    try:
        return input(prompt).strip()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user (Ctrl+C)")
        print("Exiting program...")
        sys.exit(0)


def main():
    """Main function with improved exception handling"""
    try:
        print("DRONE VISION NAVIGATION SYSTEM (Enhanced)")
        print("-" * 50)

        # Create system
        drone = DroneVisionSystem()

        if drone.client is None:
            print("Cannot connect to simulator, exiting")
            return

        # User configuration
        print("\nMISSION CONFIGURATION:")
        print("1. Test mode (60 seconds)")
        print("2. Short mission (5 minutes)")
        print("3. Long mission (10 minutes)")
        print("4. Custom time")

        choice = safe_input("\nSelect mode (1-4): ")

        if choice == '1':
            mission_time = 60
        elif choice == '2':
            mission_time = 300
        elif choice == '3':
            mission_time = 600
        elif choice == '4':
            try:
                time_input = safe_input("Enter mission time (seconds): ")
                mission_time = int(time_input)
            except ValueError:
                mission_time = 300
                print(f"Invalid input, using default: {mission_time} seconds")
        else:
            mission_time = 300
            print(f"Using default: {mission_time} seconds")

        print(f"\nMISSION SETTINGS:")
        print(f"  Time: {mission_time} seconds ({mission_time / 60:.1f} minutes)")
        print(f"  Image saving: Auto")
        print(f"  Logging: Enabled")

        confirm = safe_input("\nStart mission? (y/n): ").lower()

        if confirm == 'y':
            drone.run_mission(mission_time)
        else:
            print("Mission cancelled")

        print("\nProgram ended gracefully")

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user (Ctrl+C)")
        print("Exiting...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()