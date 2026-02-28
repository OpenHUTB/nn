# Import necessary libraries
import cv2  # OpenCV for computer vision tasks (video capture, image processing)
import numpy as np  # NumPy for numerical operations and array manipulation
from tensorflow.keras.models import load_model  # Load pre-trained Keras model
from tensorflow.keras.preprocessing.image import img_to_array  # Convert image to array for model input
import random  # For simulating random emergency conditions (low battery, etc.)
import time  # For time-related operations (timeouts, sleep)


# DroneBattery Class to manage battery state and operations
class DroneBattery:
    """Class to simulate and manage drone battery operations"""

    def __init__(self, max_capacity=100, current_charge=100):
        """
        Initialize battery parameters
        Args:
            max_capacity: Maximum battery capacity (default 100%)
            current_charge: Current battery charge level (default 100%)
        """
        self.max_capacity = max_capacity
        self.current_charge = current_charge

    def display_battery_status(self):
        """Display current battery percentage"""
        print(f"Battery Status: {self.current_charge}%")

    def charge_battery(self, charge_rate=10):
        """
        Simulate battery charging process
        Args:
            charge_rate: Percentage to charge per iteration (default 10%)
        """
        while self.current_charge < self.max_capacity:
            self.current_charge += charge_rate
            if self.current_charge > self.max_capacity:
                self.current_charge = self.max_capacity
            print(f"Charging... {self.current_charge}%")
            time.sleep(1)  # Simulate time delay for charging
        print("Battery fully charged!")

    def discharge_battery(self, discharge_rate=10):
        """
        Simulate battery discharging process
        Args:
            discharge_rate: Percentage to discharge per iteration (default 10%)
        """
        while self.current_charge > 0:
            self.current_charge -= discharge_rate
            if self.current_charge < 0:
                self.current_charge = 0
            print(f"Discharging... {self.current_charge}%")
            time.sleep(1)  # Simulate time delay for discharging
        print("Battery completely drained!")

    def is_battery_low(self):
        """
        Check if battery level is critically low
        Returns:
            Boolean: True if battery < 20%, False otherwise
        """
        return self.current_charge < 20


# Load and compile the pre-trained machine learning model
print("📦 Loading model...")
# Load model from specified file path
model = load_model(r"C:\Users\hp\DroneNavVision\data\best_model.h5")
# Compile model with optimizer and loss function for inference
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class names corresponding to model output categories
class_names = ['Animal', 'City', 'Fire', 'Forest', 'Vehicle', 'Water']


def check_emergency():
    """
    Simulate random emergency condition checks
    Returns:
        String: Randomly selected emergency condition
    """
    emergency_conditions = ['low_battery', 'emergency']
    return random.choice(emergency_conditions)


def handle_low_battery(drone_battery):
    """
    Handle low battery emergency procedure
    Args:
        drone_battery: DroneBattery instance to manage charging
    """
    print("🔋 Low battery! Returning to base.")
    drone_battery.charge_battery(charge_rate=15)  # Fast charge at 15% per second
    exit()  # Exit program after charging


def preprocess_frame(frame):
    """
    Preprocess video frame for model input
    Args:
        frame: Raw image frame from camera
    Returns:
        numpy array: Preprocessed image ready for model prediction
    """
    # Resize frame to match model input dimensions (128x128)
    resized = cv2.resize(frame, (128, 128))
    # Normalize pixel values to [0, 1] and convert to array
    img_array = img_to_array(resized) / 255.0
    # Add batch dimension (1, 128, 128, 3)
    return np.expand_dims(img_array, axis=0)


def decide_navigation(predicted_class):
    """
    Determine navigation action based on detected class
    Args:
        predicted_class: String of detected object/environment class
    """
    # Define navigation strategies for each detected class
    if predicted_class == 'Fire':
        print("🔥 Fire detected! Navigate away.")
    elif predicted_class == 'Animal':
        print("🦌 Animal ahead. Hovering.")
    elif predicted_class == 'Forest':
        print("🌲 Forest zone detected. Reduce speed.")
    elif predicted_class == 'Water':
        print("🌊 Water body detected. Maintain altitude and avoid descent.")
    elif predicted_class == 'Vehicle':
        print("🚗 Vehicle detected. Hover and wait.")
    elif predicted_class == 'City':
        print("🏙️ Urban area detected. Enable obstacle avoidance and slow navigation.")
    else:
        print("✅ Clear path. Continue normal navigation.")


def main():
    """
    Main function to run drone vision system
    Handles video capture, frame processing, prediction, and navigation decisions
    """
    print("🚁 Starting the drone vision process...")
    start_time = time.time()  # Record start time for timeout condition

    # Video source configuration - adjust based on drone camera setup
    # Options: RTSP stream, IP camera, or local camera
    VIDEO_SOURCE = "http://192.168.1.3:4747/video"  # IP camera stream URL

    # Initialize video capture with FFMPEG backend for streaming
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    # Configure video capture properties
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Increase buffer size for smoother streaming
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate to 30 FPS

    # Validate video stream connection
    if not cap.isOpened():
        print("❌ Failed to open video source. Check connection or URL.")
        return
    else:
        print("✅ Video source opened successfully.")

    # Initialize drone battery management
    drone_battery = DroneBattery()

    # Main processing loop
    while True:
        print("📸 Processing frame...")

        # Emergency condition check: Low battery
        if drone_battery.is_battery_low():
            handle_low_battery(drone_battery)

        # Timeout condition: Stop after 5 minutes (300 seconds)
        elapsed_time = time.time() - start_time
        if elapsed_time > 300:
            print("⏰ Timeout reached! Stopping the drone.")
            break

        # Capture frame from video stream
        ret, frame = cap.read()
        if not ret:  # Check if frame capture was successful
            print("❌ Failed to capture frame.")
            break

        # Process frame and make prediction
        processed = preprocess_frame(frame)  # Preprocess for model
        pred = model.predict(processed)  # Run model inference
        predicted_class = class_names[np.argmax(pred)]  # Get class with highest probability
        confidence = np.max(pred) * 100  # Convert confidence to percentage

        # Display prediction on video feed
        cv2.putText(frame, f"{predicted_class} ({confidence:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Drone Vision Feed", frame)  # Show annotated video

        # Make navigation decision based on prediction
        decide_navigation(predicted_class)

        # Exit condition: Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Manual stop initiated by the user.")
            break

    # Cleanup resources
    cap.release()  # Release video capture device
    cv2.destroyAllWindows()  # Close all OpenCV windows


# Standard Python idiom to run main function when script is executed directly
if __name__ == "__main__":
    main()