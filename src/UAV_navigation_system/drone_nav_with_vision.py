import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import random  # For simulating low battery and emergency conditions
import time  # For the timeout condition

# DroneBattery Class to manage battery
class DroneBattery:
    def __init__(self, max_capacity=100, current_charge=100):
        self.max_capacity = max_capacity
        self.current_charge = current_charge
        
    def display_battery_status(self):
        print(f"Battery Status: {self.current_charge}%")
        
    def charge_battery(self, charge_rate=10):
        while self.current_charge < self.max_capacity:
            self.current_charge += charge_rate
            if self.current_charge > self.max_capacity:
                self.current_charge = self.max_capacity
            print(f"Charging... {self.current_charge}%")
            time.sleep(1)
        print("Battery fully charged!")
        
    def discharge_battery(self, discharge_rate=10):
        while self.current_charge > 0:
            self.current_charge -= discharge_rate
            if self.current_charge < 0:
                self.current_charge = 0
            print(f"Discharging... {self.current_charge}%")
            time.sleep(1)
        print("Battery completely drained!")
    
    def is_battery_low(self):
        return self.current_charge < 20

# ✅ Load and compile model
print("📦 Loading model...")
model = load_model(r"C:\Users\hp\DroneNavVision\data\best_model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # ✅ Compile step added

class_names = ['Animal', 'City', 'Fire', 'Forest', 'Vehicle', 'Water']

def check_emergency():
    emergency_conditions = ['low_battery', 'emergency']
    return random.choice(emergency_conditions)

# Handle low battery condition
def handle_low_battery(drone_battery):
    print("🔋 Low battery! Returning to base.")
    drone_battery.charge_battery(charge_rate=15)
    exit()

def preprocess_frame(frame):
    resized = cv2.resize(frame, (128, 128))
    img_array = img_to_array(resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def decide_navigation(predicted_class):
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
    print("🚁 Starting the drone vision process...")
    start_time = time.time()
    
    # Use the correct video source based on your setup
    VIDEO_SOURCE = "http://192.168.1.3:4747/video"  # Or set your IP stream (e.g., "rtsp://<IP_ADDRESS>/stream")

    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Increase buffer size
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set the desired FPS (adjust as needed)


    # Check if video stream is opened successfully
    if not cap.isOpened():
        print("❌ Failed to open video source. Check connection or URL.")
        return
    else:
        print("✅ Video source opened successfully.")


    drone_battery = DroneBattery()

    while True:
        print("📸 Processing frame...")
        
        # Check if battery is low
        if drone_battery.is_battery_low():
            handle_low_battery(drone_battery)

        elapsed_time = time.time() - start_time
        if elapsed_time > 300:
            print("⏰ Timeout reached! Stopping the drone.")
            break

        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        processed = preprocess_frame(frame)
        pred = model.predict(processed)
        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        cv2.putText(frame, f"{predicted_class} ({confidence:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Drone Vision Feed", frame)

        decide_navigation(predicted_class)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Manual stop initiated by the user.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
