# test_basic.py
"""
Basic connection test
"""

import airsim
import cv2
import numpy as np
import time


def test_connection():
    print("=" * 50)
    print("BASIC CONNECTION TEST")
    print("=" * 50)

    # Test AirSim connection
    try:
        print("Connecting to simulator...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✓ Connection successful!")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

    # Test image capture
    try:
        print("Testing image capture...")
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene)
        ])

        if responses:
            response = responses[0]
            print(f"✓ Image size: {response.width}x{response.height}")

            # Convert to numpy array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img = img1d.reshape(response.height, response.width, 3)

            # Save test image
            cv2.imwrite("test_capture.jpg", img)
            print("✓ Test image saved: test_capture.jpg")
    except Exception as e:
        print(f"✗ Image capture failed: {e}")

    # Test drone control
    try:
        print("Testing drone control...")
        client.enableApiControl(True)
        client.armDisarm(True)
        print("✓ Drone unlocked")

        print("Simple takeoff test...")
        client.takeoffAsync().join()
        time.sleep(2)
        print("✓ Takeoff successful")

        print("Hovering for 2 seconds...")
        time.sleep(2)

        print("Landing...")
        client.landAsync().join()
        print("✓ Landing successful")

    except Exception as e:
        print(f"✗ Control test failed: {e}")
        print("You may need to switch to drone mode in simulator")

    print("\n" + "=" * 50)
    print("TEST COMPLETE!")
    print("=" * 50)

    return True


if __name__ == "__main__":
    test_connection()
    input("\nPress Enter to exit...")