"""Lidar sensor test script for AirSim drone.

This script tests the Lidar sensor on a drone vehicle in AirSim simulation.
It reads Lidar data for 10 iterations and prints the point cloud information.
"""

import time

import airsim

# Configuration matching settings.json
VEHICLE_NAME = "Drone_1"
LIDAR_NAME = "lidar_1"

# Test parameters
NUM_READINGS = 10
READING_INTERVAL = 1.0  # seconds
POINTS_PER_COORDINATE = 3  # x, y, z


def test_lidar_sensor(client: airsim.MultirotorClient,
                      vehicle_name: str,
                      lidar_name: str,
                      num_readings: int = NUM_READINGS) -> bool:
    """Test Lidar sensor by reading data multiple times.

    Args:
        client: AirSim multirotor client.
        vehicle_name: Name of the vehicle to test.
        lidar_name: Name of the Lidar sensor.
        num_readings: Number of readings to take.

    Returns:
        True if test completed successfully, False if failed.
    """
    for i in range(num_readings):
        lidar_data = client.getLidarData(lidar_name=lidar_name,
                                         vehicle_name=vehicle_name)

        if lidar_data and lidar_data.point_cloud:
            num_points = len(lidar_data.point_cloud) // POINTS_PER_COORDINATE
            print(f"  > OK! Reading {i + 1}/{num_readings}: Found {num_points} points.")

            # Print first detected point
            first_point = lidar_data.point_cloud[0:3]
            print(f"    - First point (relative to Lidar): "
                  f"X={first_point[0]:.2f}, Y={first_point[1]:.2f}, Z={first_point[2]:.2f}")

        elif lidar_data and not lidar_data.point_cloud:
            print(f"  > Lidar '{lidar_name}' is working, but detected 0 points.")

        else:
            print(f"  > FAILED to get Lidar data. Check names and restart UE.")
            return False

        time.sleep(READING_INTERVAL)

    return True


def main() -> None:
    """Main test function."""
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Enable API control to get sensor data
    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)

    print(f"Connected. Testing Lidar '{LIDAR_NAME}' on vehicle '{VEHICLE_NAME}'...")
    print("Please check the UE simulation window for RED debug points.")

    try:
        success = test_lidar_sensor(client, VEHICLE_NAME, LIDAR_NAME)
        if success:
            print("Lidar test completed successfully!")
        else:
            print("Lidar test failed!")
    except airsim.AirSimException as e:
        print(f"AirSim error occurred: {e}")
    finally:
        client.enableApiControl(False, vehicle_name=VEHICLE_NAME)
        print("Test complete.")


if __name__ == "__main__":
    main()