import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import carla
import time
import numpy as np

def test_carla_connection():
    print("=" * 60)
    print("Testing CARLA Connection and Environment Setup")
    print("=" * 60)
    
    # Connect to CARLA
    try:
        print("\n1. Connecting to CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"✅ Connected! Current map: {world.get_map().name}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    # Keep current map (Town10HD_Opt)
    try:
        print("\n2. Using current map...")
        print(f"✅ Using map: {world.get_map().name}")
    except Exception as e:
        print(f"❌ Failed to get map: {e}")
        return False
    
    # Set synchronous mode
    try:
        print("\n3. Setting synchronous mode...")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        print("✅ Synchronous mode enabled")
    except Exception as e:
        print(f"❌ Failed to set synchronous mode: {e}")
        return False
    
    # Spawn a vehicle
    try:
        print("\n4. Spawning vehicle...")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        
        if spawn_points:
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
            print("✅ Vehicle spawned successfully")
        else:
            print("❌ No spawn points available")
            return False
    except Exception as e:
        print(f"❌ Failed to spawn vehicle: {e}")
        return False
    
    # Test simulation step
    try:
        print("\n5. Testing simulation step...")
        for i in range(5):
            world.tick()
            time.sleep(0.1)
            location = vehicle.get_location()
            print(f"   Step {i+1}: Vehicle at ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})")
        print("✅ Simulation steps working")
    except Exception as e:
        print(f"❌ Simulation step failed: {e}")
        return False
    
    # Clean up
    try:
        print("\n6. Cleaning up...")
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! CARLA environment is working correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_carla_connection()
    sys.exit(0 if success else 1)