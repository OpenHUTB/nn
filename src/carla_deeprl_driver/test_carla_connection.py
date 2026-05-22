import carla
import sys

try:
    print("Attempting to connect to CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    print("Connected to CARLA!")
    
    world = client.get_world()
    print(f"Current world: {world.get_map().name}")
    
    print("Reloading world...")
    client.reload_world(False)
    print("World reloaded successfully!")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
