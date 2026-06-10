import carla
import sys
import time

class NightModeSystem:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        
        self.is_night_mode = False
        self.current_light = 10000
        self.lights_on = False
        
        self.set_daytime()

    def set_daytime(self):
        weather = carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_azimuth_angle=90.0,
            sun_altitude_angle=75.0,
            fog_density=0.0,
            fog_distance=0.0,
            fog_falloff=0.1,
            wetness=0.0,
            scattering_intensity=1.0,
            mie_scattering_scale=0.03,
            rayleigh_scattering_scale=0.03,
            dust_storm=0.0
        )
        self.world.set_weather(weather)
        self.current_light = 10000
        self.is_night_mode = False
        self.turn_off_lights()

    def set_nighttime(self):
        weather = carla.WeatherParameters(
            cloudiness=95.0,
            precipitation=5.0,
            precipitation_deposits=10.0,
            wind_intensity=5.0,
            sun_azimuth_angle=180.0,
            sun_altitude_angle=-45.0,
            fog_density=40.0,
            fog_distance=20.0,
            fog_falloff=0.1,
            wetness=20.0,
            scattering_intensity=0.1,
            mie_scattering_scale=0.1,
            rayleigh_scattering_scale=0.01,
            dust_storm=0.0
        )
        self.world.set_weather(weather)
        self.current_light = 500
        self.is_night_mode = True
        self.turn_on_lights()

    def turn_on_lights(self):
        lights = carla.VehicleLightState(0)
        lights |= carla.VehicleLightState.Headlight
        lights |= carla.VehicleLightState.Taillight
        lights |= carla.VehicleLightState.Fog
        lights |= carla.VehicleLightState.Position
        
        self.vehicle.set_light_state(lights)
        self.lights_on = True
        print("[LIGHTS] Headlights, taillights, and fog lights turned ON")

    def turn_off_lights(self):
        lights = carla.VehicleLightState(0)
        self.vehicle.set_light_state(lights)
        self.lights_on = False
        print("[LIGHTS] All lights turned OFF")

    def toggle_mode(self):
        if self.is_night_mode:
            self.set_daytime()
            print("[NIGHT MODE] Switched to DAY MODE")
        else:
            self.set_nighttime()
            print("[NIGHT MODE] Switched to NIGHT MODE")

    def get_status(self):
        return {
            'is_night': self.is_night_mode,
            'light_level': self.current_light,
            'lights_on': self.lights_on
        }

def spawn_vehicle(world, blueprint_library):
    """尝试在多个生成点生成车辆"""
    tesla_bp = blueprint_library.find("vehicle.tesla.model3")
    tesla_bp.set_attribute("color", "0, 0, 0")
    
    spawn_points = world.get_map().get_spawn_points()
    
    for i, spawn_point in enumerate(spawn_points[:10]):
        try:
            vehicle = world.spawn_actor(tesla_bp, spawn_point)
            print(f"[INFO] Vehicle spawned at spawn point {i}")
            return vehicle
        except RuntimeError:
            print(f"[INFO] Spawn point {i} occupied, trying next...")
            continue
    
    print("[ERROR] All spawn points are occupied")
    return None

def main():
    print("=" * 60)
    print("CARLA - Night Mode System")
    print("=" * 60)
    
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[INFO] Connected to CARLA server")
        
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        vehicle = spawn_vehicle(world, blueprint_library)
        
        if not vehicle:
            print("[ERROR] Failed to spawn vehicle")
            sys.exit(1)
        
        night_mode = NightModeSystem(vehicle)
        
        vehicle.set_autopilot(True)
        
        print("[INFO] Night mode system activated")
        print("[INFO] Press Ctrl+C to stop")
        print("[INFO] System starts in DAY mode")
        print("[INFO] Toggling night mode in 5 seconds...")
        
        time.sleep(5)
        night_mode.toggle_mode()
        
        try:
            while True:
                status = night_mode.get_status()
                mode = "🌙 NIGHT" if status['is_night'] else "☀️ DAY"
                lights = "💡 ON" if status['lights_on'] else "⬛ OFF"
                
                print(f"\r[INFO] Mode: {mode} | "
                      f"Light Level: {status['light_level']} | "
                      f"Lights: {lights}", end="")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            print("\n[INFO] Cleaning up...")
            vehicle.destroy()
            print("[INFO] Done")
            
    except RuntimeError as e:
        print(f"[ERROR] Runtime error: {e}")
        print("[INFO] Make sure CARLA server is running")
        sys.exit(1)

if __name__ == "__main__":
    main()