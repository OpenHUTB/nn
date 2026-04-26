import sys
import math
from colorama import Fore, Back, Style, Cursor

class Logger:
    def __init__(self) -> None:
        pass

    def vehicle_info(self, ego_vehicle, world, vehicle_detect_range=100.0):
        vehicles = world.get_actors().filter("*vehicle*")

        active_vehicle_num = 0
        ego_location = ego_vehicle.get_transform()
        for vehicle in vehicles:
            t = vehicle.get_transform()
            get_distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            distance = get_distance(ego_location.location)
            if distance < vehicle_detect_range:
                active_vehicle_num += 1

        vehicle_speed = ego_vehicle.get_velocity().length() * 3.6

        nearby_info = f"{Fore.GREEN}{Style.BRIGHT}Vehicles nearby: {(active_vehicle_num - 1)}{Style.RESET_ALL}"
        speed_info = f"{Fore.YELLOW}{Style.BRIGHT}Speed: {vehicle_speed:6.2f} km/h{Style.RESET_ALL}"
        
        return nearby_info, speed_info

        
    def push_info(self, weather_manager, ego_vehicle, world, loop_counter):
        weather_info = weather_manager.weather_info()
        nearby_info, speed_info = self.vehicle_info(ego_vehicle, world, 100.0)

        if loop_counter > 0:
            sys.stdout.write(Cursor.UP(5))
        
        sys.stdout.write(f"\n{nearby_info.ljust(50)}\n{speed_info.ljust(50)}\n{weather_info}\n")    
        sys.stdout.flush()