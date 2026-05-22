import re
import sys
import math
from colorama import Fore, Style, Cursor


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class Logger:
    INFO_WIDTH = 50

    def __init__(self) -> None:
        pass

    @staticmethod
    def _distance_squared(location_a, location_b):
        dx = location_a.x - location_b.x
        dy = location_a.y - location_b.y
        dz = location_a.z - location_b.z
        return dx * dx + dy * dy + dz * dz

    @staticmethod
    def _visible_ljust(text, width):
        visible_length = len(ANSI_ESCAPE_RE.sub("", text))
        return text + (" " * max(0, width - visible_length))

    def vehicle_info(self, ego_vehicle, world, vehicle_detect_range=100.0):
        vehicles = world.get_actors().filter("*vehicle*")

        active_vehicle_num = 0
        ego_location = ego_vehicle.get_transform().location
        detect_range_squared = vehicle_detect_range * vehicle_detect_range
        for vehicle in vehicles:
            vehicle_location = vehicle.get_transform().location
            if self._distance_squared(ego_location, vehicle_location) < detect_range_squared:
                active_vehicle_num += 1

        vehicle_speed = ego_vehicle.get_velocity().length() * 3.6
        nearby_vehicle_num = max(0, active_vehicle_num - 1)

        nearby_info = f"{Fore.GREEN}{Style.BRIGHT}Vehicles nearby: {nearby_vehicle_num}{Style.RESET_ALL}"
        speed_info = f"{Fore.YELLOW}{Style.BRIGHT}Speed: {vehicle_speed:6.2f} km/h{Style.RESET_ALL}"
        
        return nearby_info, speed_info

        
    def push_info(self, weather_manager, ego_vehicle, world, loop_counter):
        weather_info = weather_manager.weather_info()
        nearby_info, speed_info = self.vehicle_info(ego_vehicle, world, 100.0)

        if loop_counter > 0:
            sys.stdout.write(Cursor.UP(5))
        
        sys.stdout.write(
            f"\n{self._visible_ljust(nearby_info, self.INFO_WIDTH)}\n"
            f"{self._visible_ljust(speed_info, self.INFO_WIDTH)}\n"
            f"{weather_info}\n"
        )
        sys.stdout.flush()