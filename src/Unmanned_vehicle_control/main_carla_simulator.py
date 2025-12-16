import carla
import numpy as np

from src.config import MAX_BRAKING_M_S_2, MAX_WHEEL_ANGLE_RAD, MAX_ACCELERATION_M_S_2


class CarlaSimulator:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.ego_vehicle = None
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()

    def load_world(self, map_name):
        self.client.load_world(map_name)

    def spawn_ego_vehicle(self, vehicle_name, x=0, y=0, z=0, pitch=0, yaw=0, roll=0):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_name)[0]
        spawn_location = carla.Location(x, y, z)
        spawn_rotation = carla.Rotation(pitch, yaw,roll)
        spawn_transform = carla.Transform(location=spawn_location, rotation=spawn_rotation)
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)

    def set_spectator(self, x=0, y=0, z=0, pitch=0, yaw=0, roll=0):
        spectator = self.world.get_spectator()
        location = carla.Location(x=x, y=y, z=z)
        rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        spectator_transform = carla.Transform(location, rotation)
        spectator.set_transform(spectator_transform)

    def clean(self):
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()

    def draw_trajectory(self, x_traj, y_traj, height = 0, thickness = 0.1, red = 0, green = 0, blue = 0, life_time = 0.1):
        for i in range(len(x_traj) - 1):
            start_point = carla.Location(x=x_traj[i], y=y_traj[i], z=height)
            end_point = carla.Location(x=x_traj[i + 1], y=y_traj[i + 1], z=height)
            self.world.debug.draw_line(start_point, end_point, thickness=thickness, color=carla.Color(red, green, blue), life_time=life_time)

    def get_main_ego_vehicle_state(self):
        transform = self.ego_vehicle.get_transform()
        x = transform.location.x
        y = transform.location.y
        theta = np.deg2rad(transform.rotation.yaw)
        v = np.sqrt(self.ego_vehicle.get_velocity().x**2 + self.ego_vehicle.get_velocity().y**2)
        return x, y, theta, v

    def apply_control(self, steer, throttle, brake):
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    @staticmethod
    def process_control_inputs(wheel_angle_rad, acceleration_m_s_2):
        if acceleration_m_s_2 == 0:
            throttle = 0
            brake = 0
        elif acceleration_m_s_2 < 0:
            throttle = 0
            brake = acceleration_m_s_2 / MAX_BRAKING_M_S_2
        else:
            throttle = acceleration_m_s_2 / MAX_ACCELERATION_M_S_2
            brake = 0
        steer = wheel_angle_rad / MAX_WHEEL_ANGLE_RAD
        return throttle, brake, steer

    def print_ego_vehicle_characteristics(self):
        if not self.ego_vehicle:
            print("Vehicle not spawned yet!")
            return None

        physics_control = self.ego_vehicle.get_physics_control()

        print("Vehicle Physics Information.\n")

        print("Wheel Information:")
        for i, wheel in enumerate(physics_control.wheels):
            print(f" Wheel {i + 1}:")
            print(f"   Tire Friction: {wheel.tire_friction}")
            print(f"   Damping Rate: {wheel.damping_rate}")
            print(f"   Max Steer Angle: {wheel.max_steer_angle}")
            print(f"   Radius: {wheel.radius}")
            print(f"   Max Brake Torque: {wheel.max_brake_torque}")
            print(f"   Max Handbrake Torque: {wheel.max_handbrake_torque}")
            print(f"   Position (x, y, z): ({wheel.position.x}, {wheel.position.y}, {wheel.position.z})")

        print(f" Torque Curve:")
        for point in physics_control.torque_curve:
            print(f"RPM: {point.x}, Torque: {point.y}")
        print(f" Max RPM: {physics_control.max_rpm}")
        print(f" MOI (Moment of Inertia): {physics_control.moi}")
        print(f" Damping Rate Full Throttle: {physics_control.damping_rate_full_throttle}")
        print(f" Damping Rate Zero Throttle Clutch Engaged: {physics_control.damping_rate_zero_throttle_clutch_engaged}")
        print(f" Damping Rate Zero Throttle Clutch Disengaged: {physics_control.damping_rate_zero_throttle_clutch_disengaged}")
        print(f" If True, the vehicle will have an automatic transmission: {physics_control.use_gear_autobox}")
        print(f" Gear Switch Time: {physics_control.gear_switch_time}")
        print(f" Clutch Strength: {physics_control.clutch_strength}")
        print(f" Final Ratio: {physics_control.final_ratio}")
        print(f" Mass: {physics_control.mass}")
        print(f" Drag coefficient: {physics_control.drag_coefficient}")
        print(f" Steering Curve:")
        for point in physics_control.steering_curve:
            print(f"Speed: {point.x}, Steering: {point.y}")