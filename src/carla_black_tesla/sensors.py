import carla
import time

class SensorCollector:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.imu_sensor = None
        self.gnss_sensor = None
        self.speed = 0
        self.acceleration = [0, 0, 0]
        self.location = None
        self.throttle = 0
        self.brake = 0
        self.steer = 0

    def setup_sensors(self, world):
        blueprint = world.get_blueprint_library()
        
        imu_bp = blueprint.find('sensor.other.imu')
        self.imu_sensor = world.spawn_actor(
            imu_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.imu_sensor.listen(lambda data: self._update_imu(data))
        
        gnss_bp = blueprint.find('sensor.other.gnss')
        self.gnss_sensor = world.spawn_actor(
            gnss_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.gnss_sensor.listen(lambda data: self._update_gnss(data))

    def _update_imu(self, data):
        self.acceleration = [data.accelerometer.x, data.accelerometer.y, data.accelerometer.z]

    def _update_gnss(self, data):
        self.location = (data.latitude, data.longitude, data.altitude)

    def update(self):
        velocity = self.vehicle.get_velocity()
        self.speed = ((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5) * 3.6
        
        transform = self.vehicle.get_transform()
        if self.location is None:
            self.location = (transform.location.x, transform.location.y, transform.location.z)
        
        control = self.vehicle.get_control()
        self.throttle = control.throttle
        self.brake = control.brake
        self.steer = control.steer

    def display(self):
        print(f"Speed: {self.speed:.1f} km/h | "
              f"Accel: ({self.acceleration[0]:.1f}, {self.acceleration[1]:.1f}, {self.acceleration[2]:.1f}) | "
              f"Throttle: {self.throttle:.2f} | Brake: {self.brake:.2f} | Steer: {self.steer:.2f}")

    def destroy(self):
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.gnss_sensor:
            self.gnss_sensor.destroy()

def main():
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        bp_lib = world.get_blueprint_library()
        tesla_bp = bp_lib.find("vehicle.tesla.model3")
        tesla_bp.set_attribute("color", "0, 0, 0")
        
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(tesla_bp, spawn_points[0])
        vehicle.set_autopilot(True)
        
        sensors = SensorCollector(vehicle)
        sensors.setup_sensors(world)
        print("Sensor collection enabled!")
        
        while True:
            sensors.update()
            sensors.display()
            time.sleep(0.1)
    except KeyboardInterrupt:
        sensors.destroy()
        vehicle.destroy()

if __name__ == "__main__":
    main()
