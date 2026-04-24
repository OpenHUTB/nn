import argparse
import csv
import os
import queue
import threading
import time
import io
import math
import numpy as np

try:
    import carla
except ImportError:
    raise RuntimeError("CARLA Python API not found. Ensure carla module is on PYTHONPATH.")

try:
    import pygame
except ImportError:
    pygame = None

try:
    from PIL import Image
except ImportError:
    Image = None

from http.server import BaseHTTPRequestHandler, HTTPServer

# --- Shared state for server mode ---
_latest_frame_lock = threading.Lock()
_latest_frame_rgb = None
_latest_frame_jpeg = None
_latest_dims = (0, 0)


class CarlaSync:
    """Synchronous CARLA simulation context manager"""
    def __init__(self, world, sensors, fps=20.0):
        self.world = world
        self.sensors = sensors
        self.delta = 1.0 / fps
        self._original_settings = None
        self.queues = []

    def __enter__(self):
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta
        self.world.apply_settings(settings)

        # Setup sensor queues
        self.queues = []
        for s in self.sensors:
            q = queue.Queue()
            s.listen(q.put)
            self.queues.append(q)
        return self

    def tick(self, timeout=2.0):
        frame = self.world.tick()
        data = []
        end_time = time.time() + timeout
        for q in self.queues:
            remaining = max(end_time - time.time(), 0.001)
            while True:
                item = q.get(timeout=remaining)
                if getattr(item, "frame", None) == frame:
                    data.append(item)
                    break
        return frame, data

    def __exit__(self, exc_type, exc_val, exc_tb):
        for s in self.sensors:
            s.stop()
        if self._original_settings:
            self.world.apply_settings(self._original_settings)


def render_pygame(surface, image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    surface.blit(surf, (0, 0))


def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


class FrameHTTPHandler(BaseHTTPRequestHandler):
    """HTTP server for serving the latest camera frame"""
    def do_GET(self):
        global _latest_frame_rgb, _latest_frame_jpeg, _latest_dims
        if self.path.startswith("/frame"):
            with _latest_frame_lock:
                if _latest_frame_rgb is None:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"no frame yet")
                    return
                # Send JPEG if Pillow is available
                if Image is not None:
                    if _latest_frame_jpeg is None:
                        buf = io.BytesIO()
                        Image.fromarray(_latest_frame_rgb).save(buf, format="JPEG", quality=85)
                        _latest_frame_jpeg = buf.getvalue()
                    data = _latest_frame_jpeg
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    data = _latest_frame_rgb.tobytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("X-Format", "rgb24")
                    self.send_header("X-Width", str(_latest_dims[0]))
                    self.send_header("X-Height", str(_latest_dims[1]))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK: /frame endpoint available")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA RGB camera frame viewer/recorder/server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", default=2000, type=int)
    parser.add_argument("--town", default=None)
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument("--width", default=400, type=int)
    parser.add_argument("--height", default=224, type=int)
    parser.add_argument("--fov", default=90.0, type=float)
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--log-every", default=20, type=int)
    parser.add_argument("--mode", choices=["display", "record", "server"], default="display")
    parser.add_argument("--traffic-vehicles", default=0, type=int)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", default=8080, type=int)
    return parser.parse_args()


def choose_vehicle_bp(bp_library):
    for name in ["vehicle.tesla.model3", "vehicle.lincoln.mkz_2020",
                 "vehicle.audi.tt", "vehicle.mercedes.coupe",
                 "vehicle.bmw.grandtourer", "vehicle.tesla.cybertruck"]:
        bp = bp_library.find(name)
        if bp:
            return bp
    return bp_library.filter("vehicle.*")[0]


def main():
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    if args.town and (world.get_map() is None or args.town not in world.get_map().name):
        world = client.load_world(args.town)

    bp_library = world.get_blueprint_library()

    # Spawn ego vehicle
    vehicle_bp = choose_vehicle_bp(bp_library)
    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "ego")
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0] if spawn_points else carla.Transform()
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point) or world.spawn_actor(vehicle_bp, spawn_point)

    # Spawn traffic vehicles
    traffic_vehicles = []
    if args.traffic_vehicles > 0:
        vehicle_bps = bp_library.filter("vehicle.*")
        spawn_points = world.get_map().get_spawn_points()
        for _ in range(args.traffic_vehicles):
            bp = np.random.choice(vehicle_bps)
            sp = np.random.choice(spawn_points)
            v = world.try_spawn_actor(bp, sp)
            if v:
                v.set_autopilot(True)
                traffic_vehicles.append(v)

    # Camera setup
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.width))
    cam_bp.set_attribute("image_size_y", str(args.height))
    cam_bp.set_attribute("fov", str(args.fov))
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.0), carla.Rotation(pitch=-5.0))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    actors = [vehicle, camera] + traffic_vehicles

    # Mode flags
    is_record = args.mode == "record"
    is_server = args.mode == "server"
    want_display = (args.mode == "display") or (args.mode == "server" and not args.no_display)

    # Display setup
    if want_display:
        if pygame is None:
            raise RuntimeError("pygame not installed")
        pygame.init()
        display = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("CARLA RGB Camera")
        clock = pygame.time.Clock()
    else:
        display = None
        clock = None

    # Dataset paths
    if is_record:
        images_dir = os.path.join(args.out, "images")
        ensure_dir(images_dir)
        csv_file = open(os.path.join(args.out, "data.csv"), "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "image_path", "steer", "throttle", "brake", "speed_mps",
                         "x", "y", "z", "roll", "pitch", "yaw"])
    else:
        images_dir = None
        writer = None
        csv_file = None

    # Start HTTP server
    if is_server:
        def run_server():
            HTTPServer((args.server_host, args.server_port), FrameHTTPHandler).serve_forever()
        threading.Thread(target=run_server, daemon=True).start()
        print(f"[server] hosting camera at http://{args.server_host}:{args.server_port}/frame")

    try:
        with CarlaSync(world, [camera], fps=args.fps) as sync:
            frame_count = 0
            running = True
            while running:
                if display:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            running = False

                frame, data = sync.tick()
                image = data[0]

                # Display
                if display:
                    render_pygame(display, image)
                    pygame.display.flip()
                    if clock:
                        clock.tick(args.fps)

                # Update server frame
                if is_server:
                    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
                    rgb = arr[:, :, :3][:, :, ::-1].copy()
                    with _latest_frame_lock:
                        global _latest_frame_rgb, _latest_frame_jpeg, _latest_dims
                        _latest_frame_rgb = rgb
                        _latest_dims = (image.width, image.height)
                        if Image:
                            buf = io.BytesIO()
                            Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
                            _latest_frame_jpeg = buf.getvalue()

                # Record
                if is_record:
                    filename = f"{frame:06d}.png"
                    abs_path = os.path.join(images_dir, filename)
                    image.save_to_disk(abs_path)
                    ctrl = vehicle.get_control()
                    speed = get_speed(vehicle)
                    loc = vehicle.get_transform().location
                    rot = vehicle.get_transform().rotation
                    writer.writerow([frame, filename, ctrl.steer, ctrl.throttle, ctrl.brake,
                                     speed, loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw])
                    csv_file.flush()
                    frame_count += 1
                    if args.log_every > 0 and frame_count % args.log_every == 0:
                        print(f"Recorded {frame_count} frames")

    except KeyboardInterrupt:
        pass
    finally:
        if csv_file:
            csv_file.close()
        if display and pygame:
            pygame.quit()
        for actor in actors:
            try:
                actor.destroy()
            except Exception:
                pass
        if is_record:
            print(f"Stopped. Recorded {frame_count} frames.")


if __name__ == "__main__":
    main()
