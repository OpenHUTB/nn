"""MuJoCo simulation backend.

The vehicle is a planar-constrained rigid body (two slide joints + one
yaw hinge) driven by a dynamic bicycle model with a yaw-rate servo.
Collisions with the track walls are resolved by MuJoCo's contact solver,
so bumping into a wall really does slow the car down.

Sensors
-------
* ``front`` camera, 70 deg vertical FOV, mounted on the roof (RGB frames
  are what the neural networks consume);
* a 2-D lidar made of 64 horizontal :func:`mujoco.mj_ray` casts, used by
  the mapping / SLAM pipeline;
* proprioceptive state (speed, yaw rate, lateral and heading error).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .track import (
    CAR_LENGTH,
    CAR_WIDTH,
    ROAD_HALF_WIDTH,
    WALL_HEIGHT,
    WALL_THICKNESS,
    Track,
    build_track,
    start_pose,
)

# ---------------------------------------------------------------- dynamics
MASS = 1500.0
WHEELBASE = 2.7
INERTIA_Z = MASS * (CAR_LENGTH**2 + CAR_WIDTH**2) / 12.0
MAX_STEER = 0.55
MAX_ACCEL = 4.0
MAX_BRAKE = 7.0
MAX_LAT_ACCEL = 7.0
MAX_SPEED = 16.0
DRAG_C = 0.55
ROLL_RESIST = 0.02
YAW_SERVO = 12.0
TAU_MAX = 12000.0
SUBSTEPS = 5  # 100 Hz physics, 20 Hz control

LIDAR_BEAMS = 64
LIDAR_RANGE = 20.0
LIDAR_HEIGHT = 0.05
IMAGE_SIZE = 64

SCENES = {
    # name: (light diffuse, light ambient, haze, haze_color)
    "day": ((1.0, 0.98, 0.92), (0.45, 0.45, 0.45), 0.05, (0.55, 0.70, 0.90)),
    "dusk": ((1.0, 0.72, 0.45), (0.30, 0.26, 0.22), 0.18, (0.45, 0.35, 0.40)),
    "fog": ((0.85, 0.88, 0.92), (0.50, 0.50, 0.50), 0.45, (0.75, 0.78, 0.82)),
}


def _quat_from_mat(mat: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z) quaternion."""
    m = np.asarray(mat, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def camera_quat(pitch_deg: float = -6.0) -> np.ndarray:
    """Orientation of a forward-looking camera in the car body frame."""
    # Camera convention: looks along -Z, up along +Y, right along +X.
    # In the body frame (x forward, y left, z up) this means
    #   camera x (right)  = body -y
    #   camera y (up)     = body +z
    #   camera z (back)   = body -x
    base = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    p = np.deg2rad(pitch_deg)
    rot_x = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return _quat_from_mat(base @ rot_x)


@dataclass
class CarState:
    x: float
    y: float
    yaw: float
    speed: float  # forward speed m/s
    lateral: float  # signed distance to centre line, +left
    heading_err: float  # rad, +left
    s: float  # progress along track
    lateral_vel: float = 0.0
    yaw_rate: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.speed / MAX_SPEED, self.lateral / ROAD_HALF_WIDTH, self.heading_err / 0.6, self.yaw_rate / 1.0],
            dtype=np.float32,
        )


@dataclass
class SimConfig:
    scene: str = "day"
    seed: int = 0
    start_offset: float = 0.0
    randomize_start: bool = False
    offset_range: float = 1.2
    allow_reverse: bool = False
    range_noise: float = 0.02


class CarSim:
    """Thin, deterministic wrapper around a MuJoCo car + track scene."""

    def __init__(self, track: Track | None = None, config: SimConfig | None = None):
        self.track = track if track is not None else build_track()
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.model = mujoco.MjModel.from_xml_string(self._xml(), assets=self._assets())
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.front_cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        self.wall_group = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        self.renderer = mujoco.Renderer(self.model, IMAGE_SIZE, IMAGE_SIZE)
        self.renderer._scene_option.geomgroup[2] = 0  # hide own car body in ego camera
        self._free_cam = mujoco.MjvCamera()
        self._free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        self.set_scene(self.config.scene)
        self.reset(self.config.start_offset)

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _assets(self) -> dict[str, bytes]:
        road = self._strip_obj(0.0, ROAD_HALF_WIDTH)
        left = self._strip_obj(ROAD_HALF_WIDTH - 0.18, 0.18, z=0.02)
        right = self._strip_obj(-(ROAD_HALF_WIDTH - 0.18), 0.18, z=0.02)
        return {
            "road.obj": road.encode("ascii"),
            "edge_left.obj": left.encode("ascii"),
            "edge_right.obj": right.encode("ascii"),
        }

    def _strip_obj(self, offset: float, half_width: float, z: float = 0.012) -> str:
        """Road / lane strip as OBJ text (quad strip along the centre line)."""
        idx = np.arange(0, len(self.track.points), 2)
        left = self.track.points[idx] + (offset + half_width) * self.track.normals[idx]
        right = self.track.points[idx] + (offset - half_width) * self.track.normals[idx]
        n = len(idx)
        lines = []
        for (lx, ly), (rx, ry) in zip(left, right):
            lines.append(f"v {lx:.4f} {ly:.4f} {z:.4f}")
            lines.append(f"v {rx:.4f} {ry:.4f} {z:.4f}")
        for i in range(n):
            a, b = 2 * i + 1, 2 * i + 2
            c, d = 2 * ((i + 1) % n) + 1, 2 * ((i + 1) % n) + 2
            lines.append(f"f {a} {b} {d}")
            lines.append(f"f {a} {d} {c}")
        return "\n".join(lines) + "\n"

    def _xml(self) -> str:
        walls = []
        for xy, yaw, hl in self.track.wall_poses(spacing=2.0):
            for (x, y), angle in zip(xy, yaw):
                walls.append(
                    f'<geom type="box" pos="{x:.3f} {y:.3f} {WALL_HEIGHT / 2:.3f}" '
                    f'euler="0 0 {angle:.4f}" size="{hl:.3f} {WALL_THICKNESS / 2:.3f} {WALL_HEIGHT / 2:.3f}" '
                    f'material="wall" group="1"/>'
                )

        # Dashed centre line: 3 m dash every 6 m.
        dashes = []
        spacing, dash_len = 6.0, 3.0
        n = int(self.track.length // spacing)
        for k in range(n):
            s = k * spacing
            idx = int(round(s / self.track.length * len(self.track.points))) % len(self.track.points)
            x, y = self.track.points[idx]
            angle = np.arctan2(self.track.tangents[idx, 1], self.track.tangents[idx, 0])
            dashes.append(
                f'<geom type="box" pos="{x:.3f} {y:.3f} 0.021" euler="0 0 {angle:.4f}" '
                f'size="{dash_len / 2:.3f} 0.09 0.004" material="line" contype="0" conaffinity="0"/>'
            )

        q = camera_quat()
        start = self.track.points[0]
        yaw0 = np.arctan2(self.track.tangents[0, 1], self.track.tangents[0, 0])
        qy = np.array([np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)])

        return f"""
<mujoco model="mujoco_car_nn">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  <option timestep="0.01" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" impratio="20"/>
  <visual>
    <global offwidth="1280" offheight="720" fovy="45"/>
    <quality shadowsize="2048" offsamples="2"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.15 0.15 0.15"/>
    <rgba haze="0.6 0.72 0.85 1"/>
    <map znear="0.0005" zfar="3.0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.74 0.95" rgb2="0.16 0.24 0.38" width="256" height="1536"/>
    <texture name="grass" type="2d" builtin="checker" rgb1="0.32 0.46 0.26" rgb2="0.27 0.41 0.22" width="256" height="256"/>
    <material name="grass" texture="grass" texrepeat="120 120" reflectance="0.05"/>
    <material name="asphalt" rgba="0.20 0.21 0.24 1" specular="0.2" shininess="0.25" reflectance="0.05"/>
    <material name="line" rgba="0.96 0.96 0.92 1" specular="0.1"/>
    <material name="wall" rgba="0.82 0.28 0.24 1" specular="0.2"/>
    <material name="body" rgba="0.12 0.34 0.72 1" specular="0.6" shininess="0.7"/>
    <material name="glass" rgba="0.10 0.13 0.18 1" specular="0.9" shininess="0.9"/>
    <material name="tyre" rgba="0.06 0.06 0.06 1"/>
    <material name="rim" rgba="0.75 0.76 0.78 1" specular="0.8"/>
    <mesh name="road" file="road.obj" inertia="shell"/>
    <mesh name="edge_left" file="edge_left.obj" inertia="shell"/>
    <mesh name="edge_right" file="edge_right.obj" inertia="shell"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 60" dir="0 0 -1" directional="true" diffuse="1 0.98 0.92" ambient="0.45 0.45 0.45" castshadow="false"/>
    <geom name="ground" type="plane" size="120 120 0.1" material="grass" condim="1" contype="0" conaffinity="0" group="0"/>
    <geom name="road" type="mesh" mesh="road" material="asphalt" contype="0" conaffinity="0" group="0"/>
    <geom name="edge_left" type="mesh" mesh="edge_left" material="line" contype="0" conaffinity="0" group="0"/>
    <geom name="edge_right" type="mesh" mesh="edge_right" material="line" contype="0" conaffinity="0" group="0"/>
    {chr(10).join(dashes)}
    {chr(10).join(walls)}
    <body name="car" pos="0 0 0">
      <inertial pos="0 0 0.55" mass="1.0" diaginertia="0.1 0.1 0.1"/>
      <joint name="car_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="car_y" type="slide" axis="0 1 0" damping="0"/>
      <body name="chassis" pos="0 0 0.55">
        <joint name="car_yaw" type="hinge" axis="0 0 1" damping="0"/>
        <geom name="chassis" type="box" size="{CAR_LENGTH / 2:.3f} {CAR_WIDTH / 2:.3f} 0.30" material="body" mass="{MASS * 0.8:.1f}" group="2"/>
        <geom name="cabin" type="box" pos="-0.15 0 0.46" size="0.85 0.78 0.22" material="glass" mass="{MASS * 0.2:.1f}" group="2"/>
        <geom name="wheel_fl" type="cylinder" pos="1.35 0.82 -0.20" euler="1.5708 0 0" size="0.35 0.12" material="tyre" contype="0" conaffinity="0" group="2"/>
        <geom name="wheel_fr" type="cylinder" pos="1.35 -0.82 -0.20" euler="1.5708 0 0" size="0.35 0.12" material="tyre" contype="0" conaffinity="0" group="2"/>
        <geom name="wheel_rl" type="cylinder" pos="-1.35 0.82 -0.20" euler="1.5708 0 0" size="0.35 0.12" material="tyre" contype="0" conaffinity="0" group="2"/>
        <geom name="wheel_rr" type="cylinder" pos="-1.35 -0.82 -0.20" euler="1.5708 0 0" size="0.35 0.12" material="tyre" contype="0" conaffinity="0" group="2"/>
        <camera name="front" pos="1.45 0 0.85" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" fovy="70"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, lateral_offset: float = 0.0) -> CarState:
        if self.config.randomize_start:
            lateral_offset = float(self.rng.uniform(-self.config.offset_range, self.config.offset_range))
        xy, yaw = start_pose(self.track, lateral_offset)
        self.data.qpos[:] = 0.0
        self.data.qpos[0] = xy[0]
        self.data.qpos[1] = xy[1]
        self.data.qpos[2] = yaw
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def set_scene(self, name: str) -> None:
        diffuse, ambient, haze, color = SCENES[name]
        self.model.light_diffuse[0] = diffuse
        self.model.light_ambient[0] = ambient
        self.model.vis.map.haze = haze
        self.model.vis.rgba.haze[:3] = color
        self.config.scene = name

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _apply_control(self, steer: float, throttle: float) -> None:
        d = self.data
        yaw = float(d.qpos[2])
        c, s = np.cos(yaw), np.sin(yaw)
        v_fwd = c * d.qvel[0] + s * d.qvel[1]
        v_lat = -s * d.qvel[0] + c * d.qvel[1]
        omega = float(d.qvel[2])

        delta = float(np.clip(steer, -1.0, 1.0)) * MAX_STEER
        speed = max(abs(v_fwd), 0.6)
        omega_des = v_fwd * np.tan(delta) / WHEELBASE
        omega_max = MAX_LAT_ACCEL / speed
        omega_des = float(np.clip(omega_des, -omega_max, omega_max))
        tau = float(np.clip(INERTIA_Z * (omega_des - omega) * YAW_SERVO, -TAU_MAX, TAU_MAX))

        throttle = float(np.clip(throttle, -1.0, 1.0))
        accel = throttle * MAX_ACCEL if throttle >= 0 else throttle * MAX_BRAKE
        if throttle < 0.0 and v_fwd <= 0.05 and not self.config.allow_reverse:
            accel = 0.0  # evaluation never drives backwards; teleoperation may
        f_long = MASS * accel - DRAG_C * v_fwd * abs(v_fwd) - ROLL_RESIST * MASS * 9.81 * np.sign(v_fwd)
        if throttle <= 0.0 and abs(v_fwd) < 0.05 and not self.config.allow_reverse:
            f_long = 0.0
        if v_fwd > MAX_SPEED:
            f_long -= MASS * (v_fwd - MAX_SPEED) * 2.0
        f_lat = float(np.clip(-MASS * v_lat * 9.0, -1.0 * MASS * 9.81, 1.0 * MASS * 9.81))

        d.qfrc_applied[0] = c * f_long - s * f_lat
        d.qfrc_applied[1] = s * f_long + c * f_lat
        d.qfrc_applied[2] = tau

    def step(self, action, substeps: int = SUBSTEPS) -> CarState:
        steer, throttle = float(action[0]), float(action[1])
        for _ in range(substeps):
            self._apply_control(steer, throttle)
            mujoco.mj_step(self.model, self.data)
        return self.state()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def state(self) -> CarState:
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        yaw = float(self.data.qpos[2])
        c, s = np.cos(yaw), np.sin(yaw)
        vx = c * self.data.qvel[0] + s * self.data.qvel[1]
        vy = -s * self.data.qvel[0] + c * self.data.qvel[1]
        lateral, heading_err = self.track.lateral_error(np.array([x, y]), yaw)
        idx, _ = self.track.nearest(np.array([x, y]))
        return CarState(
            x=x,
            y=y,
            yaw=yaw,
            speed=float(vx),
            lateral=lateral,
            heading_err=heading_err,
            s=float(self.track.s[idx]),
            lateral_vel=float(vy),
            yaw_rate=float(self.data.qvel[2]),
        )

    def lidar(self) -> np.ndarray:
        """64 horizontal ranges in the car frame, ``LIDAR_RANGE`` = no hit."""
        d = self.data
        yaw = float(d.qpos[2])
        origin = np.array([d.qpos[0], d.qpos[1], LIDAR_HEIGHT + 0.55])
        ranges = np.full(LIDAR_BEAMS, LIDAR_RANGE, dtype=np.float32)
        angles = yaw + np.linspace(-np.pi, np.pi, LIDAR_BEAMS, endpoint=False)
        geomid = np.zeros(1, dtype=np.int32)
        for i, a in enumerate(angles):
            vec = np.array([np.cos(a), np.sin(a), 0.0])
            dist = mujoco.mj_ray(
                self.model, d, origin, vec, self.wall_group, True, -1, geomid
            )
            if dist >= 0.0:
                ranges[i] = min(float(dist), LIDAR_RANGE)
        if self.config.range_noise > 0:
            ranges += self.rng.normal(0.0, self.config.range_noise, size=ranges.shape).astype(np.float32)
            ranges = np.clip(ranges, 0.05, LIDAR_RANGE)
        return ranges

    def camera_image(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.front_cam)
        return self.renderer.render().copy()

    # ------------------------------------------------------------------
    # Rendering for figures / GIFs
    # ------------------------------------------------------------------
    def render_free(self, width: int, height: int, lookat, azimuth: float, elevation: float, distance: float) -> np.ndarray:
        cam = self._free_cam
        cam.lookat[:] = lookat
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.distance = distance
        renderer = getattr(self, "_viz_renderer", None)
        if renderer is None or renderer.width != width or renderer.height != height:
            renderer = mujoco.Renderer(self.model, height, width)
            self._viz_renderer = renderer
        renderer.update_scene(self.data, camera=cam)
        return renderer.render().copy()

    def render_chase(self, width: int = 640, height: int = 360) -> np.ndarray:
        st = self.state()
        lookat = [st.x + 2.0 * np.cos(st.yaw), st.y + 2.0 * np.sin(st.yaw), 1.0]
        az = np.degrees(st.yaw)
        return self.render_free(width, height, lookat, az, -14.0, 11.0)

    def render_top(self, width: int = 640, height: int = 360, distance: float = 105.0) -> np.ndarray:
        return self.render_free(width, height, [0.0, 0.0, 0.0], 90.0, -90.0, distance)

    def render_front(self, width: int = 640, height: int = 360) -> np.ndarray:
        renderer = getattr(self, "_front_big", None)
        if renderer is None or renderer.width != width or renderer.height != height:
            renderer = mujoco.Renderer(self.model, height, width)
            renderer._scene_option.geomgroup[2] = 0
            self._front_big = renderer
        renderer.update_scene(self.data, camera=self.front_cam)
        return renderer.render().copy()

    def close(self) -> None:
        for name in ("renderer", "_viz_renderer", "_front_big"):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass

