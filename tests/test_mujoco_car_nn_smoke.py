"""Smoke tests: run with ``python -m pytest tests -q`` or ``python tests/test_smoke.py``."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mujoco_car_nn.expert import PurePursuitExpert  # noqa: E402
from mujoco_car_nn.models import build_model  # noqa: E402
from mujoco_car_nn.sim import IMAGE_SIZE, LIDAR_BEAMS, CarSim, SimConfig  # noqa: E402
from mujoco_car_nn.slam import GridMap  # noqa: E402
from mujoco_car_nn.track import build_track  # noqa: E402


def test_track_closed_and_smooth():
    track = build_track()
    assert 150 < track.length < 350
    # curve closes on itself
    assert np.linalg.norm(track.points[0] - track.points[-1]) < 5.0
    # minimum turning radius is drivable for a 9 m wide road
    assert 1.0 / np.abs(track.curvature).max() > 9.0
    # arc length is monotonic
    assert np.all(np.diff(track.s) > 0)


def test_sim_reset_and_step():
    sim = CarSim(build_track(), SimConfig(scene="day", randomize_start=False))
    st = sim.reset(0.0)
    assert abs(st.lateral) < 1e-6
    expert = PurePursuitExpert(sim.track)
    for _ in range(40):
        st = sim.step(expert.act(st))
    assert st.speed > 1.0, "expert should accelerate the car"
    img = sim.camera_image()
    assert img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    scan = sim.lidar()
    assert scan.shape == (LIDAR_BEAMS,)
    assert scan.min() < 10.0, "lidar should see the walls"
    sim.close()


def test_policy_shapes_and_range():
    import torch

    for name in ("cnn", "mlp"):
        net = build_model(name)
        out = net(torch.zeros(2, 3, 64, 64), torch.zeros(2, 1))
        assert out.shape == (2, 2)
        assert torch.all(out.abs() <= 1.0 + 1e-6)


def test_grid_map_update():
    grid = GridMap()
    ranges = np.full(64, 5.0, dtype=np.float32)
    angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    grid.update((0.0, 0.0, 0.0), ranges, angles)
    assert (grid.log_odds > 0).sum() > 0, "endpoints must be marked occupied"
    assert (grid.log_odds < 0).sum() > 0, "rays must clear free space"


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all tests passed")
