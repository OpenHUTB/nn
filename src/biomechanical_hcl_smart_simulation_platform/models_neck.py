"""
Render a PNG image for the MyoConverter Neck6D converted model (MuJoCo XML).

What this script does:
1) Load the converted MuJoCo model: models/mjc/Neck6D/neck6d_cvt3.xml
2) Reset and load keyframe 0 (required by MyoConverter README)
3) Step a few frames
4) Offscreen render a PNG image to output/neck6d_render.png

Optional:
- If you pass --viewer, it will open an interactive viewer (if supported).

Usage:
  python examples/render_neck6d_png.py
  python examples/render_neck6d_png.py --xml models/mjc/Neck6D/neck6d_cvt3.xml --out output/neck6d_render.png
  python examples/render_neck6d_png.py --viewer
"""

import argparse
import os
from pathlib import Path

import mujoco
from PIL import Image


def ensure_parent_dir(filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def set_camera_like_default(scene, lookat=None, distance=None, azimuth=None, elevation=None):
    """
    Simple camera setup helper.
    If your teacher wants exactly the same viewpoint as a reference image,
    you can tune these parameters.
    """
    # scene is mujoco.MjvScene; camera stored in scene.camera
    cam = scene.camera

    # cam.lookat is a 3-vector
    if lookat is not None:
        cam.lookat[:] = lookat

    # cam.distance is a float
    if distance is not None:
        cam.distance = float(distance)

    # cam.azimuth/elevation in degrees
    if azimuth is not None:
        cam.azimuth = float(azimuth)

    if elevation is not None:
        cam.elevation = float(elevation)


def offscreen_render_png(
    xml_path: str,
    out_path: str,
    width: int = 1400,
    height: int = 1000,
    steps: int = 10,
    camera_lookat=(0.0, 0.0, 1.1),
    camera_distance=2.2,
    camera_azimuth=135.0,
    camera_elevation=-15.0,
) -> None:
    """
    Offscreen render a single PNG frame.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    # Load model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # IMPORTANT: Load keyframe 0 (per README)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Step a few frames to settle (optional but often makes rendering stable)
    for _ in range(max(0, int(steps))):
        mujoco.mj_step(model, data)

    # Create renderer
    renderer = mujoco.Renderer(model, width=width, height=height)

    # Update scene (so camera edits apply)
    renderer.update_scene(data)

    # Adjust camera to a reasonable view (tune to match your reference)
    set_camera_like_default(
        renderer.scene,
        lookat=camera_lookat,
        distance=camera_distance,
        azimuth=camera_azimuth,
        elevation=camera_elevation,
    )

    # Render
    img = renderer.render()

    # Save
    ensure_parent_dir(out_path)
    Image.fromarray(img).save(out_path)
    print(f"[OK] Saved render to: {out_path}")


def optional_viewer(xml_path: str) -> None:
    """
    Launch an interactive viewer (requires a working OpenGL context).
    """
    import time
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description="Render Neck6D MuJoCo model to a PNG.")
    parser.add_argument(
        "--xml",
        type=str,
        default="models/mjc/Neck6D/neck6d_cvt3.xml",
        help="Path to MuJoCo XML model (default: models/mjc/Neck6D/neck6d_cvt3.xml)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/neck6d_render.png",
        help="Output PNG path (default: output/neck6d_render.png)",
    )
    parser.add_argument("--width", type=int, default=1400, help="Image width")
    parser.add_argument("--height", type=int, default=1000, help="Image height")
    parser.add_argument("--steps", type=int, default=10, help="Simulation steps before capture")

    # Camera parameters (easy to justify + easy for老师复现)
    parser.add_argument("--lookat-x", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=1.1)
    parser.add_argument("--distance", type=float, default=2.2)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-15.0)

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open interactive viewer after rendering (optional).",
    )

    args = parser.parse_args()

    offscreen_render_png(
        xml_path=args.xml,
        out_path=args.out,
        width=args.width,
        height=args.height,
        steps=args.steps,
        camera_lookat=(args.lookat_x, args.lookat_y, args.lookat_z),
        camera_distance=args.distance,
        camera_azimuth=args.azimuth,
        camera_elevation=args.elevation,
    )

    if args.viewer:
        optional_viewer(args.xml)


if __name__ == "__main__":
    main()
