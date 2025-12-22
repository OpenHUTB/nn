# preliminary_human_model_render_thigh_muscles.py
# Render MuJoCo MJCF with "thigh muscle" visualization as thick capsules, optionally exporting a GIF.

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

import mujoco

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None


# -----------------------------
# Config
# -----------------------------
DEFAULT_THIGH_KEYWORDS = (
    # Quadriceps
    "rectus", "rect_fem", "rf", "vasti", "vastus",
    # Hamstrings
    "ham", "biceps_fem", "bf", "semiten", "semimem",
    # Adductors / sartorius / gracilis / TFL
    "adductor", "add", "sart", "sartorius", "grac", "gracilis", "tfl", "tensor",
    # Hip flexors / glutes (often affect thigh look)
    "iliopsoas", "psoas", "iliacus", "glut", "glute",
)

RGBA_REST = np.array([0.85, 0.15, 0.15, 0.45], dtype=np.float32)   # relaxed muscle (semi-transparent red)
RGBA_ACTIVE = np.array([1.00, 0.25, 0.25, 0.85], dtype=np.float32) # active muscle (brighter, less transparent)


@dataclass
class MuscleVizSpec:
    base_radius: float = 0.010   # meters
    bulge_gain: float = 1.25     # radius multiplier gain: r = base*(1 + bulge_gain*act)
    max_capsules: int = 3000     # budget for extra geoms


# -----------------------------
# Utilities
# -----------------------------
def _safe_name(model: mujoco.MjModel, objtype: mujoco.mjtObj, idx: int) -> str:
    name = mujoco.mj_id2name(model, objtype, idx)
    return name or f"{objtype.name}_{idx}"


def _pick_actuators_by_keywords(model: mujoco.MjModel, keywords: Sequence[str]) -> List[int]:
    keys = tuple(k.lower() for k in keywords)
    picked: List[int] = []
    for i in range(model.nu):  # in MuJoCo, #controls == #actuators for typical models
        nm = _safe_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).lower()
        if any(k in nm for k in keys):
            picked.append(i)
    return picked


def _actuator_to_tendon_id(model: mujoco.MjModel, act_id: int) -> Optional[int]:
    """
    For muscle actuators using tendon transmission, actuator_trntype should be mjTRN_TENDON
    and actuator_trnid[act_id, 0] is the tendon id.
    """
    if not hasattr(model, "actuator_trntype") or not hasattr(model, "actuator_trnid"):
        return None
    trntype = int(model.actuator_trntype[act_id])
    if trntype != int(mujoco.mjtTrn.mjTRN_TENDON):
        return None
    tid = int(model.actuator_trnid[act_id, 0])
    if tid < 0 or tid >= model.ntendon:
        return None
    return tid


def _tendon_path_sites_world(model: mujoco.MjModel, data: mujoco.MjData, tendon_id: int) -> Optional[np.ndarray]:
    """
    Best-effort extraction of tendon path point(s) using wrap sites.
    If wrap-site info isn't available, returns None.
    """
    required = ("tendon_adr", "tendon_num", "wrap_type", "wrap_objid")
    if not all(hasattr(model, f) for f in required):
        return None

    adr = int(model.tendon_adr[tendon_id])
    num = int(model.tendon_num[tendon_id])
    if num <= 0:
        return None

    pts: List[np.ndarray] = []
    for w in range(adr, adr + num):
        wtype = int(model.wrap_type[w])
        if wtype == int(mujoco.mjtWrap.mjWRAP_SITE):
            sid = int(model.wrap_objid[w])
            pts.append(data.site_xpos[sid].copy())
        # NOTE: other wrap types (geom/pulley) are ignored here for simplicity.

    if len(pts) >= 2:
        return np.stack(pts, axis=0)
    return None


def _normalize01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _activation_from_ctrl(data: mujoco.MjData, act_id: int) -> float:
    if data.ctrl is None or act_id >= data.ctrl.shape[0]:
        return 0.0
    return _normalize01(float(data.ctrl[act_id]))


def _lerp_rgba(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = _normalize01(t)
    return (1.0 - t) * a + t * b


def _set_default_camera(model: mujoco.MjModel, cam: mujoco.MjvCamera) -> None:
    # Use model statistics if available to frame the whole model nicely.
    center = np.array(model.stat.center, dtype=np.float64)
    extent = float(model.stat.extent)

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = 2.2 * extent if extent > 1e-6 else 2.0
    cam.azimuth = 90.0
    cam.elevation = -20.0


def _add_capsule(scene: mujoco.MjvScene, geom_id: int, p0: np.ndarray, p1: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    """
    Uses mjv_initGeom + mjv_connector to create a capsule between p0 and p1.
    mjv_connector assumes mjv_initGeom has set other properties like rgba. :contentReference[oaicite:3]{index=3}
    """
    mujoco.mjv_initGeom(
        scene.geoms[geom_id],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[geom_id],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        width=float(radius),
        from_=p0.astype(np.float64),
        to=p1.astype(np.float64),
    )


# -----------------------------
# Offscreen render loop
# -----------------------------
def render_offscreen_gif(
    mjcf_path: str,
    out_path: str,
    seconds: float,
    fps: int,
    width: int,
    height: int,
    keywords: Sequence[str],
    viz: MuscleVizSpec,
) -> None:
    if iio is None:
        raise RuntimeError("imageio 未安装：请先 `pip install imageio`（或 imageio[ffmpeg] 生成 mp4）。")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # Load keyframe 0 as recommended for MyoConverter outputs. :contentReference[oaicite:4]{index=4}
    try:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
    except Exception:
        # If no keyframe exists, continue.
        mujoco.mj_forward(model, data)

    # Pick thigh-related actuators and map to tendons
    act_ids = _pick_actuators_by_keywords(model, keywords)
    act_to_tendon = {a: _actuator_to_tendon_id(model, a) for a in act_ids}
    act_to_tendon = {a: t for a, t in act_to_tendon.items() if t is not None}

    if not act_to_tendon:
        print("[WARN] 没找到匹配关键词且使用 tendon transmission 的 actuator；将只渲染骨架/身体，不画肌肉胶囊。")

    # Create OpenGL context for offscreen rendering. :contentReference[oaicite:5]{index=5}
    gl_ctx = mujoco.GLContext(width, height)
    gl_ctx.make_current()

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=viz.max_capsules)  # allocate lots of geom slots
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    viewport = mujoco.MjrRect(0, 0, width, height)

    _set_default_camera(model, cam)

    # Make sure we have an offscreen buffer sized correctly
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
    mujoco.mjr_resizeOffscreen(width, height, con)

    nframes = int(math.ceil(seconds * fps))
    dt = 1.0 / fps

    frames: List[np.ndarray] = []

    # simple “demo” control signal so you can see bulging: sine-wave activations
    phases = np.linspace(0.0, 2.0 * math.pi, num=max(1, len(act_to_tendon)), endpoint=False)

    for k in range(nframes):
        t = k * dt

        # drive the selected thigh muscles a bit (0..1)
        if model.nu > 0:
            data.ctrl[:] = 0.0
            for j, (act_id, _) in enumerate(act_to_tendon.items()):
                data.ctrl[act_id] = 0.5 + 0.5 * math.sin(2.0 * math.pi * 0.8 * t + float(phases[j]))

        mujoco.mj_step(model, data)

        # Update base scene (standard pipeline). :contentReference[oaicite:6]{index=6}
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

        # Append custom "muscle" capsules after the model geoms
        geom_id = int(scn.ngeom)
        for act_id, tendon_id in act_to_tendon.items():
            pts = _tendon_path_sites_world(model, data, tendon_id)
            if pts is None:
                continue

            act = _activation_from_ctrl(data, act_id)
            radius = float(viz.base_radius * (1.0 + viz.bulge_gain * act))
            rgba = _lerp_rgba(RGBA_REST, RGBA_ACTIVE, act)

            for p0, p1 in zip(pts[:-1], pts[1:]):
                if geom_id >= scn.maxgeom:
                    break
                # skip tiny segments
                if float(np.linalg.norm(p1 - p0)) < 1e-4:
                    continue
                _add_capsule(scn, geom_id, p0, p1, radius, rgba)
                geom_id += 1

        scn.ngeom = geom_id

        mujoco.mjr_render(viewport, scn, con)

        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, viewport, con)  # :contentReference[oaicite:7]{index=7}
        rgb = np.flipud(rgb)  # OpenGL origin is bottom-left
        frames.append(rgb)

    iio.imwrite(out_path, np.stack(frames, axis=0), fps=fps)
    gl_ctx.free()

    print(f"[OK] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", required=True, help="Path to MJCF xml (e.g. models/mjc/Leg6Dof9Musc/leg6dof9musc_cvt3.xml)")
    parser.add_argument("--out", default="thigh_muscles.gif", help="Output GIF path")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--base-radius", type=float, default=0.010)
    parser.add_argument("--bulge-gain", type=float, default=1.25)
    parser.add_argument("--keyword", action="append", default=None,
                        help="Add a muscle name keyword to match actuators (can repeat). If not set, uses a thigh keyword list.")
    parser.add_argument("--maxgeom", type=int, default=4000, help="MjvScene maxgeom budget (increase if many muscles)")

    args = parser.parse_args()

    keywords = args.keyword if args.keyword else list(DEFAULT_THIGH_KEYWORDS)
    viz = MuscleVizSpec(base_radius=args.base_radius, bulge_gain=args.bulge_gain, max_capsules=args.maxgeom)

    render_offscreen_gif(
        mjcf_path=args.mjcf,
        out_path=args.out,
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        keywords=keywords,
        viz=viz,
    )


if __name__ == "__main__":
    main()
