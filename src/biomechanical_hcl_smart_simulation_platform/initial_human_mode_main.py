#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
initial_human_model_render.py
Render a MyoConverter MuJoCo human model (MJCF) with enhanced visuals.

Key points (from README):
- Converted XML contains a keyframe that should be loaded at init.
  Use mujoco.mj_resetDataKeyframe(model, data, 0). :contentReference[oaicite:1]{index=1}

Features:
- Patches MJCF into a "render-friendly" version (skybox/ground/lights/quality).
- Offscreen render to PNG/GIF/MP4.
- Optional: render leg/thigh muscle "volume" as semi-transparent capsules along tendon site paths.

Examples:
  # Still image
  python initial_human_model_render.py --xml path/to/model.xml --out render.png

  # Orbit GIF
  python initial_human_model_render.py --xml path/to/model.xml --out render.gif --seconds 4 --orbit

  # Add muscle volume (legs by default)
  python initial_human_model_render.py --xml path/to/model.xml --out render.gif --orbit --muscle-volume

Headless:
  export MUJOCO_GL=egl   # (or osmesa)
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import mujoco

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


# -------------------------
# MJCF patching (for realism)
# -------------------------

def _abspath_if_relative(p: str, base_dir: str) -> str:
    if not p:
        return p
    if p.startswith(("http://", "https://", "file://", "${")):
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def _indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for ch in elem:
            _indent(ch, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def patch_mjcf_for_render(in_xml: Path, out_xml: Path) -> None:
    """
    - Keeps the model intact (including keyframe).
    - Adds skybox, ground texture/material, floor geom, lights, and visual quality knobs.
    - Makes file=... paths absolute so writing to a new folder won't break assets.
    """
    base_dir = str(in_xml.parent.resolve())
    tree = ET.parse(str(in_xml))
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"Not an MJCF <mujoco> root: {in_xml}")

    # Absolutize all file="" references
    for el in root.iter():
        if "file" in el.attrib:
            el.set("file", _abspath_if_relative(el.get("file", ""), base_dir))

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Skybox texture (no external file)
    if asset.find("./texture[@type='skybox']") is None:
        ET.SubElement(asset, "texture", {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.72 0.86 1.0",
            "rgb2": "0.12 0.20 0.35",
            "width": "512",
            "height": "512",
        })

    # Ground checker texture + material
    if asset.find("./texture[@name='tex_ground']") is None:
        ET.SubElement(asset, "texture", {
            "name": "tex_ground",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.80 0.80 0.80",
            "rgb2": "0.63 0.63 0.63",
            "width": "512",
            "height": "512",
            "mark": "cross",
            "markrgb": "0.35 0.35 0.35",
        })

    if asset.find("./material[@name='mat_ground']") is None:
        ET.SubElement(asset, "material", {
            "name": "mat_ground",
            "texture": "tex_ground",
            "texrepeat": "6 6",
            "reflectance": "0.15",
            "shininess": "0.25",
            "specular": "0.35",
        })

    # Visual quality (shadows, AA, fog)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")

    if visual.find("quality") is None:
        ET.SubElement(visual, "quality", {
            "shadowsize": "4096",
            "offsamples": "4",
            "numslices": "28",
            "numstacks": "28",
        })

    if visual.find("map") is None:
        ET.SubElement(visual, "map", {
            "znear": "0.01",
            "zfar": "60",
            "fogstart": "8",
            "fogend": "25",
        })

    # Add a floor + lights into existing worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Floor (avoid duplicates)
    has_floor = any((g.tag == "geom" and g.get("name", "") == "floor") for g in worldbody.findall("geom"))
    if not has_floor:
        ET.SubElement(worldbody, "geom", {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 0",
            "size": "10 10 0.1",
            "material": "mat_ground",
            "rgba": "1 1 1 1",
        })

    # Simple 3-point lights (avoid duplicates by name)
    existing_light_names = {l.get("name", "") for l in worldbody.findall("light")}
    if "key_light" not in existing_light_names:
        ET.SubElement(worldbody, "light", {
            "name": "key_light",
            "pos": "2.5 -2.5 3.0",
            "dir": "-0.6 0.6 -0.7",
            "diffuse": "0.9 0.9 0.9",
            "specular": "0.4 0.4 0.4",
        })
    if "fill_light" not in existing_light_names:
        ET.SubElement(worldbody, "light", {
            "name": "fill_light",
            "pos": "-2.0 2.0 2.5",
            "dir": "0.5 -0.5 -0.7",
            "diffuse": "0.45 0.45 0.50",
            "specular": "0.15 0.15 0.15",
        })
    if "rim_light" not in existing_light_names:
        ET.SubElement(worldbody, "light", {
            "name": "rim_light",
            "pos": "0 3.5 2.8",
            "dir": "0 -1 -0.6",
            "diffuse": "0.35 0.35 0.40",
            "specular": "0.20 0.20 0.20",
        })

    _indent(root)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_xml), encoding="utf-8", xml_declaration=True)


# -------------------------
# Muscle volume (capsules along tendon site path)
# -------------------------

DEFAULT_LEG_KEYWORDS = (
    "rect", "vasti", "vastus", "ham", "biceps", "semiten", "semimem",
    "adductor", "sart", "grac", "tfl", "iliopsoas", "psoas", "iliacus",
    "glut", "soleus", "gastro", "tibialis", "perone", "fibular",
)

RGBA_REST = np.array([0.85, 0.15, 0.15, 0.40], dtype=np.float32)
RGBA_ACTIVE = np.array([1.00, 0.28, 0.28, 0.85], dtype=np.float32)


def _safe_name(model: mujoco.MjModel, objtype: mujoco.mjtObj, idx: int) -> str:
    nm = mujoco.mj_id2name(model, objtype, idx)
    return nm if nm else f"{objtype.name}_{idx}"


def _pick_actuators_by_keywords(model: mujoco.MjModel, keywords: Sequence[str]) -> List[int]:
    keys = tuple(k.lower() for k in keywords)
    picked = []
    for aid in range(model.nu):
        nm = _safe_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid).lower()
        if any(k in nm for k in keys):
            picked.append(aid)
    return picked


def _actuator_to_tendon_id(model: mujoco.MjModel, aid: int) -> Optional[int]:
    if not (hasattr(model, "actuator_trntype") and hasattr(model, "actuator_trnid")):
        return None
    if int(model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_TENDON):
        return None
    tid = int(model.actuator_trnid[aid, 0])
    if 0 <= tid < model.ntendon:
        return tid
    return None


def _tendon_sites_world(model: mujoco.MjModel, data: mujoco.MjData, tid: int) -> Optional[np.ndarray]:
    # Best-effort: take SITE wrap points only
    need = ("tendon_adr", "tendon_num", "wrap_type", "wrap_objid")
    if not all(hasattr(model, n) for n in need):
        return None
    adr = int(model.tendon_adr[tid])
    num = int(model.tendon_num[tid])
    if num <= 0:
        return None

    pts: List[np.ndarray] = []
    for w in range(adr, adr + num):
        if int(model.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_SITE):
            sid = int(model.wrap_objid[w])
            pts.append(np.array(data.site_xpos[sid], dtype=np.float64))
    if len(pts) >= 2:
        return np.stack(pts, axis=0)
    return None


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(max(0.0, min(1.0, t)))
    return (1.0 - t) * a + t * b


def _activation_from_ctrl(data: mujoco.MjData, aid: int) -> float:
    if data.ctrl is None or aid >= data.ctrl.shape[0]:
        return 0.0
    return float(max(0.0, min(1.0, float(data.ctrl[aid]))))


def _add_capsule(scene: mujoco.MjvScene, geom_id: int, p0: np.ndarray, p1: np.ndarray, radius: float, rgba: np.ndarray) -> None:
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


# -------------------------
# Rendering
# -------------------------

def _init_camera(model: mujoco.MjModel, azimuth: float, elevation: float, dist_scale: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    center = np.array(model.stat.center, dtype=np.float64)
    extent = float(model.stat.extent) if float(model.stat.extent) > 1e-9 else 1.0
    cam.lookat[:] = center
    cam.distance = max(0.05, dist_scale * extent)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    return cam


def _init_vis_options(show_tendons: bool) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    # Realism-ish flags
    opt.flags[mujoco.mjtVisFlag.mjVIS_SHADOW] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_REFLECTION] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_SKYBOX] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_FOG] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

    # Tendons (red lines) — can turn off when drawing muscle volume
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1 if show_tendons else 0
    return opt


def render(
    xml_path: Path,
    out_path: Path,
    seconds: float,
    fps: int,
    width: int,
    height: int,
    orbit: bool,
    azimuth: float,
    elevation: float,
    dist_scale: float,
    ctrl_mode: str,
    muscle_volume: bool,
    muscle_keywords: Sequence[str],
    base_radius: float,
    bulge_gain: float,
    maxgeom: int,
) -> None:
    if imageio is None and out_path.suffix.lower() in (".gif", ".mp4", ".mov", ".mkv"):
        raise RuntimeError("需要 imageio 导出 gif/mp4：请 `pip install imageio`。")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Load keyframe 0 (README recommendation). :contentReference[oaicite:2]{index=2}
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Offscreen GL context
    gl = mujoco.GLContext(width, height)
    gl.make_current()

    cam = _init_camera(model, azimuth=azimuth, elevation=elevation, dist_scale=dist_scale)
    opt = _init_vis_options(show_tendons=not muscle_volume)

    scn = mujoco.MjvScene(model, maxgeom=maxgeom)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    viewport = mujoco.MjrRect(0, 0, width, height)

    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
    mujoco.mjr_resizeOffscreen(width, height, con)

    # Precompute muscle->tendon mapping
    act_to_tendon: Dict[int, int] = {}
    if muscle_volume and model.nu > 0 and model.ntendon > 0:
        act_ids = _pick_actuators_by_keywords(model, muscle_keywords)
        for aid in act_ids:
            tid = _actuator_to_tendon_id(model, aid)
            if tid is not None:
                act_to_tendon[aid] = tid
        if not act_to_tendon:
            print("[WARN] muscle-volume=ON 但未找到 tendon transmission 的 actuator；将仅使用原始渲染。")

    # output
    ext = out_path.suffix.lower()
    nframes = 1 if ext == ".png" else max(1, int(round(seconds * fps)))

    frames: List[np.ndarray] = []

    phases = np.linspace(0.0, 2.0 * math.pi, num=max(1, len(act_to_tendon)), endpoint=False)

    for k in range(nframes):
        t = k / max(1, fps)

        # orbit camera
        if orbit:
            cam.azimuth = float(azimuth + 360.0 * (k / max(1, nframes)))

        # drive controls (optional, for visible "muscle bulge")
        if ctrl_mode != "none" and model.nu > 0:
            data.ctrl[:] = 0.0
            if ctrl_mode == "sine":
                # only drive the selected actuators (if muscle volume); otherwise drive nothing
                for j, aid in enumerate(act_to_tendon.keys()):
                    data.ctrl[aid] = 0.5 + 0.5 * math.sin(2.0 * math.pi * 0.8 * t + float(phases[j]))

        mujoco.mj_step(model, data)

        # update scene
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

        # add muscle capsules after base geoms
        if muscle_volume and act_to_tendon:
            base_ngeom = int(scn.ngeom)
            geom_id = base_ngeom

            for aid, tid in act_to_tendon.items():
                pts = _tendon_sites_world(model, data, tid)
                if pts is None:
                    continue

                act = _activation_from_ctrl(data, aid)
                radius = float(base_radius * (1.0 + bulge_gain * act))
                rgba = _lerp(RGBA_REST, RGBA_ACTIVE, act)

                for p0, p1 in zip(pts[:-1], pts[1:]):
                    if geom_id >= scn.maxgeom:
                        break
                    if float(np.linalg.norm(p1 - p0)) < 1e-4:
                        continue
                    _add_capsule(scn, geom_id, p0, p1, radius, rgba)
                    geom_id += 1

            scn.ngeom = geom_id

        mujoco.mjr_render(viewport, scn, con)

        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, viewport, con)
        rgb = np.flipud(rgb)  # OpenGL origin bottom-left
        frames.append(rgb)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".png":
        imageio.imwrite(str(out_path), frames[0])
    else:
        imageio.mimsave(str(out_path), frames, fps=fps)

    gl.free()
    print(f"[OK] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to MyoConverter MJCF xml (e.g. models/mjc/Gait2354Simbody/gait2354_cvt3.xml)")
    ap.add_argument("--out", default="human_render.gif", help="Output: .png/.gif/.mp4 (gif recommended here)")

    ap.add_argument("--workdir", default="./_render_build", help="Where to write patched XML")
    ap.add_argument("--no-patch", action="store_true", help="Do not patch MJCF (use original as-is)")

    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--orbit", action="store_true", help="Orbit camera around the model")
    ap.add_argument("--azimuth", type=float, default=95.0)
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--dist-scale", type=float, default=2.0)

    ap.add_argument("--ctrl-mode", choices=["none", "sine"], default="sine",
                    help="Control signal for visualizing muscle bulge (sine is just a demo)")

    ap.add_argument("--muscle-volume", action="store_true",
                    help="Draw semi-transparent muscle volume capsules along tendon site path (leg keywords by default).")
    ap.add_argument("--muscle-keyword", action="append", default=None,
                    help="Add keyword for selecting actuators (repeatable). If not set, uses a leg/thigh keyword set.")
    ap.add_argument("--base-radius", type=float, default=0.010, help="Base capsule radius (m)")
    ap.add_argument("--bulge-gain", type=float, default=1.25, help="Radius gain by activation")
    ap.add_argument("--maxgeom", type=int, default=20000, help="Scene maxgeom budget")

    args = ap.parse_args()

    src_xml = Path(args.xml).resolve()
    out_path = Path(args.out).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if args.no_patch:
        patched_xml = src_xml
    else:
        patched_xml = workdir / (src_xml.stem + "_render.xml")
        patch_mjcf_for_render(src_xml, patched_xml)

    keywords = args.muscle_keyword if args.muscle_keyword else list(DEFAULT_LEG_KEYWORDS)

    render(
        xml_path=patched_xml,
        out_path=out_path,
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        orbit=args.orbit,
        azimuth=args.azimuth,
        elevation=args.elevation,
        dist_scale=args.dist_scale,
        ctrl_mode=args.ctrl_mode,
        muscle_volume=args.muscle_volume,
        muscle_keywords=keywords,
        base_radius=args.base_radius,
        bulge_gain=args.bulge_gain,
        maxgeom=args.maxgeom,
    )


if __name__ == "__main__":
    main()
