#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render a connected (welded) leg + neck composite MJCF.

What this script does:
1) Load leg.xml and neck.xml (original files).
2) Reset each model to keyframe(0) if present (README-style init).
3) Export namespaced copies (leg_, neck_) WITHOUT <keyframe> to avoid nq mismatch when composing.
   Also converts all `file="..."` paths to absolute paths, so assets still load from anywhere.
4) Compute relpose for an equality/weld constraint from the keyframe poses.
5) Generate a scene MJCF that <include>s the two namespaced parts and adds the weld.
6) Load the scene, copy qpos/qvel/act from each part’s keyframe into the composite model, then render.

Usage:
  python scripts/render_leg_neck_connected.py \
    --leg /path/to/leg.xml \
    --neck /path/to/neck.xml \
    --out  /path/to/out.gif

Notes:
- If auto-detected root body is not what you want, pass --leg_root_body / --neck_root_body.
- Output scene + temp MJCFs are written under --workdir (default: ./_build_leg_neck).
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# imageio is optional unless you save gif/mp4
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import mujoco
except Exception as e:
    raise SystemExit(
        "Cannot import `mujoco`. Please install official MuJoCo python bindings, e.g.\n"
        "  pip install mujoco\n"
        f"Original error: {e}"
    )


# -------------------------
# XML helpers
# -------------------------

FILE_ATTR_WHITELIST = {"file"}  # attributes that should be treated as filesystem paths


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print indentation (in-place)."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _strip_keyframes(root: ET.Element) -> None:
    """Remove <keyframe> blocks entirely."""
    for kf in list(root.findall("keyframe")):
        root.remove(kf)


def _collect_rename_tokens(root: ET.Element) -> Dict[str, str]:
    """
    Collect all tokens that should be namespaced:
    - any 'name'
    - any class-related tokens ('class', 'childclass') so defaults don't collide
    """
    tokens: set[str] = set()
    for el in root.iter():
        for k in ("name", "class", "childclass"):
            v = el.get(k)
            if v and v.strip():
                tokens.add(v.strip())
    return {t: t for t in tokens}


def _abspath_if_relative(path: str, base_dir: str) -> str:
    # ignore already-absolute or URLs or mujoco-style placeholders
    if not path:
        return path
    if path.startswith(("http://", "https://", "file://")):
        return path
    if path.startswith("${"):  # environment var style
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def namespace_mjcf(
    in_xml: str,
    out_xml: str,
    prefix: str,
    remove_keyframe: bool = True,
    absolutize_files: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """
    Create a namespaced copy of MJCF:
    - prefixes 'name' / 'class' / 'childclass' values
    - replaces references by token matching (whitespace-separated)
    - optionally removes <keyframe>
    - optionally converts all file="..." to absolute paths based on in_xml directory

    Returns:
      (root_body_name_original, rename_map_old_to_new)
    """
    base_dir = os.path.dirname(os.path.abspath(in_xml))
    tree = ET.parse(in_xml)
    root = tree.getroot()

    if remove_keyframe:
        _strip_keyframes(root)

    # Build rename map
    tokens = _collect_rename_tokens(root)
    rename_map: Dict[str, str] = {old: f"{prefix}{old}" for old in tokens.keys()}

    # Apply rename on attributes
    for el in root.iter():
        # absolutize file paths
        if absolutize_files:
            for attr, val in list(el.attrib.items()):
                if attr in FILE_ATTR_WHITELIST and val:
                    el.set(attr, _abspath_if_relative(val, base_dir))

        # rename primary naming attrs
        for k in ("name", "class", "childclass"):
            v = el.get(k)
            if v and v in rename_map:
                el.set(k, rename_map[v])

        # rename references by token substitution
        for attr, val in list(el.attrib.items()):
            # skip file paths and root model name
            if attr in FILE_ATTR_WHITELIST:
                continue
            if el is root and attr == "model":
                continue

            if not val or not val.strip():
                continue

            parts = val.split()
            changed = False
            for i, tok in enumerate(parts):
                if tok in rename_map:
                    parts[i] = rename_map[tok]
                    changed = True
            if changed:
                el.set(attr, " ".join(parts))

    _indent(root)
    os.makedirs(os.path.dirname(os.path.abspath(out_xml)), exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    return out_xml, rename_map


def detect_root_body_name(xml_path: str) -> str:
    """
    Heuristic: pick the first <body> directly under <worldbody>.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> found in: {xml_path}")

    bodies = worldbody.findall("body")
    if not bodies:
        raise ValueError(f"No <body> directly under <worldbody> in: {xml_path}")

    # prefer named body
    for b in bodies:
        nm = b.get("name")
        if nm and nm.strip():
            return nm.strip()
    # fallback
    return bodies[0].get("name", "").strip() or "unnamed_root_body"


# -------------------------
# MuJoCo state helpers
# -------------------------

@dataclass
class ModelState:
    joint_qpos: Dict[str, np.ndarray]
    joint_qvel: Dict[str, np.ndarray]
    actuator_act: Dict[str, np.ndarray]
    body_pose_world: Dict[str, Tuple[np.ndarray, np.ndarray]]  # name -> (pos(3), quat(4))


def _joint_qpos_len(jnt_type: int) -> int:
    # mjtJoint: FREE=0, BALL=1, SLIDE=2, HINGE=3 (MuJoCo convention)
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def _joint_dof_len(jnt_type: int) -> int:
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1


def load_and_reset(model_xml: str) -> Tuple["mujoco.MjModel", "mujoco.MjData"]:
    m = mujoco.MjModel.from_xml_path(model_xml)
    d = mujoco.MjData(m)

    if m.nkey > 0:
        # README-style: reset to keyframe 0
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
    else:
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.qpos0
        mujoco.mj_forward(m, d)
    return m, d


def extract_state(m: "mujoco.MjModel", d: "mujoco.MjData") -> ModelState:
    joint_qpos: Dict[str, np.ndarray] = {}
    joint_qvel: Dict[str, np.ndarray] = {}

    # joints
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not jname:
            continue
        jtype = int(m.jnt_type[jid])
        qpos_adr = int(m.jnt_qposadr[jid])
        dof_adr = int(m.jnt_dofadr[jid])

        qpos_len = _joint_qpos_len(jtype)
        dof_len = _joint_dof_len(jtype)

        joint_qpos[jname] = np.array(d.qpos[qpos_adr:qpos_adr + qpos_len], dtype=float)
        joint_qvel[jname] = np.array(d.qvel[dof_adr:dof_adr + dof_len], dtype=float)

    # actuator activations (if any)
    actuator_act: Dict[str, np.ndarray] = {}
    for aid in range(m.na):
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if not aname:
            continue
        actadr = int(m.actuator_actadr[aid])
        actnum = int(m.actuator_actnum[aid])
        if actnum > 0:
            actuator_act[aname] = np.array(d.act[actadr:actadr + actnum], dtype=float)

    # body poses
    body_pose_world: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for bid in range(m.nbody):
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not bname:
            continue
        pos = np.array(d.xpos[bid], dtype=float)
        quat = np.array(d.xquat[bid], dtype=float)  # w x y z
        body_pose_world[bname] = (pos, quat)

    return ModelState(
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        actuator_act=actuator_act,
        body_pose_world=body_pose_world,
    )


# -------------------------
# Quaternion math
# -------------------------

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product. q = [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_norm(q: np.ndarray) -> float:
    return float(np.linalg.norm(q))


def quat_inv(q: np.ndarray) -> np.ndarray:
    n2 = float(np.dot(q, q))
    if n2 <= 0:
        return np.array([1, 0, 0, 0], dtype=float)
    return quat_conj(q) / n2


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) to rotation matrix."""
    w, x, y, z = q
    # normalize
    n = quat_norm(q)
    if n == 0:
        return np.eye(3)
    w, x, y, z = (q / n).tolist()

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def compute_relpose(body1_pos, body1_quat, body2_pos, body2_quat) -> Tuple[np.ndarray, np.ndarray]:
    """
    relpose is pose of body2 in body1 frame:
      p_rel = R1^T (p2 - p1)
      q_rel = inv(q1) * q2
    """
    R1 = quat_to_mat(body1_quat)
    p_rel = R1.T @ (body2_pos - body1_pos)
    q_rel = quat_mul(quat_inv(body1_quat), body2_quat)

    # normalize quaternion
    n = quat_norm(q_rel)
    if n > 0:
        q_rel = q_rel / n
    else:
        q_rel = np.array([1, 0, 0, 0], dtype=float)

    return p_rel, q_rel


# -------------------------
# Composite scene generation
# -------------------------

def write_scene_xml(
    scene_xml: str,
    leg_part_xml: str,
    neck_part_xml: str,
    leg_root_body: str,
    neck_root_body: str,
    relpose_pos: np.ndarray,
    relpose_quat: np.ndarray,
) -> None:
    """
    Create a scene MJCF which includes two parts and adds an equality/weld between their root bodies.
    """
    os.makedirs(os.path.dirname(os.path.abspath(scene_xml)), exist_ok=True)

    rp = list(relpose_pos.flatten()) + list(relpose_quat.flatten())
    rp_str = " ".join(f"{x:.8g}" for x in rp)

    # Keep this scene minimal; you can add cameras/lights as needed.
    mjcf = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="leg_neck_connected_scene">
  <include file="{os.path.abspath(leg_part_xml)}"/>
  <include file="{os.path.abspath(neck_part_xml)}"/>

  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.92 0.92 0.92 1"/>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="cam" mode="trackcom" pos="2.5 0 1.5" xyaxes="0 1 0 -0.35 0 0.94"/>
  </worldbody>

  <equality>
    <weld name="leg_neck_weld"
          body1="{leg_root_body}"
          body2="{neck_root_body}"
          relpose="{rp_str}"/>
  </equality>
</mujoco>
"""
    with open(scene_xml, "w", encoding="utf-8") as f:
        f.write(mjcf)


def apply_state_to_composite(
    composite_model: "mujoco.MjModel",
    composite_data: "mujoco.MjData",
    src_state: ModelState,
    prefix: str,
) -> None:
    """
    Copy qpos/qvel/act by name from a source model state into the composite model,
    assuming composite names are prefixed.
    """
    # joints
    for jname, qpos in src_state.joint_qpos.items():
        tgt_name = prefix + jname
        jid = mujoco.mj_name2id(composite_model, mujoco.mjtObj.mjOBJ_JOINT, tgt_name)
        if jid < 0:
            continue
        jtype = int(composite_model.jnt_type[jid])
        qpos_adr = int(composite_model.jnt_qposadr[jid])
        qpos_len = _joint_qpos_len(jtype)
        if len(qpos) == qpos_len:
            composite_data.qpos[qpos_adr:qpos_adr + qpos_len] = qpos

    for jname, qvel in src_state.joint_qvel.items():
        tgt_name = prefix + jname
        jid = mujoco.mj_name2id(composite_model, mujoco.mjtObj.mjOBJ_JOINT, tgt_name)
        if jid < 0:
            continue
        jtype = int(composite_model.jnt_type[jid])
        dof_adr = int(composite_model.jnt_dofadr[jid])
        dof_len = _joint_dof_len(jtype)
        if len(qvel) == dof_len:
            composite_data.qvel[dof_adr:dof_adr + dof_len] = qvel

    # actuator activation states
    for aname, act in src_state.actuator_act.items():
        tgt_name = prefix + aname
        aid = mujoco.mj_name2id(composite_model, mujoco.mjtObj.mjOBJ_ACTUATOR, tgt_name)
        if aid < 0:
            continue
        actadr = int(composite_model.actuator_actadr[aid])
        actnum = int(composite_model.actuator_actnum[aid])
        if actnum > 0 and len(act) == actnum:
            composite_data.act[actadr:actadr + actnum] = act


# -------------------------
# Rendering
# -------------------------

def render_gif(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    out_path: str,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    frames: int = 180,
    camera: str = "cam",
    settle_steps: int = 20,
) -> None:
    if imageio is None:
        raise SystemExit("imageio is not installed. Please `pip install imageio` to save GIF/MP4.")

    # Let constraints settle a bit
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, width=width, height=height)
    imgs = []

    for _ in range(frames):
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera)
        img = renderer.render()  # HxWx3 uint8
        imgs.append(img)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()

    if ext == ".gif":
        imageio.mimsave(out_path, imgs, fps=fps)
    elif ext in (".mp4", ".mkv", ".mov"):
        imageio.mimsave(out_path, imgs, fps=fps)  # imageio chooses writer by extension
    else:
        raise ValueError(f"Unsupported output format: {ext} (use .gif or .mp4)")


# -------------------------
# Main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, help="Path to leg MJCF (original).")
    ap.add_argument("--neck", required=True, help="Path to neck MJCF (original).")
    ap.add_argument("--out", required=True, help="Output .gif/.mp4 path.")
    ap.add_argument("--workdir", default="./_build_leg_neck", help="Directory for generated temp MJCFs and scene.")
    ap.add_argument("--leg_prefix", default="leg_", help="Namespace prefix for leg model.")
    ap.add_argument("--neck_prefix", default="neck_", help="Namespace prefix for neck model.")

    ap.add_argument("--leg_root_body", default="", help="Override: body name in leg model to weld from.")
    ap.add_argument("--neck_root_body", default="", help="Override: body name in neck model to weld to.")

    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--settle_steps", type=int, default=20)

    args = ap.parse_args()

    leg_xml = os.path.abspath(args.leg)
    neck_xml = os.path.abspath(args.neck)
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)

    # 1) Detect root bodies (in ORIGINAL names)
    leg_root = args.leg_root_body.strip() or detect_root_body_name(leg_xml)
    neck_root = args.neck_root_body.strip() or detect_root_body_name(neck_xml)

    # 2) Load originals and reset keyframe 0 (README-style)
    leg_m, leg_d = load_and_reset(leg_xml)
    neck_m, neck_d = load_and_reset(neck_xml)
    leg_state = extract_state(leg_m, leg_d)
    neck_state = extract_state(neck_m, neck_d)

    if leg_root not in leg_state.body_pose_world:
        raise SystemExit(f"Cannot find body '{leg_root}' in leg model after reset. Try --leg_root_body.")
    if neck_root not in neck_state.body_pose_world:
        raise SystemExit(f"Cannot find body '{neck_root}' in neck model after reset. Try --neck_root_body.")

    leg_pos, leg_quat = leg_state.body_pose_world[leg_root]
    neck_pos, neck_quat = neck_state.body_pose_world[neck_root]

    # 3) Compute relpose (neck body pose in leg body frame) from keyframe poses
    rel_p, rel_q = compute_relpose(leg_pos, leg_quat, neck_pos, neck_quat)

    # 4) Write namespaced, no-keyframe copies (assets file paths become absolute)
    leg_ns_xml = os.path.join(workdir, "leg_ns_nokey.xml")
    neck_ns_xml = os.path.join(workdir, "neck_ns_nokey.xml")

    _, leg_map = namespace_mjcf(leg_xml, leg_ns_xml, prefix=args.leg_prefix, remove_keyframe=True, absolutize_files=True)
    _, neck_map = namespace_mjcf(neck_xml, neck_ns_xml, prefix=args.neck_prefix, remove_keyframe=True, absolutize_files=True)

    leg_root_ns = args.leg_prefix + leg_root
    neck_root_ns = args.neck_prefix + neck_root

    # 5) Generate composite scene XML
    scene_xml = os.path.join(workdir, "leg_neck_connected_scene.xml")
    write_scene_xml(
        scene_xml=scene_xml,
        leg_part_xml=leg_ns_xml,
        neck_part_xml=neck_ns_xml,
        leg_root_body=leg_root_ns,
        neck_root_body=neck_root_ns,
        relpose_pos=rel_p,
        relpose_quat=rel_q,
    )

    # 6) Load composite and apply initial state by copying from original keyframes
    comp_m = mujoco.MjModel.from_xml_path(scene_xml)
    comp_d = mujoco.MjData(comp_m)

    mujoco.mj_resetData(comp_m, comp_d)
    apply_state_to_composite(comp_m, comp_d, leg_state, prefix=args.leg_prefix)
    apply_state_to_composite(comp_m, comp_d, neck_state, prefix=args.neck_prefix)
    mujoco.mj_forward(comp_m, comp_d)

    # 7) Render
    render_gif(
        model=comp_m,
        data=comp_d,
        out_path=os.path.abspath(args.out),
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=args.frames,
        camera="cam",
        settle_steps=args.settle_steps,
    )

    print(f"[OK] Scene: {scene_xml}")
    print(f"[OK] Output: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
