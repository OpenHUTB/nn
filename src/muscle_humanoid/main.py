import mujoco
import mujoco.viewer as viewer
import numpy as np
import os
import random
import time

def main():
    xml_path = os.path.join(os.path.dirname(__file__), "humanoid.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    slide_x = model.joint("slide_x").qposadr.item()
    slide_y = model.joint("slide_y").qposadr.item()
    arm_l = model.joint("j_larm").qposadr.item()
    arm_r = model.joint("j_rarm").qposadr.item()
    hip_l = model.joint("left_hip").qposadr.item()
    hip_r = model.joint("right_hip").qposadr.item()
    knee_l = model.joint("left_knee").qposadr.item()
    knee_r = model.joint("right_knee").qposadr.item()

    STAND_KNEE = 0.0
    MOVE_LIMIT = 0.85
    MOVE_SPEED = 0.0001
    DANGER_Z = 0.35
    DETECT_RANGE = 0.7
    BALL_INTERVAL = 3.5
    CUBE_INTERVAL = 4.2

    patrol_points = [
        [-0.8, -0.3],
        [0.8, -0.3],
        [0.8, 0.3],
        [-0.8, 0.3]
    ]
    patrol_idx = 0
    pos_x, pos_y = 0.0, 0.0
    last_ball = time.time()
    last_cube = time.time()
    swing_t = 0.0

    data.qpos[slide_x] = pos_x
    data.qpos[slide_y] = pos_y
    data.qpos[arm_l] = 0.0
    data.qpos[arm_r] = 0.0
    data.qpos[hip_l] = 0.0
    data.qpos[hip_r] = 0.0
    data.qpos[knee_l] = STAND_KNEE
    data.qpos[knee_r] = STAND_KNEE
    data.qvel[:] = 0

    v = viewer.launch_passive(model, data)
    v.cam.distance = 6.2
    v.cam.elevation = -18
    v.cam.lookat[:] = [0, 0, 0.6]

    while v.is_running():
        swing_t += 1

        if time.time() - last_ball > BALL_INTERVAL:
            last_ball = time.time()
            idx = random.randint(0,2)
            jid = model.joint(idx).qposadr.item()
            data.qpos[jid]     = random.uniform(-0.75,0.75)
            data.qpos[jid+1]   = random.uniform(-0.75,0.75)
            data.qpos[jid+2]   = 4.2
            data.qvel[jid:jid+3] = 0

        if time.time() - last_cube > CUBE_INTERVAL:
            last_cube = time.time()
            cube_jid = model.joint(3).qposadr.item()
            data.qpos[cube_jid]     = random.uniform(-0.75,0.75)
            data.qpos[cube_jid+1]   = random.uniform(-0.75,0.75)
            data.qpos[cube_jid+2]   = 4.2
            data.qvel[cube_jid:cube_jid+3] = 0

        dx, dy = 0, 0
        danger_flag = False

        for i in range(3):
            bx, by, bz = data.xpos[model.body(i+1).id]
            dist = np.hypot(bx-pos_x, by-pos_y)
            if bz > DANGER_Z and dist < DETECT_RANGE:
                dx = -np.sign(bx-pos_x)*MOVE_SPEED
                dy = -np.sign(by-pos_y)*MOVE_SPEED
                danger_flag = True
                break
        if not danger_flag:
            cx, cy, cz = data.xpos[model.body(4).id]
            dist_cube = np.hypot(cx-pos_x, cy-pos_y)
            if cz > DANGER_Z and dist_cube < DETECT_RANGE:
                dx = -np.sign(cx-pos_x)*MOVE_SPEED
                dy = -np.sign(cy-pos_y)*MOVE_SPEED
                danger_flag = True

        if not danger_flag:
            tx, ty = patrol_points[patrol_idx]
            dx = np.sign(tx-pos_x)*MOVE_SPEED
            dy = np.sign(ty-pos_y)*MOVE_SPEED
            if abs(pos_x-tx) < 0.04 and abs(pos_y-ty) <0.04:
                patrol_idx = (patrol_idx+1) % 4

        pos_x += dx
        pos_y += dy
        pos_x = np.clip(pos_x, -MOVE_LIMIT, MOVE_LIMIT)
        pos_y = np.clip(pos_y, -MOVE_LIMIT, MOVE_LIMIT)

        data.qpos[slide_x] = pos_x
        data.qpos[slide_y] = pos_y

        swing_k = 0.0001
        arm_amp = 0.45
        leg_amp = 0.11

        data.qpos[knee_l] = STAND_KNEE
        data.qpos[knee_r] = STAND_KNEE

        if not danger_flag:
            s = np.sin(swing_t * swing_k)
            data.qpos[arm_l] = arm_amp * s
            data.qpos[arm_r] = -arm_amp * s
            data.qpos[hip_l] = leg_amp * s
            data.qpos[hip_r] = -leg_amp * s
        else:
            data.qpos[arm_l] *= 0.92
            data.qpos[arm_r] *= 0.92
            data.qpos[hip_l] *= 0.92
            data.qpos[hip_r] *= 0.92

        mujoco.mj_step(model, data)
        v.sync()

if __name__ == "__main__":
    main()