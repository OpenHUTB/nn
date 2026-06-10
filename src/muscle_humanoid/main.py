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

    r1_slide_x = model.joint("r1_slide_x").qposadr.item()
    r1_slide_y = model.joint("r1_slide_y").qposadr.item()
    r1_arm_l = model.joint("r1_j_larm").qposadr.item()
    r1_arm_r = model.joint("r1_j_rarm").qposadr.item()
    r1_hip_l = model.joint("r1_left_hip").qposadr.item()
    r1_hip_r = model.joint("r1_right_hip").qposadr.item()
    r1_knee_l = model.joint("r1_left_knee").qposadr.item()
    r1_knee_r = model.joint("r1_right_knee").qposadr.item()

    r2_slide_x = model.joint("r2_slide_x").qposadr.item()
    r2_slide_y = model.joint("r2_slide_y").qposadr.item()
    r2_arm_l = model.joint("r2_j_larm").qposadr.item()
    r2_arm_r = model.joint("r2_j_rarm").qposadr.item()
    r2_hip_l = model.joint("r2_left_hip").qposadr.item()
    r2_hip_r = model.joint("r2_right_hip").qposadr.item()
    r2_knee_l = model.joint("r2_left_knee").qposadr.item()
    r2_knee_r = model.joint("r2_right_knee").qposadr.item()

    STAND_KNEE = 0.0
    MOVE_LIMIT = 0.85
    MOVE_SPEED = 0.0001
    DANGER_Z = 0.35
    DETECT_RANGE = 0.7
    ROBOT_SAFE_DIST = 0.42
    ESCAPE_SPEED = 0.00035
    BALL_INTERVAL = 3.5
    CUBE_INTERVAL = 4.2

    # 分区巡逻点位：R1左区、R2右区
    patrol1 = [[-0.75, -0.3], [-0.3, -0.3], [-0.3, 0.3], [-0.75, 0.3]]
    patrol2 = [[0.3, -0.3], [0.75, -0.3], [0.75, 0.3], [0.3, 0.3]]
    idx1, idx2 = 0, 0
    r1_x, r1_y = -0.5, 0
    r2_x, r2_y = 0.5, 0

    last_ball = time.time()
    last_cube = time.time()
    swing_t = 0.0
    data.qpos[r1_slide_x] = r1_x
    data.qpos[r1_slide_y] = r1_y
    data.qpos[r2_slide_x] = r2_x
    data.qpos[r2_slide_y] = r2_y
    data.qpos[r1_knee_l] = STAND_KNEE
    data.qpos[r1_knee_r] = STAND_KNEE
    data.qpos[r2_knee_l] = STAND_KNEE
    data.qpos[r2_knee_r] = STAND_KNEE
    data.qvel[:] = 0

    v = viewer.launch_passive(model, data)
    v.cam.distance = 7.2
    v.cam.elevation = -22
    v.cam.lookat[:] = [0, 0, 0.4]
    swing_k = 0.0001
    arm_amp = 0.45
    leg_amp = 0.11

    while v.is_running():
        swing_t += 1
        # 高空障碍刷新
        if time.time() - last_ball > BALL_INTERVAL:
            last_ball = time.time()
            for i in range(3):
                jid = model.joint(i).qposadr.item()
                data.qpos[jid:jid+3] = [random.uniform(-0.7,0.7),random.uniform(-0.7,0.7),4.2]
                data.qvel[jid:jid+3]=0
        if time.time() - last_cube > CUBE_INTERVAL:
            last_cube = time.time()
            cid = model.joint(3).qposadr.item()
            data.qpos[cid:cid+3]=[random.uniform(-0.7,0.7),random.uniform(-0.7,0.7),4.2]
            data.qvel[cid:cid+3]=0

        danger_pos = []
        for i in range(4):
            px,py,pz = data.xpos[model.body(i+1).id]
            if pz>DANGER_Z: danger_pos.append([px,py])
        dist_robot = np.hypot(r2_x-r1_x,r2_y-r1_y)

        # R1左区巡逻
        t1x,t1y = patrol1[idx1]
        dx1,dy1 = np.sign(t1x-r1_x)*MOVE_SPEED, np.sign(t1y-r1_y)*MOVE_SPEED
        danger1=False
        for ox,oy in danger_pos:
            if np.hypot(ox-r1_x,oy-r1_y)<DETECT_RANGE:
                dx1=-np.sign(ox-r1_x)*MOVE_SPEED
                dy1=-np.sign(oy-r1_y)*MOVE_SPEED
                danger1=True;break
        if dist_robot<ROBOT_SAFE_DIST:
            dx1=0;dy1=ESCAPE_SPEED;danger1=True
        if abs(r1_x-t1x)<0.05 and abs(r1_y-t1y)<0.05:
            idx1=(idx1+1)%4
        r1_x += dx1; r1_y += dy1
        r1_x = np.clip(r1_x,-MOVE_LIMIT,MOVE_LIMIT)
        r1_y = np.clip(r1_y,-MOVE_LIMIT,MOVE_LIMIT)

        # R2右区巡逻
        t2x,t2y = patrol2[idx2]
        dx2,dy2 = np.sign(t2x-r2_x)*MOVE_SPEED, np.sign(t2y-r2_y)*MOVE_SPEED
        danger2=False
        for ox,oy in danger_pos:
            if np.hypot(ox-r2_x,oy-r2_y)<DETECT_RANGE:
                dx2=-np.sign(ox-r2_x)*MOVE_SPEED
                dy2=-np.sign(oy-r2_y)*MOVE_SPEED
                danger2=True;break
        if dist_robot<ROBOT_SAFE_DIST:
            dx2=0;dy2=-ESCAPE_SPEED;danger2=True
        if abs(r2_x-t2x)<0.05 and abs(r2_y-t2y)<0.05:
            idx2=(idx2+1)%4
        r2_x += dx2; r2_y += dy2
        r2_x = np.clip(r2_x,-MOVE_LIMIT,MOVE_LIMIT)
        r2_y = np.clip(r2_y,-MOVE_LIMIT,MOVE_LIMIT)

        data.qpos[r1_slide_x]=r1_x;data.qpos[r1_slide_y]=r1_y
        data.qpos[r2_slide_x]=r2_x;data.qpos[r2_slide_y]=r2_y
        data.qpos[r1_knee_l]=data.qpos[r1_knee_r]=STAND_KNEE
        data.qpos[r2_knee_l]=data.qpos[r2_knee_r]=STAND_KNEE

        s = np.sin(swing_t*swing_k)
        if not danger1:
            data.qpos[r1_arm_l]=arm_amp*s;data.qpos[r1_arm_r]=-arm_amp*s
            data.qpos[r1_hip_l]=leg_amp*s;data.qpos[r1_hip_r]=-leg_amp*s
        else:
            data.qpos[r1_arm_l:r1_arm_r+1]*=0.92
            data.qpos[r1_hip_l:r1_hip_r+1]*=0.92
        if not danger2:
            data.qpos[r2_arm_l]=arm_amp*s;data.qpos[r2_arm_r]=-arm_amp*s
            data.qpos[r2_hip_l]=leg_amp*s;data.qpos[r2_hip_r]=-leg_amp*s
        else:
            data.qpos[r2_arm_l:r2_arm_r+1]*=0.92
            data.qpos[r2_hip_l:r2_hip_r+1]*=0.92

        mujoco.mj_step(model,data)
        v.sync()

if __name__ == "__main__":
    main()