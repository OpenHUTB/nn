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
    lk = model.joint("left_knee").qposadr.item()
    rk = model.joint("right_knee").qposadr.item()

    STAND_KNEE = 0.0
    MOVE_LIMIT = 0.85      # 边界放大，支持更远巡逻
    MOVE_SPEED = 0.0001      # 速度放慢（原来0.006→0.0001，走路变慢）
    DANGER_Z = 0.35
    DETECT_RANGE = 0.7
    BALL_INTERVAL = 3.0

    # 巡逻点位拉远：-0.8 ~ +0.8，巡逻路程变长
    pointA = -0.8
    pointB = 0.8
    target_point = pointB
    pos_x, pos_y = 0.0, 0.0
    last_ball = time.time()

    data.qpos[slide_x] = pos_x
    data.qpos[slide_y] = pos_y
    data.qpos[lk] = STAND_KNEE
    data.qpos[rk] = STAND_KNEE
    data.qvel[:] = 0

    v = viewer.launch_passive(model, data)
    v.cam.distance = 5.8
    v.cam.elevation = -18
    v.cam.lookat[:] = [0, 0, 0.6]
    print("远距离慢速巡逻+遇球躲避")

    while v.is_running():
        # 定时生成小球
        if time.time() - last_ball > BALL_INTERVAL:
            last_ball = time.time()
            idx = random.randint(0, 2)
            jid = model.joint(idx).qposadr.item()
            data.qpos[jid] = random.uniform(-0.42, 0.42)
            data.qpos[jid+1] = random.uniform(-0.42, 0.42)
            data.qpos[jid+2] = 4.0
            data.qvel[jid:jid+3] = 0

        dx, dy = 0, 0
        danger_flag = False

        # 检测危险落球
        for i in range(3):
            bx, by, bz = data.xpos[model.body(i+1).id]
            dist = np.hypot(bx-pos_x, by-pos_y)
            if bz > DANGER_Z and dist < DETECT_RANGE:
                dx = -np.sign(bx-pos_x) * MOVE_SPEED
                dy = -np.sign(by-pos_y) * MOVE_SPEED
                danger_flag = True
                break

        # 无危险继续往返巡逻
        if not danger_flag:
            dx = np.sign(target_point - pos_x) * MOVE_SPEED
            dy = 0
            if abs(pos_x - target_point) < 0.03:
                target_point = pointA if target_point == pointB else pointB

        # 边界锁死不出画面
        pos_x += dx
        pos_y += dy
        pos_x = np.clip(pos_x, -MOVE_LIMIT, MOVE_LIMIT)
        pos_y = np.clip(pos_y, -MOVE_LIMIT, MOVE_LIMIT)

        data.qpos[slide_x] = pos_x
        data.qpos[slide_y] = pos_y
        data.qpos[lk] = STAND_KNEE
        data.qpos[rk] = STAND_KNEE

        mujoco.mj_step(model, data)
        v.sync()

if __name__ == "__main__":
    main()