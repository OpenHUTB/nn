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

    # 关节索引
    slide_x = model.joint("slide_x").qposadr.item()
    lk = model.joint("left_knee").qposadr.item()
    rk = model.joint("right_knee").qposadr.item()

    # 参数：膝盖角度固定为0，强制站直
    STAND_KNEE = 0.0
    MOVE_LIMIT = 0.35
    MOVE_SPEED = 0.007
    DETECT_RANGE = 0.7
    BALL_INTERVAL = 2.2

    last_ball = time.time()
    current_pos = 0.0

    # 初始姿态：强制膝盖伸直，机器人站直
    data.qpos[slide_x] = 0.0
    data.qpos[lk] = STAND_KNEE
    data.qpos[rk] = STAND_KNEE
    data.qvel[:] = 0.0

    # 固定相机视角，聚焦机器人
    v = viewer.launch_passive(model, data)
    v.cam.distance = 5.8
    v.cam.elevation = -18
    v.cam.lookat[:] = [0, 0, 0.6]

    print("运行：机器人强制站直，左右移动躲避小球，不出画面、不倒地")

    while v.is_running():
        # 定时生成小球（仅在头顶小范围生成）
        if time.time() - last_ball > BALL_INTERVAL:
            last_ball = time.time()
            idx = random.randint(0, 2)
            jid = model.joint(idx).qposadr.item()
            data.qpos[jid]     = random.uniform(-0.4, 0.4)
            data.qpos[jid+1]   = 0.0
            data.qpos[jid+2]   = 4.0
            data.qvel[jid:jid+3] = 0.0

        # 检测小球位置，决定躲避方向
        target_x = 0.0
        for i in range(3):
            bx, by, bz = data.xpos[model.body(i+1).id]
            if bz > 0.3 and abs(bx) < DETECT_RANGE:
                target_x = -np.sign(bx) * MOVE_SPEED
                break

        # 更新位置，强制边界，不出画面
        current_pos += target_x
        current_pos = np.clip(current_pos, -MOVE_LIMIT, MOVE_LIMIT)
        data.qpos[slide_x] = current_pos

        # 强制膝盖保持伸直，机器人全程站直
        data.qpos[lk] = STAND_KNEE
        data.qpos[rk] = STAND_KNEE

        mujoco.mj_step(model, data)
        v.sync()

if __name__ == "__main__":
    main()