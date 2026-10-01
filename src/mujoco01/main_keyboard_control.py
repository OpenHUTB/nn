#!/usr/bin/env python3
"""MuJoCo 人形机器人键盘控制"""
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import pygame

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humanoid.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("=== Actuator 列表 ===")
actuator_map = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  [{i}] {name}")
    if name:
        actuator_map[name] = i
print("=====================")


def get_idx(keyword):
    for name, idx in actuator_map.items():
        if keyword in name:
            return idx
    return None


mujoco.mj_resetDataKeyframe(model, data, 0)
home_qpos = model.key_qpos[0].copy()

actuator_info = []
for i in range(model.nu):
    joint_id = model.actuator_trnid[i, 0]
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]
    actuator_info.append((i, qpos_adr, dof_adr))

pygame.init()
screen = pygame.display.set_mode((500, 280))
pygame.display.set_caption("MuJoCo Keyboard Control")
font = pygame.font.SysFont("monospace", 16)

KEY_BINDINGS = [
    (pygame.K_w, "abdomen_y", 1.0, "W/S : Torso Pitch"),
    (pygame.K_s, "abdomen_y", -1.0, ""),
    (pygame.K_a, "abdomen_z", 1.0, "A/D : Torso Yaw"),
    (pygame.K_d, "abdomen_z", -1.0, ""),
    (pygame.K_q, "shoulder1_right", 1.0, "Q/E : Right/Left Shoulder"),
    (pygame.K_e, "shoulder1_left", 1.0, ""),
    (pygame.K_r, "elbow_right", 1.0, "R/F : Right/Left Elbow"),
    (pygame.K_f, "elbow_left", 1.0, ""),
]

STEP = 0.05
MAX_CTRL = 1.0
KP = 40.0
KD = 4.0

keyboard_offset = np.zeros(model.nu)

print("操作说明：")
print("  点击 pygame 小窗口使其聚焦")
print("  W/S : 躯干前倾/后仰")
print("  A/D : 躯干左转/右转")
print("  Q/E : 抬右臂/抬左臂")
print("  R/F : 屈右肘/屈左肘")
print("  Space : 全部复位")
print("  ESC   : 退出")

running = True
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    keyboard_offset[:] = 0.0

        keys = pygame.key.get_pressed()
        for key, keyword, sign, _ in KEY_BINDINGS:
            idx = get_idx(keyword)
            if idx is None:
                continue
            if keys[key]:
                keyboard_offset[idx] = np.clip(keyboard_offset[idx] + sign * STEP, -MAX_CTRL, MAX_CTRL)
            else:
                keyboard_offset[idx] *= 0.95

        for i, qpos_adr, dof_adr in actuator_info:
            target = home_qpos[qpos_adr] + keyboard_offset[i]
            current = data.qpos[qpos_adr]
            velocity = data.qvel[dof_adr]
            data.ctrl[i] = KP * (target - current) - KD * velocity

        data.qpos[0:7] = home_qpos[0:7]
        data.qvel[0:6] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()

        screen.fill((30, 30, 30))
        y = 10
        for _, _, _, text in KEY_BINDINGS:
            if text:
                surf = font.render(text, True, (230, 230, 230))
                screen.blit(surf, (10, y))
                y += 24
        pygame.display.flip()
        time.sleep(0.002)

pygame.quit()
print("程序退出")
