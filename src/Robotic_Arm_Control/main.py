#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import signal
import threading
import numpy as np
import mujoco

RUNNING = True

def sig_handle(*args):
    global RUNNING
    RUNNING = False

signal.signal(signal.SIGINT, sig_handle)

class ArmController:
    def __init__(self):
        xml = """
<mujoco>
    <option timestep="0.0005" gravity="0 0 -9.81"/>
    <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1"/>
        <body name="base" pos="0 0 0">
            <geom type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
            <joint name="joint1" axis="0 0 1" pos="0 0 0.1" range="-3.14 3.14"/>
            <body name="link1" pos="0 0 0.1">
                <geom type="cylinder" size="0.04 0.18" rgba="0.2 0.8 0.2 1"/>
                <joint name="joint2" axis="0 1 0" pos="0 0 0.18" range="-1.57 1.57"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint1"/>
        <motor name="motor2" joint="joint2"/>
    </actuator>
</mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.target = np.zeros(2)

    def control(self):
        q = self.data.qpos[:2]
        err = self.target - q
        tau = 100 * err - 10 * self.data.qvel[:2]
        self.data.ctrl[:] = np.clip(tau, -5, 5)

    def run(self):
        global RUNNING
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        while RUNNING and self.viewer.is_running():
            self.control()
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        self.viewer.close()

def demo(arm):
    time.sleep(1)
    arm.target = np.array([0.5, 0.3])
    time.sleep(3)
    arm.target = np.array([0, 0])

if __name__ == "__main__":
    arm = ArmController()
    threading.Thread(target=demo, args=(arm,), daemon=True).start()
    arm.run()