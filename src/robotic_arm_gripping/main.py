import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    model = mujoco.MjModel.from_xml_path("arm_model.xml")
    data = mujoco.MjData(model)

    phase = 0
    phase_time = 0.0
    target = np.zeros(4)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 60
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat = [0.2, 0, 0.3]

        while viewer.is_running():
            phase_time += model.opt.timestep

            # 动作序列
            if phase == 0:
                # 准备姿势
                target = np.array([0.2, -0.6, 0.0, 0.0])
                if phase_time > 2.0:
                    phase = 1
                    phase_time = 0

            elif phase == 1:
                # 伸过去
                target = np.array([0.0, -1.2, 0.0, 0.0])
                if phase_time > 2.0:
                    phase = 2
                    phase_time = 0

            elif phase == 2:
                # 张开手指
                target = np.array([0.0, -1.2, 0.7, 0.7])
                if phase_time > 1.0:
                    phase = 3
                    phase_time = 0

            elif phase == 3:
                # 抓住
                target = np.array([0.0, -1.2, 0.1, 0.1])
                if phase_time > 1.0:
                    phase = 4
                    phase_time = 0

            elif phase == 4:
                # 举起
                target = np.array([-0.7, 0.6, 0.1, 0.1])
                if phase_time > 3.0:
                    phase = 5
                    phase_time = 0

            # 平滑控制
            data.ctrl[:] = 0.95 * data.ctrl + 0.05 * target

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.008)

if __name__ == "__main__":
    main()