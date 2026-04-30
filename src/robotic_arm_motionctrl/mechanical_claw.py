import mujoco
import mujoco.viewer
import time
import numpy as np
from pynput import keyboard


class StableGripperControl:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 关键修复：禁用可能导致冲突的equality约束
        self.model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EQUALITY

        # 抓取器执行器
        self.gripper_joints = [
            "finger1_slide", "finger1_hinge1", "finger1_hinge2",
            "finger2_slide", "finger2_hinge1", "finger2_hinge2",
            "finger3_slide", "finger3_hinge1", "finger3_hinge2"
        ]
        self.act_ids = {}
        for j in self.gripper_joints:
            self.act_ids[j] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{j}")

        self.running = True
        self.target_gripper_state = 0  # 0=打开, 1=关闭
        self.current_gripper_ctrl = {j: 0.0 for j in self.gripper_joints}

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.daemon = True
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.KeyCode.from_char('z'):
                self.target_gripper_state = 1
                print("关闭爪子")
            if key == keyboard.KeyCode.from_char('x'):
                self.target_gripper_state = 0
                print("打开爪子")
            if key == keyboard.Key.esc:
                self.running = False
        except:
            pass

    def update_gripper_smooth(self):
        # 目标控制量（匹配XML ctrlrange）
        slide_target = 0.5 if self.target_gripper_state == 1 else -0.5
        hinge1_target = 1.5 if self.target_gripper_state == 1 else -0.5
        hinge2_target = 1.2 if self.target_gripper_state == 1 else 0.0

        # 平滑更新：每次只改一点点，避免突变
        for j in self.gripper_joints:
            if "slide" in j:
                target = slide_target
                curr = self.current_gripper_ctrl[j]
                new_val = curr + np.clip((target - curr) * 0.2, -0.1, 0.1)
                new_val = np.clip(new_val, -0.5, 0.5)
                self.current_gripper_ctrl[j] = new_val
            elif "hinge1" in j:
                target = hinge1_target
                curr = self.current_gripper_ctrl[j]
                new_val = curr + np.clip((target - curr) * 0.2, -0.1, 0.1)
                new_val = np.clip(new_val, -0.5, 1.5)
                self.current_gripper_ctrl[j] = new_val
            elif "hinge2" in j:
                target = hinge2_target
                curr = self.current_gripper_ctrl[j]
                new_val = curr + np.clip((target - curr) * 0.2, -0.1, 0.1)
                new_val = np.clip(new_val, 0.0, 1.2)
                self.current_gripper_ctrl[j] = new_val

        # 写入控制量
        for j in self.gripper_joints:
            if self.act_ids[j] != -1:
                self.data.ctrl[self.act_ids[j]] = self.current_gripper_ctrl[j] * 10

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 1.5
            while self.running and viewer.is_running():
                self.update_gripper_smooth()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.001)


if __name__ == "__main__":
    arm = StableGripperControl("arm6dof_final.xml")
    print("✅ 稳定版：按 Z 关 | X 开，不会再变白")
    arm.run()