import time
import keyboard
import mujoco
import mujoco.viewer


class ArmEmergencyStop:
    """
    6自由度机械臂 紧急停止功能模块
    功能：急停、安全锁、复位、状态监控
    """

    def __init__(self, model_path="arm6dof_final.xml"):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 状态定义
        self.STATE_NORMAL = 0  # 正常运行
        self.STATE_EMERG = 1  # 已急停
        self.state = self.STATE_NORMAL

        # 防抖动
        self.last_estop_time = 0
        self.debounce_interval = 0.5

    def emergency_stop(self):
        """执行紧急停止：力矩清零、速度清零、锁死"""
        self.state = self.STATE_EMERG
        self.data.ctrl[:] = 0.0  # 控制量清零
        self.data.qvel[:] = 0.0  # 关节速度清零
        print("\n[EMERGENCY STOP] 紧急停止已触发！所有关节力矩已切断。")

    def reset(self):
        """复位机械臂与状态"""
        mujoco.mj_resetData(self.model, self.data)
        self.state = self.STATE_NORMAL
        print("[RESET] 系统已复位，恢复正常运行。")

    def run(self):
        """主仿真循环 """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("=" * 60)
            print(" 6-DOF Robotic Arm Emergency Stop System")
            print(" 按键 Q = 急停    |  按键 R = 复位")
            print("=" * 60)

            while viewer.is_running():
                now = time.time()

                # ========== 急停按键 ==========
                if keyboard.is_pressed('q'):
                    if now - self.last_estop_time > self.debounce_interval:
                        self.emergency_stop()
                        self.last_estop_time = now

                # ========== 复位按键 ==========
                if keyboard.is_pressed('r'):
                    if self.state == self.STATE_EMERG:
                        self.reset()
                        time.sleep(0.2)

                # ========== 状态控制 ==========
                if self.state == self.STATE_NORMAL:
                    mujoco.mj_step(self.model, self.data)
                else:
                    self.data.ctrl[:] = 0.0  # 持续锁死

                viewer.sync()
                time.sleep(0.002)


if __name__ == "__main__":
    arm = ArmEmergencyStop()
    arm.run()