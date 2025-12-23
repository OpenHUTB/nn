import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
import sys
import os

# ========== 路径适配 ==========
SCENE_PATH = os.path.join(os.path.dirname(__file__),
                          "mujoco_menagerie-main",
                          "franka_emika_panda",
                          "grab_scene.xml")

if not os.path.exists(SCENE_PATH):
    print(f"❌ 场景文件不存在：{SCENE_PATH}")
    sys.exit(1)

# ========== 智能抓取控制器 ==========
class PandaAutoGrab:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.running = True
        self.step_counter = 0
        self.current_phase = 0
        self.grab_complete = False

        # 机械臂参数
        self.ee_body_id = self.model.body("hand").id
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        self.gripper_joint_names = ["finger_joint1", "finger_joint2"]

        # 雅克比矩阵
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

        # 抓取参数
        self.cube_body_id = self.model.body("cube").id
        self.target_place_pos = np.array([0.3, 0.0, 0.1])
        self.gripper_open_pos = 0.04
        self.gripper_close_pos = 0.005
        self.safe_lift_height = 0.15
        self.grab_height = 0.05

        # 打印模型信息
        print("="*50)
        print("📌 模型Body列表：", [self.model.body(i).name for i in range(min(self.model.nbody, 10))])
        print("📌 模型Joint列表：", [self.model.joint(i).name for i in range(min(self.model.njnt, 10))])
        print("="*50)

    def get_ee_pos(self):
        """获取末端执行器位置"""
        return self.data.xpos[self.ee_body_id].copy()

    def get_cube_pos(self):
        """获取立方体位置"""
        return self.data.xpos[self.cube_body_id].copy()

    def _compute_jacobian(self):
        """计算雅克比矩阵"""
        mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, self.get_ee_pos(), self.ee_body_id)
        return self.jacp[:, self.joint_ids]

    def _move_step(self, target, tol=0.003, speed=0.3):
        """单步移动控制（修复维度匹配问题）"""
        ee_pos = self.get_ee_pos()
        error = target - ee_pos
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return True  # 到达目标

        # 计算雅克比矩阵
        jacobian = self._compute_jacobian()  # 3×7

        # ========== 修正：正确的阻尼伪逆计算 ==========
        # 方法1：使用正则化参数的伪逆（推荐）
        lambda_ = 0.01  # 阻尼系数
        jacobian_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + lambda_ * np.eye(3))

        # 方法2：若方法1仍报错，可改用numpy伪逆（自动处理维度）
        # jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-3)

        # 关节速度指令
        joint_vel_cmd = speed * jacobian_pinv @ error
        joint_vel_cmd = np.clip(joint_vel_cmd, -0.5, 0.5)  # 速度限制

        # PD力矩计算
        torque = np.zeros(7)
        for i in range(7):
            angle_error = joint_vel_cmd[i] * 0.1
            torque[i] = 250 * angle_error - 100 * self.data.qvel[self.joint_ids[i]]
            torque[i] = np.clip(torque[i], -20, 20)

        # 设置关节力矩
        for i in range(7):
            self.data.ctrl[self.joint_ids[i]] = torque[i]

        return False

    def _gripper_step(self, pos):
        """单步夹爪控制"""
        for j_name in self.gripper_joint_names:
            j_id = self.model.joint(j_name).id
            self.data.ctrl[j_id] = pos
        return True

    def _grab_phase_machine(self):
        """抓取状态机"""
        if self.current_phase == 0:
            # 阶段0：移动到初始位置
            if self._move_step(np.array([0.4, 0.0, 0.2])):
                print("\n✅ 到达初始位置")
                self.current_phase = 1
                self.step_counter = 0

        elif self.current_phase == 1:
            # 阶段1：获取立方体位置
            self.cube_pos = self.get_cube_pos()
            print(f"\n🎯 识别到立方体位置：{np.round(self.cube_pos, 3)}")
            self.current_phase = 2

        elif self.current_phase == 2:
            # 阶段2：移动到立方体上方
            if self._move_step(self.cube_pos + np.array([0, 0, self.safe_lift_height]), speed=0.4):
                print("\n✅ 到达立方体上方")
                self.current_phase = 3
                self.step_counter = 0

        elif self.current_phase == 3:
            # 阶段3：打开夹爪
            if self.step_counter == 0:
                self._gripper_step(self.gripper_open_pos)
                print("\n✋ 打开夹爪")
            if self.step_counter > 100:  # 等待夹爪动作
                self.current_phase = 4
                self.step_counter = 0
            self.step_counter += 1

        elif self.current_phase == 4:
            # 阶段4：下降抓取
            if self._move_step(self.cube_pos + np.array([0, 0, self.grab_height]), speed=0.2):
                print("\n✅ 下降到抓取高度")
                self.current_phase = 5
                self.step_counter = 0

        elif self.current_phase == 5:
            # 阶段5：闭合夹爪
            if self.step_counter == 0:
                self._gripper_step(self.gripper_close_pos)
                print("\n🤏 闭合夹爪抓取")
            if self.step_counter > 100:
                self.current_phase = 6
                self.step_counter = 0
            self.step_counter += 1

        elif self.current_phase == 6:
            # 阶段6：抬升立方体
            if self._move_step(self.cube_pos + np.array([0, 0, self.safe_lift_height + 0.05]), speed=0.3):
                print("\n✅ 抬升立方体")
                self.current_phase = 7
                self.step_counter = 0

        elif self.current_phase == 7:
            # 阶段7：移动到放置点上方
            if self._move_step(self.target_place_pos + np.array([0, 0, self.safe_lift_height]), speed=0.4):
                print("\n✅ 到达放置点上方")
                self.current_phase = 8
                self.step_counter = 0

        elif self.current_phase == 8:
            # 阶段8：下降放置
            if self._move_step(self.target_place_pos + np.array([0, 0, self.grab_height]), speed=0.2):
                print("\n✅ 下降到放置高度")
                self.current_phase = 9
                self.step_counter = 0

        elif self.current_phase == 9:
            # 阶段9：释放立方体
            if self.step_counter == 0:
                self._gripper_step(self.gripper_open_pos)
                print("\n🫳 释放立方体")
            if self.step_counter > 100:
                self.current_phase = 10
                self.step_counter = 0
            self.step_counter += 1

        elif self.current_phase == 10:
            # 阶段10：撤离机械臂
            if self._move_step(self.target_place_pos + np.array([0, 0, self.safe_lift_height]), speed=0.3):
                print("\n✅ 撤离机械臂")
                self.current_phase = 11
                self.step_counter = 0

        elif self.current_phase == 11:
            # 阶段11：返回初始位置
            if self._move_step(np.array([0.4, 0.0, 0.2]), speed=0.4):
                print("\n✅ 返回初始位置")
                self.current_phase = 12

        elif self.current_phase == 12:
            # 阶段12：抓取完成
            if not self.grab_complete:
                print("\n" + "="*50)
                print("✅ 智能抓取任务完成！")
                print("="*50)
                self.grab_complete = True

    def run(self):
        """单线程仿真主循环"""
        # 初始化Viewer
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = 70
        self.viewer.cam.elevation = -25
        self.viewer.cam.distance = 1.8
        self.viewer.cam.lookat = np.array([0.4, 0.0, 0.1])

        print("\n🚀 仿真已启动，开始自动抓取...")
        print("💡 关闭Viewer窗口可退出程序")

        # 单线程主循环
        while self.viewer.is_running():
            if self.running and not self.grab_complete:
                self._grab_phase_machine()
            else:
                # 抓取完成后归零力矩
                for i in range(7):
                    self.data.ctrl[self.joint_ids[i]] = 0

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(1/200)

        # 清理
        self.running = False
        self.viewer.close()
        print("\n👋 仿真结束")

# ========== 主函数 ==========
if __name__ == "__main__":
    try:
        panda = PandaAutoGrab()
        panda.run()
    except Exception as e:
        print(f"\n❌ 程序错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)