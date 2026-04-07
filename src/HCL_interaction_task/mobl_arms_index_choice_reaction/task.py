import numpy as np

class ChoicePanelTask:
    def __init__(self, config, sim):
        self.config = config
        self.sim = sim  # 关联你的仿真器实例
        # 从配置中读取任务参数（后续在config.yaml中配置）
        self.target_button = config.get("target_button", 0)  # 默认目标按钮0
        self.button_reward = config.get("button_reward", 10.0)  # 选对奖励
        self.effort_cost = config.get("effort_cost", 0.01)  # 动作成本系数
        self.timeout_steps = config.get("timeout", 300)  # 超时步数
        
        self.step_count = 0  # 记录当前任务步数
        self.done = False    # 任务是否完成
        self.reward = 0.0    # 当前奖励值

    def reset(self):
        """重置任务状态（每次仿真/任务重新开始时调用）"""
        self.step_count = 0
        self.done = False
        self.reward = 0.0
        print(f"\n🔄 任务已重置！目标按钮：button-{self.target_button}，超时步数：{self.timeout_steps}")
        return {"reward": self.reward, "done": self.done}

    def update(self):
        """每一步仿真都调用，更新任务状态（核心逻辑）"""
        if self.done:  # 任务已完成，直接返回状态
            return {"reward": self.reward, "done": self.done}
        
        self.step_count += 1
        self.reward = 0.0

        # 1. 检测：指尖是否碰到目标按钮
        is_touch_target = self._check_button_contact(self.target_button)
        if is_touch_target:
            self.reward += self.button_reward  # 选对按钮，添加奖励
            self.done = True
            print(f"\n🎉 任务成功！指尖碰到button-{self.target_button}，奖励：{self.reward:.2f}")

        # 2. 动作成本扣分（兼容执行器为0的情况）
        ctrl_effort = np.sum(np.abs(self.sim.data.ctrl)) if self.sim.model.nu > 0 else 0.0
        effort_penalty = self.effort_cost * ctrl_effort
        self.reward -= effort_penalty
        if self.sim.model.nu == 0 and self.step_count % 50 == 0:
            print(f"\n💡 提示：当前模型无执行器，动作成本扣分暂不生效（后续添加执行器后自动启用）")

        # 3. 超时判断：超过步数未完成则任务失败
        if self.step_count >= self.timeout_steps:
            self.done = True
            self.reward -= 5.0  # 超时额外扣分
            print(f"\n⏰ 任务超时失败！未在{self.timeout_steps}步内碰到button-{self.target_button}，最终奖励：{self.reward:.2f}")

        # 4. 定期打印任务状态（每50步）
        if self.step_count % 50 == 0 and not self.done:
            print(f"\n💰 step{self.step_count} 任务状态：")
            print(f"   是否碰到目标按钮：{'是' if is_touch_target else '否'}")
            print(f"   当前奖励：{self.reward:.2f}")
            print(f"   剩余步数：{self.timeout_steps - self.step_count}")

        return {"reward": self.reward, "done": self.done}

    def _check_button_contact(self, button_id):
        """检测指尖（hand_2distph）是否碰到指定按钮（button-x）"""
        finger_geom_name = self.sim.finger_geom_name  # 从仿真器获取指尖geom名称
        target_button_name = f"button-{button_id}"    # 目标按钮geom名称

        # 遍历所有碰撞对，判断是否包含「指尖」和「目标按钮」
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            # 获取两个碰撞geom的名称
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            geom1_name = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            # 判断是否是「指尖-目标按钮」的碰撞
            if (finger_geom_name in [geom1_name, geom2_name]) and (target_button_name in [geom1_name, geom2_name]):
                return True
        return False

# 导入mujoco（避免内部函数报错）
import mujoco