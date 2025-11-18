import numpy as np
import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import time
import xml.etree.ElementTree as ET  # 用于解析XML

class GapCorridorEnvironment:
    """基于mujoco-py的带空隙走廊环境（使用自定义人形机器人模型）"""
    def __init__(self, corridor_length=100, corridor_width=10, robot_xml_path="/home/qiqi/mujoco_ros_ws/src/humanoid_motion/xml/humanoid_2.xml"):
        """
        Args:
            corridor_length: 走廊总长度
            corridor_width: 走廊宽度
            robot_xml_path: 自定义人形机器人XML文件路径
        """
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.robot_xml_path = robot_xml_path  # 自定义机器人模型路径
        self.model = self._build_model()  # 构建整合后的模型
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.timestep = self.model.opt.timestep  # 物理步长（从机器人XML读取，为0.005s）
        self.control_timestep = 0.03  # 控制步长
        self.control_steps = int(self.control_timestep / self.timestep)  # 每个控制步包含的物理步
        self._max_episode_steps = 30 / self.control_timestep  # 总步数（30秒）
        self.current_step = 0

    def _parse_robot_xml(self):
        """解析自定义机器人XML，提取需要的节点（身体、执行器、肌腱等）"""
        tree = ET.parse(self.robot_xml_path)
        root = tree.getroot()

        # 提取机器人的身体定义（<worldbody>下的torso节点）
        worldbody = root.find("worldbody")
        robot_body = worldbody.find("body[@name='torso']")  # 机器人主体
        # 调整机器人初始位置（放在走廊起点，x=2.0处，避免初始在空隙）
        robot_body.set("pos", "2.0 0 1.282")  # 保持高度不变，x方向移动到走廊起点

        # 提取执行器（<actuator>）、肌腱（<tendon>）、接触排除（<contact>）定义
        actuator = root.find("actuator")
        tendon = root.find("tendon")
        contact = root.find("contact")
        asset = root.find("asset")  # 机器人的材质/纹理定义
        visual = root.find("visual")  # 可视化配置
        keyframe = root.find("keyframe")  # 关键帧（可选，用于初始姿态）

        return {
            "robot_body": ET.tostring(robot_body, encoding="unicode"),
            "actuator": ET.tostring(actuator, encoding="unicode") if actuator is not None else "",
            "tendon": ET.tostring(tendon, encoding="unicode") if tendon is not None else "",
            "contact": ET.tostring(contact, encoding="unicode") if contact is not None else "",
            "asset": ET.tostring(asset, encoding="unicode") if asset is not None else "",
            "visual": ET.tostring(visual, encoding="unicode") if visual is not None else "",
            "keyframe": ET.tostring(keyframe, encoding="unicode") if keyframe is not None else ""
        }

    def _build_model(self):
        """构建带空隙的走廊环境，并整合自定义人形机器人模型"""
        # 解析自定义机器人XML
        robot_parts = self._parse_robot_xml()

        # 基础XML结构（走廊环境+机器人）
        xml = f"""
        <mujoco model="gap_corridor_with_custom_humanoid">
            <!-- 物理参数（使用机器人XML中的timestep） -->
            <option timestep="0.005" gravity="0 0 -9.81"/>
            
            <!-- 整合机器人的材质和可视化配置 -->
            {robot_parts['visual']}
            {robot_parts['asset']}
            
            <!-- 走廊环境的默认参数 -->
            <default>
                <joint armature="0.1" damping="1" limited="true"/>
                <geom conaffinity="0" condim="3" friction="1 0.1 0.1" 
                      solimp="0.99 0.99 0.003" solref="0.02 1"/>
            </default>
            
            <worldbody>
                <!-- 走廊地面（半透明，方便观察空隙） -->
                <geom name="floor" type="plane" size="{self.corridor_length/2} {self.corridor_width/2} 0.1" 
                      pos="{self.corridor_length/2} 0 0" rgba="0.9 0.9 0.9 0.3"/>
                
                <!-- 带空隙的走廊平台 -->
                {self._build_gaps_corridor()}
                
                <!-- 整合自定义人形机器人 -->
                {robot_parts['robot_body']}
            </worldbody>
            
            <!-- 机器人的接触排除配置 -->
            {robot_parts['contact']}
            
            <!-- 机器人的肌腱定义 -->
            {robot_parts['tendon']}
            
            <!-- 机器人的执行器（电机） -->
            {robot_parts['actuator']}
            
            <!-- 机器人的关键帧（可选） -->
            {robot_parts['keyframe']}
        </mujoco>
        """
        return load_model_from_xml(xml)

    def _build_gaps_corridor(self):
        """构建带空隙的走廊（平台+空隙交替）"""
        gaps_xml = ""
        platform_length = 2.0  # 平台长度
        gap_length = 1.0  # 空隙长度
        platform_thickness = 0.2  # 平台厚度
        current_pos = 0.0  # 起始位置
        # 交替添加平台和空隙
        while current_pos < self.corridor_length:
            # 平台左半部分（y负方向）
            gaps_xml += f"""
            <geom name="platform_left_{current_pos}" type="box" 
                  size="{platform_length/2} {self.corridor_width/4 - 0.1} {platform_thickness/2}" 
                  pos="{current_pos + platform_length/2} {-self.corridor_width/4} {platform_thickness/2}" 
                  rgba="0.4 0.4 0.8 1"/>
            """
            # 平台右半部分（y正方向）
            gaps_xml += f"""
            <geom name="platform_right_{current_pos}" type="box" 
                  size="{platform_length/2} {self.corridor_width/4 - 0.1} {platform_thickness/2}" 
                  pos="{current_pos + platform_length/2} {self.corridor_width/4} {platform_thickness/2}" 
                  rgba="0.4 0.4 0.8 1"/>
            """
            current_pos += platform_length + gap_length  # 移动到下一个平台起点
        return gaps_xml

    def reset(self):
        """重置环境到初始状态（使用机器人的默认姿态）"""
        self.current_step = 0
        # 重置到默认姿态（或关键帧"stand"，如果有的话）
        self.sim.reset()
        # 可选：设置初始姿态为"stand_on_left_leg"关键帧
        # self.sim.set_state_from_keyframe("stand_on_left_leg")
        self.sim.forward()  # 刷新物理状态
        return self._get_observation()

    def _get_observation(self):
        """获取观测（关节位置、速度、躯干位置）"""
        qpos = self.sim.data.qpos.copy()  # 关节位置
        qvel = self.sim.data.qvel.copy()  # 关节速度
        torso_pos = self.sim.data.get_body_xpos("torso")  # 躯干位置
        return np.concatenate([qpos, qvel, torso_pos])

    def _get_reward(self):
        """计算奖励：前进速度（沿走廊X轴）+ 空隙掉落惩罚"""
        # 前进速度奖励（X方向速度）
        torso_vel = self.sim.data.get_body_xvelp("torso")[0]  # X方向线速度
        reward = torso_vel * 0.1  # 速度越大奖励越高

        # 掉落惩罚（踩到空隙区域）
        fall_penalty = 0.0
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1_name = self.model.geom_names[contact.geom1]
            geom2_name = self.model.geom_names[contact.geom2]
            # 若接触的不是平台（即踩到空隙的地面），扣分
            if "platform" not in geom1_name and "platform" not in geom2_name:
                fall_penalty -= 0.3
        return reward + fall_penalty

    def step(self, action):
        """执行动作并推进环境（动作维度需与机器人执行器数量匹配）"""
        self.current_step += 1
        # 应用动作到执行器（控制信号限制在[-1,1]，你的机器人有20个执行器）
        self.sim.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        # 执行多个物理步（匹配控制步长）
        for _ in range(self.control_steps):
            self.sim.step()
        # 获取观测、奖励、完成状态
        obs = self._get_observation()
        reward = self._get_reward()
        done = self.current_step >= self._max_episode_steps
        # 若机器人跌倒（躯干高度过低），提前结束
        torso_z = self.sim.data.get_body_xpos("torso")[2]
        if torso_z < 0.5:  # 躯干高度低于0.5米视为跌倒
            done = True
            reward -= 1.0  # 跌倒额外惩罚
        return obs, reward, done

    def render(self):
        """渲染画面"""
        self.viewer.render()


def random_policy(env):
    """随机策略（生成符合动作空间的随机动作，你的机器人有20个执行器）"""
    action_dim = env.model.nu  # 动作维度 = 执行器数量（20）
    return np.random.uniform(low=-1.0, high=1.0, size=action_dim)


def main():
    # 创建带空隙的走廊环境（使用自定义人形机器人）
    env = GapCorridorEnvironment(
        corridor_length=100, 
        corridor_width=10,
        robot_xml_path="/home/qiqi/mujoco_ros_ws/src/humanoid_motion/xml/humanoid_2.xml"  # 确保路径正确
    )
    # 运行环境
    while True:
        obs = env.reset()
        total_reward = 0.0
        while True:
            action = random_policy(env)  # 生成随机动作（20维）
            obs, reward, done = env.step(action)  # 执行一步
            total_reward += reward
            env.render()  # 渲染画面
            if done:
                print(f"Episode finished. Total reward: {total_reward:.2f}")
                break
        time.sleep(1)  # 每轮结束后暂停1秒

if __name__ == "__main__":
    main()