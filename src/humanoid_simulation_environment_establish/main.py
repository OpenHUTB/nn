import numpy as np
import mujoco
from mujoco import viewer
import time
from pathlib import Path
import xml.etree.ElementTree as ET


class KeyboardController:
    """键盘控制节点：使用MuJoCo viewer的key_callback处理键盘输入"""
    def __init__(self, action_dim, actuator_indices=None):
        """
        Args:
            action_dim: 动作维度（执行器数量）
            actuator_indices: 执行器名称到索引的映射
        """
        self.action_dim = action_dim
        self.actuator_indices = actuator_indices or {}
        self.current_action = np.zeros(action_dim)
        
        self.exit_flag = False
        self.paused = False
        self.reset_flag = False
        
        # 移动控制状态
        self.move_forward = False
        self.move_backward = False
        self.turn_left = False
        self.turn_right = False
        
        # 步行动作时间计数器
        self.step_time = 0.0
        self.step_frequency = 1.6  # 步频 (Hz)
        

        self._print_help()
    
    def _print_help(self):
        """打印键盘控制指令说明"""
        print("\n===== 键盘控制指令 =====")
        print("  w/↑: 前进")
        print("  s/↓: 后退")
        print("  a/←: 左转")
        print("  d/→: 右转")
        print("  空格: 暂停/继续")
        print("  r: 重置环境")
        print("  q: 退出程序")
        print("=======================")
        print("注意：请在查看器窗口内按键盘（窗口需要有焦点）\n")
    
    def key_callback(self, keycode):
        """MuJoCo viewer的键盘回调函数"""
        try:
            arrow_keys = {
                265: '\x1b[A',  # 上箭头 (Up)
                264: '\x1b[B',  # 下箭头 (Down)
                263: '\x1b[D',  # 左箭头 (Left)
                262: '\x1b[C',  # 右箭头 (Right)
            }
            
            if keycode in arrow_keys:
                key = arrow_keys[keycode]
            elif keycode == 32:  # 空格键 (Space)
                key = ' '
            elif 32 <= keycode <= 126:  # 可打印ASCII字符
                key = chr(keycode).lower()
            else:
                return
            
            self._process_key(key)
        except Exception as e:
            print(f"[错误] 处理按键时出错 (keycode={keycode}): {e}")
    
    def _set_action(self, action, name, value):
        """根据执行器名称写入动作，自动忽略缺失的执行器"""
        idx = self.actuator_indices.get(name)
        if idx is not None and 0 <= idx < self.action_dim:
            action[idx] = value
    
    def _create_walking_action(self, forward=True, turn_direction=0):
        """创建步行动作：基于周期的左右腿交替摆动"""
        action = np.zeros(self.action_dim)
        
        if not self.actuator_indices:
            return action
        
        # 计算步行动作相位
        phase = 2 * np.pi * self.step_time * self.step_frequency
        swing = np.sin(phase)
        counter_swing = np.sin(phase + np.pi)
        lift = np.maximum(0.0, np.sin(phase))
        counter_lift = np.maximum(0.0, np.sin(phase + np.pi))
        direction = 1 if forward else -1
        
        torso_pitch = 0.25 * direction
        self._set_action(action, "abdomen_y", torso_pitch)
        self._set_action(action, "abdomen_x", 0.15 * turn_direction)
        
        # 右腿
        self._set_action(action, "hip_x_right", 0.6 * direction * swing)
        self._set_action(action, "hip_y_right", -0.4 * lift)
        self._set_action(action, "knee_right", 0.7 * (0.5 - 0.5 * np.cos(phase)))
        self._set_action(action, "ankle_y_right", -0.3 * lift)
        self._set_action(action, "ankle_x_right", 0.2 * swing)
        
        # 左腿（相位相反）
        self._set_action(action, "hip_x_left", -0.6 * direction * counter_swing)
        self._set_action(action, "hip_y_left", -0.4 * counter_lift)
        self._set_action(action, "knee_left", 0.7 * (0.5 - 0.5 * np.cos(phase + np.pi)))
        self._set_action(action, "ankle_y_left", -0.3 * counter_lift)
        self._set_action(action, "ankle_x_left", -0.2 * counter_swing)
        
        # 转向控制通过髋关节外展实现
        if turn_direction != 0:
            turn_strength = 0.5 * turn_direction
            self._set_action(action, "hip_z_right", turn_strength)
            self._set_action(action, "hip_z_left", -turn_strength)
        
        return action
    
    def _process_key(self, key):
        """处理按键输入"""
        if isinstance(key, str) and key.startswith('\x1b['):
            key_char = None  # 方向键用特殊序列表示
        else:
            key_char = key if isinstance(key, str) and len(key) == 1 else None
        
        # 处理移动指令（切换模式：每次按键切换状态）
        if (key_char and key_char == 'w') or key == '\x1b[A':
            if self.move_forward:
                self.move_forward = False
                print("[键盘] 停止前进")
            else:
                self.move_forward = True
                self.move_backward = False
                print("[键盘] 前进")
            return
        elif (key_char and key_char == 's') or key == '\x1b[B':
            if self.move_backward:
                self.move_backward = False
                print("[键盘] 停止后退")
            else:
                self.move_backward = True
                self.move_forward = False
                print("[键盘] 后退")
            return
        elif (key_char and key_char == 'a') or key == '\x1b[D':
            if self.turn_left:
                self.turn_left = False
                print("[键盘] 停止左转")
            else:
                self.turn_left = True
                self.turn_right = False
                print("[键盘] 左转")
            return
        elif (key_char and key_char == 'd') or key == '\x1b[C':
            if self.turn_right:
                self.turn_right = False
                print("[键盘] 停止右转")
            else:
                self.turn_right = True
                self.turn_left = False
                print("[键盘] 右转")
            return
        
        if key == ' ':
            self.paused = not self.paused
            if self.paused:
                self.current_action = np.zeros(self.action_dim)
                self.move_forward = False
                self.move_backward = False
                self.turn_left = False
                self.turn_right = False
            print(f"[键盘] {'⏸️ 已暂停' if self.paused else '▶️ 继续'}")
        elif key_char == 'r':
            self.reset_flag = True
            print("[键盘] 🔄 重置环境")
        elif key_char == 'q':
            self.exit_flag = True
            print("[键盘] ❌ 准备退出程序...")
    
    def update_step_time(self, dt):
        """更新步行动作时间"""
        if not self.paused and (self.move_forward or self.move_backward or self.turn_left or self.turn_right):
            self.step_time += dt
        else:
            self.step_time = 0.0
    
    def get_action(self, dt=0.03):
        """获取当前控制动作"""
        if self.paused:
            return np.zeros(self.action_dim)
        
        # 更新步行动作时间
        self.update_step_time(dt)
        
        # 根据移动状态创建动作
        if self.move_forward:
            turn_dir = 0
            if self.turn_left:
                turn_dir = -1
            elif self.turn_right:
                turn_dir = 1
            self.current_action = self._create_walking_action(forward=True, turn_direction=turn_dir)
        elif self.move_backward:
            turn_dir = 0
            if self.turn_left:
                turn_dir = 1
            elif self.turn_right:
                turn_dir = -1
            self.current_action = self._create_walking_action(forward=False, turn_direction=turn_dir)
        elif self.turn_left or self.turn_right:
            turn_dir = -1 if self.turn_left else 1
            self.current_action = self._create_walking_action(forward=False, turn_direction=turn_dir)
        else:
            # 没有移动指令时，返回零动作或保持平衡的微小动作
            self.current_action = np.zeros(self.action_dim)
        
        return self.current_action.copy()
    
    def should_exit(self):
        """检查是否应该退出"""
        return self.exit_flag
    
    def should_reset(self):
        """检查是否应该重置"""
        return self.reset_flag
    
    def clear_reset_flag(self):
        """清除重置标志"""
        self.reset_flag = False


class GapCorridorEnvironment:
    """基于mujoco的带空隙走廊环境（使用自定义人形机器人模型）"""
    def __init__(self, corridor_length=100, corridor_width=10, robot_xml_path=None):
        """
        Args:
            corridor_length: 走廊总长度
            corridor_width: 走廊宽度
            robot_xml_path: 自定义人形机器人XML文件路径
        """
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        if robot_xml_path is None:
            default_path = Path(__file__).resolve().parent / "model" / "humanoid" / "humanoid.xml"
        else:
            default_path = Path(robot_xml_path)
        if not default_path.is_file():
            raise FileNotFoundError(f"无法找到机器人XML文件: {default_path}")
        self.robot_xml_path = default_path
        xml_string = self._build_model()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.timestep = self.model.opt.timestep
        self.control_timestep = 0.03
        self.control_steps = int(self.control_timestep / self.timestep)
        self._max_episode_steps = 30 / self.control_timestep
        self.current_step = 0
        self._actuator_indices = self._build_actuator_indices()

    def _parse_robot_xml(self):
        """解析自定义机器人XML，提取需要的节点（身体、执行器、肌腱等）"""
        tree = ET.parse(self.robot_xml_path)
        root = tree.getroot()
        
        robot_body = root.find("worldbody").find("body[@name='torso']")
        robot_body.set("pos", "1.0 0.5 2")
        
        # 提取XML节点并转换为字符串
        single_nodes = ["actuator", "tendon", "contact", "asset", "visual", "keyframe", "statistic"]
        parts = {"robot_body": ET.tostring(robot_body, encoding="unicode")}
        for node_name in single_nodes:
            node = root.find(node_name)
            parts[node_name] = ET.tostring(node, encoding="unicode") if node is not None else ""
        default_nodes = root.findall("default")
        parts["default"] = "".join(ET.tostring(node, encoding="unicode") for node in default_nodes)
        
        return parts

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
            {robot_parts['statistic']}
            
            <!-- 走廊环境的默认参数 -->
            <default>
                <joint armature="0.1" damping="1" limited="true"/>
                <geom conaffinity="0" condim="3" friction="1 0.1 0.1" 
                      solimp="0.99 0.99 0.003" solref="0.02 1"/>
            </default>
            {robot_parts['default']}
            
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
        return xml

    def _build_gaps_corridor(self):
        """构建带空隙的走廊（平台+空隙交替）"""
        platform_length, gap_length, platform_thickness = 2.0, 1.0, 0.2
        platform_width = self.corridor_width / 4 - 0.1
        gaps = []
        
        current_pos = 0.0
        while current_pos < self.corridor_length:
            x_pos = current_pos + platform_length / 2
            z_pos = platform_thickness / 2
            size_str = f"{platform_length/2} {platform_width} {platform_thickness/2}"
            
            for side, y_pos in [("left", -self.corridor_width/4), ("right", self.corridor_width/4)]:
                gaps.append(f"""
            <geom name="platform_{side}_{current_pos}" type="box" 
                  size="{size_str}" 
                  pos="{x_pos} {y_pos} {z_pos}" 
                  rgba="0.4 0.4 0.8 1"/>
            """)
            current_pos += platform_length + gap_length
        
        return ''.join(gaps)
    
    def _build_actuator_indices(self):
        """建立执行器名称到索引的映射，方便控制器按名称写入动作"""
        indices = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                indices[name] = i
        return indices
    
    def get_actuator_indices(self):
        return self._actuator_indices.copy()

    def reset(self):
        """重置环境到初始状态"""
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_observation()

    def _get_observation(self):
        """获取观测（关节位置、速度、躯干位置）"""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_pos = self.data.xpos[torso_id].copy()
        return np.concatenate([qpos, qvel, torso_pos])

    def _get_reward(self):
        """计算奖励：前进速度（沿走廊X轴）+ 空隙掉落惩罚"""
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        
        geom_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, 
            self.data, 
            mujoco.mjtObj.mjOBJ_BODY, 
            torso_id, 
            geom_vel, 
            0
        )
        reward = geom_vel[0] * 0.1
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            geom_names = [geom1_name, geom2_name]
            if not any(name and "platform" in name for name in geom_names):
                reward -= 0.3
                break
        return reward

    def step(self, action):
        """执行动作并推进环境"""
        self.current_step += 1
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        
        for _ in range(self.control_steps):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_observation()
        reward = self._get_reward()
        done = self.current_step >= self._max_episode_steps
        
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_z = self.data.xpos[torso_id][2]
        if torso_z < 0.5:
            done = True
            reward -= 1.0
        return obs, reward, done

    def render(self, viewer_handle=None):
        """渲染画面"""
        if viewer_handle is not None:
            with viewer_handle.lock():
                viewer_handle.sync()


def main():
    env = GapCorridorEnvironment(corridor_length=100, corridor_width=10)
    
    print("\n环境已初始化")
    print(f"执行器数量: {env.model.nu}")
    print(f"关节数量: {env.model.nq}")
    
    controller = KeyboardController(env.model.nu, env.get_actuator_indices())
    obs = env.reset()
    total_reward = 0.0
    
    print("\n启动MuJoCo交互式查看器...")
    print("按 ESC 或关闭窗口退出程序")
    
    try:
        viewer_handle = mujoco.viewer.launch_passive(
            env.model, 
            env.data,
            key_callback=controller.key_callback,
            show_left_ui=True,
            show_right_ui=True
        )
        
        print("\n查看器已启动，开始仿真循环...")
        
        step = 0
        while viewer_handle.is_running() and not controller.should_exit():
            if controller.should_reset():
                obs = env.reset()
                total_reward = 0.0
                step = 0
                # 重置移动状态
                controller.move_forward = False
                controller.move_backward = False
                controller.turn_left = False
                controller.turn_right = False
                controller.step_time = 0.0
                controller.clear_reset_flag()
            
            # 获取动作（传入控制步长以更新步行动作）
            action = controller.get_action(dt=env.control_timestep)
            obs, reward, done = env.step(action)
            total_reward += reward
            
            env.render(viewer_handle)
            
            if step % 100 == 0:
                torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                torso_pos = env.data.xpos[torso_id]
                print(f"Step {step}: 躯干位置 = {torso_pos}, 累计奖励 = {total_reward:.2f}")
            
            if done:
                print(f"\nEpisode finished. Total reward: {total_reward:.2f}")
                obs = env.reset()
                total_reward = 0.0
                step = 0
            
            step += 1
            time.sleep(0.01)
        
        viewer_handle.close()
        print("\n查看器已关闭")
        
    except Exception as e:
        print(f"无法启动查看器: {e}")
        import traceback
        traceback.print_exc()
    
    print("程序已退出")

if __name__ == "__main__":
    main()