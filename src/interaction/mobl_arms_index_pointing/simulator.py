import mujoco
import yaml
import numpy as np
import cv2
import time

class Simulator:
    def __init__(self, config_path, model_path):
        # 1. 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # 2. 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # 3. 初始化渲染器（使用兼容性更好的方式）
        self.viewer = self._init_viewer()
        # 4. 初始化MediaPipe手势追踪
        self.init_mediapipe()
        self.start_time = time.time()  # 记录开始时间
        self.target_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_x"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_y"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_z")
        ]

        self.finger_tip_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "index3")  # 食指末端body名称
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")  # 目标小球body名称

    def _init_viewer(self):
        """初始化viewer，适配不同版本的mujoco"""
        try:
            # 尝试使用较新版本的API
            from mujoco import viewer
            return viewer.launch_passive(self.model, self.data)
        except (ImportError, AttributeError):
            try:
                # 尝试使用旧版本API
                from mujoco.viewer import MujocoViewer
                return MujocoViewer(self.model, self.data)
            except (ImportError, AttributeError):
                try:
                    # 尝试直接使用Viewer类
                    from mujoco import Viewer
                    return Viewer(self.model, self.data)
                except (ImportError, AttributeError):
                    # 如果都失败了，提供一个虚拟viewer
                    print("Warning: 无法初始化图形界面，将运行在无头模式下")
                    return None

    def init_mediapipe(self):
        """初始化MediaPipe手部追踪"""
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        self.cap = cv2.VideoCapture(0)

    def map_gesture_to_finger(self):
        """将MediaPipe捕捉的手势坐标映射到虚拟手指关节"""
        ret, frame = self.cap.read()
        if not ret:
            return np.zeros(5)
        
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            joint_angles = np.array([
                index_tip.x * 2 - 1,
                index_tip.y * 2 - 1,
                0, 0, 0
            ])
            return joint_angles
        return np.zeros(5)

    def update_target_position(self):
        """根据配置更新小球位置"""
        if "target_movement" not in self.config:
            return
            
        t = time.time() - self.start_time
        movement = self.config["target_movement"]
        freq = movement["frequency"]
        amp = movement["amplitude"]
        
        # 正弦曲线运动（可根据需要修改运动轨迹）
        if movement["mode"] == "sinusoidal":
            x = self.config["target_pos"][0] + amp * np.sin(2 * np.pi * freq * t)
            y = self.config["target_pos"][1] + amp * np.cos(2 * np.pi * freq * t) * 0.5
            z = self.config["target_pos"][2]
        else:
            x, y, z = self.config["target_pos"]
            
        # 设置小球关节位置
        self.data.qpos[self.target_joint_ids[0]] = x
        self.data.qpos[self.target_joint_ids[1]] = y
        self.data.qpos[self.target_joint_ids[2]] = z

    def step(self):
        """推进仿真：手势映射→驱动手指→渲染"""
        # 1. 获取手势映射的关节角度
        joint_angles = self.map_gesture_to_finger()
        # 2. 应用关节角度到虚拟手指
        self.data.qpos[:len(joint_angles)] = joint_angles
        
        self.update_target_position()
        # 3. 推进仿真
        mujoco.mj_step(self.model, self.data)
        # 4. 渲染界面
        if self.viewer:
            try:
                # 新版本API
                self.viewer.sync()
                return not self.viewer.close_requested()
            except AttributeError:
                try:
                    # 旧版本API
                    return self.viewer.is_alive
                except AttributeError:
                    # 默认继续运行
                    return True
        return True  # 无头模式下始终返回True

    def close(self):
        if self.viewer:
            try:
                self.viewer.close()
            except:
                pass
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()