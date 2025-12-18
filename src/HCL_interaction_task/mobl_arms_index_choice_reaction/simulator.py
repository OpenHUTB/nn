import mujoco
import mujoco.viewer  # 强制导入viewer（放弃兼容旧版本，优先保证运行）
import numpy as np
import yaml
import time
import os

# 解决Linux GLX错误：优先用osmesa，若不行则用glfw（备选）
os.environ['MUJOCO_GL'] = 'osmesa'  
# 增加环境变量，避免viewer闪退
os.environ['MJPYTHON_FRAMEWORK'] = 'gtk3'

class IndexSimulator:
    def __init__(self, config_path, model_path):
        # 1. 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 2. 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 3. 检查qpos长度（关键！避免索引错误）
        self.qpos_len = len(self.data.qpos)
        print(f"✅ 模型加载成功！qpos长度：{self.qpos_len}")
        self.move_joints = 3 if self.qpos_len >=3 else self.qpos_len

        # 4. 初始化核心变量
        self.is_running = True
        self.viewer = None
        self.button_names = ["button-0", "button-1", "button-2", "button-3"]
        self.button_touched = {name: False for name in self.button_names}
        self.finger_geom_name = "hand_2distph"

    def reset(self):
        """重置仿真：安全赋值qpos"""
        mujoco.mj_resetData(self.model, self.data)
        # 安全赋值：只给存在的关节赋值
        if self.qpos_len >= 1:
            self.data.qpos[0] = 0.0  # X方向
        if self.qpos_len >= 2:
            self.data.qpos[1] = 0.0  # Y方向
        if self.qpos_len >= 3:
            self.data.qpos[2] = 0.0  # Z方向
        self.button_touched = {name: False for name in self.button_names}
        self.is_running = True
        return self.data.qpos.copy()

    def step(self, action=None):
        """每一步仿真：安全移动手指"""
        # 1. 默认动作：向按钮移动（放慢速度，避免瞬移）
        if action is None:
            action = np.array([0.001, 0.0, 0.0])  # 把步长从0.01改成0.001，移动更慢更稳

        # 2. 安全更新qpos（只更新存在的关节）
        for i in range(min(len(action), self.move_joints)):
            self.data.qpos[i] += action[i]

        # 3. 推进仿真
        mujoco.mj_step(self.model, self.data)

        # 4. 检测碰撞
        self._check_button_collision()

        return self.data.qpos.copy()

    def run_simulation(self):
        """
        核心修改：主动创建Viewer+强制循环，直到手动关闭
        """
        print("✅ 仿真器初始化成功！")
        print("👉 可视化窗口已弹出，手指正在向按钮移动...（按Ctrl+C终止）")
        self.reset()

        # 主动创建Viewer实例（放弃passive模式，改用主动模式）
        try:
            with mujoco.viewer.launch(self.model, self.data) as self.viewer:
                # 调整视角
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -15
                self.viewer.cam.distance = 1.2
                self.viewer.cam.lookat = [0.4, 0.0, 0.4]

                # 强制循环：直到手动关闭窗口/按Ctrl+C
                while self.is_running and self.viewer.is_running():
                    # 执行仿真步
                    self.step()
                    # 控制帧率（每秒50帧，不会卡）
                    time.sleep(0.02)
                    # 同步viewer（主动模式必须手动sync）
                    self.viewer.sync()

        except KeyboardInterrupt:
            # 按Ctrl+C优雅退出
            print("\n⚠️ 检测到Ctrl+C，正在退出仿真...")
        except Exception as e:
            print(f"⚠️ 可视化启动失败，改用无窗口模式运行：{e}")
            # 无窗口模式：循环100秒后退出
            start_time = time.time()
            while time.time() - start_time < 100:
                self.step()
                time.sleep(0.02)

        self.close()

    def _check_button_collision(self):
        """检测碰撞（简化版）"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name

            for btn_name in self.button_names:
                if (btn_name in [geom1, geom2]) and (self.finger_geom_name in [geom1, geom2]):
                    if not self.button_touched[btn_name]:
                        color_map = {"button-0":"红", "button-1":"绿", "button-2":"蓝", "button-3":"黄"}
                        print(f"🎉 碰到【{color_map[btn_name]}按钮】！（继续运行，不会停止）")
                        self.button_touched[btn_name] = True

    def close(self):
        """关闭资源"""
        self.is_running = False
        if self.viewer:
            try:
                self.viewer.close()
            except:
                pass
        print("\n👋 仿真结束～")