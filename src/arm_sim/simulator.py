import mujoco
import mujoco.viewer  
import numpy as np
import yaml
import time
import os

class IndexSimulator:
    def __init__(self, config_path, model_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 1. 模型关键信息+关节range（稳定核心）
        print("\n模型核心信息：")
        print(f"关节总数：{self.model.njnt} | 执行器数：{self.model.nu} | qpos长度：{len(self.data.qpos)}")

        # 在__init__里新增：打印所有geom
        print("\n所有geom名称：")
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            print(f"geom{i}：{geom_name}")
        
        self.key_joints = {
            15: {
                "name": "shoulder_rot (肩关节旋转)",
                "range": (-1.57, 0.349),
                "step": 0.05,
                "current_step_dir": 0.05  # 运动方向
            },
            16: {
                "name": "elbow_flexion (肘关节弯曲)",
                "range": (0, 2.26893),
                "step": 0.08,
                "current_step_dir": 0.08
            },
            17: {
                "name": "pro_sup (前臂旋前旋后)",
                "range": (-1.5708, 1.5708),
                "step": 0.04,
                "current_step_dir": 0.04
            }
        }

        # 打印关节初始信息
        print("\n关键关节（带range限制）：")
        for jnt_id, info in self.key_joints.items():
            if jnt_id < len(self.data.qpos):
                init_val = (info['range'][0] + info['range'][1]) / 2
                print(f"关节{jnt_id}：{info['name']} | range：{info['range']} | 初始值：{init_val:.2f} rad")

        # 2. 仿真状态（极简，避免段错误）
        self.is_running = True
        self.viewer = None
        self.print_interval = 50
        self.current_step = 0
        self.button_touched = {"button-0":False, "button-1":False, "button-2":False, "button-3":False}
        self.finger_geom_name = "hand_2distph"

        # 新增：找到screen对应的geom ID（控制面板颜色）
        self.screen_geom_name = "screen"
        self.screen_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, self.screen_geom_name
        )
        if self.screen_geom_id == -1:
            print(f"未找到面板geom（{self.screen_geom_name}），颜色控制失效")
        else:
            print(f"找到面板geom（ID：{self.screen_geom_id}），已启用颜色切换")

        # 新增：面板要切换的4种颜色（和按钮颜色对应）
        self.screen_colors = [
            [0.8, 0.1, 0.1, 1.0],  # 红色（对应button-0）
            [0.1, 0.8, 0.1, 1.0],  # 绿色（对应button-1）
            [0.1, 0.1, 0.8, 1.0],  # 蓝色（对应button-2）
            [0.8, 0.8, 0.1, 1.0]   # 黄色（对应button-3）
        ]
        self.color_switch_interval = 100  # 每隔100步切换一次颜色（≈3秒）

    def reset(self):
        """重置关节到range中间值"""
        mujoco.mj_resetData(self.model, self.data)
        for jnt_id, info in self.key_joints.items():
            if jnt_id < len(self.data.qpos):
                init_val = (info['range'][0] + info['range'][1]) / 2
                self.data.qpos[jnt_id] = init_val
                self.key_joints[jnt_id]['current_step_dir'] = info['step']  # 重置方向
        self.current_step = 0
        self.is_running = True
        return self.data.qpos.copy()

    def step(self):
        """单步仿真：稳定+慢速，无错误"""
        if not self.is_running:
            return self.data.qpos.copy()

        # 新增：每隔N步随机切换面板颜色（兼容所有MuJoCo版本）
        if self.screen_geom_id != -1 and self.current_step % self.color_switch_interval == 0:
            # 随机选一种颜色
            random_color = self.screen_colors[np.random.randint(0, len(self.screen_colors))]
            # 关键修改：直接赋值model.geom_rgba（替代过时的mj_geom_rgba）
            self.model.geom_rgba[self.screen_geom_id] = random_color
            # 打印颜色变化
            color_map = {
                tuple(self.screen_colors[0]): "红",
                tuple(self.screen_colors[1]): "绿",
                tuple(self.screen_colors[2]): "蓝",
                tuple(self.screen_colors[3]): "黄"
            }
            color_name = color_map[tuple(random_color)]
            print(f"\nstep{self.current_step} 面板颜色切换为：{color_name}色")

        # 1. 关节慢速摆动（不超range）
        for jnt_id, info in self.key_joints.items():
            if jnt_id >= len(self.data.qpos):
                continue
            
            current_val = self.data.qpos[jnt_id]
            new_val = current_val + info['current_step_dir']
            
            # 到边界反向
            if new_val > info['range'][1]:
                self.key_joints[jnt_id]['current_step_dir'] = -info['step']
                new_val = info['range'][1] - 0.01
            elif new_val < info['range'][0]:
                self.key_joints[jnt_id]['current_step_dir'] = info['step']
                new_val = info['range'][0] + 0.01
            
            # 安全赋值
            self.data.qpos[jnt_id] = np.clip(new_val, info['range'][0], info['range'][1])

        # 2. 推进仿真
        mujoco.mj_step(self.model, self.data)

        # 3. 定期打印（减少刷屏）
        if self.current_step % self.print_interval == 0:
            print(f"\nstep{self.current_step} 关节角度（直观度数）：")
            for jnt_id, info in self.key_joints.items():
                if jnt_id < len(self.data.qpos):
                    rad = self.data.qpos[jnt_id]
                    deg = rad * 57.3  # 弧度转角度
                    print(f"   {info['name']}：{rad:.2f} rad ≈ {deg:.0f}°")

        # 4. 渲染（容错处理，避免段错误）
        if self.viewer:
            try:
                self.viewer.sync()
                time.sleep(0.03)  # 慢速，肉眼看清
            except Exception as e:
                print(f"渲染同步警告（不影响仿真）：{e}")
                pass

        self.current_step += 1
        return self.data.qpos.copy()

    def run_simulation(self):
        """核心：仅Ctrl+C中断，无窗口检测（解决段错误）"""
        print("\n仿真启动成功！")
        print("手臂会慢速稳定摆动，按【Ctrl+C】终止仿真（关闭窗口需手动Ctrl+C）")
        print("关节严格限制在安全范围，无NaN/段错误！")
        self.reset()

        # 启动可视化（最大容错）
        viewer_enabled = False
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # 优化视角
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -15
            self.viewer.cam.distance = 0.6
            self.viewer.cam.lookat = [0.45, -0.15, 0.8]
            viewer_enabled = True
            print("可视化窗口启动成功！能看到手臂慢速摆动")
        except Exception as e:
            print(f"可视化启动失败（不影响动作）：{e}")
            print("无窗口模式：终端打印关节角度，确认动作真实发生！")
            self.viewer = None

        # ========== 终极循环：仅Ctrl+C中断，无窗口检测 ==========
        try:
            while self.is_running:
                self.step()
                if viewer_enabled and self.viewer:
                    try:
                        self.viewer.sync()
                    except:
                        pass
        except KeyboardInterrupt:
            print("\n\n检测到Ctrl+C，正在优雅退出仿真...")
            self.is_running = False

        # 清理资源
        self.close()
        print(f"\n仿真正常结束！共运行{self.current_step}步")
        print("最终关节角度：")
        for jnt_id, info in self.key_joints.items():
            if jnt_id < len(self.data.qpos):
                rad = self.data.qpos[jnt_id]
                deg = rad * 57.3
                print(f"   {info['name']}：{rad:.2f} rad ≈ {deg:.0f}°")

    def close(self):
        """安全关闭资源，避免段错误"""
        self.is_running = False
        if self.viewer:
            try:
                # 不同版本MuJoCo的viewer关闭方式兼容
                if hasattr(self.viewer, 'close'):
                    self.viewer.close()
                else:
                    mujoco.viewer.close(self.viewer)
            except:
                pass

if __name__ == "__main__":
    # 替换为你的配置和模型路径
    CONFIG_PATH = "config.yaml"
    MODEL_PATH = "simulation.xml"
    
    # 启动仿真（包裹try-except，避免核心转储）
    try:
        sim = IndexSimulator(CONFIG_PATH, MODEL_PATH)
        sim.run_simulation()
    except Exception as e:
        print(f"\n仿真启动异常：{e}")
        print("请检查模型文件路径/配置文件是否正确")
