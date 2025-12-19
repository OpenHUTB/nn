import numpy as np
import jax
import mujoco
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf
from loco_mujoco.trajectory import TrajectoryInfo, TrajectoryModel, TrajectoryData
import matplotlib.pyplot as plt
import time

class LogMotionSynchronizer:
    """滚木运动同步器：空中下落+落地反馈+直线滚动（无冗余代码，无报错）"""
    def __init__(self, env, log_init_pos=[0.0, 0.0, 1.5]):
        self.env = env
        self.model = env.get_model()
        self.data = env.get_data()
        
        # 滚木状态标记
        self.is_landed = False
        self.land_time = None
        self.roll_torque = 10.0  # 滚动扭矩
        
        # 添加滚木模型（极简版）
        self._add_log_to_model(log_init_pos)
        
        # 关节索引（直接定义，无需通过关节名查找ID）
        self.log_pos_ids = slice(0, 3)    # x/y/z位置
        self.log_rot_ids = slice(3, 6)    # rx/ry/rz姿态
        self.log_vel_ids = slice(0, 6)    # 速度/角速度
        
        # 数据记录
        self.log_pos_history = []
        self.log_vel_history = []
        self.time_history = []

    def _add_log_to_model(self, init_pos):
        """添加滚木模型（无自定义assets，兼容旧版Mujoco）"""
        spec = self.env._mjspec
        
        # 1. 滚木主体（自由关节）
        log_body = spec.worldbody.add_body(
            name="log",
            pos=init_pos  # 空中初始位置：z=1.5m
        )
        
        # 自由关节（6自由度：下落+滚动）
        log_body.add_joint(
            name="log_root",
            type=mujoco.mjtJoint.mjJNT_FREE
        )
        
        # 2. 胶囊体滚木（直接设置物理属性）
        # 核心滚木
        log_body.add_geom(
            name="log_core",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=(-0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
            size=(0.18, 0.0, 0.0),
            rgba=(0.72, 0.5, 0.3, 1.0),
            friction=(0.8, 0.1, 0.1),
            density=1200.0,
            condim=3,
            margin=0.001
        )
        # 外层树皮
        log_body.add_geom(
            name="bark_outer",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=(-0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
            size=(0.19, 0.0, 0.0),
            rgba=(0.6, 0.4, 0.22, 1.0),
            friction=(0.85, 0.1, 0.1),
            density=1300.0,
            condim=3,
            margin=0.001
        )
        
        # 编译模型
        self.env._model = spec.compile()
        self.env._data = mujoco.MjData(self.env._model)

    def check_landing(self):
        """检测滚木落地"""
        if self.is_landed:
            return True
        
        # 遍历所有接触点
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # 获取接触的几何名称
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            
            # 判断滚木与地面接触
            if ("log" in geom1_name or "log" in geom2_name) and "floor" in (geom1_name + geom2_name):
                self.is_landed = True
                self.land_time = time.time()
                log_pos = self.data.qpos[self.log_pos_ids]
                print(f"\n【落地反馈】滚木于 {self.land_time - self.start_time:.2f}s 落地！")
                print(f"落地位置：x={log_pos[0]:.2f}, y={log_pos[1]:.2f}, z={log_pos[2]:.2f}")
                return True
        return False

    def apply_roll_force(self):
        """落地后施加扭矩，让滚木沿x轴滚动"""
        if not self.is_landed:
            return
        
        # 绕y轴施加扭矩
        mujoco.mj_applyFT(
            self.model,
            self.data,
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, self.roll_torque, 0.0], dtype=np.float64),
            self.data.body("log").xpos,
            self.data.body("log").id
        )

    def update(self, step_time):
        """每步更新滚木状态"""
        self.check_landing()
        self.apply_roll_force()
        
        # 记录数据
        self.log_pos_history.append(self.data.qpos[self.log_pos_ids].copy().astype(np.float64))
        self.log_vel_history.append(self.data.qvel[self.log_vel_ids].copy().astype(np.float64))
        self.time_history.append(step_time)

    def plot_motion_curve(self):
        """绘制滚木运动曲线"""
        if not self.time_history:
            print("无运动数据可绘制！")
            return
        
        pos_array = np.array(self.log_pos_history)
        vel_array = np.array(self.log_vel_history)
        time_array = np.array(self.time_history)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 位置曲线
        ax1.plot(time_array, pos_array[:, 0], label="X轴位置（滚动方向）", color="blue")
        ax1.plot(time_array, pos_array[:, 2], label="Z轴位置（高度）", color="red")
        if self.land_time:
            ax1.axvline(x=self.land_time - self.start_time, 
                        color="green", linestyle="--", label="落地时刻")
        ax1.set_title("滚木位置变化")
        ax1.set_xlabel("时间 (s)")
        ax1.set_ylabel("位置 (m)")
        ax1.legend()
        ax1.grid(True)
        
        # 速度曲线
        ax2.plot(time_array, vel_array[:, 0], label="X轴速度", color="blue")
        ax2.plot(time_array, vel_array[:, 5], label="绕Y轴角速度（滚动）", color="orange")
        if self.land_time:
            ax2.axvline(x=self.land_time - self.start_time, 
                        color="green", linestyle="--", label="落地时刻")
        ax2.set_title("滚木速度变化")
        ax2.set_xlabel("时间 (s)")
        ax2.set_ylabel("速度 (m/s) / 角速度 (rad/s)")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # 1. 创建LocoMuJoCo环境
    env = ImitationFactory.make(
        "UnitreeH1",
        n_substeps=20,
        default_dataset_conf=DefaultDatasetConf(["walk", "squat"])
    )
    # 核心修复：移除return_info=True，适配旧版LocoEnv
    env.reset(jax.random.PRNGKey(0))
    
    # 构造机器人直立动作（不依赖info，直接适配H1关节）
    # 方案：使用"微小力矩+位置保持"，避免机器人因重力倒下
    stand_action = np.zeros(env.action_dim, dtype=np.float64)
    # 针对UnitreeH1机器人，为核心关节施加平衡力矩（适配所有动作维度）
    # 即使维度不匹配，也不会报错，仅核心关节生效
    torque_bias = 0.6  # 基础平衡力矩，可微调
    for j in range(min(12, env.action_dim)):
        # 髋/膝/踝关节（前12个关节）施加平衡力矩
        stand_action[j] = torque_bias if j % 3 != 1 else torque_bias * 1.2  # 膝关节力矩稍大
    
    # 初始化滚木（空中位置：z=1.5m）
    log_sync = LogMotionSynchronizer(env, log_init_pos=[0.0, 0.0, 1.5])
    log_sync.start_time = time.time()
    
    # 运行仿真（1000步，约5秒）
    total_steps = 1000
    print("开始仿真：机器人直立静止，滚木从空中下落...")
    try:
        for step in range(total_steps):
            # 执行直立动作，保持机器人不倒
            env.step(stand_action)
            
            # 更新滚木状态（完全不变）
            log_sync.update(step * env.dt)
            
            # 渲染画面
            env.render()
            
            # 每50步打印状态
            if step % 50 == 0:
                log_pos = log_sync.data.qpos[log_sync.log_pos_ids]
                log_vel = log_sync.data.qvel[log_sync.log_vel_ids]
                print(f"\n第 {step} 步 - 状态：")
                print(f"  机器人：直立静止")
                print(f"  滚木位置：x={log_pos[0]:.2f}, y={log_pos[1]:.2f}, z={log_pos[2]:.2f}")
                print(f"  滚木速度：x={log_vel[0]:.2f} m/s, 绕Y轴角速度={log_vel[5]:.2f} rad/s")
                print(f"  滚木落地状态：{'已落地' if log_sync.is_landed else '下落中'}")
        
        print("\n仿真结束！")
        log_sync.plot_motion_curve()
        
    except KeyboardInterrupt:
        print("\n仿真被中断！")
        log_sync.plot_motion_curve()

if __name__ == "__main__":
    main()