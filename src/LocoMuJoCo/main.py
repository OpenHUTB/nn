from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


# # example --> you can add as many datasets as you want in the lists!
env = ImitationFactory.make("UnitreeH1",
                            default_dataset_conf=DefaultDatasetConf(["squat"]),
                            lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
                            # if SMPL and AMASS are installed, you can use the following:
                            #amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"])
                            )

env.play_trajectory(n_episodes=3, n_steps_per_episode=1000, render=True)


import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf

# 自定义机器人行走环境（彻底修复所有报错）
class WalkerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # 创建UnitreeH1环境（简化数据集，加速加载）
        self.env = ImitationFactory.make(
            "UnitreeH1",
            default_dataset_conf=DefaultDatasetConf(["walk"]),
            lafan1_dataset_conf=LAFAN1DatasetConf(["walk1_subject1"])
        )
        
        # 修复Box空间精度警告：显式转换为float32
        obs_low = self.env.info.observation_space.low.astype(np.float32)
        obs_high = self.env.info.observation_space.high.astype(np.float32)
        obs_shape = self.env.info.observation_space.shape
        act_low = self.env.info.action_space.low.astype(np.float32)
        act_high = self.env.info.action_space.high.astype(np.float32)
        act_shape = self.env.info.action_space.shape
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, shape=act_shape, dtype=np.float32)
        
        # 核心属性（避免访问self.env.sim）
        self.render_mode = render_mode
        self.robot_model = self.env.model  # 直接获取模型
        self.robot_data = self.env.data     # 直接获取数据
        
        # 脚部和地面几何ID（容错获取）
        self.foot_ids = []
        for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
            geom_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id != -1:
                self.foot_ids.append(geom_id)
        self.ground_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        # 状态跟踪
        self.episode_steps = 0
        self.max_episode_steps = 500  # 缩短单episode步数

    def step(self, action):
        # 执行动作
        observation, _, _, _, info = self.env.step(action)
        
        # 获取机器人关键状态
        x_vel = self.robot_data.qvel[0]
        torso_z = self.robot_data.qpos[2]
        torso_angle = self.robot_data.qpos[3:6]
        
        # 优化后的奖励函数
        total_reward = 0.0
        total_reward += x_vel * 15.0  # 前进奖励
        total_reward += -abs(torso_z - 1.1) * 3.0  # 直立奖励
        total_reward += -np.sum(np.square(torso_angle)) * 3.0  # 姿态稳定奖励
        total_reward += -np.sum(np.square(action)) * 0.005  # 动作平滑惩罚
        total_reward += 1.5  # 存活奖励
        
        # 触地惩罚
        if self.ground_id != -1:
            for i in range(self.robot_data.ncon):
                contact = self.robot_data.contact[i]
                geom1, geom2 = contact.geom1, contact.geom2
                is_foot_contact = (geom1 in self.foot_ids) or (geom2 in self.foot_ids)
                is_ground_contact = (geom1 == self.ground_id) or (geom2 == self.ground_id)
                if is_ground_contact and not is_foot_contact:
                    total_reward -= 40.0
        
        # 终止条件
        self.episode_steps += 1
        done = False
        if torso_z < 0.6 or np.any(np.abs(torso_angle) > 0.8) or self.episode_steps >= self.max_episode_steps:
            done = True
            if torso_z < 0.6 or np.any(np.abs(torso_angle) > 0.8):
                total_reward -= 80.0
        
        # 渲染
        if self.render_mode == "human":
            self.render()
            
        return observation, total_reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.env.reset()
        self.episode_steps = 0
        return observation, {}

    def render(self):
        self.env.render()

    def close(self):
        # 彻底容错的资源释放（不访问任何可能不存在的属性）
        try:
            self.env.close()
        except AttributeError:
            # 1. 不访问self.env.sim，直接跳过渲染窗口关闭（避免报错）
            # （loco-mujoco的渲染窗口会自动随程序退出释放）
            pass
        
        # 2. 安全释放Mujoco模型和数据（先检查属性是否存在）
        if hasattr(self, 'robot_model') and self.robot_model is not None:
            try:
                mujoco.mj_deleteModel(self.robot_model)
            except:
                pass
            del self.robot_model
        
        if hasattr(self, 'robot_data') and self.robot_data is not None:
            try:
                mujoco.mj_deleteData(self.robot_data)
            except:
                pass
            del self.robot_data
        
        # 3. 清空环境引用
        if hasattr(self, 'env'):
            del self.env

# 优化后的训练函数（30万步，20-30分钟）
def train_ppo():
    env = WalkerEnv(render_mode=None)
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./logs/",
        name_prefix="walker_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # 评估回调
    eval_env = WalkerEnv(render_mode=None)
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1
    )
    
    # 优化后的PPO超参数
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        n_steps=1024,
        batch_size=32,
        n_epochs=5,
        gamma=0.98,
        gae_lambda=0.92,
        clip_range=0.25,
        ent_coef=0.02,
        device="auto"
    )
    
    print("开始训练（优化版，预计20-30分钟）...")
    model.learn(
        total_timesteps=300000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    model.save("unitree_h1_walker_optimized")
    print("训练完成！模型已保存为 unitree_h1_walker_optimized")

# 修复后的测试函数
def test_ppo():
    from stable_baselines3 import PPO
    # 使用上下文管理器自动管理环境（避免手动调用close()）
    with WalkerEnv(render_mode="human") as env:
        model = PPO.load("unitree_h1_walker_optimized", env=env)
        print("开始测试...（按ESC退出）")
        obs, _ = env.reset()
        for _ in range(10000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
    # 退出with块自动调用close()，无需手动调用

if __name__ == "__main__":
    train_ppo()
    test_ppo()