# CARLA 自动驾驶强化学习 - 训练运行日志记录
## 项目运行说明
本文件为项目训练过程的核心日志存档，包含训练初始化、多轮迭代过程、性能统计等关键信息，用于证明项目可正常运行。
完整运行截图已同步存放至 docs 文件夹，日志内容均来自真实训练终端输出。

---

## 一、训练启动与初始化信息
===== 开始启动 CARLA 强化学习训练 =====
请确保 CarlaUE4 模拟器已提前开启

Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.

UserWarning: WARN: Box bound precision lowered by casting to float32
Using cuda device
Logging to ./tensorboard_logs/trained_ssl_rl/carla_ppo_20260514_125914/carla_ppo_20260514_125914_0

---

## 二、训练过程（Episode 1~10）
Episode 1 | Reason: REWARD_BASED | Reward: -8.6 | Steps: 93 | Speed: 16.1 km/h
Episode 1 completed (reason: reward_based)

Episode 2 | Reason: REWARD_BASED | Reward: -12.4 | Steps: 53 | Speed: 7.2 km/h
Episode 2 completed (reason: reward_based)

Episode 3 | Reason: REWARD_BASED | Reward: 5.1 | Steps: 91 | Speed: 18.7 km/h
Episode 3 completed (reason: reward_based)

Episode 4 | Reason: REWARD_BASED | Reward: 6.8 | Steps: 119 | Speed: 18.5 km/h
Episode 4 completed (reason: reward_based)

Episode 5 | Reason: REWARD_BASED | Reward: 5.8 | Steps: 123 | Speed: 19.4 km/h
Episode 5 completed (reason: reward_based)

Episode 6 | Reason: REWARD_BASED | Reward: 4.3 | Steps: 108 | Speed: 16.5 km/h
Episode 6 completed (reason: reward_based)

Episode 7 | Reason: REWARD_BASED | Reward: 4.6 | Steps: 102 | Speed: 14.7 km/h
Episode 7 completed (reason: reward_based)

Episode 8 | Reason: REWARD_BASED | Reward: 0.6 | Steps: 150 | Speed: 21.7 km/h
Episode 8 completed (reason: reward_based)

Episode 9 | Reason: REWARD_BASED | Reward: 1.0 | Steps: 101 | Speed: 17.2 km/h
Episode 9 completed (reason: reward_based)

Episode 10 | Reason: REWARD_BASED | Reward: -1.4 | Steps: 1218 | Speed: 0.2 km/h
Episode 10 completed (reason: reward_based)

---

## 三、训练统计结果
| rollout/          |       |
|-------------------|-------|
| ep_len_mean       | 104   |
| ep_rew_mean       | 0.793 |
| time/             |       |
| fps               | 2     |
| iterations        | 1     |
| time_elapsed      | 813   |
| total_timesteps   | 2048  |

---

## 四、验证说明
1.  训练环境正常初始化，CUDA GPU 加速已启用
2.  智能体成功与 CARLA 模拟器交互并完成多轮训练
3.  奖励函数、终止条件、步数统计均正常运行
4.  本文件为纯文本日志记录，无冗余大文件，符合项目提交规范