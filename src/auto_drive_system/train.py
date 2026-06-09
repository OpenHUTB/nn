# import warnings
# import os
#
# warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# import argparse
# import config
# import time
#
# parser = argparse.ArgumentParser(description="Trains a CARLA agent")
# parser.add_argument("--host", default="127.0.0.1", type=str, help="IP of the host server (default: 127.0.0.1)")
# parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
# parser.add_argument("--town", default="Town01", type=str, help="Name of the map in CARLA")
# parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timestep to train for")
# parser.add_argument("--reload_model", type=str, default="", help="Path to a model to reload")
# parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
# parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
# parser.add_argument("--num_checkpoints", type=int, default=10, help="Checkpoint frequency")
# parser.add_argument("--config", type=str, default="1", help="Config to use (default: 1)")
#
# args = vars(parser.parse_args())
# config.set_config(args["config"])
#
# from stable_baselines3 import PPO, DDPG, SAC
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.logger import configure
# from agent.env import CarlaEnv
#
# from agent.rewards import reward_functions
# from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class
#
# from config import CONFIG
#
# log_dir = 'tensorboard'
# os.makedirs(log_dir, exist_ok=True)
# reload_model = args["reload_model"]
# total_timesteps = args["total_timesteps"]
#
# algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC}
# if CONFIG["algorithm"] not in algorithm_dict:
#     raise ValueError("Invalid algorithm name")
#
# AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]
#
# env = CarlaEnv(host=args["host"], port=args["port"], town=args["town"],
#                 fps=args["fps"], obs_sensor=CONFIG["obs_sensor"], obs_res=CONFIG["obs_res"],
#                     reward_fn=reward_functions[CONFIG["reward_fn"]],
#                     view_res=(1120, 560), action_smoothing=CONFIG["action_smoothing"],
#                     allow_spectator=True, allow_render=args["no_render"])
#
# if reload_model == "":
#     model = AlgorithmRL('CnnPolicy', env, verbose=2, tensorboard_log=log_dir, device='cuda',
#                         **CONFIG["algorithm_params"])
#     model_suffix = f"{int(time.time())}_id{args['config']}"
# else:
#     model = AlgorithmRL.load(reload_model, env=env, device='cuda', **CONFIG["algorithm_params"])
#     model_suffix = f"{reload_model.split('/')[-2].split('_')[-1]}_finetuning"
#
# model_name = f'{model.__class__.__name__}_{model_suffix}'
#
# model_dir = os.path.join(log_dir, model_name)
# new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
# model.set_logger(new_logger)
# write_json(CONFIG, os.path.join(model_dir, 'config.json'))
#
# model.learn(total_timesteps=total_timesteps,
#             callback=[HParamCallback(CONFIG), TensorboardCallback(1), CheckpointCallback(
#                 save_freq=total_timesteps // args["num_checkpoints"],
#                 save_path=model_dir,
#                 name_prefix="model")], reset_num_timesteps=False)

# 导入必要的库
import warnings
import os

# 忽略所有警告信息
warnings.filterwarnings("ignore")
# 设置环境变量，屏蔽TensorFlow的日志输出 (3表示只显示ERROR级别日志)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import config
import time

# 创建参数解析器，用于处理命令行参数
parser = argparse.ArgumentParser(description="Trains a CARLA agent")
# 定义CARLA服务器IP地址参数
parser.add_argument("--host", default="127.0.0.1", type=str, help="IP of the host server (default: 127.0.0.1)")
# 定义CARLA服务器TCP端口参数
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
# 定义要加载的CARLA地图名称参数
parser.add_argument("--town", default="Town01", type=str, help="Name of the map in CARLA")
# 定义训练的总时间步数参数
parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timestep to train for")
# 定义用于继续训练或微调的模型路径参数
parser.add_argument("--reload_model", type=str, default="", help="Path to a model to reload")
# 定义是否渲染环境的参数 (默认渲染，若指定--no_render则不渲染)
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
# 定义环境渲染的帧率参数
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
# 定义模型检查点的保存频率参数
parser.add_argument("--num_checkpoints", type=int, default=10, help="Checkpoint frequency")
# 定义要使用的配置文件编号参数
parser.add_argument("--config", type=str, default="1", help="Config to use (default: 1)")

# 将解析后的参数转换为字典
args = vars(parser.parse_args())
# 根据命令行参数设置配置
config.set_config(args["config"])

# 从stable_baselines3库导入强化学习算法
from stable_baselines3 import PPO, DDPG, SAC
# 导入检查点回调，用于定期保存模型
from stable_baselines3.common.callbacks import CheckpointCallback
# 导入日志配置工具
from stable_baselines3.common.logger import configure
# 从自定义模块导入CARLA环境类
from agent.env import CarlaEnv

# 从自定义模块导入奖励函数
from agent.rewards import reward_functions
# 从自定义模块导入工具函数
from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class

# 从配置模块导入配置字典
from config import CONFIG

# 定义TensorBoard日志保存目录
log_dir = 'tensorboard'
# 如果目录不存在则创建
os.makedirs(log_dir, exist_ok=True)
# 获取要重新加载的模型路径
reload_model = args["reload_model"]
# 获取总训练时间步数
total_timesteps = args["total_timesteps"]

# 创建一个字典，将算法名称字符串映射到对应的算法类
algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC}
# 检查配置文件中指定的算法是否在支持的算法列表中
if CONFIG["algorithm"] not in algorithm_dict:
    # 如果不在，则抛出值错误异常
    raise ValueError("Invalid algorithm name")

# 根据配置选择对应的强化学习算法类
AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]

# 初始化CARLA环境
env = CarlaEnv(host=args["host"], port=args["port"], town=args["town"],
                fps=args["fps"], obs_sensor=CONFIG["obs_sensor"], obs_res=CONFIG["obs_res"],
                    reward_fn=reward_functions[CONFIG["reward_fn"]],
                    view_res=(1120, 560), action_smoothing=CONFIG["action_smoothing"],
                    allow_spectator=True, allow_render=args["no_render"])

# 判断是否是重新加载模型进行训练
if reload_model == "":
    # 如果不是，则创建一个新的模型实例
    model = AlgorithmRL('CnnPolicy', env, verbose=2, tensorboard_log=log_dir, device='cuda',
                        **CONFIG["algorithm_params"])
    # 生成一个基于当前时间的模型后缀名
    model_suffix = f"{int(time.time())}_id{args['config']}"
else:
    # 如果是，则从指定路径加载已保存的模型
    model = AlgorithmRL.load(reload_model, env=env, device='cuda', **CONFIG["algorithm_params"])
    # 生成一个基于已加载模型ID的微调后缀名
    model_suffix = f"{reload_model.split('/')[-2].split('_')[-1]}_finetuning"

# 组合生成完整的模型名称
model_name = f'{model.__class__.__name__}_{model_suffix}'

# 拼接生成模型的保存目录路径
model_dir = os.path.join(log_dir, model_name)
# 配置模型的新日志记录器，指定输出格式
new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
# 将新配置的日志记录器设置给模型
model.set_logger(new_logger)
# 将当前配置写入JSON文件，保存在模型目录中
write_json(CONFIG, os.path.join(model_dir, 'config.json'))

# 开始训练模型
model.learn(total_timesteps=total_timesteps,
            # 传入一个包含多个回调函数的列表
            callback=[HParamCallback(CONFIG), TensorboardCallback(1), CheckpointCallback(
                # 设置检查点保存频率
                save_freq=total_timesteps // args["num_checkpoints"],
                # 设置检查点保存路径
                save_path=model_dir,
                # 设置模型文件名前缀
                name_prefix="model")],
            # 设置为False，表示不从0开始计数，而是继续之前的训练步数
            reset_num_timesteps=False)