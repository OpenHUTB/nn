"""
run_lyapunov.py - Lyapunov理论塑形训练脚本

基于原有的run.py，添加Lyapunov奖励系统支持

使用方法：
    # Lyapunov塑形训练
    python run_lyapunov.py --cfg_file tools/cfgs/your_config.yaml --agent_id 100 --reward_system lyapunov

    # 原有系统训练（对比）
    python run_lyapunov.py --cfg_file tools/cfgs/your_config.yaml --agent_id 101 --reward_system improved
"""

import os
import sys
import git
import gym
import carla_gym
import inspect
import argparse
import numpy as np
import os.path as osp
from pathlib import Path

# ⭐ 添加：设置标准输出编码为UTF-8（Windows兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentPath + '/agents/reinforcement_learning/')

import shutil
import carla_gym.envs

from stable_baselines.bench import Monitor
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.ddpg.policies import CnnPolicy as DDPGCnnPolicy
from stable_baselines.common.policies import MlpPolicy as CommonMlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy as CommonMlpLstmPolicy
from stable_baselines.common.policies import CnnPolicy as CommonCnnPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines import TRPO
from stable_baselines import A2C
from stable_baselines.common.policies import nature_cnn, sequence_1d_cnn, sequence_1d_cnn_ego_bypass_tc

from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file


def parse_args_cfgs():
    parser = argparse.ArgumentParser(description='Train with Lyapunov Shaping')

    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--env', help='environment ID', type=str, default='CarlaGymEnv-v1')

    # ⭐ 添加 baseline 选项
    parser.add_argument('--reward_system', type=str, default='improved',
                        choices=['baseline', 'improved', 'lyapunov'],
                        help='Reward system to use: baseline (Code1), improved (enhanced), or lyapunov (theory-based)')



    parser.add_argument('--log_interval', help='Log interval (model)', type=int, default=10)
    parser.add_argument('--agent_id', type=int, default=None)
    parser.add_argument('--num_timesteps', type=float, default=1e7)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=0)
    parser.add_argument('--verbosity', help='Terminal mode: 0:Off, 1:Action,Reward 2:All', default=0, type=int)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--test_model', help='test model file name', type=str, default='')
    parser.add_argument('--test_last', help='test model best or last?', action='store_true', default=False)
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic Manager TCP port to listen to (default: 8000)')
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')

    args = parser.parse_args()

    args.num_timesteps = int(args.num_timesteps)

    if args.test and args.cfg_file is None:
        path = 'logs/agent_{}/'.format(args.agent_id)
        conf_list = [cfg_file for cfg_file in os.listdir(path) if '.yaml' in cfg_file]
        args.cfg_file = path + conf_list[0]

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    if args.test:
        args.play_mode = True

    return args, cfg


if __name__ == '__main__':
    args, cfg = parse_args_cfgs()

    # 打印配置信息
    print('\n' + '=' * 80)
    print('[MULTI-REWARD TRAINING SCRIPT]')
    print('=' * 80)
    print('Reward System: {}'.format(args.reward_system.upper()))
    if args.reward_system == 'baseline':
        print('  (Reproducing Code1 original reward mechanism)')
    elif args.reward_system == 'improved':
        print('  (Enhanced reward with curriculum learning)')
    elif args.reward_system == 'lyapunov':
        print('  (Theory-driven potential shaping)')
    print('Agent ID: {}'.format(args.agent_id))
    print('Total Timesteps: {:,}'.format(args.num_timesteps))
    print('Config File: {}'.format(args.cfg_file))
    print('=' * 80 + '\n')

    print('Env is starting')

    # ⭐ 创建环境时传入reward_system参数
    from carla_gym.envs.carla_env_v1 import CarlaGymEnv
    env = CarlaGymEnv(reward_system_type=args.reward_system)

    if args.play_mode:
        env.enable_auto_render()
    env.begin_modules(args)
    n_actions = env.action_space.shape[-1]

    # info_keywords保持不变
    info_keywords = (
        'training_step','rew_total', 'rew_base',
        'rew_speed', 'rew_safety', 'rew_lane_keep',
        'rew_comfort', 'rew_efficiency', 'rew_progress',
        'rew_shaping', 'rew_milestone', 'rew_improvement',
        'milestone_count',
        'stage_id',
        'collision_penalty', 'off_road_penalty',
        'ep_total_rew', 'ep_speed_viols', 'ep_safety_viols',
        'ep_comfort_viols', 'ep_succ_lane_changes',
        'ego_speed', 'ego_acc',
        'lead_exists', 'lead_distance', 'lead_speed',
        'distance_traveled',
        'lanechange', 'lane_change_just_completed', 'lane_change_safe',
        'collision', 'off_road', 'track_finished',
        'no_leading_steps', 'episode_steps', 'no_leading_ratio',
    )

    # 策略选择
    if cfg.POLICY.NAME == 'DDPG':
        policy = {'MLP': DDPGMlpPolicy, 'CNN': DDPGCnnPolicy}
    else:
        policy = {'MLP': CommonMlpPolicy, 'LSTM': CommonMlpLstmPolicy, 'CNN': CommonCnnPolicy}

    if not args.test:  # training
        if args.agent_id is not None:
            # ⭐ 创建日志文件夹（添加奖励系统后缀）
            log_dir = 'logs/agent_{}_{}/'.format(args.agent_id, args.reward_system)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(log_dir + 'models/', exist_ok=True)
            save_path = log_dir + 'models/'

            # ⭐ 确保 Monitor 正确记录 episode 信息
            from stable_baselines.bench import Monitor

            env = Monitor(
                env,
                log_dir,
                allow_early_resets=True,
                reset_keywords=(),
                info_keywords=info_keywords
            )

            # ⭐ 添加：确保环境正确包装
            print(f"Environment wrapped with Monitor: {log_dir}")

            # log commit id and config
            try:
                repo = git.Repo(search_parent_directories=False)
                commit_id = repo.head.object.hexsha
            except git.exc.InvalidGitRepositoryError:
                print("Warning: Not a git repository, skipping git commit logging.")
                commit_id = "N/A"

            with open(log_dir + 'reproduction_info.txt', 'w', encoding='utf-8') as f:
                f.write('Git commit id: {}\n\n'.format(commit_id))
                f.write('Program arguments:\n\n{}\n\n'.format(args))
                f.write('Configuration file:\n\n{}'.format(cfg))
                # ⭐ 添加奖励系统信息
                f.write('\n\nReward System: {}\n'.format(args.reward_system))
                f.close()

            # save a copy of config file
            original_adr = currentPath + '/tools/cfgs/' + args.cfg_file.split('/')[-1]
            target_adr = log_dir + args.cfg_file.split('/')[-1]
            shutil.copyfile(original_adr, target_adr)

        else:
            save_path = 'logs/'
            env = Monitor(
                env,
                'logs/',
                allow_early_resets=True,
                reset_keywords=(),
                info_keywords=info_keywords
            )

        model_dir = save_path + '{}_final_model'.format(cfg.POLICY.NAME)

        # 创建模型
        if cfg.POLICY.NAME == 'DDPG':
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=float(cfg.POLICY.ACTION_NOISE) * np.ones(n_actions)
            )
            param_noise = AdaptiveParamNoiseSpec(
                initial_stddev=float(cfg.POLICY.PARAM_NOISE_STD),
                desired_action_stddev=float(cfg.POLICY.PARAM_NOISE_STD)
            )
            model = DDPG(
                policy[cfg.POLICY.NET],
                env,
                verbose=1,
                param_noise=param_noise,
                action_noise=action_noise,
                policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)}
            )
        elif cfg.POLICY.NAME == 'PPO2':
            model = PPO2(
                policy[cfg.POLICY.NET],
                env,
                verbose=1,
                model_dir=save_path,
                policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)}
            )
        elif cfg.POLICY.NAME == 'TRPO':
            model = TRPO(
                policy[cfg.POLICY.NET],
                env,
                verbose=1,
                model_dir=save_path,
                policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)}
            )
        elif cfg.POLICY.NAME == 'A2C':
            model = A2C(
                policy[cfg.POLICY.NET],
                env,
                verbose=1,
                model_dir=save_path,
                policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)}
            )
        else:
            print(cfg.POLICY.NAME)
            raise Exception('Algorithm name is not defined!')

        print('Model is Created')
        print('[Training with {} reward system]'.format(args.reward_system.upper()))

        try:
            print('Training Started')
            if cfg.POLICY.NAME == 'DDPG':
                model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval, save_path=save_path)
            else:
                model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval)
        finally:
            print(100 * '*')
            print('FINISHED TRAINING; saving model...')
            print(100 * '*')
            model.save(model_dir)
            env.destroy()
            print('model has been saved.')

    else:  # test
        if args.agent_id is not None:
            # ⭐ 测试时也支持指定奖励系统
            save_path = 'logs/agent_{}_{}/models/'.format(args.agent_id, args.reward_system)
            if not os.path.exists(save_path):
                # 尝试不带后缀的路径
                save_path = 'logs/agent_{}/models/'.format(args.agent_id)
        else:
            save_path = 'logs/'

        if args.test_model == '':
            best_last = 'best'
            if args.test_last:
                best_last = 'step'
            best_s = [int(best[5:-4]) for best in os.listdir(save_path) if best_last in best]
            best_s.sort()
            args.test_model = best_last + '_{}'.format(best_s[-1])

        model_dir = save_path + args.test_model
        print('{} is Loading...'.format(args.test_model))

        if cfg.POLICY.NAME == 'DDPG':
            model = DDPG.load(model_dir)
            model.action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=np.zeros(n_actions)
            )
            model.param_noise = None
        elif cfg.POLICY.NAME == 'PPO2':
            model = PPO2.load(model_dir)
        elif cfg.POLICY.NAME == 'TRPO':
            model = TRPO.load(model_dir)
        elif cfg.POLICY.NAME == 'A2C':
            model = A2C.load(model_dir)
        else:
            print(cfg.POLICY.NAME)
            raise Exception('Algorithm name is not defined!')

        print('Model is loaded')
        try:
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()
                if done:
                    obs = env.reset()
        finally:
            env.destroy()