#!/usr/bin/env python3
"""
CARLA Frenet Trajectory Planning - Main Entry Point
===================================================

This is the main entry point for running the CARLA RL-based Frenet trajectory planning project.
It supports training and testing of various reinforcement learning algorithms.

Usage:
    python main.py [options]

Training:
    python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1

Testing:
    python main.py --test --agent_id 1 --test_model best_1000000

For more options:
    python main.py --help
"""

import os
import sys
import inspect
import argparse
import os.path as osp
from pathlib import Path

# Get current path
currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))

# Add project root and local stable_baselines to path
# For the carla_autonomous_car module, we need to add the agents path
sys.path.insert(0, currentPath)
sys.path.insert(0, currentPath + '/agents/reinforcement_learning/')

# Default availability flags
CARLA_AVAILABLE = False
CONFIG_AVAILABLE = False

def import_modules():
    """Import CARLA and RL modules dynamically."""
    global CARLA_AVAILABLE, CONFIG_AVAILABLE
    global gym, carla_gym, carla_gym_envs, Monitor
    global DDPGMlpPolicy, DDPGCnnPolicy, CommonMlpPolicy, CommonMlpLstmPolicy, CommonCnnPolicy
    global NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
    global DDPG, PPO2, TRPO, A2C
    global nature_cnn, sequence_1d_cnn, sequence_1d_cnn_ego_bypass_tc
    global cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
    global np, datetime, json, shutil

    try:
        import numpy as np
        from datetime import datetime
        import json
        import shutil

        import gym
        import carla_gym
        import carla_gym.envs
        from stable_baselines.bench import Monitor
        from stable_baselines.common.policies import MlpPolicy as CommonMlpPolicy
        from stable_baselines.common.policies import MlpLstmPolicy as CommonMlpLstmPolicy
        from stable_baselines.common.policies import CnnPolicy as CommonCnnPolicy
        from stable_baselines import PPO2
        # Import CNN extractors for use with eval() in config
        from stable_baselines.common.policies import nature_cnn, sequence_1d_cnn, sequence_1d_cnn_ego_bypass_tc

        CARLA_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import CARLA modules: {e}")
        print("Please make sure CARLA is installed. See README.md for installation instructions.")
        CARLA_AVAILABLE = False
        return False

    try:
        from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
        CONFIG_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import config module: {e}")
        CONFIG_AVAILABLE = False
        return False

    return True


def parse_args_cfgs(cfg):
    """Parse command line arguments and configuration files."""
    parser = argparse.ArgumentParser(
        description='CARLA RL Frenet Trajectory Planning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training with baseline configuration
  python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1

  # Training with improved configuration
  python main.py --cfg_file configs/experiment_improved.yaml --agent_id 2

  # Training with Lyapunov configuration
  python main.py --cfg_file configs/experiment_lyapunov.yaml --agent_id 3

  # Testing with saved model
  python main.py --test --agent_id 1 --test_model best_1000000

  # Testing with visualization
  python main.py --test --agent_id 1 --play_mode 1
        """
    )

    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--env', help='environment ID', type=str, default='CarlaGymEnv-v1')
    parser.add_argument('--log_interval', help='Log interval (model)', type=int, default=100)
    parser.add_argument('--agent_id', type=int, default=None, help='Agent ID for logging and model saving')
    parser.add_argument('--num_timesteps', type=float, default=1e7, help='Number of training timesteps')
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=0)
    parser.add_argument('--verbosity', help='Terminal mode: 0:Off, 1:Action,Reward 2:All', default=0, type=int)
    parser.add_argument('--test', default=False, action='store_true', help='Run in test mode')
    parser.add_argument('--test_model', help='test model file name', type=str, default='')
    parser.add_argument('--test_last', help='test model best or last?', action='store_true', default=False)
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic Manager TCP port to listen to (default: 8000)')
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    parser.add_argument('--list_cfgs', action='store_true', help='List available configuration files')

    args = parser.parse_args()

    args.num_timesteps = int(args.num_timesteps)

    # Auto-detect config file for testing if not specified
    if args.test and args.cfg_file is None and args.agent_id is not None:
        path = 'logs/agent_{}/'.format(args.agent_id)
        if os.path.exists(path):
            conf_list = [cfg_file for cfg_file in os.listdir(path) if cfg_file.endswith('.yaml')]
            if conf_list:
                args.cfg_file = path + conf_list[0]
                print(f"Auto-detected config file: {args.cfg_file}")

    if args.cfg_file:
        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.TAG = Path(args.cfg_file).stem
        cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # visualize all test scenarios
    if args.test:
        args.play_mode = True

    return args, cfg


def list_config_files():
    """List available configuration files."""
    cfg_dir = 'configs'
    if not os.path.exists(cfg_dir):
        print(f"Config directory not found: {cfg_dir}")
        return

    print("\nAvailable Configuration Files:")
    print("=" * 50)
    for cfg_file in sorted(os.listdir(cfg_dir)):
        if cfg_file.endswith('.yaml'):
            print(f"  - {cfg_file}")
    print("=" * 50)


def save_lane_change_stats_to_file(env, agent_id, phase='training'):
    """Save lane change statistics to JSON file"""
    if agent_id is None:
        return

    try:
        stats = env.get_lane_change_stats()
        stats_file = f'logs/agent_{agent_id}/lane_change_stats_{phase}.json'

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        print(f'✅ Lane change statistics saved to: {stats_file}')
    except Exception as e:
        print(f"Warning: Could not save lane change statistics: {e}")


def print_training_header(args, cfg):
    """Print formatted training header"""
    print('\n' + '=' * 80)
    print('CARLA RL FRENET TRAJECTORY PLANNING - TRAINING')
    print('=' * 80)
    print(f'Algorithm:        {cfg.POLICY.NAME}')
    print(f'Agent ID:         {args.agent_id}')
    print(f'Total Timesteps:  {args.num_timesteps:,}')
    print(f'Config File:      {args.cfg_file}')
    print('=' * 80 + '\n')


def print_testing_header(args, cfg):
    """Print formatted testing header"""
    print('\n' + '=' * 80)
    print('CARLA RL FRENET TRAJECTORY PLANNING - TESTING')
    print('=' * 80)
    print(f'Algorithm:        {cfg.POLICY.NAME}')
    print(f'Agent ID:         {args.agent_id}')
    print(f'Model File:       {args.test_model}')
    print(f'Config File:      {args.cfg_file}')
    print('=' * 80 + '\n')


def create_model(args, cfg, env, n_actions, save_path):
    """Create RL model based on configuration."""
    # Policy selection
    if cfg.POLICY.NAME == 'DDPG':
        policy = {'MLP': DDPGMlpPolicy, 'CNN': DDPGCnnPolicy}   # DDPG does not have LSTM policy
    else:
        policy = {'MLP': CommonMlpPolicy, 'LSTM': CommonMlpLstmPolicy, 'CNN': CommonCnnPolicy}

    # Model creation
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
        print(f"Unsupported algorithm: {cfg.POLICY.NAME}")
        raise Exception('Algorithm name is not defined!')

    return model


def load_model(args, cfg, env, n_actions, save_path):
    """Load saved RL model for testing."""
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

    return model


def setup_training_environment(args, cfg, env):
    """Setup environment for training with proper monitoring."""
    if args.agent_id is not None:
        # Create log folder
        os.makedirs(currentPath + '/logs/agent_{}/'.format(args.agent_id), exist_ok=True)
        os.makedirs(currentPath + '/logs/agent_{}/models/'.format(args.agent_id), exist_ok=True)
        save_path = 'logs/agent_{}/models/'.format(args.agent_id)

        # Extended info keywords for better monitoring
        info_keywords = (
            'training_step', 'rew_total', 'rew_base',
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

        env = Monitor(
            env,
            'logs/agent_{}/'.format(args.agent_id),
            allow_early_resets=True,
            reset_keywords=(),
            info_keywords=info_keywords
        )

        # Log git commit info
        try:
            import git
            repo = git.Repo(search_parent_directories=False)
            commit_id = repo.head.object.hexsha
        except ImportError:
            print("Warning: GitPython not available, skipping git commit logging.")
            commit_id = "N/A"
        except git.exc.InvalidGitRepositoryError:
            print("Warning: Not a git repository, skipping git commit logging.")
            commit_id = "N/A"

        # Save reproduction info
        training_start_time = datetime.now()
        with open('logs/agent_{}/reproduction_info.txt'.format(args.agent_id), 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('TRAINING CONFIGURATION\n')
            f.write('=' * 80 + '\n\n')
            f.write('Algorithm:     {}\n'.format(cfg.POLICY.NAME))
            f.write('Agent ID:      {}\n'.format(args.agent_id))
            f.write('Total Steps:   {:,}\n'.format(args.num_timesteps))
            f.write('Start Time:    {}\n\n'.format(training_start_time.strftime('%Y-%m-%d %H:%M:%S')))
            f.write('Git commit id: {}\n\n'.format(commit_id))
            f.write('Program arguments:\n\n{}\n\n'.format(args))
            f.write('Configuration file:\n\n{}'.format(cfg))

        # Save a copy of config file
        if args.cfg_file:
            original_adr = currentPath + '/' + args.cfg_file
            target_adr = currentPath + '/logs/agent_{}/'.format(args.agent_id) + args.cfg_file.split('/')[-1]
            shutil.copyfile(original_adr, target_adr)

    else:
        save_path = 'logs/'
        env = Monitor(env, 'logs/', info_keywords=('reserved',))

    return env, save_path


def run_training(args, cfg, env):
    """Run training loop."""
    print_training_header(args, cfg)

    n_actions = env.action_space.shape[-1]
    env, save_path = setup_training_environment(args, cfg, env)

    model = create_model(args, cfg, env, n_actions, save_path)
    model_dir = save_path + '{}_final_model'.format(cfg.POLICY.NAME)

    print('Model is Created')
    print(f'Training with {cfg.POLICY.NAME}\n')

    training_start_time = datetime.now()

    try:
        print('=' * 80)
        print('TRAINING STARTED')
        print('=' * 80 + '\n')

        if cfg.POLICY.NAME == 'DDPG':
            model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval, save_path=save_path)
        else:
            model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval)

    finally:
        # Training finished
        training_end_time = datetime.now()
        training_duration = training_end_time - training_start_time

        print('\n' + '=' * 80)
        print('TRAINING FINISHED')
        print('=' * 80)
        print(f'Training Duration: {training_duration}')
        print(f'Start Time:        {training_start_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'End Time:          {training_end_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 80 + '\n')

        # Save model
        print('Saving model...')
        model.save(model_dir)
        print(f'✅ Model saved to: {model_dir}\n')

        # Save training summary
        if args.agent_id is not None:
            training_summary = {
                'algorithm': cfg.POLICY.NAME,
                'agent_id': args.agent_id,
                'total_timesteps': args.num_timesteps,
                'training_start': training_start_time.isoformat(),
                'training_end': training_end_time.isoformat(),
                'training_duration_seconds': training_duration.total_seconds(),
            }

            summary_file = f'logs/agent_{args.agent_id}/training_summary.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(training_summary, f, indent=2)
            print(f'✅ Training summary saved to: {summary_file}\n')

            # Save lane change stats
            save_lane_change_stats_to_file(env, args.agent_id, 'training')

        # Destroy environment
        env.destroy()

        print('\n' + '=' * 80)
        print('ALL DONE!')
        print('=' * 80 + '\n')


def run_testing(args, cfg, env):
    """Run testing loop."""
    print_testing_header(args, cfg)

    n_actions = env.action_space.shape[-1]

    # Determine model path
    if args.agent_id is not None:
        save_path = 'logs/agent_{}/models/'.format(args.agent_id)
    else:
        save_path = 'logs/'

    # Auto-select model if not specified
    if args.test_model == '':
        best_last = 'best'
        if args.test_last:
            best_last = 'step'

        if os.path.exists(save_path):
            best_s = [int(best[5:-4]) for best in os.listdir(save_path) if best_last in best and best.endswith('.pkl')]
            best_s.sort()
            if best_s:
                args.test_model = best_last + '_{}'.format(best_s[-1])
                print(f"Auto-selected model: {args.test_model}")

    model = load_model(args, cfg, env, n_actions, save_path)

    print('Model is loaded')
    print(f'\n{"=" * 80}')
    print('TESTING STARTED')
    print(f'{"=" * 80}\n')

    try:
        obs = env.reset()
        episode_count = 0

        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()

            if done:
                episode_count += 1
                print(f'\n{"=" * 60}')
                print(f'Episode {episode_count} finished')
                print(f'{"=" * 60}\n')
                obs = env.reset()

    except KeyboardInterrupt:
        print('\n\nTesting interrupted by user')

    finally:
        print(f'\n{"=" * 80}')
        print('TESTING FINISHED')
        print(f'{"=" * 80}')
        print(f'Total Episodes: {episode_count}\n')

        # Save test statistics
        print('Collecting lane change statistics...')
        save_lane_change_stats_to_file(env, args.agent_id, 'testing')

        # Destroy environment
        env.destroy()

        print(f'\n{"=" * 80}')
        print('ALL DONE!')
        print(f'{"=" * 80}\n')


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("CARLA FRENET TRAJECTORY PLANNING - MAIN ENTRY")
    print("=" * 80 + "\n")

    # Import modules first
    if not import_modules():
        print("\nFailed to import required modules.")
        print("Cannot continue without CARLA and config modules.")
        return

    # Check config availability
    try:
        from config import cfg
    except ImportError:
        print("ERROR: Cannot import config module!")
        return

    args, cfg = parse_args_cfgs(cfg)

    # List configs if requested
    if args.list_cfgs:
        list_config_files()
        return

    if not CARLA_AVAILABLE:
        print("ERROR: CARLA modules are not available!")
        print("Please make sure CARLA is installed. See README.md for installation instructions.")
        return

    if not args.cfg_file and not args.test:
        print("ERROR: Configuration file is required for training!")
        print("Use --cfg_file to specify a configuration file or --list_cfgs to see available configs.")
        print("\nExample:")
        print("  python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1")
        return

    print('Env is starting...')
    try:
        env = gym.make(args.env)
        if args.play_mode:
            env.enable_auto_render()
        env.begin_modules(args)
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return

    try:
        if args.test:
            run_testing(args, cfg, env)
        else:
            run_training(args, cfg, env)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        if 'env' in locals():
            env.destroy()


if __name__ == '__main__':
    main()