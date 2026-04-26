#!/usr/bin/env python3
"""
CARLA Autonomous Car - First Release
====================================

Complete autonomous car system with DDPG algorithm. This is the first release
that allows the car to drive autonomously in CARLA simulator.

Usage:
    python main.py [options]

Training:
    python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1 --train

Testing:
    python main.py --agent_id 1 --test --test_model best_100000 --play_mode 1

Quick Start:
    # 1. Start CARLA server first (terminal 1)
    ./CarlaUE4.sh -carla-server -fps=20 -world-port=2000

    # 2. Train the car (terminal 2)
    python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1 --train --num_timesteps 100000

    # 3. Test the trained car
    python main.py --agent_id 1 --test --test_model best_100000 --play_mode 1
"""

import os
import sys
import inspect
import argparse
import os.path as osp
from pathlib import Path
import time
import json
import numpy as np

# Get current path
currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))

# Add local stable_baselines to path (contains custom CNN extractors)
sys.path.insert(0, currentPath + '/agents/reinforcement_learning/')

# Import CARLA and RL modules
try:
    import tensorflow as tf
    import gym
    import carla_gym
    import carla_gym.envs
    from stable_baselines.bench import Monitor
    from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
    from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
    from stable_baselines import DDPG
    from config import cfg, cfg_from_yaml_file
    CARLA_RL_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error: CARLA RL modules not available: {e}")
    print("💡 Please make sure CARLA is installed. See README.md for installation instructions.")
    CARLA_RL_AVAILABLE = False


def parse_args():
    """Parse command line arguments for autonomous car system."""
    parser = argparse.ArgumentParser(
        description='CARLA Autonomous Car - DDPG Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Start:
  # Start CARLA server first (terminal 1)
  ./CarlaUE4.sh -carla-server -fps=20 -world-port=2000

  # Train the car (terminal 2)
  python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1 --train

  # Test the trained car
  python main.py --agent_id 1 --test --test_model best_100000 --play_mode 1

Examples:
  python main.py --help
  python main.py --list_cfgs
  python main.py --version
  python main.py --validate-config configs/experiment_baseline.yaml

  # Training with different configurations
  python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1 --train --num_timesteps 100000
  python main.py --cfg_file configs/experiment_improved.yaml --agent_id 2 --train

  # Testing with visualization
  python main.py --agent_id 1 --test --play_mode 1
  python main.py --agent_id 1 --test --play_mode 2
        """
    )

    parser.add_argument('--list_cfgs', action='store_true',
                       help='List available configuration files')
    parser.add_argument('--version', action='version',
                       version='CARLA Autonomous Car 1.0.0',
                       help='Show version information')
    parser.add_argument('--cfg_dir', type=str, default='configs',
                       help='Configuration directory (default: configs)')
    parser.add_argument('--validate-config', type=str, metavar='FILE',
                       help='Validate a configuration file')

    # Training/Testing modes
    parser.add_argument('--train', action='store_true',
                       help='Run in training mode')
    parser.add_argument('--test', action='store_true',
                       help='Run in testing mode')
    parser.add_argument('--cfg_file', type=str, default=None,
                       help='Configuration file for training/testing')
    parser.add_argument('--agent_id', type=int, default=None,
                       help='Agent ID for logging and model saving')
    parser.add_argument('--test_model', type=str, default='',
                       help='Test model file name (without extension)')
    parser.add_argument('--play_mode', type=int, default=0,
                       help='Display mode: 0:off, 1:2D, 2:3D (default: 0)')
    parser.add_argument('--num_timesteps', type=int, default=int(1e5),
                       help='Number of training timesteps (default: 1e5)')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Log interval during training (default: 100)')

    return parser.parse_args()


def validate_config(config_file):
    """Validate a YAML configuration file."""
    print(f"\n🔍 Validating configuration: {config_file}")

    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return None

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print("✅ Configuration file syntax is valid")

        # Check required sections
        required_sections = ['POLICY', 'ENV', 'TRAINING']
        for section in required_sections:
            if section in config:
                print(f"✅ Found required section: {section}")
            else:
                print(f"⚠️  Missing section: {section}")

        # Show some key values
        print(f"\n📊 Configuration Summary:")
        if 'POLICY' in config and 'NAME' in config['POLICY']:
            print(f"  Algorithm: {config['POLICY']['NAME']}")
        if 'ENV' in config and 'NAME' in config['ENV']:
            print(f"  Environment: {config['ENV']['NAME']}")
        if 'TRAINING' in config and 'TIMESTEPS' in config['TRAINING']:
            print(f"  Timesteps: {config['TRAINING']['TIMESTEPS']:,}")

        return config

    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return None

def list_config_files(cfg_dir):
    """List available configuration files."""
    if not os.path.exists(cfg_dir):
        print(f"❌ Config directory not found: {cfg_dir}")
        return False

    yaml_files = [f for f in os.listdir(cfg_dir) if f.endswith('.yaml')]

    if not yaml_files:
        print(f"❌ No configuration files found in {cfg_dir}")
        return False

    print(f"\n📋 Available Configuration Files in '{cfg_dir}':")
    print("=" * 60)
    for i, cfg_file in enumerate(sorted(yaml_files), 1):
        print(f"  {i}. {cfg_file}")
    print("=" * 60)
    return True


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
    """Main entry point for CARLA autonomous car system."""
    print("\n" + "=" * 60)
    print("🚗 CARLA AUTONOMOUS CAR SYSTEM - v1.0.0")
    print("=" * 60 + "\n")

    args = parse_args()

    # Basic utilities
    if args.list_cfgs:
        list_config_files(args.cfg_dir)
        return

    if args.validate_config:
        validate_config(args.validate_config)
        return

    # Check CARLA availability
    if not CARLA_RL_AVAILABLE:
        print("❌ Cannot run: CARLA RL modules not available")
        print("💡 Please install CARLA first:")
        print("   https://github.com/carla-simulator/carla/releases")
        return

    # Determine mode
    if args.train:
        if not args.cfg_file:
            print("❌ Configuration file required for training")
            print("💡 Use --cfg_file to specify a configuration file")
            print("💡 Use --list_cfgs to see available configs")
            return

        # Load configuration
        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.TAG = Path(args.cfg_file).stem

        # Run training
        try:
            env = gym.make(cfg.ENV.NAME)
            env.begin_modules(args)
            run_training(args, cfg, env)
        except Exception as e:
            print(f"❌ Failed to start environment: {e}")
            import traceback
            traceback.print_exc()

    elif args.test:
        if not args.agent_id:
            print("❌ Agent ID required for testing")
            print("💡 Use --agent_id to specify the trained agent")
            return

        # Auto-detect config if not specified
        if not args.cfg_file:
            path = 'logs/agent_{}/'.format(args.agent_id)
            if os.path.exists(path):
                conf_list = [f for f in os.listdir(path) if f.endswith('.yaml')]
                if conf_list:
                    args.cfg_file = path + conf_list[0]
                    print(f"🤖 Auto-detected config: {args.cfg_file}")

        if not args.cfg_file:
            print("❌ Configuration file not found")
            print("💡 Use --cfg_file to specify a configuration file")
            return

        # Load configuration
        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.TAG = Path(args.cfg_file).stem

        # Run testing
        try:
            env = gym.make(cfg.ENV.NAME)
            if args.play_mode:
                env.enable_auto_render()
            env.begin_modules(args)
            run_testing(args, cfg, env)
        except Exception as e:
            print(f"❌ Failed to start environment: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("💡 No mode specified. Use --train or --test")
        print("\nAvailable commands:")
        print("  --train              Run training mode")
        print("  --test               Run testing mode")
        print("  --list_cfgs          List configuration files")
        print("  --validate-config    Validate a config file")
        print("  --help               Show detailed help")


if __name__ == '__main__':
    main()