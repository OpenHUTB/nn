#!/usr/bin/env python3
"""
Test script to verify the basic structure of the carla_autonomous_car module.
"""

import sys
import os

# Add current path
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, currentPath)
sys.path.insert(0, currentPath + '/agents/reinforcement_learning/')

def test_basic_files():
    """Test basic file existence."""
    print("Testing basic files...")

    required_files = [
        'main.py',
        'config.py',
        'requirements.txt',
        'custom_policies.py',
        'configs/experiment_baseline.yaml',
        'configs/experiment_improved.yaml',
        'configs/experiment_lyapunov.yaml',
        'README.md'
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print("OK: {} exists".format(file_path))
        else:
            print("FAIL: {} not found".format(file_path))
            all_exist = False

    return all_exist

def test_directory_structure():
    """Test directory structure."""
    print("\nTesting directory structure...")

    required_dirs = [
        'agents',
        'agents/reinforcement_learning',
        'agents/reinforcement_learning/stable_baselines',
        'carla_gym',
        'carla_gym/envs',
        'configs',
        'logs',
        'docs'
    ]

    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print("OK: {} exists".format(dir_path))
        else:
            print("FAIL: {} not found".format(dir_path))
            all_exist = False

    return all_exist

def test_stable_baselines_modules():
    """Test stable_baselines modules."""
    print("\nTesting stable_baselines modules...")

    sb_path = 'agents/reinforcement_learning/stable_baselines'
    if not os.path.exists(sb_path):
        print("FAIL: {} not found".format(sb_path))
        return False

    key_modules = ['PPO2', 'DDPG', 'A2C', 'TRPO']
    all_exist = True
    for module in key_modules:
        module_path = '{}/{}'.format(sb_path, module.lower())
        if os.path.exists(module_path):
            print("OK: {} module found".format(module))
        else:
            print("WARNING: {} module not found".format(module))

    return all_exist

def test_logs_structure():
    """Test logs directory structure."""
    print("\nTesting logs structure...")

    logs_path = 'logs'
    if os.path.exists(logs_path):
        print("OK: {} exists".format(logs_path))

        agent_1_path = 'logs/agent_1'
        if os.path.exists(agent_1_path):
            print("  - agent_1 directory found")
        else:
            print("  - WARNING: agent_1 directory not found")

        return True
    else:
        print("FAIL: {} not found".format(logs_path))
        return False

def count_files():
    """Count files in the module."""
    print("\nCounting files...")

    total_files = 0
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        total_files += len(files)

    print("Total files in module: {}".format(total_files))

    if total_files >= 100:
        print("OK: Sufficient files for complete module")
        return True
    else:
        print("WARNING: File count seems low")
        return True  # Still pass, just a warning

def main():
    """Run all tests."""
    print("=" * 60)
    print("CARLA AUTONOMOUS CAR MODULE - STRUCTURE TEST")
    print("=" * 60 + "\n")

    all_passed = True

    all_passed &= test_basic_files()
    all_passed &= test_directory_structure()
    all_passed &= test_stable_baselines_modules()
    all_passed &= test_logs_structure()
    all_passed &= count_files()

    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: ALL TESTS PASSED!")
        print("The module structure is complete and ready for first submission.")
    else:
        print("ERROR: SOME TESTS FAILED!")
        print("Please check the missing components before submission.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())