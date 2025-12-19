#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepMind Humanoid Robot Simulation - Main Launcher
Supports: Dynamic Obstacle Avoidance + Moving Target Tracking
UTF-8 encoded, GitHub compatible, cross-platform support
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# ====================== Global Configuration (保留原有配置，不新增) ======================
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['MUJOCO_QUIET'] = '1'
os.environ['LC_ALL'] = 'en_US.UTF-8' if platform.system() != 'Windows' else ''


def setup_console_encoding():
    """Configure console for UTF-8 output (保留原有逻辑，不修改)"""
    if platform.system() == 'Windows':
        try:
            os.system("chcp 65001 > nul")
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
    else:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')


# ====================== Core Launch Functions (仅最小化适配，不新增冗余逻辑) ======================
def validate_directory_structure():
    """Validate required directory structure and files (保留原有校验逻辑，仅明确路径)"""
    # 完全保留原有目录校验逻辑，仅对齐文件名称，不新增额外检查
    project_root = Path(__file__).resolve().parent
    robot_walk_dir = project_root / "robot_walk"

    if not robot_walk_dir.exists():
        print(f"❌ Missing required directory: {robot_walk_dir}")
        print("📋 Expected structure: embodied_robot/robot_walk/")
        return False, None, None

    # 保留原有必要文件校验，不新增其他文件检查
    required_files = [
        ("Robot control script", robot_walk_dir / "move_straight.py"),
        ("Mujoco model file", robot_walk_dir / "Robot_move_straight.xml")
    ]

    missing_files = []
    for desc, file_path in required_files:
        if not file_path.exists():
            missing_files.append(f"{desc}: {file_path}")

    if missing_files:
        print("\n❌ Missing required files:")
        for missing in missing_files:
            print(f"   - {missing}")
        print("\n📋 Ensure robot_walk directory contains:")
        print("   1. move_straight.py (updated dynamic version)")
        print("   2. Robot_move_straight.xml (with dynamic targets)")
        return False, None, None

    # 保留原有输出，不新增额外信息
    print(f"✅ Directory structure validated successfully")
    print(f"   Project root: {project_root}")
    print(f"   Robot walk dir: {robot_walk_dir}")
    return True, project_root, robot_walk_dir


def check_python_environment():
    """Check Python version and required packages (恢复原有逻辑，删除无效依赖检查)"""
    # 保留原有Python版本检查
    py_version = sys.version_info
    if py_version < (3, 8):
        print(f"❌ Unsupported Python version: {py_version.major}.{py_version.minor}")
        print("   Required: Python 3.8 or higher")
        return False

    print(f"✅ Python version validated: {py_version.major}.{py_version.minor}.{py_version.micro}")

    # 恢复原有依赖检查，删除collections无效检查
    required_packages = [
        ("mujoco", "mujoco"),
        ("numpy", "numpy")
    ]

    missing_packages = []
    for pkg_import, pkg_name in required_packages:
        try:
            __import__(pkg_import)
        except ImportError:
            missing_packages.append(pkg_name)

    if missing_packages:
        print("\n❌ Missing required Python packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")

        # 保留原有自动安装逻辑，不修改
        try:
            user_input = input("\n📥 Auto-install missing packages? (y/n): ").lower().strip()
            if user_input == 'y':
                print("\n📦 Installing packages...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "pip"] + missing_packages,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding='utf-8'
                )
                print("✅ Packages installed successfully")
            elif user_input != 'n':
                print("⚠️  Invalid input - skipping auto-install")
        except subprocess.CalledProcessError as e:
            print(f"❌ Package installation failed: {e.stderr}")
            return False
        except KeyboardInterrupt:
            print("\n🛑 Input interrupted - skipping auto-install")

    return True


def launch_simulation(robot_walk_dir):
    """Launch the robot simulation (恢复原有启动逻辑，删除多余配置)"""
    script_path = robot_walk_dir / "move_straight.py"

    # 保留原有启动提示，不新增额外信息
    print("\n🚀 Launching robot simulation...")
    print("=" * 60)
    print("📌 Features:")
    print("   • Dynamic Obstacle Avoidance")
    print("   • Moving Target Tracking")
    print("   • Real-time Target Position Updates")
    print("   • Intelligent Path Planning")
    print("=" * 60)
    print("💡 Press Ctrl+C in the console to stop the simulation")
    print("=" * 60 + "\n")

    try:
        # 恢复原有环境变量，删除自定义PYTHONPATH配置
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['MUJOCO_QUIET'] = '1'

        # 保留原有启动逻辑，不新增兜底检查
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(robot_walk_dir),
            env=env,
            encoding='utf-8'
        )

        if result.returncode == 0:
            print("\n🏁 Simulation completed successfully")
        else:
            print(f"\n❌ Simulation exited with error code: {result.returncode}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Simulation failed: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error launching simulation: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main launcher function (完全恢复原有逻辑，不修改)"""
    setup_console_encoding()

    # 保留原有欢迎信息
    print("=" * 60)
    print("🤖 DeepMind Humanoid Robot Simulation Launcher")
    print("📅 Version: 2.0 (Dynamic Target + Obstacle Avoidance)")
    print("=" * 60 + "\n")

    # 保留原有三步流程，不修改
    print("🔍 Step 1/3: Validating directory structure...")
    valid_structure, project_root, robot_walk_dir = validate_directory_structure()
    if not valid_structure:
        sys.exit(1)

    print("\n🔍 Step 2/3: Checking Python environment...")
    valid_env = check_python_environment()
    if not valid_env:
        sys.exit(1)

    print("\n🔍 Step 3/3: Launching simulation...")
    launch_simulation(robot_walk_dir)

    print("\n✅ Launcher completed successfully")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Launcher interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Launcher error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)