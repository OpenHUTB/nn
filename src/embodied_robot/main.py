#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepMind Humanoid Robot Simulation - Main Launcher
Supports: Dynamic Obstacle Avoidance + Moving Target Tracking
UTF-8 encoded, GitHub compatible, cross-platform support
适配：robot_walk目录下的move_straight.py + Robot_move_straight.xml
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil

# ====================== Global Configuration ======================
# Force UTF-8 encoding for all operations
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['MUJOCO_QUIET'] = '1'  # Disable Mujoco logs
os.environ['LC_ALL'] = 'en_US.UTF-8' if platform.system() != 'Windows' else ''


# Set console encoding (Windows fix)
def setup_console_encoding():
    """Configure console for UTF-8 output (cross-platform)"""
    if platform.system() == 'Windows':
        try:
            # Set Windows console to UTF-8
            os.system("chcp 65001 > nul")
            # Reconfigure stdout/stderr for Python 3.7+
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
    else:
        # Linux/Mac encoding setup
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')


# ====================== Core Launch Functions ======================
def validate_directory_structure():
    """Validate required directory structure and files"""
    # Fixed structure:
    # embodied_robot/
    #   ├── main.py (当前文件)
    #   └── robot_walk/ (子目录)
    #       ├── move_straight.py (稳定控制脚本)
    #       └── Robot_move_straight.xml (Mujoco模型)

    project_root = Path(__file__).resolve().parent  # embodied_robot/
    robot_walk_dir = project_root / "robot_walk"     # embodied_robot/robot_walk/

    # Check robot_walk directory exists
    if not robot_walk_dir.exists():
        print(f"❌ Missing required directory: {robot_walk_dir}")
        print("📋 Please create 'robot_walk' subdirectory under project root!")
        print(f"   Project root: {project_root}")
        return False, None, None

    # Check core files (严格匹配两个目标文件)
    required_files = [
        ("Robot stable control script", robot_walk_dir / "move_straight.py"),
        ("Mujoco humanoid model file", robot_walk_dir / "Robot_move_straight.xml")
    ]

    missing_files = []
    for desc, file_path in required_files:
        if not file_path.exists():
            missing_files.append(f"{desc}: {file_path.name} (路径: {file_path.parent})")

    if missing_files:
        print("\n❌ Missing required files in robot_walk directory:")
        for missing in missing_files:
            print(f"   - {missing}")
        print("\n📋 Please place these 2 files into robot_walk/:")
        print("   1. move_straight.py (stable robot control script)")
        print("   2. Robot_move_straight.xml (Mujoco humanoid model)")
        return False, None, None

    # All checks passed
    print(f"✅ Directory structure validated successfully")
    print(f"   Project root: {project_root}")
    print(f"   Robot control dir: {robot_walk_dir}")
    print(f"   ✔ move_straight.py exists")
    print(f"   ✔ Robot_move_straight.xml exists")
    return True, project_root, robot_walk_dir


def check_python_environment():
    """Check Python version and required packages (适配Mujoco依赖)"""
    # Check Python version (Mujoco requires 3.8+, 与控制脚本一致)
    py_version = sys.version_info
    if py_version < (3, 8):
        print(f"❌ Unsupported Python version: {py_version.major}.{py_version.minor}")
        print("   Required: Python 3.8 or higher (for Mujoco compatibility)")
        return False

    print(f"✅ Python version validated: {py_version.major}.{py_version.minor}.{py_version.micro}")

    # Check required packages (严格匹配控制脚本依赖)
    required_packages = [
        ("mujoco", "mujoco"),       # Mujoco仿真引擎
        ("numpy", "numpy"),         # 数值计算依赖
        ("collections", "collections")  # 队列依赖（Python内置，兜底检查）
    ]

    missing_packages = []
    for pkg_import, pkg_name in required_packages:
        try:
            __import__(pkg_import)
        except ImportError:
            missing_packages.append(pkg_name)

    # 过滤内置包（避免误报）
    missing_packages = [pkg for pkg in missing_packages if pkg not in ["collections"]]

    if missing_packages:
        print("\n❌ Missing required Python packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")

        # Auto-install prompt (友好适配)
        try:
            user_input = input("\n📥 Auto-install missing packages? (y/n): ").lower().strip()
            if user_input == 'y':
                print("\n📦 Installing packages (pip upgrade + missing packages)...")
                # 先升级pip，再安装依赖
                pip_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
                subprocess.run(pip_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                # 安装缺失包
                install_cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + missing_packages
                subprocess.run(install_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                print("✅ All missing packages installed successfully!")
            elif user_input != 'n':
                print("⚠️  Invalid input - skipping auto-install (please install manually)")
        except subprocess.CalledProcessError as e:
            print(f"❌ Package installation failed: {e.stderr}")
            print("💡 Please install manually with: pip install " + " ".join(missing_packages))
            return False
        except KeyboardInterrupt:
            print("\n🛑 Input interrupted - skipping auto-install")

    return True


def launch_simulation(robot_walk_dir):
    """Launch the robot simulation (完美适配robot_walk目录下的脚本)"""
    script_path = robot_walk_dir / "move_straight.py"
    model_path = robot_walk_dir / "Robot_move_straight.xml"

    # 再次确认脚本和模型存在（兜底检查）
    if not script_path.exists() or not model_path.exists():
        print("\n❌ Fatal error: Simulation files missing suddenly!")
        return

    print("\n🚀 Launching DeepMind Humanoid Simulation (Stable Version)")
    print("=" * 65)
    print("📌 Supported Features:")
    print("   • Enhanced Balance Control (Fix Fall-Down Issue)")
    print("   • Dynamic Obstacle Avoidance (3 Dynamic + 1 Fixed Obstacle)")
    print("   • Moving Patrol Target Tracking (5 Dynamic Targets)")
    print("   • Slow & Stable Gait (Prevent Imbalance)")
    print("   • Real-time COM (Center of Mass) Monitoring")
    print("=" * 65)
    print(f"📂 Simulation script: {script_path.name}")
    print(f"📂 Mujoco model: {model_path.name}")
    print(f"💡 Tip: Press Ctrl+C in console to stop simulation")
    print("=" * 65 + "\n")

    try:
        # Set environment variables for child process (继承+扩展)
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['MUJOCO_QUIET'] = '1'
        # 添加项目路径到PythonPath，确保脚本可导入依赖
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + env.get('PYTHONPATH', '')

        # 启动仿真脚本（指定工作目录为robot_walk，避免文件路径问题）
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(robot_walk_dir),  # 关键：工作目录切换到robot_walk
            env=env,
            encoding='utf-8'
        )

        # 检查退出码
        if result.returncode == 0:
            print("\n🏁 Simulation completed successfully!")
        else:
            print(f"\n❌ Simulation exited with error code: {result.returncode}")
            print("💡 Please check if move_straight.py and Robot_move_straight.xml are intact")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Simulation failed to run: {e.stderr}")
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user (Ctrl+C)")
    except FileNotFoundError:
        print(f"\n❌ Python interpreter not found: {sys.executable}")
    except Exception as e:
        print(f"\n❌ Unexpected error launching simulation: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main launcher function (流程化执行)"""
    # Step 0: Setup console encoding first
    setup_console_encoding()

    # Welcome message
    print("=" * 65)
    print("🤖 DeepMind Humanoid Robot Simulation Launcher (v2.0)")
    print("📁 Adapted for: embodied_robot/robot_walk/ directory")
    print("=" * 65 + "\n")

    # Step 1: Validate directory structure
    print("🔍 Step 1/3: Validating directory structure...")
    valid_structure, project_root, robot_walk_dir = validate_directory_structure()
    if not valid_structure:
        sys.exit(1)

    # Step 2: Check Python environment
    print("\n🔍 Step 2/3: Checking Python environment & dependencies...")
    valid_env = check_python_environment()
    if not valid_env:
        sys.exit(1)

    # Step 3: Launch simulation
    print("\n🔍 Step 3/3: Launching robot simulation...")
    launch_simulation(robot_walk_dir)

    # Exit successfully
    print("\n✅ Launcher completed all operations successfully!")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Launcher interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Launcher fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)