# -*- coding: utf-8 -*-  # 强制声明文件编码为UTF-8
# 或简化版：
# coding=utf-8
import os
import sys
import subprocess
import platform
from pathlib import Path


def setup_environment():
    """
    Initialize runtime environment - adapt to directory structure:
    main.py is at the same level as robot_walk, scripts/models in robot_walk subdirectory
    """
    # Get project root (directory of main.py)
    project_root = Path(__file__).resolve().parent
    print(f"📁 Project root directory: {project_root}")

    # Define paths
    robot_walk_dir = project_root / "robot_walk"
    script_file = robot_walk_dir / "move_straight.py"
    model_file = robot_walk_dir / "Robot_move_straight.xml"

    # Check subdirectory existence
    if not robot_walk_dir.exists():
        print(f"\n❌ Missing subdirectory: {robot_walk_dir}")
        print("📋 Required directory structure:")
        print("   embodied_robot/")
        print("   ├── main.py")
        print("   └── robot_walk/")
        print("       ├── move_straight.py")
        print("       └── Robot_move_straight.xml")
        sys.exit(1)
    print(f"✅ Found subdirectory: {robot_walk_dir}")

    # Check file existence
    files_to_check = [
        ("Robot control script", script_file),
        ("Mujoco model file", model_file)
    ]

    missing_files = []
    for file_desc, file_path in files_to_check:
        if not file_path.exists():
            missing_files.append(f"{file_desc}: {file_path}")
        else:
            print(f"✅ {file_desc} found: {file_path}")

    # Handle missing files
    if missing_files:
        print("\n❌ Missing required files:")
        for missing in missing_files:
            print(f"   - {missing}")
        print("\n📋 Ensure robot_walk directory contains:")
        print("   1. move_straight.py (Robot control script)")
        print("   2. Robot_move_straight.xml (Mujoco model file)")
        sys.exit(1)

    return project_root, robot_walk_dir, script_file, model_file


def get_python_executable():
    """
    Get correct Python interpreter path (priority to virtual environment)
    """
    python_exe = sys.executable
    print(f"\n🐍 Using Python interpreter: {python_exe}")

    # Verify Python version
    try:
        version_result = subprocess.run(
            [python_exe, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        python_version = version_result.stdout.strip()
        print(f"🔍 Python version: {python_version}")

        # Check minimum version (3.8+)
        version_parts = python_version.split()[1].split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        if major < 3 or (major == 3 and minor < 8):
            print("⚠️  Warning: Mujoco recommends Python 3.8+, compatibility issues may occur")
    except Exception as e:
        print(f"⚠️  Failed to detect Python version: {e}")

    return python_exe


def check_dependencies():
    """
    Check required packages installation
    """
    required_packages = [
        "mujoco",
        "numpy"
    ]

    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✅ Package {pkg} is installed")
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print("\n❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Install missing packages with:")
        print(f"   {sys.executable} -m pip install {' '.join(missing_packages)}")

        # Ask for auto-install
        if input("\n📥 Auto-install missing packages? (y/n): ").lower() == 'y':
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + missing_packages,
                    check=True
                )
                print("✅ Packages installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Package installation failed: {e}")
                sys.exit(1)


def run_robot_simulation(python_exe, robot_walk_dir, script_file,extra_args):
    """
    Launch robot simulation script (run in robot_walk directory for correct path resolution)
    """
    print("\n🚀 Starting robot patrol simulation...")
    print("=" * 50)

    try:
        # Set environment variables (no logs)
        env = os.environ.copy()
        env['MUJOCO_QUIET'] = '1'
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + env.get('PYTHONPATH', '')

        # 核心修改点：将 extra_args 拼接到启动命令列表中
        cmd = [python_exe, str(script_file)] + extra_args

        # Run script in robot_walk directory
        result = subprocess.run(
            cmd,  # 使用拼接好的命令
            cwd=str(robot_walk_dir),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True
        )

        print("=" * 50)
        print("🏁 Simulation completed successfully")
        return result.returncode

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Simulation error, return code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """
    Main launcher function
    """

    print("=" * 50)
    print("🤖 DeepMind Humanoid Robot Simulation Launcher")
    print("📌 Multi-target patrol + Dynamic obstacle avoidance")
    print("=" * 50)

    # 1. Setup environment
    try:
        project_root, robot_walk_dir, script_file, model_file = setup_environment()
    except Exception as e:
        print(f"\n❌ Environment initialization failed: {e}")
        sys.exit(1)

    # 2. Get Python executable
    python_exe = get_python_executable()

    # 3. Check dependencies
    print("\n🔍 Checking dependencies...")
    check_dependencies()

    # 核心修改点：获取 sys.argv 中除了脚本名(main.py)之外的所有参数
    extra_args = sys.argv[1:]
    if extra_args:
        print(f"📥 Received passthrough arguments: {' '.join(extra_args)}")

    # 4. Run simulation
    exit_code = run_robot_simulation(python_exe, robot_walk_dir, script_file,extra_args)

    # 5. Exit
    sys.exit(exit_code)


if __name__ == "__main__":
    # Fix Windows encoding issues
    if platform.system() == "Windows":
        try:
            os.system("chcp 65001 > nul")
        except:
            pass

    # Launch main program
    main()