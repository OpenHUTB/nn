import os
import sys
import subprocess
import platform
from pathlib import Path


def setup_environment():
    """
    初始化运行环境，适配实际目录结构：
    main.py 与 robot_walk 同级，脚本/模型在 robot_walk 子目录中
    """
    # 获取main.py所在目录（项目根目录：embodied_robot）
    project_root = Path(__file__).resolve().parent
    print(f"📁 项目根目录：{project_root}")

    # 定义子目录和关键文件路径（适配你的目录结构）
    robot_walk_dir = project_root / "robot_walk"
    script_file = robot_walk_dir / "move_straight.py"
    model_file = robot_walk_dir / "Robot_move_straight.xml"

    # 检查目录是否存在
    if not robot_walk_dir.exists():
        print(f"\n❌ 缺失子目录：{robot_walk_dir}")
        print("📋 请确保目录结构正确：")
        print("   embodied_robot/")
        print("   ├── main.py")
        print("   └── robot_walk/")
        print("       ├── move_straight.py")
        print("       └── Robot_move_straight.xml")
        sys.exit(1)
    print(f"✅ 找到子目录：{robot_walk_dir}")

    # 检查文件是否存在
    files_to_check = [
        ("机器人控制脚本", script_file),
        ("Mujoco模型文件", model_file)
    ]

    missing_files = []
    for file_desc, file_path in files_to_check:
        if not file_path.exists():
            missing_files.append(f"{file_desc}: {file_path}")
        else:
            print(f"✅ {file_desc} 已找到：{file_path}")

    # 如果有缺失文件，报错并退出
    if missing_files:
        print("\n❌ 缺失必要文件：")
        for missing in missing_files:
            print(f"   - {missing}")
        print("\n📋 请确保 robot_walk 目录下包含：")
        print("   1. move_straight.py (机器人控制脚本)")
        print("   2. Robot_move_straight.xml (Mujoco模型文件)")
        sys.exit(1)

    return project_root, robot_walk_dir, script_file, model_file


def get_python_executable():
    """
    获取正确的Python解释器路径（优先使用虚拟环境）
    """
    # 优先使用当前环境的Python
    python_exe = sys.executable
    print(f"\n🐍 使用Python解释器：{python_exe}")

    # 验证Python版本
    try:
        version_result = subprocess.run(
            [python_exe, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        python_version = version_result.stdout.strip()
        print(f"🔍 Python版本：{python_version}")

        # 检查是否至少是Python 3.8+（Mujoco要求）
        version_parts = python_version.split()[1].split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        if major < 3 or (major == 3 and minor < 8):
            print("⚠️  警告：Mujoco推荐使用Python 3.8+，可能存在兼容性问题")
    except Exception as e:
        print(f"⚠️  无法检测Python版本：{e}")

    return python_exe


def check_dependencies():
    """
    检查必要的依赖包是否安装
    """
    required_packages = [
        "mujoco",
        "numpy"
    ]

    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✅ 依赖包 {pkg} 已安装")
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print("\n❌ 缺失依赖包：")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 请运行以下命令安装：")
        print(f"   {sys.executable} -m pip install {' '.join(missing_packages)}")

        # 询问是否自动安装
        if input("\n📥 是否自动安装缺失的依赖包？(y/n): ").lower() == 'y':
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + missing_packages,
                    check=True
                )
                print("✅ 依赖包安装完成")
            except subprocess.CalledProcessError as e:
                print(f"❌ 依赖包安装失败：{e}")
                sys.exit(1)


def run_robot_simulation(python_exe, robot_walk_dir, script_file):
    """
    启动机器人仿真脚本（切换到robot_walk目录运行，确保路径正确）
    """
    print("\n🚀 启动机器人多目标点巡逻仿真...")
    print("=" * 50)

    try:
        # 设置环境变量（确保无日志、路径正确）
        env = os.environ.copy()
        env['MUJOCO_QUIET'] = '1'
        # 将项目根目录加入Python路径，确保脚本能正确导入模块
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + env.get('PYTHONPATH', '')

        # 切换到robot_walk目录运行脚本（关键：确保脚本能找到同目录的模型文件）
        result = subprocess.run(
            [python_exe, str(script_file)],
            cwd=str(robot_walk_dir),  # 运行目录切换到robot_walk
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True
        )

        print("=" * 50)
        print("🏁 仿真运行完成")
        return result.returncode

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 仿真运行出错，返回码：{e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n🛑 仿真被用户中断")
        return 0
    except Exception as e:
        print(f"\n❌ 未知错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """
    主启动函数
    """
    # 打印欢迎信息
    print("=" * 50)
    print("🤖 DeepMind Humanoid 机器人仿真启动器")
    print("📌 多目标点巡逻 + 动态障碍避障")
    print("=" * 50)

    # 1. 初始化环境和路径（适配你的目录结构）
    try:
        project_root, robot_walk_dir, script_file, model_file = setup_environment()
    except Exception as e:
        print(f"\n❌ 环境初始化失败：{e}")
        sys.exit(1)

    # 2. 检查Python解释器
    python_exe = get_python_executable()

    # 3. 检查依赖包
    print("\n🔍 检查依赖包...")
    check_dependencies()

    # 4. 运行仿真（切换到robot_walk目录）
    exit_code = run_robot_simulation(python_exe, robot_walk_dir, script_file)

    # 5. 退出
    sys.exit(exit_code)


if __name__ == "__main__":
    # 设置Windows控制台编码（解决中文乱码）
    if platform.system() == "Windows":
        try:
            os.system("chcp 65001 > nul")
        except:
            pass

    # 启动主程序
    main()