"""
CARLA 工具模块 - 用于自动查找和配置 CARLA 路径
"""
import sys
import os
import glob
import argparse


def find_carla_egg():
    """自动查找CARLA的egg文件"""
    common_paths = [
        os.path.expanduser("~/carla/*"),
        os.path.expanduser("~/Desktop/carla/*"),
        os.path.expanduser("~/Documents/carla/*"),
        "/opt/carla/*",
        "C:/carla/*",
        "D:/carla/*",
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "carla"),
    ]

    for path in common_paths:
        egg_pattern = os.path.join(path, "PythonAPI", "carla", "dist", "carla-*.egg")
        egg_files = glob.glob(egg_pattern, recursive=True)
        if egg_files:
            return egg_files[0]
        egg_pattern = os.path.join(path, "dist", "carla-*.egg")
        egg_files = glob.glob(egg_pattern, recursive=True)
        if egg_files:
            return egg_files[0]
    return None


def setup_carla_path():
    """设置CARLA路径并返回配置信息"""
    print("\n[1/6] 初始化CARLA环境...")

    path_parser = argparse.ArgumentParser(add_help=False)
    path_parser.add_argument('--carla-path', type=str, help='CARLA的egg文件路径或dist目录路径')
    args_path, remaining_argv = path_parser.parse_known_args()

    carla_egg_path = None

    if args_path.carla_path:
        if os.path.isfile(args_path.carla_path) and args_path.carla_path.endswith('.egg'):
            carla_egg_path = args_path.carla_path
        elif os.path.isdir(args_path.carla_path):
            egg_files = glob.glob(os.path.join(args_path.carla_path, "carla-*.egg"))
            if egg_files:
                carla_egg_path = egg_files[0]

        if carla_egg_path:
            sys.path.append(os.path.dirname(carla_egg_path))
            print(f"✓ 通过命令行参数加载CARLA egg文件: {carla_egg_path}")
            return carla_egg_path, remaining_argv
        else:
            print(f"✗ 命令行参数指定的路径中未找到CARLA egg文件: {args_path.carla_path}")
            sys.exit(1)

    elif os.getenv("CARLA_PYTHON_PATH"):
        env_carla_path = os.getenv("CARLA_PYTHON_PATH")
        if os.path.isfile(env_carla_path) and env_carla_path.endswith('.egg'):
            carla_egg_path = env_carla_path
        elif os.path.isdir(env_carla_path):
            egg_files = glob.glob(os.path.join(env_carla_path, "carla-*.egg"))
            if egg_files:
                carla_egg_path = egg_files[0]

        if carla_egg_path:
            sys.path.append(os.path.dirname(carla_egg_path))
            print(f"✓ 通过环境变量加载CARLA egg文件: {carla_egg_path}")
            return carla_egg_path, remaining_argv
        else:
            print(f"✗ 环境变量CARLA_PYTHON_PATH中未找到CARLA egg文件: {env_carla_path}")
            sys.exit(1)

    else:
        carla_egg_path = find_carla_egg()
        if carla_egg_path:
            sys.path.append(os.path.dirname(carla_egg_path))
            print(f"✓ 自动找到CARLA egg文件: {carla_egg_path}")
            return carla_egg_path, remaining_argv
        else:
            print("✗ 未找到CARLA egg文件！")
            print("提示：请通过以下方式之一配置CARLA路径：")
            print("  1. 命令行参数：--carla-path <CARLA的egg文件/ dist目录>")
            print("  2. 环境变量：设置CARLA_PYTHON_PATH=<CARLA的egg文件/ dist目录>")
            print("  3. 将CARLA放在用户目录carla/、桌面carla/或/opt/carla/（自动查找）")
            sys.exit(1)


def import_carla_module():
    """导入CARLA模块"""
    print("\n[2/6] 导入CARLA模块...")
    try:
        import carla
        print("✓ CARLA模块导入成功")
        return carla
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        sys.exit(1)