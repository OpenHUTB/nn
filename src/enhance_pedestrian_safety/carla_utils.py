"""
CARLA 工具模块 - 用于自动查找和配置 CARLA 路径（增强版）
"""
import sys
import os
import glob
import argparse
import subprocess
import time


def find_carla_egg():
    """自动查找CARLA的egg文件（增强版）"""
    common_paths = [
        os.path.expanduser("~/carla/*"),
        os.path.expanduser("~/Desktop/carla/*"),
        os.path.expanduser("~/Documents/carla/*"),
        "/opt/carla/*",
        "C:/carla/*",
        "D:/carla/*",
        "E:/carla/*",
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "carla"),
        os.getcwd(),
        os.path.join(os.getcwd(), "carla"),
    ]

    # 添加环境变量路径
    if os.getenv("CARLA_ROOT"):
        common_paths.insert(0, os.getenv("CARLA_ROOT"))

    egg_files = []

    for path in common_paths:
        # 检查路径是否存在
        if not os.path.exists(path.replace("*", "")):
            continue

        # 搜索.egg文件
        egg_patterns = [
            os.path.join(path, "PythonAPI", "carla", "dist", "carla-*.egg"),
            os.path.join(path, "PythonAPI", "carla", "dist", "carla-*%d*.egg" % sys.version_info.major),
            os.path.join(path, "dist", "carla-*.egg"),
            os.path.join(path, "dist", "carla-*%d*.egg" % sys.version_info.major),
            os.path.join(path, "carla-*.egg"),
            os.path.join(path, "PythonAPI", "dist", "carla-*.egg"),
        ]

        for pattern in egg_patterns:
            found = glob.glob(pattern, recursive=True)
            if found:
                egg_files.extend(found)

    # 去重并排序（最新的优先）
    egg_files = list(set(egg_files))
    egg_files.sort(key=os.path.getmtime, reverse=True)

    return egg_files[0] if egg_files else None


def check_carla_server():
    """检查CARLA服务器是否运行"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 2000))
        sock.close()
        return result == 0
    except:
        return False


def start_carla_server(carla_path=None):
    """启动CARLA服务器"""
    if check_carla_server():
        print("CARLA服务器已在运行")
        return True

    print("正在启动CARLA服务器...")

    # 查找CarlaUE4可执行文件
    carla_binaries = []

    if carla_path:
        search_paths = [carla_path]
    else:
        search_paths = [
            os.path.expanduser("~/carla"),
            "/opt/carla",
            "C:/carla",
            "D:/carla",
            "E:/carla",
        ]

    for path in search_paths:
        if os.path.exists(path):
            # Linux
            linux_bin = os.path.join(path, "CarlaUE4", "Binaries", "Linux", "CarlaUE4-Linux-Shipping")
            if os.path.exists(linux_bin):
                carla_binaries.append(linux_bin)

            # Windows
            win_bin = os.path.join(path, "CarlaUE4.exe")
            if os.path.exists(win_bin):
                carla_binaries.append(win_bin)

    if not carla_binaries:
        print("未找到CARLA可执行文件")
        return False

    # 启动服务器
    try:
        cmd = [carla_binaries[0], "-quality-level=Low", "-fps=10"]
        if os.name == 'posix':  # Linux/Unix
            subprocess.Popen(cmd, start_new_session=True)
        else:  # Windows
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

        # 等待服务器启动
        print("等待CARLA服务器启动...")
        for i in range(30):  # 等待30秒
            if check_carla_server():
                print("CARLA服务器启动成功")
                time.sleep(2)  # 额外等待2秒确保完全启动
                return True
            time.sleep(1)

        print("CARLA服务器启动超时")
        return False

    except Exception as e:
        print(f"启动CARLA服务器失败: {e}")
        return False


def setup_carla_path():
    """设置CARLA路径并返回配置信息（增强版）"""
    print("\n" + "=" * 50)
    print("[1/6] 初始化CARLA环境...")
    print("=" * 50)

    path_parser = argparse.ArgumentParser(add_help=False)
    path_parser.add_argument('--carla-path', type=str, help='CARLA的egg文件路径或CARLA根目录路径')
    path_parser.add_argument('--start-server', action='store_true', help='自动启动CARLA服务器')
    path_parser.add_argument('--server-timeout', type=int, default=30, help='服务器启动超时时间（秒）')
    args_path, remaining_argv = path_parser.parse_known_args()

    carla_egg_path = None

    # 1. 首先检查命令行参数
    if args_path.carla_path:
        print(f"使用命令行指定的CARLA路径: {args_path.carla_path}")

        if os.path.isfile(args_path.carla_path) and args_path.carla_path.endswith('.egg'):
            carla_egg_path = args_path.carla_path
            egg_dir = os.path.dirname(carla_egg_path)
            if egg_dir not in sys.path:
                sys.path.append(egg_dir)
                print(f"✓ 添加egg文件目录到系统路径: {egg_dir}")

        elif os.path.isdir(args_path.carla_path):
            # 搜索.egg文件
            egg_files = glob.glob(os.path.join(args_path.carla_path, "**", "carla-*.egg"), recursive=True)

            # 优先选择与Python版本匹配的egg
            py_version = f"{sys.version_info.major}{sys.version_info.minor}"
            version_matched = [f for f in egg_files if py_version in f]

            if version_matched:
                carla_egg_path = version_matched[0]
            elif egg_files:
                carla_egg_path = egg_files[0]

            if carla_egg_path:
                egg_dir = os.path.dirname(carla_egg_path)
                if egg_dir not in sys.path:
                    sys.path.append(egg_dir)
                    print(f"✓ 添加egg文件目录到系统路径: {egg_dir}")
            else:
                # 如果目录下没有.egg文件，尝试将其添加到Python路径
                pythonapi_path = os.path.join(args_path.carla_path, "PythonAPI")
                if os.path.exists(pythonapi_path):
                    sys.path.append(pythonapi_path)
                    print(f"✓ 添加PythonAPI目录到系统路径: {pythonapi_path}")
                else:
                    sys.path.append(args_path.carla_path)
                    print(f"✓ 添加CARLA目录到系统路径: {args_path.carla_path}")

    # 2. 检查环境变量
    if not carla_egg_path and os.getenv("CARLA_PYTHON_PATH"):
        env_carla_path = os.getenv("CARLA_PYTHON_PATH")
        print(f"使用环境变量CARLA_PYTHON_PATH: {env_carla_path}")

        if os.path.isfile(env_carla_path) and env_carla_path.endswith('.egg'):
            carla_egg_path = env_carla_path
            egg_dir = os.path.dirname(carla_egg_path)
            if egg_dir not in sys.path:
                sys.path.append(egg_dir)
                print(f"✓ 通过环境变量添加egg文件目录: {egg_dir}")

        elif os.path.isdir(env_carla_path):
            egg_files = glob.glob(os.path.join(env_carla_path, "**", "carla-*.egg"), recursive=True)
            if egg_files:
                carla_egg_path = egg_files[0]
                egg_dir = os.path.dirname(carla_egg_path)
                if egg_dir not in sys.path:
                    sys.path.append(egg_dir)
                    print(f"✓ 通过环境变量添加egg文件目录: {egg_dir}")
            else:
                # 将目录添加到Python路径
                sys.path.append(env_carla_path)
                print(f"✓ 通过环境变量添加目录到系统路径: {env_carla_path}")

    # 3. 自动查找
    if not carla_egg_path:
        print("正在自动查找CARLA egg文件...")
        carla_egg_path = find_carla_egg()

        if carla_egg_path:
            egg_dir = os.path.dirname(carla_egg_path)
            if egg_dir not in sys.path:
                sys.path.append(egg_dir)
                print(f"✓ 自动找到并添加CARLA egg文件目录: {egg_dir}")
        else:
            print("⚠ 未找到CARLA egg文件，尝试继续...")

    # 4. 检查CARLA模块是否可导入
    print("\n[2/6] 检查CARLA模块...")
    try:
        import carla
        print("✓ CARLA模块导入成功")

        # 检查服务器连接
        print("\n[3/6] 检查CARLA服务器连接...")
        if args_path.start_server:
            if not start_carla_server(args_path.carla_path):
                print("✗ CARLA服务器启动失败")
                print("提示：请手动启动CARLA服务器后再运行程序")
                sys.exit(1)
        elif not check_carla_server():
            print("⚠ CARLA服务器未运行")
            print("提示：请启动CARLA服务器或使用 --start-server 参数")
            # 不退出，让用户选择是否继续

        return carla_egg_path, remaining_argv

    except ImportError as e:
        print(f"✗ CARLA模块导入失败: {e}")
        print("\n" + "=" * 50)
        print("CARLA配置指南:")
        print("=" * 50)
        print("请通过以下方式之一配置CARLA路径：")
        print("\n1. 命令行参数（推荐）：")
        print("   python main.py --carla-path <CARLA安装目录>")
        print("\n2. 环境变量：")
        print("   export CARLA_PYTHON_PATH=<CARLA安装目录>/PythonAPI")
        print("\n3. 自动查找（将CARLA放在以下目录之一）：")
        print("   ~/carla/")
        print("   ~/Desktop/carla/")
        print("   /opt/carla/")
        print("   C:/carla/")
        print("\n4. 启动CARLA服务器：")
        print("   python main.py --start-server")
        print("\n提示：确保CARLA版本与Python版本匹配")
        print("=" * 50)
        sys.exit(1)


def import_carla_module():
    """导入CARLA模块"""
    print("\n[4/6] 导入CARLA模块...")
    try:
        import carla
        print("✓ CARLA模块导入成功")

        # 打印CARLA版本信息
        if hasattr(carla, '__version__'):
            print(f"  CARLA版本: {carla.__version__}")
        else:
            print("  CARLA版本: 未知")

        # 测试基本功能
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(5.0)
            world = client.get_world()
            print(f"  CARLA服务器: 已连接 ({world.get_map().name})")
            client.reload_world(False)  # 重置但不重新生成交通
        except Exception as e:
            print(f"  CARLA服务器连接: 失败 ({str(e)[:50]})")
            print("  提示：确保CARLA服务器正在运行")

        return carla
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        sys.exit(1)


def setup_carla_environment():
    """设置完整的CARLA环境"""
    print("\n" + "=" * 50)
    print("CARLA环境设置")
    print("=" * 50)

    # 1. 设置路径
    carla_egg_path, remaining_argv = setup_carla_path()

    # 2. 导入模块
    carla_module = import_carla_module()

    # 3. 验证环境
    print("\n[5/6] 验证环境配置...")

    # 检查Python版本兼容性
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"  Python版本: {py_version}")

    # 检查必要依赖
    required_modules = ['numpy', 'PIL', 'opencv-python']
    for module in required_modules:
        try:
            __import__(module.replace('-', '_'))
            print(f"  {module}: ✓")
        except ImportError:
            print(f"  {module}: ✗ (建议安装: pip install {module})")

    # 4. 最终状态
    print("\n[6/6] 环境设置完成")
    print("-" * 30)
    print("状态汇总:")
    print(f"  CARLA路径: {carla_egg_path or '使用系统路径'}")
    print(f"  剩余参数: {remaining_argv}")
    print("=" * 50)
    print()

    return carla_module, remaining_argv


def cleanup_carla():
    """清理CARLA资源"""
    print("\n清理CARLA环境...")
    try:
        import carla
        # 这里可以添加自定义的清理逻辑
        print("✓ CARLA环境清理完成")
    except:
        print("⚠ CARLA环境清理跳过")


if __name__ == "__main__":
    # 测试函数
    carla_module, args = setup_carla_environment()
    print(f"CARLA模块: {carla_module}")
    print(f"剩余参数: {args}")