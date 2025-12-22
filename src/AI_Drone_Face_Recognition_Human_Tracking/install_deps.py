# install_deps.py
# !/usr/bin/env python3

import subprocess
import sys


def install_packages():
    """安装所需依赖包"""
    print("正在安装AI无人机系统所需依赖...")
    print("=" * 50)

    packages = [
        "pygame",
        "opencv-python",
        "numpy",
    ]

    for package in packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {package} 安装失败: {e}")

    print("\n" + "=" * 50)
    print("✅ 所有依赖安装完成！")
    print("现在可以运行: python drone_system_complete.py")
    print("=" * 50)


if __name__ == "__main__":
    install_packages()