# install_deps.py
# !/usr/bin/env python3

import subprocess
import sys


def install_packages():
    """安装所需依赖包"""
    packages = [
        "pygame==2.5.2",
        "opencv-python==4.9.0.80",
        "numpy==1.24.4",
        "pyttsx3==2.90",
    ]

    print("正在安装依赖包...")

    for package in packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败: {e}")

    print("\n所有依赖安装完成！")
    print("现在可以运行: python main_final.py")


if __name__ == "__main__":
    install_packages()