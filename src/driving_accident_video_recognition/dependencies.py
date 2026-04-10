"""
依赖管理工具：自动检查并安装缺失的依赖包，包含容错逻辑
"""
import sys
import subprocess

def install_dependencies(packages, mirror):
    """
    自动安装依赖包
    :param packages: 依赖包列表（来自config.py）
    :param mirror: PyPI镜像源（来自config.py）
    """
    for package in packages:
        try:
            # 提取包名（去掉版本号）
            package_name = package.split('>=')[0]
            # 检查包是否已安装
            if package_name == "ultralytics":
                import ultralytics
            elif package_name == "opencv-python":
                import cv2
            elif package_name == "numpy":
                import numpy
            elif package_name == "torch":
                import torch
            print(f"✅ 依赖包 {package_name} 已安装")
        except ImportError:
            print(f"⚠️ 缺少依赖包 {package_name}，正在自动安装...")
            try:
                # 使用镜像源安装，设置超时
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-i", mirror, package],
                    timeout=300  # 5分钟超时
                )
                print(f"✅ 依赖包 {package_name} 安装成功")
            except subprocess.TimeoutExpired:
                print(f"❌ 安装 {package_name} 超时，尝试直接安装...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except Exception as e:
                print(f"❌ 安装 {package_name} 失败：{e}")
                sys.exit(1)

# 供外部导入的函数
__all__ = ["install_dependencies"]