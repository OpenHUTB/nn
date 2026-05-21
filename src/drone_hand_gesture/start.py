#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机手势控制系统 - 一键启动脚本
跨平台支持：Windows、macOS、Linux
"""

import sys
import os
import subprocess
from pathlib import Path

def get_script_dir():
    """获取脚本所在目录"""
    return Path(__file__).resolve().parent

def check_python():
    """检查 Python 环境"""
    print("[1/3] 检查 Python 环境...")
    try:
        version = sys.version_info
        print(f"[成功] Python {version.major}.{version.minor}.{version.micro}")
        return True
    except Exception as e:
        print(f"[错误] 检查 Python 环境失败: {e}")
        return False

def check_dependencies(script_dir):
    """检查依赖"""
    print("\n[2/3] 检查依赖...")
    requirements_file = script_dir / "requirements.txt"
    
    # 检查 pygame
    try:
        import pygame
        print(f"[成功] Pygame 已安装")
    except ImportError:
        print(f"[提示] Pygame 未安装")
        if requirements_file.exists():
            print(f"[提示] 正在安装依赖...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", 
                    str(requirements_file)
                ])
                print(f"[成功] 依赖安装完成")
            except Exception as e:
                print(f"[警告] 安装依赖失败: {e}")
        else:
            print(f"[提示] 未找到 requirements.txt")
    
    return True

def show_menu():
    """显示菜单"""
    print("\n" + "="*42)
    print("    无人机手势控制系统 - 一键启动")
    print("="*42)
    print()
    print("请选择启动模式:")
    print("  1. 使用启动器（推荐）")
    print("  2. 直接运行新版仿真 (main_v2.py)")
    print("  3. 直接运行旧版仿真 (main.py)")
    print("  4. 运行 AirSim 版本 (main_airsim.py)")
    print("  5. 打开配置编辑器")
    print("  0. 退出")
    print()

def launch_launcher(script_dir):
    """启动启动器"""
    print("\n正在启动启动器...")
    os.chdir(script_dir)
    subprocess.run([sys.executable, "launcher.py"])

def launch_main_v2(script_dir):
    """启动新版仿真"""
    print("\n正在启动新版仿真...")
    os.chdir(script_dir)
    subprocess.run([sys.executable, "main_v2.py"])

def launch_main(script_dir):
    """启动旧版仿真"""
    print("\n正在启动旧版仿真...")
    os.chdir(script_dir)
    subprocess.run([sys.executable, "main.py"])

def launch_airsim(script_dir):
    """启动 AirSim 版本"""
    print("\n正在启动 AirSim 版本...")
    os.chdir(script_dir)
    subprocess.run([sys.executable, "main_airsim.py"])

def launch_config_ui(script_dir):
    """启动配置编辑器"""
    print("\n正在打开配置编辑器...")
    os.chdir(script_dir)
    subprocess.run([sys.executable, "config_ui.py"])

def main():
    script_dir = get_script_dir()
    
    print("\n" + "="*42)
    print("    无人机手势控制系统 - 一键启动")
    print("="*42)
    print()
    
    if not check_python():
        print("\n请先安装 Python！")
        input("按回车键退出...")
        return
    
    if not check_dependencies(script_dir):
        print("\n依赖检查失败，继续尝试启动...")
    
    while True:
        show_menu()
        
        try:
            choice = input("请输入选项 (0-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n退出程序")
            break
        
        if choice == "0":
            print("\n退出程序")
            break
        elif choice == "1":
            launch_launcher(script_dir)
        elif choice == "2":
            launch_main_v2(script_dir)
        elif choice == "3":
            launch_main(script_dir)
        elif choice == "4":
            launch_airsim(script_dir)
        elif choice == "5":
            launch_config_ui(script_dir)
        else:
            print("\n[错误] 无效选项，请重新选择！")
            continue
        
        break
    
    print("\n程序已退出")
    try:
        input("按回车键退出...")
    except (EOFError, KeyboardInterrupt):
        pass

if __name__ == "__main__":
    main()
