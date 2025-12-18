"""
主程序入口：整合所有模块，支持命令行参数控制
"""
import sys
import os
import argparse  # 新增：导入命令行解析模块
# 关键：将当前脚本所在目录加入Python模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config import REQUIRED_PACKAGES, PYPI_MIRROR, SUPPORTED_LANGUAGES
from utils.dependencies import install_dependencies
from core.detector import AccidentDetector

def parse_args():
    """新增：解析命令行参数"""
    parser = argparse.ArgumentParser(description="驾驶事故视频识别工具")
    # 检测源：-s/--source，支持摄像头（数字）或视频路径（字符串）
    parser.add_argument(
        "--source", "-s", 
        default=None, 
        help=f"检测源（0=摄像头，或视频路径如'test.mp4'），默认使用config.py配置"
    )
    # 语言：-l/--lang，支持中文(zh)、英文(en)
    parser.add_argument(
        "--lang", "-l", 
        default="zh", 
        choices=SUPPORTED_LANGUAGES,
        help=f"标注语言，支持{SUPPORTED_LANGUAGES}，默认中文(zh)"
    )
    return parser.parse_args()

def main():
    """主函数：执行依赖安装 → 解析参数 → 启动检测"""
    args = parse_args()  # 新增：解析命令行参数
    try:
        print("🚀 启动驾驶事故视频识别工具...")
        # 第一步：自动安装依赖
        install_dependencies(REQUIRED_PACKAGES, PYPI_MIRROR)
        # 第二步：初始化检测器
        detector = AccidentDetector()
        # 第三步：启动检测（传递命令行参数：检测源、语言）
        detector.run_detection(
            source=args.source,  # 优先使用命令行指定的检测源
            language=args.lang   # 传递语言参数
        )
    except KeyboardInterrupt:
        print("\n🛑 用户强制中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
    finally:
        print("👋 程序正常退出")

if __name__ == "__main__":
    main()
